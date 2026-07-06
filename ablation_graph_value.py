"""
ablation_graph_value.py — Does the k-NN graph term actually add value, or is
fit_quality mostly driven by each ticker's own autocorrelation (W_self)?

Trains two models on IDENTICAL data, identical train/test day split, and
identical random seed (so initialization is the same) — differing in
exactly ONE thing:

    Model A (graph)    : NeighborAvg(t) = A(t) @ X(t)     (real k-NN graph)
    Model B (no graph) : NeighborAvg(t) = 0                (self-only, same
                                                             architecture,
                                                             neighbor term
                                                             forced dead)

The difference in OUT-OF-SAMPLE (held-out days, chronological split) R^2
between A and B is the graph's genuine value-add. In-sample-only
comparisons can be misleading: either model could simply be fitting noise
in-sample, so this focuses on the held-out test days specifically.

Usage:
    python ablation_graph_value.py --universe FI_COMMODITIES --window 504
    python ablation_graph_value.py --universe EQUITY_SECTORS --window 252

Each run pushes its result to HF as
results/ablation_{universe}_{window}d_{date}.json for the "🔬 Graph Value
Ablation" dashboard tab. Re-running the same universe/window combination
adds a new dated file rather than overwriting — the dashboard groups by
(universe, window) and shows the most recent run for each.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_manager
import push_results
from tnn_engine import build_knn_adjacency, TNNModel


def build_days(prices_a: pd.DataFrame, window: int):
    M, H = config.ROLLING_FEATURE_WINDOW, config.PRED_HORIZON
    log_ret_full = np.log(prices_a / prices_a.shift(1)).dropna()
    log_ret = log_ret_full.iloc[-window:]
    R = log_ret.values
    T = len(R)
    ret_mu, ret_sd = R.mean(), R.std() + 1e-8
    R_norm = (R - ret_mu) / ret_sd

    X_days, A_days, Y_days = [], [], []
    for t in range(M - 1, T - H):
        X_t = R_norm[t - M + 1: t + 1, :].T
        A_t, _ = build_knn_adjacency(X_t, config.K_NEIGHBORS)
        Y_t = R_norm[t + 1: t + 1 + H, :].mean(axis=0)
        X_days.append(X_t)
        A_days.append(A_t)
        Y_days.append(Y_t)
    return X_days, A_days, Y_days


def train_model(X_days, A_days, Y_days, use_graph: bool, rng, epochs, lr, batch_days):
    in_dim = X_days[0].shape[1]
    model = TNNModel(in_dim, rng)
    state = model.init_adam()
    n_days = len(X_days)
    step = 0

    for epoch in range(epochs):
        idx = rng.permutation(n_days)
        for i in range(0, n_days, batch_days):
            day_idx = idx[i:i + batch_days]
            if len(day_idx) < 1:
                continue

            batch_grads = None
            for d in day_idx:
                X_t, A_t, Y_t = X_days[d], A_days[d], Y_days[d]
                A_eff = A_t if use_graph else np.zeros_like(A_t)
                pred = model.forward(X_t, A_eff)
                resid = pred - Y_t
                dpred = 2.0 * resid / len(resid)
                grads = model.backward(dpred)

                if batch_grads is None:
                    batch_grads = grads
                else:
                    batch_grads["gcn"]["W_self"] = (
                        batch_grads["gcn"]["W_self"][0] + grads["gcn"]["W_self"][0],
                        batch_grads["gcn"]["W_self"][1] + grads["gcn"]["W_self"][1])
                    batch_grads["gcn"]["W_neigh"] = (
                        batch_grads["gcn"]["W_neigh"][0] + grads["gcn"]["W_neigh"][0],
                        batch_grads["gcn"]["W_neigh"][1] + grads["gcn"]["W_neigh"][1])
                    batch_grads["head"] = (
                        batch_grads["head"][0] + grads["head"][0],
                        batch_grads["head"][1] + grads["head"][1])

            n_b = len(day_idx)
            batch_grads["gcn"]["W_self"] = tuple(g / n_b for g in batch_grads["gcn"]["W_self"])
            batch_grads["gcn"]["W_neigh"] = tuple(g / n_b for g in batch_grads["gcn"]["W_neigh"])
            batch_grads["head"] = tuple(g / n_b for g in batch_grads["head"])

            step += 1
            model.apply_adam(batch_grads, state, step, lr)

    return model


def evaluate(model, X_days, A_days, Y_days, use_graph: bool):
    preds, targets = [], []
    for X_t, A_t, Y_t in zip(X_days, A_days, Y_days):
        A_eff = A_t if use_graph else np.zeros_like(A_t)
        preds.append(model.forward(X_t, A_eff))
        targets.append(Y_t)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    corr = float(np.corrcoef(preds, targets)[0, 1]) if np.std(preds) > 1e-8 else 0.0
    return r2, corr


def run_ablation(universe: str, window: int, epochs=None, seed: int = 123):
    epochs = epochs or config.TNN_EPOCHS

    df = data_manager.load_master_data()
    tickers = config.UNIVERSES[universe]
    prices = data_manager.prepare_prices(df, tickers)
    avail = [t for t in tickers if t in prices.columns]
    prices_a = prices[avail].dropna(how="any")

    X_days, A_days, Y_days = build_days(prices_a, window)
    n_days = len(X_days)
    split = int(n_days * 0.8)
    X_train, A_train, Y_train = X_days[:split], A_days[:split], Y_days[:split]
    X_test,  A_test,  Y_test  = X_days[split:], A_days[split:], Y_days[split:]

    print(f"Universe={universe}  window={window}d  "
          f"train_days={len(X_train)}  test_days={len(X_test)}  tickers={len(avail)}")

    rng_a = np.random.default_rng(seed)
    model_a = train_model(X_train, A_train, Y_train, use_graph=True, rng=rng_a,
                           epochs=epochs, lr=config.TNN_LR, batch_days=config.TNN_BATCH_DAYS)
    train_r2_a, train_corr_a = evaluate(model_a, X_train, A_train, Y_train, use_graph=True)
    test_r2_a,  test_corr_a  = evaluate(model_a, X_test,  A_test,  Y_test,  use_graph=True)

    rng_b = np.random.default_rng(seed)   # SAME seed -> identical initialization
    model_b = train_model(X_train, A_train, Y_train, use_graph=False, rng=rng_b,
                           epochs=epochs, lr=config.TNN_LR, batch_days=config.TNN_BATCH_DAYS)
    train_r2_b, train_corr_b = evaluate(model_b, X_train, A_train, Y_train, use_graph=False)
    test_r2_b,  test_corr_b  = evaluate(model_b, X_test,  A_test,  Y_test,  use_graph=False)

    print()
    print(f"{'':20s} {'Train R2':>10s} {'Train Corr':>12s} {'Test R2':>10s} {'Test Corr':>10s}")
    print(f"{'Model A (graph)':20s} {train_r2_a:10.4f} {train_corr_a:12.4f} {test_r2_a:10.4f} {test_corr_a:10.4f}")
    print(f"{'Model B (no graph)':20s} {train_r2_b:10.4f} {train_corr_b:12.4f} {test_r2_b:10.4f} {test_corr_b:10.4f}")

    test_r2_gain = test_r2_a - test_r2_b
    print()
    print(f"Out-of-sample R2 gain from the graph: {test_r2_gain:+.4f}")
    if test_r2_gain > 0.02:
        verdict = "Graph appears to add genuine out-of-sample value."
    elif test_r2_gain < -0.02:
        verdict = "Graph appears to HURT out-of-sample generalization (possible overfitting to neighbor noise)."
    else:
        verdict = ("Graph's out-of-sample contribution is negligible — fit_quality is likely "
                   "driven mostly by each ticker's own autocorrelation (W_self), not the neighbor term.")
    print(f"=> {verdict}")

    result = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "universe": universe, "window": window,
        "train_days": len(X_train), "test_days": len(X_test),
        "tickers": avail,
        "model_a": {"train_r2": train_r2_a, "train_corr": train_corr_a,
                    "test_r2": test_r2_a, "test_corr": test_corr_a},
        "model_b": {"train_r2": train_r2_b, "train_corr": train_corr_b,
                    "test_r2": test_r2_b, "test_corr": test_corr_b},
        "test_r2_gain": test_r2_gain,
        "verdict": verdict,
    }

    Path("results").mkdir(exist_ok=True)
    json_path = Path(f"results/ablation_{universe}_{window}d_{result['run_date']}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    push_results.push_daily_result(json_path)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="FI_COMMODITIES", choices=list(config.UNIVERSES.keys()))
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--epochs", type=int, default=None,
                         help="override config.TNN_EPOCHS for a faster/slower run")
    args = parser.parse_args()
    run_ablation(args.universe, args.window, epochs=args.epochs)


if __name__ == "__main__":
    main()
