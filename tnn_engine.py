"""
tnn_engine.py — Temporal Neighbourhood Network (TNN) Engine
================================================================

Theory
------
**Edge construction: temporal proximity in return space.** At each day t,
each ticker i's feature vector is its own rolling window of recent returns:

    x_i(t) = [r_i(t-M+1), ..., r_i(t)]     (M-dimensional)

The graph's edges at day t connect each ticker to its k NEAREST NEIGHBORS
by EUCLIDEAN DISTANCE between these return-trajectory vectors:

    edges(t) = { (i,j) : x_j(t) is among the k closest to x_i(t) }

This is a fundamentally different similarity notion from correlation or
sector membership:
- Correlation is scale-invariant and measures only co-movement DIRECTION.
  Two tickers can be perfectly correlated (always move together) while
  being Euclidean-DISTANT in return space (one moves 5x as much).
- Sector membership is static categorical metadata, not derived from
  actual return behavior at all.
- Euclidean k-NN in return space captures literal proximity of realized
  return trajectories — magnitude included, no linearity assumption, and
  recomputed fresh every day.

**Graph topology changes daily.** The k-NN neighbor set for each ticker is
NOT fixed and NOT slowly drifting like a rolling correlation matrix — it
is recomputed from scratch at every timestep from that day's return-space
features. Whether ticker i's neighbor set today looks like its neighbor
set a week ago is itself informative, and is tracked explicitly as the
`neighborhood_stability` diagnostic below.

**Message passing (deliberately simple).** A single-layer graph
convolution aggregates each ticker's own features with the mean of its
k-NN neighbors' features:

    NeighborAvg(t) = A(t) @ X(t)             (A(t): row-normalized k-NN adjacency)
    H(t) = tanh( X(t) @ W_self^T + NeighborAvg(t) @ W_neigh^T + b )
    pred(t) = H(t) @ W_out^T + b_out

A(t) is treated as a fixed (non-differentiable) input at each t — standard
practice for k-NN-based graph construction; the discreteness of "who are
my k nearest neighbors today" is not something gradients flow through.
The message-passing mechanism is deliberately kept minimal (unlike
GRAPH-TRANSFORMER, HYPERGRAPH-TRANSFORMER, or MAGAT elsewhere in this
suite) so that the graph CONSTRUCTION choice — not a fancier aggregation
scheme — is the isolated variable this engine studies.

**Jointly trained across the whole universe.** Unlike the per-ticker
independent models elsewhere in this suite (N-HiTS, EDMD, Hyena), a graph
inherently couples multiple tickers — a single shared model is trained
across all (ticker, day) pairs in the window at once. The external
interface (a DataFrame of score + diagnostics per ticker) stays identical
to every other engine in the suite despite this different internal
structure.

**Score construction**

    score = 0.50*forecast_signal + 0.25*neighborhood_stability*sign(forecast_signal) + 0.25*fit_quality

| Component               | Meaning                                                              |
|----------------------------|--------------------------------------------------------------------------|
| forecast_signal            | Predicted mean forward return from the graph-aggregated representation |
| neighborhood_stability     | 1 - neighbor_turnover vs. TURNOVER_LOOKBACK days ago — is this ticker's return-space peer group itself stable right now? |
| fit_quality                | R^2 of the shared model across all (ticker, day) training pairs        |

References
----------
- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with Graph
  Convolutional Networks. ICLR 2017.
- Cover, T. & Hart, P. (1967). Nearest Neighbor Pattern Classification.
  IEEE Transactions on Information Theory.
"""

import numpy as np
import pandas as pd
from typing import List

import config


# ── Basic differentiable layer ─────────────────────────────────────────────────

class Linear:
    def __init__(self, in_d: int, out_d: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_d)
        self.W = rng.normal(0, scale, (in_d, out_d))
        self.b = np.zeros(out_d)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY: np.ndarray):
        X = self.X
        X2  = X.reshape(-1, X.shape[-1])
        dY2 = dY.reshape(-1, dY.shape[-1])
        dW  = X2.T @ dY2
        db  = dY2.sum(axis=0)
        dX  = dY @ self.W.T
        return dX, dW, db


# ── Daily k-NN graph construction ─────────────────────────────────────────────

def build_knn_adjacency(X_t: np.ndarray, k: int):
    """
    X_t: (n, M) return-space feature vectors for all tickers at day t.
    Returns (A_norm (n,n) row-normalized adjacency excluding self,
             knn_idx (n,k) neighbor indices per ticker).
    """
    n = X_t.shape[0]
    diffs = X_t[:, None, :] - X_t[None, :, :]
    dist2 = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    k_eff = min(k, n - 1)
    knn_idx = np.argsort(dist2, axis=1)[:, :k_eff]

    A = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k_eff)
    cols = knn_idx.flatten()
    A[rows, cols] = 1.0 / k_eff
    return A, knn_idx


# ── GCN layer (single message-passing step) ────────────────────────────────────

class GCNLayer:
    def __init__(self, in_dim: int, hidden_dim: int, rng: np.random.Generator):
        self.W_self  = Linear(in_dim, hidden_dim, rng)
        self.W_neigh = Linear(in_dim, hidden_dim, rng)

    def forward(self, X: np.ndarray, NeighborAvg: np.ndarray) -> np.ndarray:
        z_self  = self.W_self.forward(X)
        z_neigh = self.W_neigh.forward(NeighborAvg)
        z = z_self + z_neigh
        h = np.tanh(z)
        self.cache = h
        return h

    def backward(self, dh: np.ndarray):
        h = self.cache
        dz = dh * (1 - h ** 2)
        _, dWs_W, dWs_b = self.W_self.backward(dz)
        _, dWn_W, dWn_b = self.W_neigh.backward(dz)
        return {"W_self": (dWs_W, dWs_b), "W_neigh": (dWn_W, dWn_b)}


class TNNModel:
    def __init__(self, in_dim: int, rng: np.random.Generator):
        self.gcn  = GCNLayer(in_dim, config.HIDDEN_DIM, rng)
        self.head = Linear(config.HIDDEN_DIM, 1, rng)

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """X: (n,M) day's features. A: (n,n) that day's k-NN adjacency. Returns (n,) preds."""
        NeighborAvg = A @ X
        H = self.gcn.forward(X, NeighborAvg)
        pred = self.head.forward(H)
        self._cache_A = A
        return pred[:, 0]

    def backward(self, dpred: np.ndarray):
        """dpred: (n,)."""
        dH, dW_head, db_head = self.head.backward(dpred[:, None])
        gcn_grads = self.gcn.backward(dH)
        return {"gcn": gcn_grads, "head": (dW_head, db_head)}

    def _param_list(self):
        return [
            (self.gcn.W_self, "W"), (self.gcn.W_self, "b"),
            (self.gcn.W_neigh, "W"), (self.gcn.W_neigh, "b"),
            (self.head, "W"), (self.head, "b"),
        ]

    def init_adam(self):
        return [(np.zeros_like(getattr(o, a)), np.zeros_like(getattr(o, a)))
                for o, a in self._param_list()]

    def apply_adam(self, grads, state, step, lr,
                    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        flat = [
            grads["gcn"]["W_self"][0], grads["gcn"]["W_self"][1],
            grads["gcn"]["W_neigh"][0], grads["gcn"]["W_neigh"][1],
            grads["head"][0], grads["head"][1],
        ]
        params = self._param_list()
        for i, ((obj, attr), grad) in enumerate(zip(params, flat)):
            m, v = state[i]
            m[:] = b1 * m + (1 - b1) * grad
            v[:] = b2 * v + (1 - b2) * grad ** 2
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            update = lr * mh / (np.sqrt(vh) + eps)
            setattr(obj, attr, getattr(obj, attr) - update)


# ── Training over (day, whole-universe-graph) examples ────────────────────────

def _train_tnn(X_days: list, A_days: list, Y_days: list, rng: np.random.Generator) -> TNNModel:
    """X_days[t]: (n,M), A_days[t]: (n,n), Y_days[t]: (n,) — one entry per training day."""
    n_days = len(X_days)
    B = config.TNN_BATCH_DAYS
    if n_days < B:
        raise ValueError("insufficient days for TNN training")

    in_dim = X_days[0].shape[1]
    model = TNNModel(in_dim, rng)
    state = model.init_adam()
    step = 0

    for epoch in range(config.TNN_EPOCHS):
        idx = rng.permutation(n_days)
        epoch_loss, n_b = 0.0, 0

        for i in range(0, n_days, B):
            day_idx = idx[i:i + B]
            if len(day_idx) < 1:
                continue

            batch_grads = None
            batch_loss = 0.0
            for d in day_idx:
                X_t, A_t, Y_t = X_days[d], A_days[d], Y_days[d]
                pred = model.forward(X_t, A_t)
                resid = pred - Y_t
                loss = float(np.mean(resid ** 2))
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
                batch_loss += loss

            n_in_batch = len(day_idx)
            batch_grads["gcn"]["W_self"] = tuple(g / n_in_batch for g in batch_grads["gcn"]["W_self"])
            batch_grads["gcn"]["W_neigh"] = tuple(g / n_in_batch for g in batch_grads["gcn"]["W_neigh"])
            batch_grads["head"] = tuple(g / n_in_batch for g in batch_grads["head"])

            step += 1
            model.apply_adam(batch_grads, state, step, lr=config.TNN_LR)

            epoch_loss += batch_loss / n_in_batch
            n_b += 1

        if (epoch + 1) % 15 == 0:
            print(f"    epoch {epoch+1}/{config.TNN_EPOCHS}  loss={epoch_loss/max(n_b,1):.6f}")

    return model


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_tnn_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.DataFrame:
    """
    Build a daily k-NN return-space graph across the whole universe, train a
    single shared GCN model jointly across all (ticker, day) pairs in the
    window, and extract per-ticker forecast + neighbor-turnover diagnostics.
    Returns a DataFrame of score + diagnostics (cross-sectional z-scored on
    the composite), same interface as every other engine in the suite.
    """
    cols = ["score", "forecast_signal", "neighborhood_stability", "fit_quality"]
    avail = [t for t in tickers if t in prices.columns]
    n_tickers = len(avail)
    if n_tickers < config.K_NEIGHBORS + 2:
        return pd.DataFrame(columns=cols)

    M, H, k = config.ROLLING_FEATURE_WINDOW, config.PRED_HORIZON, config.K_NEIGHBORS
    min_rows = window + M + H + config.TNN_BATCH_DAYS
    if len(prices) < min_rows:
        return pd.DataFrame(columns=cols)

    prices_a = prices[avail].dropna(how="any")
    if len(prices_a) < min_rows:
        return pd.DataFrame(columns=cols)

    log_ret_full = np.log(prices_a / prices_a.shift(1)).dropna()
    log_ret = log_ret_full.iloc[-window:]
    R = log_ret.values                        # (T, n_tickers)
    T = len(R)

    ret_mu, ret_sd = R.mean(), R.std() + 1e-8
    R_norm = (R - ret_mu) / ret_sd

    n_samples = T - M - H + 1
    if n_samples < config.TNN_BATCH_DAYS * 2:
        return pd.DataFrame(columns=cols)

    # ── Build per-day feature matrices, graphs, and targets ────────────────────
    X_days, A_days, Y_days, knn_days = [], [], [], []
    for t in range(M - 1, T - H):
        X_t = R_norm[t - M + 1: t + 1, :].T          # (n_tickers, M)
        A_t, knn_idx_t = build_knn_adjacency(X_t, k)
        Y_t = R_norm[t + 1: t + 1 + H, :].mean(axis=0)  # (n_tickers,) forward mean return
        X_days.append(X_t)
        A_days.append(A_t)
        Y_days.append(Y_t)
        knn_days.append(knn_idx_t)

    rng = np.random.default_rng(42)
    print(f"    Training TNN jointly across {len(avail)} tickers, {len(X_days)} days")
    try:
        model = _train_tnn(X_days, A_days, Y_days, rng)
    except Exception as e:
        print(f"    Failed: {e}")
        return pd.DataFrame(columns=cols)

    # ── In-sample fit quality across all (ticker, day) pairs ───────────────────
    all_preds, all_targets = [], []
    for X_t, A_t, Y_t in zip(X_days, A_days, Y_days):
        all_preds.append(model.forward(X_t, A_t))
        all_targets.append(Y_t)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ss_res = np.sum((all_preds - all_targets) ** 2)
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    fit_quality = float(1.0 - np.clip(ss_res / (ss_tot + 1e-10), 0.0, 1.0))

    # ── Inference: today's graph and forecast ──────────────────────────────────
    X_today = R_norm[-M:, :].T
    A_today, knn_today = build_knn_adjacency(X_today, k)
    pred_today_norm = model.forward(X_today, A_today)
    forecast_signal_all = pred_today_norm * ret_sd + ret_mu

    # ── Neighbor turnover: compare today's neighbor set to TURNOVER_LOOKBACK days ago ──
    lookback = min(config.TURNOVER_LOOKBACK, len(knn_days) - 1)
    knn_past = knn_days[-1 - lookback] if len(knn_days) > lookback else knn_today

    raw_scores = {}
    for i, ticker in enumerate(avail):
        today_set = set(knn_today[i].tolist())
        past_set = set(knn_past[i].tolist())
        overlap = len(today_set & past_set)
        turnover = 1.0 - (overlap / max(len(today_set), 1))
        neighborhood_stability = float(1.0 - turnover)

        forecast_signal = float(forecast_signal_all[i])
        sign = np.sign(forecast_signal) if forecast_signal != 0 else 1.0

        composite = (
            config.WEIGHT_FORECAST  * forecast_signal
            + config.WEIGHT_STABILITY * neighborhood_stability * sign
            + config.WEIGHT_FIT        * fit_quality
        )
        raw_scores[ticker] = {
            "composite": composite,
            "forecast_signal": forecast_signal,
            "neighborhood_stability": neighborhood_stability,
            "fit_quality": fit_quality,
        }
        print(f"    {ticker}: forecast={forecast_signal:.5f}  "
              f"stability={neighborhood_stability:.3f}  fit={fit_quality:.3f}")

    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
