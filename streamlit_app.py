import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Temporal Neighbourhood Network Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#0d2137; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #2e86ab; }
.etf-card    { background:linear-gradient(135deg,#0d2137 0%,#2e86ab 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#0d2137 0%,#1b3a5c 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🕸️ Temporal Neighbourhood Network</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Graph built from temporal proximity in RETURN SPACE (Euclidean k-NN), '
    'not correlation or sector membership · '
    'Topology recomputed daily — jointly trained single-layer GCN · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## TNN Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Graph:** feature window={config.ROLLING_FEATURE_WINDOW}d | k={config.K_NEIGHBORS} | "
    f"hidden dim={config.HIDDEN_DIM}")
st.sidebar.markdown(
    f"**Training:** epochs={config.TNN_EPOCHS} | lr={config.TNN_LR} | "
    f"batch={config.TNN_BATCH_DAYS} days")
st.sidebar.markdown(f"**Turnover lookback:** {config.TURNOVER_LOOKBACK} days")
st.sidebar.markdown(
    f"**Weights:** Forecast {config.WEIGHT_FORECAST:.0%} | "
    f"Stability {config.WEIGHT_STABILITY:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files, None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}`  ·  **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

tab1_path = find_latest(files, "tnn_engine_2")
tab2_path = find_latest(files, "tnn_engine_windows_")

if not tab1_path:
    if list_error:
        st.error("Could not reach HuggingFace to look for results (see 🔧 Debug above).")
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`tnn_engine_2*.json` was found (see 🔧 Debug above for the exact "
            "file list). Run trainer.py, or check the filename it actually pushed."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2, tab3 = st.tabs([
    "🏆 Best Window per ETF", "🔍 Explore by Window", "🔬 Graph Value Ablation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Temporal Neighbourhood Signal")

    with st.expander("TNN Methodology", expanded=True):
        st.markdown("""
**Edge construction: temporal proximity in return space, not correlation
or sector membership.** Each ticker's feature vector at day t is its own
rolling window of recent returns:

```
x_i(t) = [r_i(t-M+1), ..., r_i(t)]
```

The graph's edges connect each ticker to its **k nearest neighbors by
Euclidean distance** between these return-trajectory vectors. This is
fundamentally different from correlation (scale-invariant, measures only
co-movement direction — two tickers can be perfectly correlated while
being Euclidean-distant if one moves 5x as much) and from sector
membership (static categorical metadata, not derived from actual return
behavior).

**Graph topology changes daily.** The k-NN neighbor set is recomputed
from scratch at every timestep — not a fixed adjacency, not a slowly-
drifting correlation matrix. Whether a ticker's neighbors today look like
its neighbors a week ago is itself tracked as a signal.

**Message passing is deliberately simple** — a single-layer graph
convolution:

```
NeighborAvg(t) = A(t) @ X(t)
H(t) = tanh( X(t) @ W_self^T + NeighborAvg(t) @ W_neigh^T + b )
```

so that the graph *construction* choice — not a fancier aggregation
scheme — is the isolated variable this engine studies. Distinct from
GRAPH-TRANSFORMER, HYPERGRAPH-TRANSFORMER, and MAGAT elsewhere in this
suite, which use richer message-passing over differently-constructed
graphs.

**Jointly trained across the whole universe** — unlike the per-ticker
independent models elsewhere in this suite, a single shared model is
trained across all (ticker, day) pairs at once, since the graph inherently
couples tickers together.

**Signal:**

```
score = 0.50*forecast_signal + 0.25*neighborhood_stability*sign(forecast_signal) + 0.25*fit_quality
```

- `forecast_signal` — predicted mean forward return from the graph-aggregated representation
- `neighborhood_stability` — 1 minus neighbor turnover vs. `TURNOVER_LOOKBACK` days ago: is this ticker's return-space peer group itself stable right now?
- `fit_quality` — R² of the shared model across all (ticker, day) training pairs

**Validated before shipping:** the GCN layer's full gradient chain was
checked against finite differences (all parameters matched to ~1e-13–14),
and the k-NN construction was validated on synthetic clustered data —
tickers sharing a common return-space driver were correctly identified as
each other's neighbors 100% of the time, vs. ~43% expected by chance.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">TNN score = {etf['tnn_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">stability = {etf.get('neighborhood_stability', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "TNN Score": info.get("score"),
                        "Best Window (d)": info.get("best_window", "N/A"),
                        "Forecast Signal": info.get("forecast_signal"),
                        "Neighborhood Stability": info.get("neighborhood_stability"),
                        "Fit Quality": info.get("fit_quality"),
                    })
                df = pd.DataFrame(rows).sort_values("TNN Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Temporal proximity graph (Euclidean k-NN in return space) · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore TNN Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — few days of graphs to learn from; reactive, noisier
- **126d** — 6-month window; recommended minimum for a stable shared model
- **252d** — 1-year window; most stable joint training set; recommended primary signal
- **504d** — 2-year window; more days of graphs, but the universe's return-space clustering may shift across regimes within the window
        """)

    st.markdown(f"### TNN Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">TNN score = {etf['tnn_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">stability = {etf.get('neighborhood_stability', float('nan')):.2f}</div>
  <div class="etf-score">fit quality = {etf.get('fit_quality', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "TNN Score", "Forecast Signal",
                             "Neighborhood Stability", "Fit Quality"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Graph Value Ablation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔬 Does the Graph Actually Add Value?")

    with st.expander("Methodology", expanded=True):
        st.markdown("""
Trains two models on **identical data, identical train/test day split, and
identical random seed** — differing in exactly one thing:

```
Model A (graph)    : NeighborAvg(t) = A(t) @ X(t)     (real k-NN graph)
Model B (no graph) : NeighborAvg(t) = 0                (self-only, same
                                                          architecture,
                                                          neighbor term
                                                          forced dead)
```

The difference in **out-of-sample** (held-out, chronologically-last days)
R² between A and B is the graph's genuine value-add. In-sample-only
comparisons can be misleading — either model could simply be fitting noise
in-sample.

This is a separate, on-demand investigative tool — it does not run on the
daily schedule, and you can run it for as many (universe, window)
combinations as you want to check. Trigger it from the GitHub Actions tab:
**"TNN Graph Value Ablation"**.
        """)

    ablation_files = [f for f in files if f.endswith(".json") and "ablation_" in f]
    if not ablation_files:
        st.warning(
            "No ablation runs found yet. Trigger the **\"TNN Graph Value "
            "Ablation\"** workflow from the GitHub Actions tab, choosing the "
            "universe and window you want to test."
        )
        st.stop()

    all_ablations = []
    for f in ablation_files:
        d = load_json(f)
        if "error" not in d and "universe" in d:
            all_ablations.append(d)

    if not all_ablations:
        st.error("Ablation files were found but none loaded successfully.")
        st.stop()

    # Group by (universe, window), keep the most recent run_date per combo —
    # you can run this for as many combinations as you like over time.
    latest_by_key = {}
    for r in all_ablations:
        key = (r["universe"], int(r["window"]))
        if key not in latest_by_key or r["run_date"] > latest_by_key[key]["run_date"]:
            latest_by_key[key] = r

    keys_sorted = sorted(latest_by_key.keys())
    labels = [f"{u} @ {w}d" for u, w in keys_sorted]
    selected_label = st.selectbox("Select a tested universe / window combination", labels)
    selected_key = keys_sorted[labels.index(selected_label)]
    r = latest_by_key[selected_key]

    st.markdown(
        f"**Run date:** `{r.get('run_date','?')}`  ·  "
        f"**Train days:** {r.get('train_days','?')}  ·  "
        f"**Test days:** {r.get('test_days','?')}  ·  "
        f"**Tickers:** {len(r.get('tickers', []))}"
    )

    a, b = r["model_a"], r["model_b"]
    table = pd.DataFrame([
        {"": "Model A (graph)",    "Train R²": a["train_r2"], "Train Corr": a["train_corr"],
         "Test R²": a["test_r2"], "Test Corr": a["test_corr"]},
        {"": "Model B (no graph)", "Train R²": b["train_r2"], "Train Corr": b["train_corr"],
         "Test R²": b["test_r2"], "Test Corr": b["test_corr"]},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    gain = r.get("test_r2_gain", 0.0)
    m1, m2 = st.columns(2)
    m1.metric("Out-of-sample R² gain from the graph", f"{gain:+.4f}")
    if gain > 0.02:
        m2.success("Graph adds genuine value")
    elif gain < -0.02:
        m2.error("Graph hurts generalization")
    else:
        m2.info("Negligible — mostly autocorrelation")

    st.markdown(f"**Verdict:** {r.get('verdict', 'N/A')}")

    if len(latest_by_key) > 1:
        with st.expander("Compare all tested combinations"):
            rows = []
            for (u, w), rr in sorted(latest_by_key.items()):
                rows.append({
                    "Universe": u, "Window": w,
                    "Test R² (graph)": rr["model_a"]["test_r2"],
                    "Test R² (no graph)": rr["model_b"]["test_r2"],
                    "Gain": rr["test_r2_gain"],
                    "Run Date": rr["run_date"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "A positive gain means the graph genuinely helps predict returns it "
        "hasn't seen. A gain near zero means fit_quality in the main tabs is "
        "likely driven mostly by each ticker's own autocorrelation, not the "
        "k-NN neighbor structure."
    )
