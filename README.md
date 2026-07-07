# 🕸️ P2-ETF-TNN

**Temporal Neighbourhood Network Engine**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine builds a graph among ETFs from **temporal proximity in return
space** — Euclidean k-NN between rolling return-trajectory vectors — not
correlation and not sector membership, and recomputes that graph's
topology fresh every single day. A single-layer GCN, jointly trained
across the whole universe (since a graph inherently couples multiple
tickers), aggregates each ticker's own recent behavior with its current
k-NN neighbors' behavior to produce a forecast, while an explicit
**neighbor turnover** diagnostic tracks how much a ticker's return-space
peer group has itself been shifting.

---

## Theory

### Edge Construction: Temporal Proximity in Return Space

Each ticker's feature vector at day t is its own rolling window of recent
returns:

```
x_i(t) = [r_i(t-M+1), ..., r_i(t)]
```

The graph's edges connect each ticker to its **k nearest neighbors by
Euclidean distance** between these vectors:

```
edges(t) = { (i,j) : x_j(t) is among the k closest to x_i(t) }
```

This is a fundamentally different similarity notion from:
- **Correlation** — scale-invariant, measures only co-movement direction.
  Two tickers can be perfectly correlated (always move together) while
  being Euclidean-*distant* in return space if one moves 5x as much.
- **Sector membership** — static categorical metadata, not derived from
  actual return behavior at all.

### Graph Topology Changes Daily

The k-NN neighbor set for each ticker is recomputed from scratch at every
timestep from that day's return-space features — not a fixed adjacency
computed once, and not a slowly-drifting correlation matrix. Whether a
ticker's neighbors today resemble its neighbors a week ago is itself
tracked explicitly (see `neighborhood_stability` below), rather than
assumed away.

### Message Passing (Deliberately Simple)

```
NeighborAvg(t) = A(t) @ X(t)                    (A(t): row-normalized k-NN adjacency)
H(t) = tanh( X(t) @ W_self^T + NeighborAvg(t) @ W_neigh^T + b )
pred(t) = H(t) @ W_out^T + b_out
```

`A(t)` is treated as a fixed (non-differentiable) input at each t —
standard practice for k-NN-based graph construction. The message-passing
mechanism is deliberately minimal, unlike GRAPH-TRANSFORMER, HYPERGRAPH-
TRANSFORMER, or MAGAT elsewhere in this suite, so the graph *construction*
choice is the isolated variable this engine studies.

### Jointly Trained Across the Whole Universe

Unlike the per-ticker independent models elsewhere in this suite (N-HiTS,
EDMD, Hyena), a graph inherently couples multiple tickers together — a
single shared model is trained across all (ticker, day) pairs in the
window at once. The external interface (a DataFrame of score +
diagnostics per ticker) stays identical to every other engine in the
suite despite this different internal structure.

### Score Construction

```
score = 0.50*forecast_signal + 0.25*neighborhood_stability*sign(forecast_signal) + 0.25*fit_quality
```

| Component | Meaning |
|-----------|---------|
| forecast_signal | Predicted mean forward return from the graph-aggregated representation |
| neighborhood_stability | 1 - neighbor turnover vs. `TURNOVER_LOOKBACK` days ago — is this ticker's return-space peer group itself stable right now? |
| fit_quality | R² of the shared model across all (ticker, day) training pairs |

### Validation

- **GCN gradient chain** checked against finite differences — all
  parameters (both weight matrices, both bias vectors, output head)
  matched to ~1e-13–1e-14 precision.
- **k-NN construction validated on synthetic clustered data**: tickers
  sharing a common return-space driver were correctly identified as each
  other's nearest neighbors 100% of the time, versus ~43% expected by
  chance — confirming the graph construction captures genuine structure,
  not noise.

---

## Distinction from Other Graph Engines in the Suite

| Engine | Edge construction | Message passing |
|--------|---------------------|--------------------|
| EUCLID-GCN | (see that repo) | Graph convolution |
| GRAPH-TRANSFORMER | (see that repo) | Attention over graph |
| HYPERGRAPH-TRANSFORMER | (see that repo) | Attention over hyperedges |
| MAGAT | (see that repo) | Multi-hop graph attention |
| **TNN (this engine)** | **Temporal Euclidean k-NN in return space, rebuilt daily** | **Single-layer GCN (deliberately simple)** |

This engine isolates edge construction as the variable of interest by
keeping message passing minimal — the novelty here is entirely in *how
the graph is built*, not in a more elaborate aggregation mechanism.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d · 1008d`

---

## Repository Structure

```
P2-ETF-TNN/
├── config.py          # Universes, TNN hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── tnn_engine.py        # Core: daily k-NN graph, GCN layer, joint training
├── trainer.py            # Orchestrator
├── push_results.py       # HfApi.upload_file wrapper
├── streamlit_app.py       # Two-tab Streamlit dashboard
├── us_calendar.py        # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml     # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-TNN
cd P2-ETF-TNN
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-tnn-results`

---

## References

- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with Graph
  Convolutional Networks. ICLR 2017.
- Cover, T. & Hart, P. (1967). Nearest Neighbor Pattern Classification.
  IEEE Transactions on Information Theory.
