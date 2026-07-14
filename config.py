import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-tnn-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
# 1008d (~4 years) added to test whether persistent structural issues (e.g.
# FI_COMMODITIES showing negative out-of-sample correlation at both 252d and
# 504d in ablation testing) are a genuine sample-size problem that more data
# fixes, or something more structural (e.g. K_NEIGHBORS being too large
# relative to a small universe). Needs len(prices) >= 1008 + ROLLING_FEATURE_
# WINDOW + PRED_HORIZON + TNN_BATCH_DAYS (~1060 trading days, ~4.2 years) —
# windows without enough history are skipped automatically, same as always.
WINDOWS = [63, 126, 252, 504, 1008]

# ── Temporal Neighbourhood Network hyperparameters ────────────────────────────
# The graph is built from TEMPORAL PROXIMITY IN RETURN SPACE, not correlation
# and not sector membership:
#
#   x_i(t) = [r_i(t-M+1), ..., r_i(t)]     (ticker i's own rolling return
#                                            trajectory, an M-dim vector)
#   edges(t) = k-NN(i) = the k tickers j whose x_j(t) is closest to x_i(t)
#              in EUCLIDEAN distance
#
# This is a fundamentally different similarity notion from correlation:
# correlation is scale-invariant and measures only co-movement DIRECTION,
# while Euclidean distance in return space measures actual proximity of the
# realized return TRAJECTORIES themselves, magnitude included. Two tickers
# can be perfectly correlated (always move together) while being Euclidean-
# distant (one moves 5x as much), and vice versa.
#
# GRAPH TOPOLOGY CHANGES DAILY: the k-NN neighbor set for ticker i is
# recomputed at every timestep t from that day's return-space features, so
# WHO is connected to whom evolves day by day — not a fixed adjacency
# computed once and reused, and not a slowly-drifting correlation matrix.
#
# Distinct from other graph engines in this suite (EUCLID-GCN, GRAPH-
# TRANSFORMER, HYPERGRAPH-TRANSFORMER, MAGAT): the novelty here is
# specifically in EDGE CONSTRUCTION (temporal k-NN in return space, rebuilt
# daily), not in the message-passing mechanism, which is kept deliberately
# simple (a single-layer graph convolution) so the graph-construction
# choice itself is the isolated variable of interest.

ROLLING_FEATURE_WINDOW = 15    # M: length of each ticker's return-space feature vector
K_NEIGHBORS             = 4     # k: number of nearest neighbors per ticker, per day
HIDDEN_DIM              = 16    # GCN hidden dimension

PRED_HORIZON = 21        # H: forward return horizon defining the regression target

TNN_EPOCHS     = 60
TNN_LR         = 3e-3
TNN_BATCH_DAYS = 16       # minibatch = a set of DAYS; each day processes the
                           # whole universe's graph jointly in one shot

TURNOVER_LOOKBACK = 5     # days back used to measure how much a ticker's
                            # k-NN neighbor set has changed (topology dynamism)

# ── Score construction ────────────────────────────────────────────────────────
# forecast_signal        : predicted mean forward return from the graph-
#                          aggregated (self + k-NN neighbor) representation
# neighborhood_stability : 1 - neighbor_turnover, where neighbor_turnover is
#                          the fraction of a ticker's CURRENT k-NN neighbors
#                          that differ from its neighbors TURNOVER_LOOKBACK
#                          days ago. A stable neighborhood means the model's
#                          learned neighbor-aggregation pattern is more
#                          likely to still be valid; a churning neighborhood
#                          means the ticker's "peer group" in return space is
#                          itself unstable right now.
# fit_quality            : R^2 of the shared model across all (ticker, day)
#                          pairs in the training window

WEIGHT_FORECAST    = 0.50
WEIGHT_STABILITY    = 0.25
WEIGHT_FIT           = 0.25

TOP_N = 3
