# Mixup Barcodes for Topology-Aware Financial Decision Making

A reproducible, offline-first pipeline that uses persistent homology and
**mixup barcodes** to build a topological "stress score" for markets, and uses
that score to convert a classical Golden Cross trend rule into a continuous,
risk-sensitive position-sizing strategy. Evaluated on the S&P 500 (SPY) and
Bitcoin (BTC-USD).

```
python tda_stress_gc.py --asset all --stage all --jobs -1
```

---

## What this is

Standard topological summaries of a market (persistence diagrams, Betti
numbers, entropy) describe a single point cloud in isolation — how *unusual*
the current regime looks, but not *how* it has been disrupted relative to a
specific reference. This pipeline implements the **mixup barcode** of
Wagner, Arustamyan, Wheeler & Bubenik (2024) to answer that relational
question directly: how much of a calm reference regime's loop topology
($H_1$) is destroyed when the current market state is mixed into the same
space.

The mixup percentage $MP_1$ is combined with a Wasserstein distance term and
a persistence-entropy divergence into a single stress score $S(t) \in [0,1]$,
smoothed and clipped, which then gates a 50/200-day Golden Cross: trend sets
the *direction* of exposure, topology sets its *magnitude*.

## Key features

- **Exact mixup barcode via Algorithm 2**, not an approximate diagram-matching
  heuristic — builds the union filtration once, reduces two $\mathbb{F}_2$
  boundary matrices, and recovers the canonical matching of the stability
  theorem with no tolerance parameter.
- **Offline-by-design.** The internet is touched by exactly one function,
  once. Every other stage refuses to open a network connection and tells you
  the exact command to run if the cache is missing.
- **Integrity-checked data cache.** Every cached CSV is paired with a SHA-256
  digest, row count, and observed date range in `manifest.json`; a changed
  cache produces a loud warning instead of silently different numbers.
- **Seven-strategy benchmark suite** (Buy & Hold, Golden Cross, EMA
  crossover, MACD+RSI, RSI mean reversion, Bollinger Bands, MACD+RSI dual)
  sharing identical trend-filter parameters with the TDA strategy for a fair
  comparison.
- **Deterministic and parallel-safe.** A `--jobs 1` serial run and a
  `--jobs -1` parallel run agree exactly; the smoke test asserts this.
- **Built-in smoke test.** Unit checks, a ground-truth mixup validation
  suite, an independent cross-check of the Algorithm-2 reduction against
  Ripser, and an end-to-end run on synthetic data — all offline, in about two
  minutes.

## Method summary

| Component | What it measures |
|---|---|
| $\widetilde{W}_1$ | 1-Wasserstein distance between $H_0$ diagrams (reference vs. current) — connectivity / small-scale clustering |
| $MP_1$ | Mixup percentage — fraction of reference $H_1$ loop lifetime destroyed by the current state |
| $\widetilde{\Delta H}$ | Divergence in $H_1$ persistence entropy — structural complexity |
| $S(t)$ | $0.45\,\widetilde{W}_1 + 0.50\,MP_1 + 0.05\,\widetilde{\Delta H}$, expanding-min–max normalized, 30-day trailing mean, clipped to $[0,1]$ |
| $\alpha(t)$ | Position size: $1.0$ below $\theta_{\text{lo}}=0.30$, linear ramp to $0.0$ above $\theta_{\text{hi}}=0.60$ |

The current cloud is a Takens delay embedding of a trailing window of
z-scored log prices (not returns), updated daily. The reference cloud is
built once, from the middle window of the first `ref_length + window`
observations, and held fixed for the rest of the run.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. `yfinance` is only needed for the one-time data fetch;
everything else runs with it uninstalled once `data/raw/` is populated.

## Quick start

```bash
# 1. Sanity check — offline, ~2 minutes, no data needed
python tda_stress_gc.py --smoke

# 2. One-time download (needs a connection)
python tda_stress_gc.py --stage fetch --asset all

# 3. Everything else — offline from here on
python tda_stress_gc.py --asset all --stage all --jobs -1
```

## CLI reference

```
python tda_stress_gc.py [--asset {spy,btc,all}] [--stage STAGE] [--jobs N]
                         [--allow-download] [--force-refetch] [--smoke]
```

| Stage | Produces |
|---|---|
| `fetch` | `data/raw/*.csv`, `manifest.json` — the only stage that touches the network |
| `backtest` | `stress_scores.csv`, `positions.csv`, `strategy_comparison.csv` |
| `grid` | `grid_search.csv` — full 72-configuration Takens-parameter sweep per asset |
| `robustness` | reference/current-window sweeps, weight perturbation, threshold grid, `robustness_summary.json` |
| `walkforward` | out-of-sample fold metrics |
| `ml` | exploratory ML experiment over mixup features (not used in headline results) |

Outputs land in `results/<asset>/{tables,figures}/`; the exact configuration
used for a run is written to `results/config_used.json`.

If a cache is missing and you didn't run `fetch`, the pipeline stops with the
exact command to fix it rather than silently downloading fresh (possibly
revised) prices:

```
No cached data for SPY at data/raw/SPY_2010-01-01_2024-12-31.csv.
This run is offline by design. Connect once and run:

    python tda_stress_gc.py --stage fetch --asset all
```

Pass `--allow-download` if you genuinely want an analysis stage to fetch
missing data itself, or `--force-refetch` (with `fetch`) to deliberately
refresh an existing cache.

## Configuration

All parameters live in the `Config` / `AssetConfig` dataclasses at the top of
`tda_stress_gc.py`. Selected embedding parameters (window `w`, dimension `d`,
delay `τ`) are the in-sample argmax of the grid search:

| Asset | `w` | `d` | `τ` |
|---|---|---|---|
| SPY | 90 | 4 | 3 |
| BTC-USD | 60 | 4 | 2 |

Other defaults: 5 bps one-way transaction cost, 28-point cloud subsample
(fixed seed), Vietoris–Rips filtration capped at edge length 2.5,
$(\theta_{\text{lo}}, \theta_{\text{hi}}) = (0.30, 0.60)$, 30-day score
smoothing, weights $(0.45, 0.50, 0.05)$.

## Reproducibility notes

- Positions and their transaction costs are both applied with a one-day lag,
  so a trade is charged on the first day its exposure is actually held.
- `n_signals = n_entries + n_exits`, which need not be even: a position still
  open at the end of the sample has an entry with no matching exit.
- A `--jobs -1` run and a `--jobs 1` run are asserted to produce identical
  component values in the smoke test — parallelism is for speed only.
- Because embedding parameters are selected by maximizing in-sample Sharpe
  over the same sample used for reported performance, the headline numbers
  from `backtest` are an in-sample upper bound; use the `grid` and
  `robustness` stages to see the full sensitivity, and `walkforward` for an
  out-of-sample estimate.

## Citation

If this code is useful in your work, please cite the accompanying paper,
*Mixup Barcodes for Topology-Aware Financial Decision Making*, and the
underlying method:

> Wagner, H., Arustamyan, N., Wheeler, M., & Bubenik, P. (2024). Mixup
> barcodes: quantifying geometric-topological interactions between point
> clouds. *arXiv:2402.15058*.
