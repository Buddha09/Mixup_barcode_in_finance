#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Mixup Barcodes for Topology-Aware Financial Decision Making
 Consolidated, reproducible pipeline for S&P 500 (SPY) and Bitcoin (BTC-USD)
================================================================================

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from itertools import combinations as _combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist as _cdist

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tda_stress_gc")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class AssetConfig:
    """Per-asset settings.  Embedding parameters are the in-sample grid argmax."""
    ticker: str
    name: str
    start: str
    end: str
    window: int        # current-window length w (trading days)
    dimension: int     # Takens embedding dimension d
    delay: int         # Takens delay tau


@dataclass
class Config:
    # ---- assets -------------------------------------------------------------
    assets: dict = field(default_factory=lambda: {
        "spy": AssetConfig(
            ticker="SPY", name="S&P 500 ETF",
            start="2010-01-01", end="2024-12-31",
            window=90, dimension=4, delay=3,      # [A1] grid argmax for SPY
        ),
        "btc": AssetConfig(
            ticker="BTC-USD", name="Bitcoin",
            start="2017-01-01", end="2024-12-31",
            window=60, dimension=4, delay=2,      # [A1] grid argmax for BTC
        ),
    })
    price_col: str = "Close"

    # ---- embedding / homology ----------------------------------------------
    max_points: int = 28          # [B8] point-cloud subsample size (disclosed)
    subsample_seed: int = 0
    max_edge_length: float = 2.5  # [B3] Vietoris-Rips filtration cap (disclosed)
    mixup_eps: float = 1e-10      # [B11] epsilon_0 in the MP_k definition

    # ---- stress score -------------------------------------------------------
    ref_length: int = 120         # [B4] reference slice = ref_length + window days
    h0_weight: float = 0.45
    h1_mixup_weight: float = 0.50
    entropy_weight: float = 0.05
    smooth_window: int = 30       # [B7] trailing mean applied to s_raw

    # ---- strategy -----------------------------------------------------------
    ma_fast: int = 50
    ma_slow: int = 200
    s_low: float = 0.30
    s_high: float = 0.60

    # ---- backtest -----------------------------------------------------------
    transaction_cost: float = 0.0005   # [C1] 5 bps ONE-WAY on |delta position|
    risk_free_rate: float = 0.0
    trading_days: int = 252

    # ---- appendix grid (single source of truth) [A2] ------------------------
    grid_windows: tuple = (30, 45, 60, 90)
    grid_dimensions: tuple = (2, 3, 4)
    grid_delays: tuple = (1, 2, 3, 4, 5, 6)

    # ---- robustness sweeps [D] ---------------------------------------------
    rob_ref_lengths: tuple = (60, 90, 120, 150, 180)
    rob_windows: tuple = (20, 30, 45, 60, 90)
    rob_weight_pct: float = 0.10
    rob_theta_lo: tuple = (0.20, 0.25, 0.30, 0.35, 0.40)
    rob_theta_hi: tuple = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)

    # ---- walk-forward -------------------------------------------------------
    wf_train: int = 252
    wf_test: int = 63
    wf_step: int = 63

    # ---- ML -----------------------------------------------------------------
    ml_horizon: int = 5
    ml_test_size: float = 0.30
    ml_seed: int = 42

    # ---- statistics ---------------------------------------------------------
    bootstrap_reps: int = 1000
    block_size: int = 20
    bootstrap_seed: int = 42
    alpha: float = 0.05

    # ---- io -----------------------------------------------------------------
    data_dir: str = "data/raw"
    results_dir: str = "results"
    n_jobs: int = -1

    def weights(self) -> tuple:
        return (self.h0_weight, self.h1_mixup_weight, self.entropy_weight)


CFG = Config()


# ==============================================================================
# 2. DATA
# ==============================================================================

CACHE_MANIFEST = "manifest.json"


def _cache_path(cache_dir: str, ticker: str, start: str, end: str) -> Path:
    return Path(cache_dir) / f"{ticker.replace('=', '_').replace('/', '_')}_{start}_{end}.csv"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(cache_dir: str) -> dict:
    mf = Path(cache_dir) / CACHE_MANIFEST
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text())
    except Exception:                                   # pragma: no cover
        log.warning("Cache manifest at %s is unreadable; ignoring it.", mf)
        return {}


def _write_manifest(cache_dir: str, manifest: dict) -> None:
    (Path(cache_dir) / CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _parse_cached_csv(cache_file: Path, price_col: str) -> pd.Series:
    """Read a cached CSV into a clean float Series indexed by date."""
    df = pd.read_csv(cache_file, index_col=0)
    # A multi-row yfinance header leaves 'Ticker'/'Date' rows under the column
    # names; detect an unparseable index and re-read without them.
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().any():
        df = pd.read_csv(cache_file, index_col=0, skiprows=[1, 2])
        idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[idx.notna()]
    df.index = pd.DatetimeIndex(idx[idx.notna()])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in {cache_file.name}. "
                       f"Available: {list(df.columns)}")

    series = df[price_col]
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(f"Expected one '{price_col}' column, got {series.shape[1]}.")
        series = series.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce").ffill().dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def fetch_price_series(ticker: str, start: str, end: str, price_col: str = "Close",
                       cache_dir: str = "data/raw", force: bool = False) -> pd.Series:
    """Download once and cache to CSV.  THE ONLY FUNCTION THAT TOUCHES THE NETWORK.

    Writes/updates `manifest.json` with a SHA-256 of the CSV, the row count and
    the observed date range, so a later offline run can prove it is using the
    same bytes that produced the published tables.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, ticker, start, end)

    if cache_file.exists() and not force:
        log.info("Cache already present for %s (%s); use --force-refetch to replace it.",
                 ticker, cache_file.name)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:                      # pragma: no cover
            raise ImportError(
                "yfinance is required to fetch data: pip install yfinance"
            ) from exc
        log.info("Downloading %s  %s -> %s ...", ticker, start, end)
        # auto_adjust=True makes 'Close' the ADJUSTED close, which is what the
        # manuscript describes and what the original notebooks used (it is the
        # yfinance default in the versions those notebooks ran under).
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise ValueError(f"No data returned for '{ticker}'. Check the ticker and dates.")
        raw.to_csv(cache_file)
        log.info("Wrote %s (%d rows)", cache_file, len(raw))

    series = _parse_cached_csv(cache_file, price_col)
    series.name = ticker

    manifest = _read_manifest(cache_dir)
    manifest[cache_file.name] = {
        "ticker": ticker, "start": start, "end": end, "price_col": price_col,
        "sha256": _sha256(cache_file), "n_rows": int(len(series)),
        "first_date": str(series.index[0].date()), "last_date": str(series.index[-1].date()),
        "fetched_utc": pd.Timestamp.utcnow().isoformat(),
    }
    _write_manifest(cache_dir, manifest)
    return series


def load_price_series(ticker: str, start: str, end: str, price_col: str = "Close",
                      cache_dir: str = "data/raw", offline: bool = True,
                      verify: bool = True) -> pd.Series:
    """Load daily closes from the local CSV cache.

    `offline=True` (the DEFAULT) means this function will NEVER open a network
    connection: if the cache is missing it raises with the exact command to run.
    Every analysis stage calls it this way, so once `--stage fetch` has completed
    the entire pipeline -- backtest, grid search, robustness, walk-forward, ML --
    runs with the network disconnected.
    """
    cache_file = _cache_path(cache_dir, ticker, start, end)

    if not cache_file.exists():
        if offline:
            raise FileNotFoundError(
                f"\nNo cached data for {ticker} at {cache_file}.\n"
                f"This run is offline by design. Connect once and run:\n\n"
                f"    python tda_stress_gc.py --stage fetch --asset all\n\n"
                f"After that every stage works with the network disconnected."
            )
        return fetch_price_series(ticker, start, end, price_col, cache_dir)

    series = _parse_cached_csv(cache_file, price_col)
    series.name = ticker

    if verify:
        entry = _read_manifest(cache_dir).get(cache_file.name)
        if entry is None:
            log.warning("No manifest entry for %s; integrity not verified. "
                        "Re-run --stage fetch to record one.", cache_file.name)
        else:
            digest = _sha256(cache_file)
            if digest != entry.get("sha256"):
                log.warning("CACHE CHANGED for %s: sha256 %s... does not match the "
                            "manifest (%s...). Results may not match previously "
                            "published tables.", cache_file.name, digest[:12],
                            str(entry.get("sha256"))[:12])
            elif int(entry.get("n_rows", -1)) != len(series):
                log.warning("Row count for %s changed (%s -> %d).", cache_file.name,
                            entry.get("n_rows"), len(series))
            else:
                log.info("Cache verified: %s (%d rows, %s -> %s, sha256 %s...)",
                         cache_file.name, len(series), entry["first_date"],
                         entry["last_date"], digest[:12])
    return series


def fetch_all_data(cfg: Config, keys: list, force: bool = False) -> None:
    """`--stage fetch`: download every asset, cache it, and verify.  Run once
    with a connection; everything afterwards is offline."""
    log.info("=" * 74)
    log.info(" FETCH STAGE -- this is the only stage that requires the internet")
    log.info("=" * 74)
    rows = []
    for key in keys:
        a = cfg.assets[key]
        s = fetch_price_series(a.ticker, a.start, a.end, cfg.price_col,
                               cfg.data_dir, force=force)
        rows.append({"asset": key, "ticker": a.ticker, "rows": len(s),
                     "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                     "file": _cache_path(cfg.data_dir, a.ticker, a.start, a.end).name})
    log.info("\n%s", pd.DataFrame(rows).to_string(index=False))
    log.info("\nCached under %s", Path(cfg.data_dir).resolve())
    log.info("Manifest: %s", (Path(cfg.data_dir) / CACHE_MANIFEST).resolve())
    log.info("\nYou can now disconnect. Run the analysis with:")
    log.info("    python tda_stress_gc.py --asset all --stage all --jobs -1")


def synthetic_price_series(n: int = 900, seed: int = 7,
                           name: str = "SYNTH") -> pd.Series:
    """Regime-switching GBM used by --smoke so the pipeline runs offline."""
    rng = np.random.default_rng(seed)
    mu = np.where((np.arange(n) // 150) % 2 == 0, 0.0006, -0.0004)
    sig = np.where((np.arange(n) // 150) % 2 == 0, 0.008, 0.022)
    ret = mu + sig * rng.standard_normal(n)
    prices = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(prices, index=idx, name=name)


# ==============================================================================
# 3. TDA CORE  -- embedding, persistence, mixup barcode
# ==============================================================================

def local_normalise(prices: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Log-transform then z-score a price window.

    NOTE [B1]: the pipeline embeds normalised LOG PRICE LEVELS, not log returns.
    The series is never differenced.  The manuscript text was corrected to match.
    """
    prices = np.asarray(prices, dtype=float)
    shifted = prices - prices.min() + 1.0 if prices.min() <= 0 else prices
    log_p = np.log(shifted + epsilon)
    mu, sigma = log_p.mean(), log_p.std()
    return (log_p - mu) / (sigma + epsilon)


def takens_embedding(series: np.ndarray, dimension: int = 3, delay: int = 4,
                     normalise: bool = True) -> np.ndarray:
    """Delay embedding  x_i = [s(i), s(i+tau), ..., s(i+(d-1)tau)]  in R^d."""
    if normalise:
        series = local_normalise(series)
    series = np.asarray(series, dtype=float)
    T = len(series)
    n_points = T - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError(f"Window too short: T={T}, d={dimension}, tau={delay}.")
    cloud = np.empty((n_points, dimension))
    for col in range(dimension):
        start = col * delay
        cloud[:, col] = series[start: start + n_points]
    return cloud


def subsample_cloud(cloud: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Uniform random subsample to at most `max_points` points (fixed seed)."""
    n = cloud.shape[0]
    if n <= max_points:
        return cloud
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return cloud[idx]


def compute_persistence(cloud: np.ndarray, max_dim: int = 1,
                        max_edge_length: float = 2.5) -> dict:
    """Vietoris-Rips H0/H1 via Ripser, with infinite deaths capped at the
    filtration threshold [B3]."""
    from ripser import ripser
    result = ripser(cloud, maxdim=max_dim, thresh=max_edge_length)
    dgms = result["dgms"]
    dgm0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    def cap_inf(d):
        d = np.asarray(d, dtype=float).copy()
        if d.size:
            d[~np.isfinite(d[:, 1]), 1] = max_edge_length
        else:
            d = d.reshape(0, 2)
        return d

    dgm0, dgm1 = cap_inf(dgm0), cap_inf(dgm1)
    thresh = 1e-6
    return {
        "dgm0": dgm0, "dgm1": dgm1,
        "betti0": int(np.sum((dgm0[:, 1] - dgm0[:, 0]) > thresh)) if dgm0.size else 0,
        "betti1": int(np.sum((dgm1[:, 1] - dgm1[:, 0]) > thresh)) if dgm1.size else 0,
    }


def persistence_entropy(dgm: np.ndarray, epsilon: float = 1e-10) -> float:
    """Shannon entropy of normalised bar lifetimes."""
    if dgm is None or dgm.size == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[pers > 0]
    if len(pers) == 0:
        return 0.0
    p = pers / (pers.sum() + epsilon)
    return max(0.0, float(-np.sum(p * np.log(p + epsilon))))


def total_persistence(dgm: np.ndarray, p: float = 1.0) -> float:
    if dgm is None or dgm.size == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    return float(np.sum(pers[pers > 0] ** p))


def diagram_summary(dgm0: np.ndarray, dgm1: np.ndarray) -> dict:
    return {
        "betti0": int(np.sum((dgm0[:, 1] - dgm0[:, 0]) > 1e-6)) if dgm0.size else 0,
        "betti1": int(np.sum((dgm1[:, 1] - dgm1[:, 0]) > 1e-6)) if dgm1.size else 0,
        "total_pers_h1": total_persistence(dgm1),
        "entropy_h1": persistence_entropy(dgm1),
        "entropy_h0": persistence_entropy(dgm0),
        "max_pers_h0": float((dgm0[:, 1] - dgm0[:, 0]).max()) if dgm0.size else 0.0,
        "max_pers_h1": float((dgm1[:, 1] - dgm1[:, 0]).max()) if dgm1.size else 0.0,
    }

"""
# ------------------------------------------------------------------------------
# 3b. Mixup barcode -- Algorithm 2 of Wagner, Arustamyan, Wheeler & Bubenik (2024)
# ------------------------------------------------------------------------------
"""
def _build_sparse_columns(cloud_a: np.ndarray, cloud_b: np.ndarray,
                          max_edge_length: float, max_dim: int = 2):
    """Union filtration + sparse Z_2 boundary columns for B_K and B_L.

    Ordering guarantee: at equal filtration value, A-simplices precede
    B-simplices, and lower dimensions precede higher ones.  Every face of an
    A-simplex is an A-simplex and has filtration value <= its coface, so the
    merged sequence is a valid filtration.
    """
    n_a = len(cloud_a)
    union = np.vstack([cloud_a, cloud_b])
    D = _cdist(union, union)
    n = len(union)

    def in_A_simplex(sigma):          # vertices are sorted; max index decides
        return sigma[-1] < n_a

    def fval(sigma):
        if len(sigma) == 1:
            return 0.0
        return float(max(D[u, v] for u, v in _combinations(sigma, 2)))

    ea, eb = [], []
    for dim in range(max_dim + 1):
        for sigma in _combinations(range(n), dim + 1):
            f = fval(sigma)
            if f <= max_edge_length:
                (ea if in_A_simplex(sigma) else eb).append((f, dim, sigma))

    ea.sort(key=lambda e: (e[0], e[1]))
    eb.sort(key=lambda e: (e[0], e[1]))

    merged, ia, ib = [], 0, 0
    while ia < len(ea) and ib < len(eb):
        if ea[ia][0] <= eb[ib][0]:
            merged.append(ea[ia]); ia += 1
        else:
            merged.append(eb[ib]); ib += 1
    merged.extend(ea[ia:])
    merged.extend(eb[ib:])

    simplices = [e[2] for e in merged]
    filt_arr = np.array([e[0] for e in merged], dtype=float)
    in_A = [in_A_simplex(s) for s in simplices]
    in_A_set = {i for i, a in enumerate(in_A) if a}
    idx = {s: i for i, s in enumerate(simplices)}
    N = len(simplices)

    cols_K = [set() for _ in range(N)]
    cols_L = [set() for _ in range(N)]
    for j, sigma in enumerate(simplices):
        if len(sigma) == 1:
            continue
        for k in range(len(sigma)):
            face = sigma[:k] + sigma[k + 1:]
            row = idx.get(face)
            if row is not None:
                cols_K[j].add(row)
                if j in in_A_set:
                    cols_L[j].add(row)
    return simplices, filt_arr, in_A, cols_K, cols_L


def _reduce_sparse(cols):
    """Standard Z_2 persistence reduction on sparse set-columns.
    XOR = symmetric difference; pivot = max row index."""
    cols = [set(c) for c in cols]
    pivots = {}
    for j in range(len(cols)):
        while cols[j]:
            p = max(cols[j])
            if p not in pivots:
                pivots[p] = j
                break
            cols[j] ^= cols[pivots[p]]
    return cols, pivots


def mixup_triples(cloud_a: np.ndarray, cloud_b: np.ndarray,
                  max_edge_length: float = 2.5, degree: int = 1):
    """Full mixup barcode of A -> A u B in degree `degree`.

    Returns a list of (birth, d_prime, death) with birth <= d_prime <= death
    (Observation 2 of Wagner et al.).
    """
    simplices, filt_arr, in_A, cols_K, cols_L = _build_sparse_columns(
        cloud_a, cloud_b, max_edge_length, max_dim=degree + 1
    )
    INF = float(max_edge_length)

    a_rows = [i for i, a in enumerate(in_A) if a]
    b_rows = [i for i, a in enumerate(in_A) if not a]
    perm = a_rows + b_rows
    inv = {old: new for new, old in enumerate(perm)}

    def remap(cols):
        return [{inv[r] for r in col} for col in cols]

    cols_L_red, piv_L = _reduce_sparse(remap(cols_L))
    cols_K_red, piv_K = _reduce_sparse(remap(cols_K))

    pivots_L = {perm[r]: c for r, c in piv_L.items()}
    pivots_K = {perm[r]: c for r, c in piv_K.items()}

    triples = []
    for sigma_idx, sigma in enumerate(simplices):
        if len(sigma) != degree + 1:
            continue
        if not in_A[sigma_idx]:
            continue
        if cols_L_red[sigma_idx]:      # non-zero column -> death simplex, not a birth
            continue

        birth = float(filt_arr[sigma_idx])
        tau_col = pivots_L.get(sigma_idx)
        death = float(filt_arr[tau_col]) if tau_col is not None else INF
        tau_p_col = pivots_K.get(sigma_idx)
        d_prime = float(filt_arr[tau_p_col]) if tau_p_col is not None else INF
        d_prime = float(np.clip(d_prime, birth, death))
        triples.append((birth, d_prime, death))
    return triples


def mixup_barcode_distance(cloud_a: np.ndarray, cloud_b: np.ndarray,
                           max_edge_length: float = 2.5,
                           eps0: float = 1e-10) -> float:
    """Mean H1 mixup percentage MP_1(A, B) in [0, 1].

    MP_1 = mean over non-trivial reference bars of (d - d') / (d - b).
    [B11] eps0 defaults to 1e-10, i.e. only exactly-zero-length bars are
    excluded; there is no practical near-diagonal filtering.
    """
    triples = mixup_triples(cloud_a, cloud_b,
                            max_edge_length=max_edge_length, degree=1)
    if not triples:
        return 0.0
    pcts = [(d - dp) / (d - b) for b, dp, d in triples if d - b > eps0]
    if not pcts:
        return 0.0
    return float(np.clip(np.mean(pcts), 0.0, 1.0))

"""
# ==============================================================================
# 4. STRESS PIPELINE
# ==============================================================================
"""
def _embed_and_persist(prices_window: np.ndarray, dimension: int, delay: int,
                       max_points: int, max_edge_length: float,
                       subsample_seed: int) -> dict:
    cloud = takens_embedding(prices_window, dimension, delay, normalise=True)
    cloud = subsample_cloud(cloud, max_points, seed=subsample_seed)
    res = compute_persistence(cloud, max_dim=1, max_edge_length=max_edge_length)
    res["cloud"] = cloud
    return res


def _day_components(prices_slice: np.ndarray, ref_cloud: np.ndarray,
                    ref_dgm0: np.ndarray, ref_entropy_h1: float,
                    dimension: int, delay: int, max_points: int,
                    max_edge_length: float, subsample_seed: int,
                    mixup_eps: float) -> dict:
    """One trading day's raw stress components.  Module-level so joblib can
    pickle it for the parallel map."""
    from persim import wasserstein as _wasserstein
    res = _embed_and_persist(prices_slice, dimension, delay, max_points,
                             max_edge_length, subsample_seed)
    # [B9] persim.wasserstein is the OPTIMAL-MATCHING 1-Wasserstein distance with
    # a EUCLIDEAN (L2) ground metric -- it sums matched distances, it does not
    # take a p=2 power mean, and it does not use the L-infinity ground metric.
    # The manuscript now calls this  W~_1  with an L2 ground metric.
    h0_dist = float(_wasserstein(ref_dgm0, res["dgm0"]))
    h1_mixup = mixup_barcode_distance(ref_cloud, res["cloud"],
                                      max_edge_length, eps0=mixup_eps)
    ent_div = abs(persistence_entropy(res["dgm1"]) - ref_entropy_h1)
    summary = diagram_summary(res["dgm0"], res["dgm1"])
    return {"h0_dist": h0_dist, "h1_mixup": h1_mixup, "ent_div": ent_div, **summary}


class TDAStressPipeline:
    """Topological stress components + composite score S(t).

    Reference cloud [B4]
    --------------------
    `_build_reference` walks every window inside the first (ref_length + window)
    observations and keeps the MIDDLE one.  The reference is therefore a single
    `window`-day cloud drawn from the middle of that opening slice -- it is NOT a
    120-day cloud, and it is NOT refreshed during the run.

    Scoring
    --------------------
      1. all THREE raw components (including MP_1, which is already in [0,1])
         are rescaled by an EXPANDING min-max -- causal, but it means the MP_1
         entering S(t) is a rescaled MP_1, not the raw mixup percentage;
      2. weighted sum -> s_raw;
      3. trailing `smooth_window`-day mean -> s_smooth;
      4. clip to [0,1] -> stress.
    """

    def __init__(self, window: int, dimension: int, delay: int,
                 max_points: int = 28, ref_length: int = 120,
                 max_edge_length: float = 2.5, subsample_seed: int = 0,
                 mixup_eps: float = 1e-10):
        self.window = int(window)
        self.dimension = int(dimension)
        self.delay = int(delay)
        self.max_points = int(max_points)
        self.ref_length = int(ref_length)
        self.max_edge_length = float(max_edge_length)
        self.subsample_seed = int(subsample_seed)
        self.mixup_eps = float(mixup_eps)

        self.ref_dgm0_ = None
        self.ref_dgm1_ = None
        self.ref_cloud_ = None
        self.ref_entropy_h1_ = None

    @classmethod
    def for_asset(cls, cfg: Config, asset: AssetConfig,
                  window: int | None = None, dimension: int | None = None,
                  delay: int | None = None, ref_length: int | None = None):
        return cls(
            window=asset.window if window is None else window,
            dimension=asset.dimension if dimension is None else dimension,
            delay=asset.delay if delay is None else delay,
            max_points=cfg.max_points,
            ref_length=cfg.ref_length if ref_length is None else ref_length,
            max_edge_length=cfg.max_edge_length,
            subsample_seed=cfg.subsample_seed,
            mixup_eps=cfg.mixup_eps,
        )

    # -- reference ------------------------------------------------------------
    def _build_reference(self, prices: np.ndarray) -> None:
        ref_slice = prices[: self.ref_length + self.window]
        dgm0s, dgm1s, clouds = [], [], []
        for end in range(self.window, len(ref_slice) + 1):
            res = _embed_and_persist(ref_slice[end - self.window: end],
                                     self.dimension, self.delay, self.max_points,
                                     self.max_edge_length, self.subsample_seed)
            dgm0s.append(res["dgm0"]); dgm1s.append(res["dgm1"]); clouds.append(res["cloud"])
        mid = len(dgm0s) // 2
        self.ref_dgm0_ = dgm0s[mid]
        self.ref_dgm1_ = dgm1s[mid]
        self.ref_cloud_ = clouds[mid]
        self.ref_entropy_h1_ = persistence_entropy(self.ref_dgm1_)

    # -- raw components -------------------------------------------------------
    def components(self, prices: pd.Series, n_jobs: int = 1,
                   verbose: bool = True) -> pd.DataFrame:
        values, dates = prices.to_numpy(dtype=float), prices.index
        n = len(values)
        start_offset = self.ref_length + self.window
        if n < start_offset + 1:
            raise ValueError(f"Series too short ({n} days); need > {start_offset}.")

        self._build_reference(values)

        ends = list(range(start_offset, n + 1))
        slices = [values[e - self.window: e] for e in ends]
        kw = dict(ref_cloud=self.ref_cloud_, ref_dgm0=self.ref_dgm0_,
                  ref_entropy_h1=self.ref_entropy_h1_, dimension=self.dimension,
                  delay=self.delay, max_points=self.max_points,
                  max_edge_length=self.max_edge_length,
                  subsample_seed=self.subsample_seed, mixup_eps=self.mixup_eps)

        if n_jobs == 1:
            it = slices
            if verbose:
                try:
                    from tqdm import tqdm
                    it = tqdm(slices, desc="TDA components", unit="day")
                except ImportError:
                    pass
            records = [_day_components(s, **kw) for s in it]
        else:
            records = Parallel(n_jobs=n_jobs, backend="loky",
                               verbose=5 if verbose else 0)(
                delayed(_day_components)(s, **kw) for s in slices
            )

        df = pd.DataFrame(records)
        df.index = pd.DatetimeIndex([dates[e - 1] for e in ends], name="date")
        return df


def score_components(components: pd.DataFrame, weights=(0.45, 0.50, 0.05),
                     smooth_window: int = 30) -> pd.DataFrame:
    """Raw components -> S(t).  Pure, cheap, and reused by every sweep."""
    df = components.copy()
    for raw_col, norm_col in (("h0_dist", "h0_norm"),
                              ("h1_mixup", "h1_norm"),
                              ("ent_div", "ent_norm")):
        col_min = df[raw_col].expanding().min()
        col_rng = (df[raw_col].expanding().max() - col_min).replace(0, 1.0)
        df[norm_col] = ((df[raw_col] - col_min) / col_rng).clip(0, 1)

    w0, w1, w2 = weights
    df["s_raw"] = w0 * df["h0_norm"] + w1 * df["h1_norm"] + w2 * df["ent_norm"]
    df["s_smooth"] = df["s_raw"].rolling(smooth_window, min_periods=1).mean()
    df["stress"] = df["s_smooth"].clip(0, 1)
    return df


# ==============================================================================
# 5. SIGNALS AND BACKTEST
# ==============================================================================

def golden_cross_signal(prices: pd.Series, fast: int = 50,
                        slow: int = 200) -> pd.DataFrame:
    ma_fast = prices.rolling(fast, min_periods=fast).mean()
    ma_slow = prices.rolling(slow, min_periods=slow).mean()
    ma_above = (ma_fast > ma_slow).astype(int)
    gc = ((ma_above == 1) & (ma_above.shift(1) == 0)).astype(int)
    dc = ((ma_above == 0) & (ma_above.shift(1) == 1)).astype(int)
    return pd.DataFrame({"ma_fast": ma_fast, "ma_slow": ma_slow,
                         "ma_above": ma_above, "gc": gc, "dc": dc},
                        index=prices.index)


def sizing_function(stress: float, s_low: float, s_high: float) -> float:
    """alpha = 1 below s_low, 0 above s_high, linear in between.
    Identical to the manuscript's (theta_hi - S)/(theta_hi - theta_lo)."""
    if stress < s_low:
        return 1.0
    if stress > s_high:
        return 0.0
    return 1.0 - (stress - s_low) / (s_high - s_low)


def tda_stress_gc_positions(prices: pd.Series, stress: pd.Series,
                            fast: int = 50, slow: int = 200,
                            s_low: float = 0.30, s_high: float = 0.60) -> pd.DataFrame:
    """EVENT-DRIVEN position sizing [B12].

    A long position is opened ONLY on the day a Golden Cross fires and only if
    S(t) < s_high on that day; a Golden Cross that fires while S(t) >= s_high is
    discarded, and the strategy stays flat until the NEXT crossover.  While
    invested the position is re-sized every day by sizing_function(S(t)).
    A Death Cross forces a flat position unconditionally.
    """
    ma_df = golden_cross_signal(prices, fast, slow)
    combined = ma_df.join(stress.rename("stress"), how="left")
    combined["stress"] = combined["stress"].ffill().fillna(0.0)

    positions, signals = [], []
    current = 0.0
    n_suppressed = 0
    for _, row in combined.iterrows():
        s, gc, dc = row["stress"], row["gc"], row["dc"]
        size = sizing_function(s, s_low, s_high)
        if gc == 1 and s < s_high:
            current, sig = size, "buy"
        elif dc == 1:
            current, sig = 0.0, "sell"
        elif gc == 1 and s >= s_high:
            sig = "suppressed"
            n_suppressed += 1
        else:
            if current > 0:
                current = size
            sig = "no_signal"
        positions.append(current)
        signals.append(sig)

    combined["position"] = positions
    combined["signal"] = signals
    combined.attrs["n_suppressed_gc"] = n_suppressed
    return combined


def compute_portfolio_returns(prices: pd.Series, positions: pd.Series,
                              transaction_cost: float = 0.0005) -> pd.Series:
    """Net daily returns.

    The position is applied with a one-day lag (trade at the close of t,
    earn the return of t+1).  The COST is now lagged by the same one day, so the
    charge for a trade lands on the first day its exposure is actually held.
    The original code charged an unlagged pos.diff(), misaligning cost and PnL
    by one bar.
    """
    returns = prices.pct_change()
    pos, ret = positions.align(returns, join="inner")
    pos = pos.astype(float).fillna(0.0)
    gross = pos.shift(1) * ret
    cost = transaction_cost * pos.diff().abs().shift(1)
    net = (gross - cost).dropna()
    net.name = "portfolio_return"
    return net


def count_trades(positions: pd.Series) -> dict:
    """[C5] Report entries+exits AND round trips.  The old `count_trades`
    returned entries+exits, which the manuscript mislabelled as round trips
    (hence the impossible odd count of 17 for BTC)."""
    # NOTE: `pos.shift(1)` is NaN on the first row and `NaN != 0` evaluates True,
    # so the original implementation booked a phantom EXIT on day 0 for every
    # series that starts flat.  Filling the lagged series with 0.0 removes it.
    pos = positions.fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    entries = int(((pos != 0) & (prev == 0)).sum())
    exits = int(((pos == 0) & (prev != 0)).sum())
    return {"n_signals": entries + exits, "n_entries": entries,
            "n_exits": exits, "n_round_trips": entries}


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0,
                        trading_days: int = 252) -> dict:
    """[C7] Sharpe and Sortino now share the SAME numerator: the annualised
    arithmetic mean excess return.  The original mixed an arithmetic Sharpe with
    a geometric Sortino."""
    keys = ("ann_return", "sharpe", "sortino", "max_dd", "calmar", "volatility",
            "win_rate", "positive_months", "total_return")
    if returns is None or returns.empty or returns.isna().all():
        return {**{k: 0.0 for k in keys}, "n_days": 0}

    r = returns.dropna()
    rf_daily = risk_free_rate / trading_days
    excess = r - rf_daily
    n_years = len(r) / trading_days
    total_ret = float((1 + r).prod())
    ann_return = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = float(r.std() * np.sqrt(trading_days))

    ann_excess = float(excess.mean() * trading_days)
    sharpe = ann_excess / vol if vol > 0 else 0.0

    downside = excess[excess < 0]
    ds_std = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 1 else 0.0
    sortino = ann_excess / ds_std if ds_std > 0 else 0.0

    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.expanding().max()
    max_dd = float(((cumulative - rolling_max) / rolling_max).min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    active = r[r != 0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pos_mo = float((monthly > 0).mean()) if len(monthly) else 0.0

    return {"ann_return": float(ann_return), "sharpe": float(sharpe),
            "sortino": float(sortino), "max_dd": max_dd, "calmar": float(calmar),
            "volatility": vol, "win_rate": win_rate, "positive_months": pos_mo,
            "n_days": int(len(r)), "total_return": float(total_ret - 1)}


def backtest(prices: pd.Series, positions: pd.Series,
             transaction_cost: float = 0.0005, risk_free_rate: float = 0.0,
             trading_days: int = 252, strategy_name: str = "Strategy") -> dict:
    returns = compute_portfolio_returns(prices, positions, transaction_cost)
    metrics = performance_metrics(returns, risk_free_rate, trading_days)
    cumulative = (1 + returns).cumprod()
    return {**metrics, **count_trades(positions), "returns": returns,
            "cumulative": cumulative, "name": strategy_name}


# ==============================================================================
# 6. BENCHMARK STRATEGIES
# ==============================================================================

def buy_and_hold(prices: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="Buy & Hold")


def golden_cross_bench(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    pos = (prices.rolling(fast, min_periods=fast).mean() >
           prices.rolling(slow, min_periods=slow).mean()).astype(float)
    pos.name = "Golden Cross"
    return pos


def ema_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    pos = (prices.ewm(span=fast, adjust=False).mean() >
           prices.ewm(span=slow, adjust=False).mean()).astype(float)
    pos.name = "EMA 20/50"
    return pos


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def macd_rsi(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
             rsi_period: int = 14, rsi_low: int = 40, rsi_high: int = 76) -> pd.Series:
    """[E4] Renamed from `macd_adx`: ADX was never computed, RSI is the filter."""
    macd_line = (prices.ewm(span=fast, adjust=False).mean()
                 - prices.ewm(span=slow, adjust=False).mean())
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    rsi = _rsi(prices, rsi_period)
    above = (macd_line > sig_line).astype(int)
    cross_up = ((above == 1) & (above.shift(1) == 0)).to_numpy()
    cross_dn = ((above == 0) & (above.shift(1) == 1)).to_numpy()
    ok = ((rsi >= rsi_low) & (rsi <= rsi_high)).to_numpy()

    out, position = np.empty(len(prices)), 0.0
    for i in range(len(prices)):
        if cross_up[i] and ok[i]:
            position = 1.0
        elif cross_dn[i]:
            position = 0.0
        out[i] = position
    return pd.Series(out, index=prices.index, name="MACD + RSI")


def rsi_mean_reversion(prices: pd.Series, period: int = 14,
                       oversold: int = 30, overbought: int = 70) -> pd.Series:
    rsi = _rsi(prices, period).to_numpy()
    out, position = np.empty(len(rsi)), 0.0
    for i in range(len(rsi)):
        r = rsi[i]
        r_prev = rsi[i - 1] if i > 0 else r
        if r_prev <= oversold < r:
            position = 1.0
        elif r_prev <= overbought < r:
            position = 0.0
        out[i] = position
    return pd.Series(out, index=prices.index, name="RSI Reversion")


def bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    lower, upper = (sma - n_std * std).to_numpy(), (sma + n_std * std).to_numpy()
    p = prices.to_numpy()
    out, position = np.empty(len(p)), 0.0
    for i in range(len(p)):
        if np.isfinite(lower[i]) and p[i] <= lower[i]:
            position = 1.0
        elif np.isfinite(upper[i]) and p[i] >= upper[i]:
            position = 0.0
        out[i] = position
    return pd.Series(out, index=prices.index, name="Bollinger Bands")


def macd_rsi_dual(prices: pd.Series, macd_fast: int = 12, macd_slow: int = 26,
                  macd_signal: int = 9, rsi_period: int = 14,
                  rsi_threshold: int = 50) -> pd.Series:
    line = (prices.ewm(span=macd_fast, adjust=False).mean()
            - prices.ewm(span=macd_slow, adjust=False).mean())
    hist = line - line.ewm(span=macd_signal, adjust=False).mean()
    pos = ((hist > 0) & (_rsi(prices, rsi_period) > rsi_threshold)).astype(float)
    pos.name = "MACD+RSI Dual"
    return pos


BENCHMARK_REGISTRY = {
    "Buy & Hold": buy_and_hold,
    "Golden Cross": golden_cross_bench,
    "EMA 20/50": ema_crossover,
    "MACD + RSI": macd_rsi,
    "RSI Reversion": rsi_mean_reversion,
    "Bollinger Bands": bollinger_bands,
    "MACD+RSI Dual": macd_rsi_dual,
}


def run_all_benchmarks(prices: pd.Series, fast: int = 50, slow: int = 200) -> dict:
    """Run every benchmark.  `fast`/`slow` are forwarded to the Golden Cross so
    the benchmark always uses the SAME moving averages as the TDA strategy; the
    notebook called it with hard-coded 50/200 defaults regardless of CFG."""
    out = {}
    for name, fn in BENCHMARK_REGISTRY.items():
        try:
            out[name] = (fn(prices, fast=fast, slow=slow)
                         if name == "Golden Cross" else fn(prices))
        except Exception as exc:                       # pragma: no cover
            log.warning("Benchmark %s failed: %s", name, exc)
    return out

"""
# ==============================================================================
# 7. STATISTICAL TESTS (Currently under work)
# ==============================================================================

def diebold_mariano_test(returns_a: pd.Series, returns_b: pd.Series,
                         alternative: str = "greater") -> dict:
    """DM test on squared-return loss differentials.

    d_t = b_t^2 - a_t^2, so a POSITIVE mean favours strategy A (smaller loss),
    which makes `alternative='greater'` read as 'A beats B'.  The original code
    used d_t = a^2 - b^2 with the same one-sided label, i.e. the sign convention
    contradicted the stated hypothesis.
    """
    a, b = returns_a.align(returns_b, join="inner")
    mask = a.notna() & b.notna()
    a, b = a[mask].to_numpy(), b[mask].to_numpy()
    if len(a) < 3:
        return {"statistic": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0,
                "n": len(a), "conclusion": "insufficient overlap"}
    d = b ** 2 - a ** 2
    n, d_bar = len(d), float(d.mean())
    var_d = float(np.var(d, ddof=1)) / n
    dm = d_bar / np.sqrt(max(var_d, 1e-16))
    if alternative == "greater":
        p = float(1 - scipy_stats.norm.cdf(dm))
    elif alternative == "less":
        p = float(scipy_stats.norm.cdf(dm))
    else:
        p = float(2 * min(scipy_stats.norm.cdf(dm), 1 - scipy_stats.norm.cdf(dm)))
    return {"statistic": float(dm), "p_value": p, "mean_loss_diff": d_bar, "n": n,
            "conclusion": ("Reject" if p < 0.05 else "Fail to reject") +
                          f" H0 at 5% (p={p:.4f}, DM={dm:.3f})"}


_STAT_FNS = {
    "sharpe": lambda r: float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
    "ann_return": lambda r: float((1 + r).prod() ** (252 / len(r)) - 1),
    "max_dd": lambda r: float(((np.cumprod(1 + r) - np.maximum.accumulate(np.cumprod(1 + r)))
                               / np.maximum.accumulate(np.cumprod(1 + r))).min()),
    "sortino": lambda r: (float(r.mean() * 252 / (r[r < 0].std() * np.sqrt(252)))
                          if (r < 0).sum() > 1 and r[r < 0].std() > 0 else 0.0),
}


def _one_bootstrap(r: np.ndarray, block_size: int, seed: int, stat: str) -> float:
    rng = np.random.default_rng(seed)
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
    sample = np.concatenate([r[s: s + block_size] for s in starts])[:n]
    return _STAT_FNS[stat](sample)


def block_bootstrap_ci(returns: pd.Series, stat: str = "sharpe", n_reps: int = 1000,
                       ci_level: float = 0.95, block_size: int = 20,
                       seed: int = 42, n_jobs: int = 1) -> dict:
    r = returns.dropna().to_numpy()
    observed = _STAT_FNS[stat](r)
    if n_jobs == 1:
        boot = [_one_bootstrap(r, block_size, seed + i, stat) for i in range(n_reps)]
    else:
        boot = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_bootstrap)(r, block_size, seed + i, stat)
            for i in range(n_reps))
    boot = np.asarray(boot, dtype=float)
    lo = float(np.percentile(boot, 100 * (1 - ci_level) / 2))
    hi = float(np.percentile(boot, 100 * (1 + ci_level) / 2))
    return {"metric": stat, "observed": float(observed), "ci_lower": lo,
            "ci_upper": hi, "std_err": float(boot.std(ddof=1))}


def holm_bonferroni(pvals: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction over the family of DM tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, prev = [], 0.0
    for i, (label, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        rows.append({"comparison": label, "p_raw": p, "p_holm": adj,
                     "reject_at_%.2f" % alpha: adj < alpha})
    return pd.DataFrame(rows)


# ==============================================================================
# 8. ML FEATURES AND MODELS
# ==============================================================================

def compute_ta_features(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    log_ret = np.log(prices / prices.shift(1))
    ma50, ma200 = prices.rolling(50).mean(), prices.rolling(200).mean()
    df["ma_ratio"] = (ma50 / ma200).replace([np.inf, -np.inf], np.nan)
    df["rsi"] = _rsi(prices, 14)
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    df["macd"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    sma20, std20 = prices.rolling(20).mean(), prices.rolling(20).std()
    bb_range = (4 * std20).replace(0, np.nan)
    df["bb_position"] = ((prices - (sma20 - 2 * std20)) / bb_range).clip(0, 1)
    df["ret_5d"] = log_ret.rolling(5).sum()
    df["ret_20d"] = log_ret.rolling(20).sum()
    vol5, vol20 = log_ret.rolling(5).std(), log_ret.rolling(20).std()
    df["vol_ratio"] = (vol5 / vol20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["atr_norm"] = prices.diff().abs().ewm(com=13, adjust=False).mean() / prices
    return df


def extract_tda_features(stress_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["stress", "h0_dist", "h1_mixup", "entropy_h1",
            "total_pers_h1", "betti0", "betti1"]
    present = [c for c in cols if c in stress_df.columns]
    return stress_df[present].rename(columns={"stress": "tda_stress"})


def build_target(prices: pd.Series, horizon: int = 5) -> pd.Series:
    future = prices.shift(-horizon) / prices - 1
    target = (future > 0).astype(float)
    target.iloc[-horizon:] = np.nan
    target.name = f"target_h{horizon}"
    return target


def build_feature_matrix(prices: pd.Series, stress_df: pd.DataFrame, horizon: int = 5):
    ta_df = compute_ta_features(prices)
    tda_df = extract_tda_features(stress_df)
    target = build_target(prices, horizon)
    common = ta_df.index.intersection(tda_df.index).intersection(target.dropna().index)
    ta_df, tda_df, target = ta_df.loc[common], tda_df.loc[common], target.loc[common]
    all_df = pd.concat([tda_df, ta_df], axis=1)
    mask = ~(ta_df.isna().any(axis=1) | tda_df.isna().any(axis=1) | target.isna())
    return ta_df[mask], tda_df[mask], all_df[mask], target[mask]


def _get_classifiers(seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                                                   random_state=seed))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                               random_state=seed, n_jobs=1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                       learning_rate=0.05,
                                                       random_state=seed),
        "MLP": Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                               random_state=seed, early_stopping=True))]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.05, random_state=seed,
                                          eval_metric="logloss", verbosity=0, n_jobs=1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=4,
                                            learning_rate=0.05, random_state=seed,
                                            verbose=-1, n_jobs=1)
    except ImportError:
        pass
    return models


def _evaluate_model(model, X_train, X_test, y_train, y_test, ret_test=None) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    proba = (model.predict_proba(X_test)[:, 1]
             if hasattr(model, "predict_proba") else y_pred.astype(float))
    acc = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = 0.5
    sharpe = 0.0
    if ret_test is not None and len(ret_test) == len(proba):
        pr = (2.0 * proba - 1.0) * ret_test.to_numpy()
        pr = pr[np.isfinite(pr)]
        if pr.size and pr.std() > 0:
            sharpe = float(pr.mean() / pr.std() * np.sqrt(252))
    return {"accuracy": acc, "auc": auc, "sharpe": sharpe}


def run_ml_experiment(f_ta, f_tda, f_all, target, prices, test_size=0.30,
                      seed=42, horizon=5, n_jobs=1):
    """[E2] `horizon` is now the prediction horizon.  The notebook passed
    CFG['ml']['seed'] (=42) here, so the forward return used for the simulated
    Sharpe was a 42-day return while the target was a 5-day direction."""
    returns = prices.pct_change(periods=horizon)
    groups = {"TA only": f_ta, "TDA only": f_tda, "TDA+TA": f_all}
    classifiers = _get_classifiers(seed)
    records, importances = [], {}

    for group_name, X in groups.items():
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = target.iloc[:split], target.iloc[split:]
        ret_test = returns.reindex(X_test.index)
        log.info("  ML group %-9s train=%d test=%d features=%d",
                 group_name, len(X_train), len(X_test), X.shape[1])
        for model_name, model in classifiers.items():
            try:
                res = _evaluate_model(model, X_train, X_test, y_train, y_test, ret_test)
                records.append({"group": group_name, "model": model_name, **res,
                                "n_train": len(X_train), "n_test": len(X_test),
                                "n_features": X.shape[1]})
                if model_name == "RandomForest" and group_name == "TDA+TA":
                    clf = getattr(model, "named_steps", {}).get("clf", model)
                    if hasattr(clf, "feature_importances_"):
                        importances["TDA+TA_RandomForest"] = pd.Series(
                            clf.feature_importances_, index=X.columns
                        ).sort_values(ascending=False)
            except Exception as exc:                    # pragma: no cover
                log.warning("    %s failed: %s", model_name, exc)
    return pd.DataFrame(records), importances


# ==============================================================================
# 9. WALK-FORWARD VALIDATION  [E3]
# ==============================================================================

def _one_fold(prices: pd.Series, start: int, train_w: int, test_w: int,
              cfg: Config, asset: AssetConfig, fold_idx: int) -> dict | None:
    n = len(prices)
    train_end = start + train_w
    test_end = min(train_end + test_w, n)
    test_prices = prices.iloc[train_end:test_end]
    if len(test_prices) < 5:
        return None
    full_prices = prices.iloc[start:test_end]
    try:
        pipe = TDAStressPipeline.for_asset(cfg, asset)
        comps = pipe.components(full_prices, n_jobs=1, verbose=False)
        scored = score_components(comps, cfg.weights(), cfg.smooth_window)
        pos_df = tda_stress_gc_positions(full_prices, scored["stress"],
                                         cfg.ma_fast, cfg.ma_slow,
                                         cfg.s_low, cfg.s_high)
        positions = pos_df["position"].reindex(test_prices.index)
        oos = compute_portfolio_returns(test_prices, positions, cfg.transaction_cost)
        m = performance_metrics(oos, cfg.risk_free_rate, cfg.trading_days)
        m.update({"fold": fold_idx,
                  "test_start": str(test_prices.index[0].date()),
                  "test_end": str(test_prices.index[-1].date())})
        return {"metrics": m, "oos": oos}
    except Exception as exc:                            # pragma: no cover
        log.warning("Fold %d failed: %s", fold_idx, exc)
        return None


def run_walkforward(prices: pd.Series, cfg: Config, asset: AssetConfig,
                    n_jobs: int = -1) -> dict:
    n = len(prices)
    need = cfg.ref_length + asset.window
    train_w = max(cfg.wf_train, need + 20)     # a fold must fit the reference slice
    starts = list(range(0, n - train_w - cfg.wf_test + 1, cfg.wf_step))
    if not starts:
        raise ValueError(f"Series too short for walk-forward (need > {train_w + cfg.wf_test}).")
    log.info("Walk-forward: %d folds (train=%d, test=%d, step=%d)",
             len(starts), train_w, cfg.wf_test, cfg.wf_step)
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_one_fold)(prices, s, train_w, cfg.wf_test, cfg, asset, i)
        for i, s in enumerate(starts))
    out = [o for o in out if o is not None]
    if not out:
        raise RuntimeError("Walk-forward produced no valid folds.")
    oos_series = pd.concat([o["oos"] for o in out]).sort_index()
    oos_series = oos_series[~oos_series.index.duplicated(keep="first")]
    return {"oos_returns": oos_series,
            "fold_metrics": [o["metrics"] for o in out],
            "aggregate": performance_metrics(oos_series, cfg.risk_free_rate,
                                             cfg.trading_days),
            "n_folds": len(out)}


# ==============================================================================
# 10. GRID SEARCH  [A2]  -- single definition feeding both Fig. 3 and Tables 1-2
# ==============================================================================

def _one_grid_run(prices: pd.Series, cfg: Config, asset: AssetConfig,
                  window: int, dimension: int, delay: int) -> dict:
    rec = {"ticker": asset.ticker, "window": window,
           "dimension": dimension, "delay": delay}
    try:
        pipe = TDAStressPipeline.for_asset(cfg, asset, window=window,
                                           dimension=dimension, delay=delay)
        comps = pipe.components(prices, n_jobs=1, verbose=False)
        scored = score_components(comps, cfg.weights(), cfg.smooth_window)
        pos_df = tda_stress_gc_positions(prices, scored["stress"], cfg.ma_fast,
                                         cfg.ma_slow, cfg.s_low, cfg.s_high)
        res = backtest(prices, pos_df["position"], cfg.transaction_cost,
                       cfg.risk_free_rate, cfg.trading_days)
        rec.update({k: res[k] for k in ("ann_return", "sharpe", "sortino",
                                        "max_dd", "calmar", "n_signals",
                                        "n_round_trips")})
        rec["error"] = None
    except Exception as exc:
        rec.update({k: np.nan for k in ("ann_return", "sharpe", "sortino",
                                        "max_dd", "calmar")})
        rec.update({"n_signals": np.nan, "n_round_trips": np.nan, "error": str(exc)})
    return rec


def run_grid_search(prices: pd.Series, cfg: Config, asset: AssetConfig,
                    n_jobs: int = -1) -> pd.DataFrame:
    combos = list(product(cfg.grid_windows, cfg.grid_dimensions, cfg.grid_delays))
    log.info("Grid search %s: %d configurations", asset.ticker, len(combos))
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_one_grid_run)(prices, cfg, asset, w, d, t) for w, d, t in combos)
    return pd.DataFrame(rows)


# ==============================================================================
# 11. ROBUSTNESS SWEEPS  [D]
# ==============================================================================
# Section 6.5 of the manuscript asserts four robustness results.  None of them
# had supporting code.  These four functions produce them.  Weight and threshold
# sweeps reuse a single component frame and therefore cost seconds; the two
# window sweeps require recomputation and are parallelised over configurations.

def _components_for(prices, cfg, asset, window=None, ref_length=None):
    pipe = TDAStressPipeline.for_asset(cfg, asset, window=window,
                                       ref_length=ref_length)
    return pipe.components(prices, n_jobs=1, verbose=False)


def _rob_recompute_run(prices, cfg, asset, window, ref_length, label) -> dict:
    rec = {"sweep": label, "window": window, "ref_length": ref_length}
    try:
        comps = _components_for(prices, cfg, asset, window=window, ref_length=ref_length)
        scored = score_components(comps, cfg.weights(), cfg.smooth_window)
        pos = tda_stress_gc_positions(prices, scored["stress"], cfg.ma_fast,
                                      cfg.ma_slow, cfg.s_low, cfg.s_high)
        res = backtest(prices, pos["position"], cfg.transaction_cost,
                       cfg.risk_free_rate, cfg.trading_days)
        rec.update({k: res[k] for k in ("ann_return", "sharpe", "sortino",
                                        "max_dd", "calmar", "n_round_trips")})
        rec["error"] = None
    except Exception as exc:
        rec.update({k: np.nan for k in ("ann_return", "sharpe", "sortino",
                                        "max_dd", "calmar", "n_round_trips")})
        rec["error"] = str(exc)
    return rec


def robustness_reference_window(prices, cfg, asset, n_jobs=-1) -> pd.DataFrame:
    """Vary T = ref_length with the current window held at its calibrated value."""
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_rob_recompute_run)(prices, cfg, asset, asset.window, T,
                                    "reference_window")
        for T in cfg.rob_ref_lengths)
    return pd.DataFrame(rows)


def robustness_current_window(prices, cfg, asset, n_jobs=-1) -> pd.DataFrame:
    """Vary w with T held at cfg.ref_length."""
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_rob_recompute_run)(prices, cfg, asset, w, cfg.ref_length,
                                    "current_window")
        for w in cfg.rob_windows)
    return pd.DataFrame(rows)


def robustness_weights(prices, cfg, asset, components: pd.DataFrame) -> pd.DataFrame:
    """Perturb each weight by +/- rob_weight_pct and renormalise to sum 1.

    27 combinations (3 multipliers on each of 3 weights).  Reuses `components`,
    so no homology is recomputed.
    """
    base = np.array(cfg.weights(), dtype=float)
    mults = (1 - cfg.rob_weight_pct, 1.0, 1 + cfg.rob_weight_pct)
    rows = []
    for m0, m1, m2 in product(mults, mults, mults):
        w = base * np.array([m0, m1, m2])
        w = w / w.sum()
        scored = score_components(components, tuple(w), cfg.smooth_window)
        pos = tda_stress_gc_positions(prices, scored["stress"], cfg.ma_fast,
                                      cfg.ma_slow, cfg.s_low, cfg.s_high)
        res = backtest(prices, pos["position"], cfg.transaction_cost,
                       cfg.risk_free_rate, cfg.trading_days)
        rows.append({"sweep": "weights", "w_h0": round(float(w[0]), 4),
                     "w_mp1": round(float(w[1]), 4), "w_ent": round(float(w[2]), 4),
                     "mult_h0": m0, "mult_mp1": m1, "mult_ent": m2,
                     **{k: res[k] for k in ("ann_return", "sharpe", "sortino",
                                            "max_dd", "calmar", "n_round_trips")}})
    return pd.DataFrame(rows)


def robustness_thresholds(prices, cfg, asset, components: pd.DataFrame) -> pd.DataFrame:
    """Sweep (theta_lo, theta_hi) on a fixed stress series, plus the Golden
    Cross benchmark Sharpe so the comparison in Section 6.5 is explicit."""
    scored = score_components(components, cfg.weights(), cfg.smooth_window)
    gc_res = backtest(prices, golden_cross_bench(prices, cfg.ma_fast, cfg.ma_slow),
                      cfg.transaction_cost, cfg.risk_free_rate, cfg.trading_days)
    gc_sharpe = gc_res["sharpe"]
    rows = []
    for lo, hi in product(cfg.rob_theta_lo, cfg.rob_theta_hi):
        if hi <= lo:
            continue
        pos = tda_stress_gc_positions(prices, scored["stress"], cfg.ma_fast,
                                      cfg.ma_slow, lo, hi)
        res = backtest(prices, pos["position"], cfg.transaction_cost,
                       cfg.risk_free_rate, cfg.trading_days)
        rows.append({"sweep": "thresholds", "theta_lo": lo, "theta_hi": hi,
                     **{k: res[k] for k in ("ann_return", "sharpe", "sortino",
                                            "max_dd", "calmar", "n_round_trips")},
                     "gc_sharpe": gc_sharpe,
                     "beats_gc": bool(res["sharpe"] > gc_sharpe)})
    return pd.DataFrame(rows)


# ==============================================================================
# 12. PLOTTING
# ==============================================================================

COLOURS = {"TDA Stress GC": "#1a6faf", "Buy & Hold": "#7f7f7f",
           "Golden Cross": "#e07b23", "EMA 20/50": "#2ca02c",
           "MACD + RSI": "#9467bd", "RSI Reversion": "#8c564b",
           "Bollinger Bands": "#e377c2", "MACD+RSI Dual": "#17becf"}


def _savefig(fig, path, dpi=150):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_price_stress(prices, stress, signals_df, s_low, s_high, title, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(prices.index, prices.to_numpy(), lw=1.0, color="#1f77b4", label=title)
    gc_d = signals_df.index[signals_df["gc"] == 1]
    dc_d = signals_df.index[signals_df["dc"] == 1]
    ax1.scatter(gc_d, prices.reindex(gc_d), marker="^", color="green", s=55,
                zorder=5, label="Golden Cross")
    ax1.scatter(dc_d, prices.reindex(dc_d), marker="v", color="red", s=55,
                zorder=5, label="Death Cross")
    ax1.set_ylabel("Price"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(stress.index, stress.to_numpy(), color="#c0392b", lw=1.0, label="Stress S(t)")
    ax2.fill_between(stress.index, 0, stress.to_numpy(), color="#c0392b", alpha=0.18)
    ax2.axhline(s_low, ls="--", lw=0.9, color="green", label=f"Low ({s_low})")
    ax2.axhline(s_high, ls="--", lw=0.9, color="grey", label=f"High ({s_high})")
    ax2.set_ylim(0, 1); ax2.set_ylabel("Stress S(t)"); ax2.set_xlabel("Date")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, path)


def plot_cumulative_wealth(strategy_returns: dict, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, r in strategy_returns.items():
        ax.plot(r.index, (1 + r).cumprod().to_numpy(), lw=1.2,
                color=COLOURS.get(name), label=name)
    ax.set_yscale("log"); ax.set_ylabel("Cumulative wealth (log)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, path)


def plot_drawdown(strategy_returns: dict, path, keys=("TDA Stress GC",
                                                      "Buy & Hold", "Golden Cross")):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in keys:
        if name not in strategy_returns:
            continue
        c = (1 + strategy_returns[name]).cumprod()
        dd = (c - c.expanding().max()) / c.expanding().max()
        ax.plot(dd.index, dd.to_numpy(), lw=1.0, color=COLOURS.get(name), label=name)
        ax.fill_between(dd.index, dd.to_numpy(), 0, alpha=0.12, color=COLOURS.get(name))
    ax.set_ylabel("Drawdown"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, path)


def plot_grid_heatmaps(grid_df: pd.DataFrame, metric: str, path,
                       highlight: dict | None = None):
    """Figure-3 style heatmap: rows = window, columns = delay grouped by dimension.
    `highlight` = {ticker: (w, d, tau)} draws a box round the main-text cell, so
    Fig. 3 and Tables 1-2 can be checked against each other by eye. [A2]"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tickers = list(grid_df["ticker"].unique())
    windows = sorted(grid_df["window"].unique())
    dims = sorted(grid_df["dimension"].unique())
    delays = sorted(grid_df["delay"].unique())

    fig, axes = plt.subplots(len(tickers), 1,
                             figsize=(1.05 * len(dims) * len(delays), 2.6 * len(tickers)),
                             squeeze=False)
    for ai, tk in enumerate(tickers):
        ax = axes[ai][0]
        sub = grid_df[grid_df["ticker"] == tk]
        mat = np.full((len(windows), len(dims) * len(delays)), np.nan)
        for _, r in sub.iterrows():
            if not np.isfinite(r[metric]):
                continue
            i = windows.index(r["window"])
            j = dims.index(r["dimension"]) * len(delays) + delays.index(r["delay"])
            mat[i, j] = r[metric]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_yticks(range(len(windows))); ax.set_yticklabels(windows)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(d) for _ in dims for d in delays], fontsize=6)
        for k in range(1, len(dims)):
            ax.axvline(k * len(delays) - 0.5, color="white", lw=2)
        for k, d in enumerate(dims):
            ax.text((k + 0.5) * len(delays) - 0.5, -0.75, f"d = {d}",
                    ha="center", fontsize=8)
        ax.set_ylabel("window w"); ax.set_title(tk, loc="left", fontsize=9)
        if highlight and tk in highlight:
            hw, hd, ht = highlight[tk]
            if hw in windows and hd in dims and ht in delays:
                i = windows.index(hw)
                j = dims.index(hd) * len(delays) + delays.index(ht)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=2.2))
        fig.colorbar(im, ax=ax, label=metric)
    axes[-1][0].set_xlabel("delay tau, grouped by embedding dimension")
    fig.tight_layout()
    return _savefig(fig, path)


def plot_walkforward_folds(fold_metrics, path):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(fold_metrics)
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.45), 4))
    ax.bar(df["fold"], df["sharpe"],
           color=["#1a6faf" if s >= 0 else "#cc3333" for s in df["sharpe"]])
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(df["sharpe"].mean(), color="orange", ls="--",
               label=f"Mean = {df['sharpe'].mean():.3f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("OOS Sharpe"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _savefig(fig, path)


def plot_bootstrap_ci(ci_df: pd.DataFrame, metric: str, path):
    import matplotlib.pyplot as plt
    sub = ci_df[ci_df["metric"] == metric]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(sub))
    obs = sub["observed"].to_numpy()
    ax.bar(x, obs, width=0.5,
           color=[COLOURS.get(s, "#555") for s in sub["strategy"]], alpha=0.85)
    ax.errorbar(x, obs, yerr=[obs - sub["ci_lower"].to_numpy(),
                              sub["ci_upper"].to_numpy() - obs],
                fmt="none", color="black", capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(sub["strategy"], rotation=15, ha="right")
    ax.set_ylabel(metric); ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _savefig(fig, path)


# ==============================================================================
# 13. STAGE DRIVERS
# ==============================================================================

def _out(cfg: Config, asset_key: str, kind: str) -> Path:
    p = Path(cfg.results_dir) / asset_key / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_backtest(prices, cfg, asset, asset_key, n_jobs=-1, make_plots=True):
    """Main-text result: stress score, positions, Tables 1/2, benchmark table."""
    log.info("[%s] computing TDA stress components (w=%d, d=%d, tau=%d)",
             asset.ticker, asset.window, asset.dimension, asset.delay)
    pipe = TDAStressPipeline.for_asset(cfg, asset)
    comps = pipe.components(prices, n_jobs=n_jobs, verbose=True)
    scored = score_components(comps, cfg.weights(), cfg.smooth_window)
    scored.to_csv(_out(cfg, asset_key, "tables") / "stress_scores.csv")

    pos_df = tda_stress_gc_positions(prices, scored["stress"], cfg.ma_fast,
                                     cfg.ma_slow, cfg.s_low, cfg.s_high)
    pos_df.to_csv(_out(cfg, asset_key, "tables") / "positions.csv")

    tda_res = backtest(prices, pos_df["position"], cfg.transaction_cost,
                       cfg.risk_free_rate, cfg.trading_days, "TDA Stress GC")
    all_metrics = {"TDA Stress GC": tda_res}
    strategy_returns = {"TDA Stress GC": tda_res["returns"]}
    for name, pos in run_all_benchmarks(prices, cfg.ma_fast, cfg.ma_slow).items():
        res = backtest(prices, pos, cfg.transaction_cost, cfg.risk_free_rate,
                       cfg.trading_days, name)
        all_metrics[name] = res
        strategy_returns[name] = res["returns"]

    drop = ("returns", "cumulative", "name")
    metrics_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in drop}
                               for k, v in all_metrics.items()}).T
    metrics_df.to_csv(_out(cfg, asset_key, "tables") / "strategy_comparison.csv")

    log.info("\n=== %s strategy comparison ===\n%s", asset.ticker,
             metrics_df[["ann_return", "sharpe", "sortino", "max_dd", "calmar",
                         "n_round_trips", "n_signals"]].round(3).to_string())
    log.info("Suppressed Golden Cross signals (S >= s_high): %d",
             pos_df.attrs.get("n_suppressed_gc", 0))
    log.info("Sample: %s -> %s (%d days); stress obs: %d",
             prices.index[0].date(), prices.index[-1].date(), len(prices), len(scored))

    if make_plots:
        figs = _out(cfg, asset_key, "figures")
        plot_price_stress(prices, scored["stress"], pos_df, cfg.s_low, cfg.s_high,
                          asset.ticker, figs / "fig_price_stress.png")
        plot_cumulative_wealth(strategy_returns, figs / "fig_cumulative_wealth.png")
        plot_drawdown(strategy_returns, figs / "fig_drawdown.png")

    return {"components": comps, "scored": scored, "positions": pos_df,
            "metrics_df": metrics_df, "all_metrics": all_metrics,
            "strategy_returns": strategy_returns}


def stage_grid(prices, cfg, asset, asset_key, n_jobs=-1):
    grid = run_grid_search(prices, cfg, asset, n_jobs=n_jobs)
    grid.to_csv(_out(cfg, asset_key, "tables") / "grid_search.csv", index=False)
    ok = grid.dropna(subset=["sharpe"])
    if not ok.empty:
        best = ok.loc[ok["sharpe"].idxmax()]
        log.info("[%s] grid argmax Sharpe=%.4f at w=%d, d=%d, tau=%d "
                 "| grid mean Sharpe=%.4f, median=%.4f, min=%.4f",
                 asset.ticker, best["sharpe"], best["window"], best["dimension"],
                 best["delay"], ok["sharpe"].mean(), ok["sharpe"].median(),
                 ok["sharpe"].min())
        if (best["window"], best["dimension"], best["delay"]) != \
           (asset.window, asset.dimension, asset.delay):
            log.warning("[%s] grid argmax differs from the configured main-text "
                        "parameters (w=%d, d=%d, tau=%d). Update AssetConfig or "
                        "the manuscript.", asset.ticker, asset.window,
                        asset.dimension, asset.delay)
    return grid


def stage_robustness(prices, cfg, asset, asset_key, components=None, n_jobs=-1):
    if components is None:
        pipe = TDAStressPipeline.for_asset(cfg, asset)
        components = pipe.components(prices, n_jobs=n_jobs, verbose=True)
    tables = _out(cfg, asset_key, "tables")

    ref_df = robustness_reference_window(prices, cfg, asset, n_jobs=n_jobs)
    win_df = robustness_current_window(prices, cfg, asset, n_jobs=n_jobs)
    wt_df = robustness_weights(prices, cfg, asset, components)
    th_df = robustness_thresholds(prices, cfg, asset, components)

    for name, df in (("robustness_reference_window", ref_df),
                     ("robustness_current_window", win_df),
                     ("robustness_weights", wt_df),
                     ("robustness_thresholds", th_df)):
        df.to_csv(tables / f"{name}.csv", index=False)

    base_sharpe = float(wt_df.loc[(wt_df["mult_h0"] == 1.0) &
                                  (wt_df["mult_mp1"] == 1.0) &
                                  (wt_df["mult_ent"] == 1.0), "sharpe"].iloc[0])
    summary = {
        "asset": asset.ticker,
        "baseline_sharpe": round(base_sharpe, 4),
        "ref_window_sharpe_range": [round(float(ref_df["sharpe"].min()), 4),
                                    round(float(ref_df["sharpe"].max()), 4)],
        "current_window_sharpe_range": [round(float(win_df["sharpe"].min()), 4),
                                        round(float(win_df["sharpe"].max()), 4)],
        "weight_perturbation_max_abs_sharpe_change":
            round(float((wt_df["sharpe"] - base_sharpe).abs().max()), 4),
        "threshold_grid_n": int(len(th_df)),
        "threshold_grid_frac_beating_gc": round(float(th_df["beats_gc"].mean()), 4),
        "threshold_grid_sharpe_range": [round(float(th_df["sharpe"].min()), 4),
                                        round(float(th_df["sharpe"].max()), 4)],
    }
    (tables / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[%s] robustness summary:\n%s", asset.ticker, json.dumps(summary, indent=2))
    return {"reference": ref_df, "window": win_df, "weights": wt_df,
            "thresholds": th_df, "summary": summary}


def stage_walkforward(prices, cfg, asset, asset_key, all_metrics, n_jobs=-1):
    tables, figs = _out(cfg, asset_key, "tables"), _out(cfg, asset_key, "figures")
    wf = run_walkforward(prices, cfg, asset, n_jobs=n_jobs)
    pd.DataFrame(wf["fold_metrics"]).to_csv(tables / "walkforward_folds.csv", index=False)
    (tables / "walkforward_aggregate.json").write_text(
        json.dumps({k: v for k, v in wf["aggregate"].items()}, indent=2))
    log.info("[%s] walk-forward OOS: Sharpe=%.3f MaxDD=%.3f over %d folds",
             asset.ticker, wf["aggregate"]["sharpe"], wf["aggregate"]["max_dd"],
             wf["n_folds"])
    plot_walkforward_folds(wf["fold_metrics"], figs / "fig_walkforward.png")

    tda_r = all_metrics["TDA Stress GC"]["returns"]
    dm_rows, pvals = [], {}
    for bench in ("Golden Cross", "Buy & Hold"):
        if bench not in all_metrics:
            continue
        dm = diebold_mariano_test(tda_r, all_metrics[bench]["returns"], "greater")
        dm_rows.append({"comparison": f"TDA vs {bench}", **dm})
        pvals[f"TDA vs {bench}"] = dm["p_value"]
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(tables / "diebold_mariano.csv", index=False)
    holm = holm_bonferroni(pvals, cfg.alpha)
    holm.to_csv(tables / "holm_bonferroni.csv", index=False)
    log.info("[%s] DM tests:\n%s", asset.ticker, dm_df.to_string(index=False))

    ci_rows = []
    for name in ("TDA Stress GC", "Golden Cross", "Buy & Hold"):
        if name not in all_metrics:
            continue
        for metric in ("sharpe", "max_dd"):
            ci = block_bootstrap_ci(all_metrics[name]["returns"], metric,
                                    cfg.bootstrap_reps, 0.95, cfg.block_size,
                                    cfg.bootstrap_seed, n_jobs=n_jobs)
            ci_rows.append({"strategy": name, **ci})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(tables / "bootstrap_ci.csv", index=False)
    for metric in ("sharpe", "max_dd"):
        plot_bootstrap_ci(ci_df, metric, figs / f"fig_ci_{metric}.png")
    log.info("[%s] bootstrap CIs:\n%s", asset.ticker, ci_df.round(4).to_string(index=False))
    return {"walkforward": wf, "dm": dm_df, "holm": holm, "ci": ci_df}


def stage_ml(prices, cfg, asset, asset_key, scored):
    f_ta, f_tda, f_all, target = build_feature_matrix(prices, scored, cfg.ml_horizon)
    summary, importances = run_ml_experiment(
        f_ta, f_tda, f_all, target, prices,
        test_size=cfg.ml_test_size, seed=cfg.ml_seed,
        horizon=cfg.ml_horizon,          # [E2] fixed
    )
    summary.to_csv(_out(cfg, asset_key, "tables") / "ml_experiment.csv", index=False)
    if importances:
        pd.DataFrame(importances).to_csv(
            _out(cfg, asset_key, "tables") / "ml_feature_importance.csv")
    log.info("[%s] ML AUC by group:\n%s", asset.ticker,
             summary.pivot_table(index="model", columns="group",
                                 values="auc").round(3).to_string())
    return summary


# ==============================================================================
# 14. SMOKE TEST
# ==============================================================================

def _smoke_mixup_validation() -> bool:
    """Wagner et al. ground-truth geometries, ported from the notebook's
    validation suite.  Cloud A is a 30-point unit circle in R^3."""
    rng = np.random.default_rng(42)
    n = 30
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])

    b1 = circle + rng.standard_normal((n, 3)) * 0.05
    r_fill = 0.7 * np.sqrt(rng.random(n)); t_fill = 2 * np.pi * rng.random(n)
    b2 = np.column_stack([r_fill * np.cos(t_fill), r_fill * np.sin(t_fill),
                          rng.standard_normal(n) * 0.05])
    b3 = np.column_stack([5 + rng.standard_normal(n) * 0.2,
                          5 + rng.standard_normal(n) * 0.2,
                          rng.standard_normal(n) * 0.2])
    b4 = circle.copy()
    grid = np.linspace(-0.8, 0.8, 8)
    xx, yy = np.meshgrid(grid, grid)
    mask = xx ** 2 + yy ** 2 < 0.65
    b5 = np.column_stack([xx[mask], yy[mask], np.zeros(int(mask.sum()))])

    cases = [("compatible (circle+noise)", b1, 0.00, 0.15),
             ("interior fill", b2, 0.40, 1.00),
             ("distant", b3, 0.00, 0.01),
             ("identical", b4, 0.00, 0.01),
             ("dense grid", b5, 0.60, 1.00)]
    ok = True
    for name, b, lo, hi in cases:
        mp1 = mixup_barcode_distance(circle, b, 2.5)
        passed = lo <= mp1 <= hi
        ok &= passed
        log.info("  [%s] %-26s MP1=%.4f  expected [%.2f, %.2f]",
                 "PASS" if passed else "FAIL", name, mp1, lo, hi)

    # monotonicity in interior fill density
    circ2 = np.column_stack([np.cos(theta), np.sin(theta)])
    rng2 = np.random.default_rng(0)
    fills = [np.array([[0.0, 0.0]]),
             rng2.standard_normal((15, 2)) * 0.3,
             np.column_stack([0.7 * np.sqrt(rng2.random(n)) * np.cos(2 * np.pi * rng2.random(n)),
                              0.7 * np.sqrt(rng2.random(n)) * np.sin(2 * np.pi * rng2.random(n))])]
    mps = [mixup_barcode_distance(circ2, f, 2.5) for f in fills]
    mono = mps[0] <= mps[1] <= mps[2]
    ok &= mono
    log.info("  [%s] monotone in fill density: %.4f <= %.4f <= %.4f",
             "PASS" if mono else "FAIL", *mps)

    # no H1 -> MP1 = 0
    a_h0 = np.array([[-3.0, 0.0], [3.0, 0.0]]); b_h0 = np.array([[0.0, 0.0]])
    mp_h0 = mixup_barcode_distance(a_h0, b_h0, 8.0)
    ok &= (mp_h0 == 0.0)
    log.info("  [%s] no H1 features -> MP1=%.4f", "PASS" if mp_h0 == 0.0 else "FAIL", mp_h0)

    # Observation 2:  birth <= d' <= death  on a random pair
    tri = mixup_triples(circle, b2, 2.5, degree=1)
    obs2 = all(b <= dp <= d + 1e-12 for b, dp, d in tri)
    ok &= obs2
    log.info("  [%s] Observation 2 (birth <= d' <= death) on %d triples",
             "PASS" if obs2 else "FAIL", len(tri))

    # INDEPENDENT VALIDATION OF ALGORITHM 2.
    # Our own Z_2 reduction produces the (birth, death) pairs of VR(A) as a
    # by-product.  Those must equal Ripser's H1 diagram of A computed by a
    # completely separate code path.  Bars with birth == death exactly are
    # diagonal artefacts of the reduction that Ripser discards, so they are
    # dropped before comparison; Ripser reduces in float32, hence the 1e-5
    # tolerance.  This is the strongest available check that the image-
    # persistence matching is wired up correctly.
    rng3 = np.random.default_rng(3)
    th24 = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    a24 = np.column_stack([np.cos(th24), np.sin(th24)]) + rng3.standard_normal((24, 2)) * 0.03
    b20 = rng3.standard_normal((20, 2)) * 0.25
    t24 = np.array(mixup_triples(a24, b20, 2.5, degree=1))
    nz = t24[(t24[:, 2] - t24[:, 0]) > 0]
    mine = np.array(sorted(map(tuple, nz[:, [0, 2]])))
    ref = np.array(sorted(map(tuple, compute_persistence(a24, 1, 2.5)["dgm1"])))
    agree = mine.shape == ref.shape and np.allclose(mine, ref, atol=1e-5)
    ok &= agree
    log.info("  [%s] Algorithm-2 bars of A == Ripser H1(A) (%d bars, max diff %.2e)",
             "PASS" if agree else "FAIL", len(mine),
             np.abs(mine - ref).max() if mine.shape == ref.shape else float("nan"))

    # every premature death d' must be a genuine death time of VR(A u B)
    uni = compute_persistence(np.vstack([a24, b20]), 1, 2.5)["dgm1"]
    uni_deaths = set(np.round(uni[:, 1].astype(float), 5)) | {2.5}
    prem = [dp for b, dp, d in t24 if d - b > 1e-6 and dp < d - 1e-6]
    in_union = all(round(float(dp), 5) in uni_deaths for dp in prem)
    ok &= in_union
    log.info("  [%s] all %d premature deaths are death times of VR(A u B)",
             "PASS" if in_union else "FAIL", len(prem))
    return bool(ok)


def _smoke_unit_checks() -> bool:
    ok = True

    # sizing function endpoints and the manuscript's closed form
    s_lo, s_hi = 0.30, 0.60
    checks = [(0.10, 1.0), (0.30, 1.0), (0.45, 0.5), (0.60, 0.0), (0.90, 0.0)]
    for s, want in checks:
        got = sizing_function(s, s_lo, s_hi)
        if abs(got - want) > 1e-12:
            ok = False; log.error("  [FAIL] sizing(%.2f)=%.4f want %.4f", s, got, want)
    mid = 0.42
    closed = (s_hi - mid) / (s_hi - s_lo)
    if abs(sizing_function(mid, s_lo, s_hi) - closed) > 1e-12:
        ok = False; log.error("  [FAIL] sizing != (theta_hi - S)/(theta_hi - theta_lo)")
    log.info("  [%s] sizing_function matches manuscript closed form", "PASS" if ok else "FAIL")

    # transaction cost accounting: one 0 -> 1 entry at 5 bps costs exactly 5 bps
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.Series(np.full(10, 100.0), index=idx)          # flat prices, zero PnL
    pos = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    net = compute_portfolio_returns(px, pos, 0.0005)
    tc_ok = abs(net.sum() + 0.0005) < 1e-12
    ok &= tc_ok
    log.info("  [%s] one entry at 5 bps costs %.6f (want -0.000500)",
             "PASS" if tc_ok else "FAIL", net.sum())

    # cost is lagged in step with the position  [E1]
    trade_day = net.index[net != 0][0]
    lag_ok = trade_day == idx[4]
    ok &= lag_ok
    log.info("  [%s] cost lands on the first day of exposure (%s)",
             "PASS" if lag_ok else "FAIL", trade_day.date())

    # trade counting: entries vs entries+exits  [C5]
    tc = count_trades(pos)
    cnt_ok = (tc["n_round_trips"] == 1 and tc["n_signals"] == 1 and tc["n_exits"] == 0)
    ok &= cnt_ok
    log.info("  [%s] count_trades on one unclosed entry: %s",
             "PASS" if cnt_ok else "FAIL", tc)

    # Sharpe / Sortino share a numerator  [C7]
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(504) * 0.01 + 0.0004,
                  index=pd.bdate_range("2020-01-01", periods=504))
    m = performance_metrics(r)
    ann_excess = r.mean() * 252
    sh_ok = abs(m["sharpe"] - ann_excess / (r.std() * np.sqrt(252))) < 1e-10
    so_ok = abs(m["sortino"] - ann_excess / (r[r < 0].std() * np.sqrt(252))) < 1e-10
    ok &= sh_ok and so_ok
    log.info("  [%s] Sharpe and Sortino share the annualised excess-mean numerator",
             "PASS" if (sh_ok and so_ok) else "FAIL")

    # scoring is a pure function of components (robustness sweeps depend on this)
    comp = pd.DataFrame({"h0_dist": rng.random(200) * 3,
                         "h1_mixup": rng.random(200),
                         "ent_div": rng.random(200)},
                        index=pd.bdate_range("2020-01-01", periods=200))
    a = score_components(comp, (0.45, 0.50, 0.05), 30)["stress"]
    b = score_components(comp, (0.45, 0.50, 0.05), 30)["stress"]
    c = score_components(comp, (0.30, 0.60, 0.10), 30)["stress"]
    pure_ok = a.equals(b) and not a.equals(c)
    ok &= pure_ok
    log.info("  [%s] score_components is pure and weight-sensitive",
             "PASS" if pure_ok else "FAIL")

    # normalisation is expanding (causal): prefix-stability  [B6]
    half = score_components(comp.iloc[:100], (0.45, 0.50, 0.05), 30)["s_raw"]
    full = score_components(comp, (0.45, 0.50, 0.05), 30)["s_raw"].iloc[:100]
    causal_ok = np.allclose(half.to_numpy(), full.to_numpy())
    ok &= causal_ok
    log.info("  [%s] expanding normalisation is causal (prefix-stable)",
             "PASS" if causal_ok else "FAIL")

    # Takens embedding shape and content
    s = np.arange(20, dtype=float)
    cl = takens_embedding(s, dimension=3, delay=4, normalise=False)
    emb_ok = cl.shape == (12, 3) and np.allclose(cl[0], [0, 4, 8])
    ok &= emb_ok
    log.info("  [%s] takens_embedding shape %s", "PASS" if emb_ok else "FAIL", cl.shape)

    # subsampling determinism
    big = rng.standard_normal((100, 3))
    det_ok = np.array_equal(subsample_cloud(big, 28, 0), subsample_cloud(big, 28, 0))
    ok &= det_ok
    log.info("  [%s] subsample_cloud is deterministic", "PASS" if det_ok else "FAIL")

    # DM sign convention: A strictly better must give a small p under 'greater'
    good = pd.Series(rng.standard_normal(500) * 0.002, index=r.index[:500])
    bad = good * 4.0
    dm = diebold_mariano_test(good, bad, "greater")
    dm_ok = dm["p_value"] < 0.05
    ok &= dm_ok
    log.info("  [%s] DM sign convention (p=%.4g for a clearly better A)",
             "PASS" if dm_ok else "FAIL", dm["p_value"])

    # Holm-Bonferroni monotonicity
    h = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.2})
    holm_ok = bool(h["p_holm"].is_monotonic_increasing) and (h["p_holm"] >= h["p_raw"]).all()
    ok &= holm_ok
    log.info("  [%s] Holm-Bonferroni is monotone and conservative",
             "PASS" if holm_ok else "FAIL")

    return bool(ok)


def run_smoke_test(n_jobs: int = 2) -> int:
    log.info("=" * 74)
    log.info(" SMOKE TEST -- offline, synthetic prices, no network required")
    log.info("=" * 74)

    log.info("\n--- unit checks -------------------------------------------------")
    unit_ok = _smoke_unit_checks()

    log.info("\n--- mixup barcode validation (Wagner et al. ground truth) --------")
    mixup_ok = _smoke_mixup_validation()

    log.info("\n--- end-to-end pipeline on synthetic data ------------------------")
    cfg = Config()
    cfg.results_dir = "results_smoke"
    cfg.ref_length = 30
    cfg.max_points = 12          # tiny clouds: the point is correctness, not fidelity
    cfg.smooth_window = 10
    cfg.bootstrap_reps = 50
    cfg.ma_fast, cfg.ma_slow = 20, 60    # so crossovers exist in a 400-day sample
    cfg.wf_train, cfg.wf_test, cfg.wf_step = 150, 40, 40
    cfg.grid_windows, cfg.grid_dimensions, cfg.grid_delays = (25, 30), (2, 3), (1, 2)
    cfg.rob_ref_lengths, cfg.rob_windows = (25, 30), (20, 25)
    cfg.rob_theta_lo, cfg.rob_theta_hi = (0.25, 0.30), (0.55, 0.60)
    asset = AssetConfig("SYNTH", "Synthetic", "2015-01-01", "2018-12-31",
                        window=25, dimension=3, delay=2)
    cfg.assets = {"synth": asset}
    prices = synthetic_price_series(400, seed=7, name="SYNTH")

    pipeline_ok = True
    try:
        bt = stage_backtest(prices, cfg, asset, "synth", n_jobs=n_jobs, make_plots=True)
        assert bt["scored"]["stress"].between(0, 1).all(), "stress outside [0,1]"
        assert bt["positions"]["position"].between(0, 1).all(), "position outside [0,1]"
        assert np.isfinite(bt["metrics_df"]["sharpe"]).all(), "non-finite Sharpe"
        log.info("  [PASS] backtest stage")

        # parallel and serial component paths must agree exactly
        pipe = TDAStressPipeline.for_asset(cfg, asset)
        c1 = pipe.components(prices.iloc[:120], n_jobs=1, verbose=False)
        c2 = pipe.components(prices.iloc[:120], n_jobs=n_jobs, verbose=False)
        assert np.allclose(c1.to_numpy(dtype=float), c2.to_numpy(dtype=float)), \
            "serial and parallel components differ"
        log.info("  [PASS] parallel components == serial components")

        g = stage_grid(prices, cfg, asset, "synth", n_jobs=n_jobs)
        assert g["sharpe"].notna().all(), "grid produced NaNs"
        plot_grid_heatmaps(g, "sharpe",
                           _out(cfg, "synth", "figures") / "fig_grid_sharpe.png",
                           highlight={asset.ticker: (asset.window, asset.dimension,
                                                     asset.delay)})
        log.info("  [PASS] grid stage + heatmap (%d configs)", len(g))

        rob = stage_robustness(prices, cfg, asset, "synth",
                               components=bt["components"], n_jobs=n_jobs)
        assert len(rob["weights"]) == 27, "weight sweep should have 27 rows"
        assert not rob["thresholds"].empty, "threshold sweep empty"
        log.info("  [PASS] robustness stage")

        ml = stage_ml(prices, cfg, asset, "synth", bt["scored"])
        assert ml["auc"].between(0, 1).all(), "AUC outside [0,1]"
        log.info("  [PASS] ML stage")

        wf = stage_walkforward(prices, cfg, asset, "synth", bt["all_metrics"],
                               n_jobs=n_jobs)
        assert wf["walkforward"]["n_folds"] >= 1, "no walk-forward folds"
        assert (wf["ci"]["ci_lower"] <= wf["ci"]["observed"]).all() and \
               (wf["ci"]["observed"] <= wf["ci"]["ci_upper"]).all(), "CI does not bracket"
        log.info("  [PASS] walk-forward + DM + bootstrap stage")
    except Exception as exc:
        pipeline_ok = False
        log.exception("  [FAIL] pipeline: %s", exc)

    log.info("\n" + "=" * 74)
    log.info(" unit checks : %s", "PASS" if unit_ok else "FAIL")
    log.info(" mixup       : %s", "PASS" if mixup_ok else "FAIL")
    log.info(" pipeline    : %s", "PASS" if pipeline_ok else "FAIL")
    log.info("=" * 74)
    return 0 if (unit_ok and mixup_ok and pipeline_ok) else 1


# ==============================================================================
# 15. MAIN
# ==============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TDA stress-gated Golden Cross: full reproducible pipeline.")
    p.add_argument("--asset", default="all", choices=["spy", "btc", "all"])
    p.add_argument("--stage", default="all",
                   choices=["fetch", "all", "backtest", "grid", "robustness",
                            "walkforward", "ml"],
                   help="'fetch' downloads and caches prices (the ONLY stage "
                        "needing the internet). Everything else is offline.")
    p.add_argument("--jobs", type=int, default=-1,
                   help="joblib n_jobs (-1 = all cores, 1 = serial).")
    p.add_argument("--data-dir", default=CFG.data_dir)
    p.add_argument("--out-dir", default=CFG.results_dir)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--force-refetch", action="store_true",
                   help="With --stage fetch, re-download even if cached.")
    p.add_argument("--allow-download", action="store_true",
                   help="Let analysis stages download missing data instead of "
                        "failing. Off by default so an offline run is loud "
                        "rather than silently different.")
    p.add_argument("--smoke", action="store_true",
                   help="Run the offline self-test and exit.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        return run_smoke_test(n_jobs=max(1, min(4, os.cpu_count() or 1)))

    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.results_dir = args.out_dir
    cfg.n_jobs = args.jobs

    keys = ["spy", "btc"] if args.asset == "all" else [args.asset]

    if args.stage == "fetch":
        fetch_all_data(cfg, keys, force=args.force_refetch)
        return 0

    stages = (["backtest", "grid", "robustness", "ml", "walkforward"]
              if args.stage == "all" else [args.stage])

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.results_dir) / "config_used.json").write_text(json.dumps(
        {**{k: v for k, v in asdict(cfg).items() if k != "assets"},
         "assets": {k: asdict(v) for k, v in cfg.assets.items()}}, indent=2))

    for key in keys:
        asset = cfg.assets[key]
        log.info("\n" + "=" * 74)
        log.info(" %s (%s)  w=%d  d=%d  tau=%d", asset.name, asset.ticker,
                 asset.window, asset.dimension, asset.delay)
        log.info("=" * 74)
        try:
            prices = load_price_series(asset.ticker, asset.start, asset.end,
                                       cfg.price_col, cfg.data_dir,
                                       offline=not args.allow_download)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        log.info("Loaded %d trading days: %s -> %s", len(prices),
                 prices.index[0].date(), prices.index[-1].date())

        bt = None
        if "backtest" in stages or "ml" in stages or "walkforward" in stages \
                or "robustness" in stages:
            bt = stage_backtest(prices, cfg, asset, key, n_jobs=cfg.n_jobs,
                                make_plots=not args.no_plots)
        if "grid" in stages:
            grid = stage_grid(prices, cfg, asset, key, n_jobs=cfg.n_jobs)
            if not args.no_plots:
                for metric in ("sharpe", "ann_return", "sortino", "max_dd", "calmar"):
                    plot_grid_heatmaps(
                        grid, metric, _out(cfg, key, "figures") / f"fig_grid_{metric}.png",
                        highlight={asset.ticker: (asset.window, asset.dimension,
                                                  asset.delay)})
        if "robustness" in stages:
            stage_robustness(prices, cfg, asset, key,
                             components=bt["components"], n_jobs=cfg.n_jobs)
        if "ml" in stages:
            stage_ml(prices, cfg, asset, key, bt["scored"])
        if "walkforward" in stages:
            stage_walkforward(prices, cfg, asset, key, bt["all_metrics"],
                              n_jobs=cfg.n_jobs)

    log.info("\nAll outputs written to %s", Path(cfg.results_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
