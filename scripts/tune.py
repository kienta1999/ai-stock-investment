#!/usr/bin/env python3
"""
Parameter tuner with train/test split and grid search.

  In-sample (tune on):     2024-04-20 → 2025-04-20  (Year 1)
  Out-of-sample (validate): 2025-04-20 → 2026-04-20  (Year 2)

MAX_SLOTS sweep (2026-04-25): tests whether running 1–4 concurrent positions
beats the single-trade baseline. Each slot is its own sub-account starting
with $10k/N and compounding independently (see backtest.simulate). Tradeoff:
more slots = more deployment + diversification, but lower-quality picks fill
slots 2-N, which can hurt win rate. 4 variations.

OHLCV data is cached to data/raw_ohlcv_2y.pkl (7-day TTL) to avoid yfinance
rate-limit hits on repeated runs. Delete the file or pass --refresh to bust.

Output is intentionally minimal: only variations with alpha >=10pp on BOTH
windows are printed in the final table (plus baseline as reference). Use the
fallback top-5-by-min-alpha if nothing qualifies.

NOTE: tune.py's per-window alphas are useful for RANKING parameter directions
but are NOT a reliable proxy for continuous backtest magnitude. Always
re-validate any winner by running scripts/backtest.py with the new defaults
before committing.
"""

import sys
import time
import pickle
import itertools
import warnings
from pathlib import Path
from datetime import date, timedelta

warnings.filterwarnings("ignore")

import yfinance as yf

from universe import load_universe
import signals as sg
from signals import build_regime_series, BENCHMARK
from backtest import simulate, CAPITAL_INIT

IN_START   = date(2024, 4, 20)
IN_END     = date(2025, 4, 20)
OUT_START  = date(2025, 4, 20)
OUT_END    = date(2026, 4, 20)

CACHE_PATH = Path(__file__).parent.parent / "data" / "raw_ohlcv_2y.pkl"
CACHE_TTL_DAYS = 7

ALPHA_REPORT_FLOOR = 10.0  # only print variations with alpha >= this on BOTH windows


# ─────────────────────────────────────────────────────────────────────────────
# Data caching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_or_load_data(tickers, start, end, refresh=False):
    if not refresh and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days < CACHE_TTL_DAYS:
            print(f"Loaded OHLCV from cache ({age_days:.1f}d old). "
                  f"Pass --refresh to bust.", flush=True)
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Cache stale ({age_days:.1f}d > {CACHE_TTL_DAYS}d TTL), "
                  f"refreshing...", flush=True)

    print(f"Downloading OHLCV ({start} → {end})...", flush=True)
    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(raw, f)
    print(f"Cached to {CACHE_PATH}", flush=True)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep helpers — core simulation lives in backtest.simulate()
# ─────────────────────────────────────────────────────────────────────────────

def apply_overrides(overrides: dict):
    for k, v in overrides.items():
        setattr(sg, k, v)


def snapshot_baseline() -> dict:
    return {k: getattr(sg, k) for k in dir(sg)
            if (k.startswith(("Q_", "L1_", "L2_", "L3_", "L4_", "RS_"))
                or k in ("MIN_QUALITY_SCORE", "MAX_SLOTS"))}


def spy_return(raw, bt_dates):
    spy = raw["Close"][BENCHMARK].dropna()
    spy_w = spy[(spy.index.date >= bt_dates[0]) & (spy.index.date <= bt_dates[-1])]
    return (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * 100


def run_one(raw, tickers, all_dates, bt_dates, bench_ret, regime):
    spy_close, spy_ma, vix_close = regime
    end_cap, trades, _ = simulate(
        raw, tickers, all_dates, bt_dates,
        spy_close=spy_close, spy_ma=spy_ma, vix_close=vix_close,
    )
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    ret = (end_cap / CAPITAL_INIT - 1) * 100
    alpha = ret - bench_ret
    return {"trades": n, "win_rate": (wins/n if n else 0),
            "return": ret, "alpha": alpha, "end_cap": end_cap}


# ─────────────────────────────────────────────────────────────────────────────
# Grid generation
# ─────────────────────────────────────────────────────────────────────────────

GRID = {
    # Portfolio-level: number of concurrent positions. 1 = current SOT baseline.
    # 2–4 split capital equally per slot; later signals fill empty slots only
    # (no swap-on-better-quality). Filling slot 2-N forces lower-quality picks,
    # so this isn't free — that's exactly what the sweep is measuring.
    "MAX_SLOTS":          [1, 2, 3, 4],
}
# = 4 variations


def build_variations():
    variations = [("Baseline", {})]
    keys = list(GRID.keys())
    for combo in itertools.product(*GRID.values()):
        overrides = dict(zip(keys, combo))
        short = f"SLOTS={overrides['MAX_SLOTS']}"
        variations.append((short, overrides))
    return variations


# ─────────────────────────────────────────────────────────────────────────────

def main():
    refresh = "--refresh" in sys.argv

    universe = load_universe()
    tickers = universe["Ticker"].tolist()
    # SMA200 needs ~290 calendar days of warm-up before the earliest scan date
    fetch_start = IN_START - timedelta(days=300)
    raw = fetch_or_load_data(
        tickers + [BENCHMARK, "^VIX"],
        start=fetch_start.isoformat(), end=OUT_END.isoformat(),
        refresh=refresh,
    )
    all_dates = sorted(set(raw.index.date))
    in_dates  = [d for d in all_dates if IN_START  <= d <= IN_END]
    out_dates = [d for d in all_dates if OUT_START <= d <= OUT_END]

    in_bench  = spy_return(raw, in_dates)
    out_bench = spy_return(raw, out_dates)
    print(f"\nIn-sample  ({IN_START} → {IN_END}):    {len(in_dates)} days, SPY {in_bench:+.1f}%")
    print(f"Out-sample ({OUT_START} → {OUT_END}):  {len(out_dates)} days, SPY {out_bench:+.1f}%")

    # Regime series — shared across all variations; pass-through to simulate()
    regime = build_regime_series(raw)
    if regime[0] is None:
        print("WARN: regime data unavailable in cached OHLCV — run with --refresh "
              "to re-download including ^VIX. Gate disabled for this sweep.")

    baseline = snapshot_baseline()
    variations = build_variations()
    n = len(variations)

    print(f"\nRunning {n} variations (each tested on both windows)...")
    print("Progress: ", end="", flush=True)

    results = []
    t0 = time.time()
    for i, (name, overrides) in enumerate(variations):
        apply_overrides(baseline)
        apply_overrides(overrides)
        in_res  = run_one(raw, tickers, all_dates, in_dates, in_bench, regime)
        out_res = run_one(raw, tickers, all_dates, out_dates, out_bench, regime)
        results.append({"name": name, "overrides": overrides,
                        "in": in_res, "out": out_res})
        print(".", end="", flush=True)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f" [{i+1}/{n}, {elapsed/60:.1f}min]", flush=True)
            print("Progress: ", end="", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f}min.\n")

    base = results[0]
    base_in_a, base_out_a = base["in"]["alpha"], base["out"]["alpha"]

    print("="*88)
    print(f"  BASELINE: in α={base_in_a:+.1f}pp  |  out α={base_out_a:+.1f}pp  "
          f"|  in trades={base['in']['trades']}, out trades={base['out']['trades']}")
    print("="*88)

    qualifying = [r for r in results[1:]
                  if r["in"]["alpha"] >= ALPHA_REPORT_FLOOR
                  and r["out"]["alpha"] >= ALPHA_REPORT_FLOOR]

    print(f"\n{len(qualifying)} of {n-1} variations cleared "
          f"the +{ALPHA_REPORT_FLOOR:.0f}pp floor on BOTH windows:\n")

    if not qualifying:
        # Fall back: show the top-5 by min(in, out) alpha so we still learn something
        ranked = sorted(results[1:],
                        key=lambda r: min(r["in"]["alpha"], r["out"]["alpha"]),
                        reverse=True)
        print("  (none qualified — showing top-5 by min-of-two-alphas as fallback)\n")
        print(f"  {'variation':<55} {'in α':>9} {'out α':>9} {'min':>9}")
        print("  " + "-"*86)
        for r in ranked[:5]:
            in_a, out_a = r["in"]["alpha"], r["out"]["alpha"]
            print(f"  {r['name']:<55} {in_a:>+8.1f}p {out_a:>+8.1f}p "
                  f"{min(in_a, out_a):>+8.1f}p")
        return

    # Sort by sum of alphas (proxy for both-window strength)
    qualifying.sort(key=lambda r: r["in"]["alpha"] + r["out"]["alpha"], reverse=True)
    print(f"  {'variation':<55} {'in α':>9} {'out α':>9} {'sum':>9}")
    print("  " + "-"*86)
    for r in qualifying:
        in_a, out_a = r["in"]["alpha"], r["out"]["alpha"]
        print(f"  {r['name']:<55} {in_a:>+8.1f}p {out_a:>+8.1f}p "
              f"{in_a + out_a:>+8.1f}p")


if __name__ == "__main__":
    main()
