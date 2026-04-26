#!/usr/bin/env python3
"""
OOS sweep — run backtest.simulate() across all 5 historical windows and print
a single alpha summary table. Used to evaluate algo-level changes (e.g. the
RS filter) against the baselines documented in README "Out-of-sample regime
sweep".

Each window downloads its own OHLCV (with SMA200 warm-up) and caches under
data/raw_oos_<start>_<end>.pkl (7-day TTL). Re-running the script reads from
cache so iteration on signals.py params is fast.

Usage:
  .venv/bin/python3 scripts/run_oos.py            # run all windows, current config
  .venv/bin/python3 scripts/run_oos.py --refresh  # bust cache
"""

import sys
import time
import pickle
import warnings
from pathlib import Path
from datetime import date, timedelta

warnings.filterwarnings("ignore")

import yfinance as yf

from universe import load_universe
import signals as sg
from signals import BENCHMARK
from backtest import simulate, CAPITAL_INIT

# Baseline alpha = current shipped config (RS filter v1, dual 63/126 ∩ top 40%).
# Update after each accepted change so future tuning measures Δ vs the latest
# state-of-the-art, not the original pre-filter numbers. Old pre-RS baselines
# are kept in the README "Out-of-sample regime sweep" section for archeology.
WINDOWS = [
    ("2024-26 bull",      date(2024, 4, 20),  date(2026, 4, 20),  128.6),
    ("2015 chop",         date(2015, 1, 1),   date(2016, 1, 1),    14.3),
    ("2018 vol shock",    date(2018, 1, 1),   date(2019, 1, 1),   28.7),
    ("2020 COVID",        date(2020, 2, 19),  date(2020, 12, 31), 16.7),
    ("2022-24 bear",      date(2022, 1, 1),   date(2024, 1, 1),   25.1),
    ("2008 GFC",          date(2007, 10, 10), date(2009, 12, 31), 81.5),
]

CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_TTL_DAYS = 7


def fetch_or_load(tickers, start, end, refresh=False):
    cache_path = CACHE_DIR / f"raw_oos_{start.isoformat()}_{end.isoformat()}.pkl"
    if not refresh and cache_path.exists():
        age = (time.time() - cache_path.stat().st_mtime) / 86400
        if age < CACHE_TTL_DAYS:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    raw = yf.download(
        tickers, start=start.isoformat(), end=end.isoformat(),
        interval="1d", auto_adjust=True, progress=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw, f)
    return raw


def run_window(name, start, end, baseline_alpha, refresh):
    universe = load_universe()
    tickers = universe["Ticker"].tolist()
    fetch_start = start - timedelta(days=300)
    raw = fetch_or_load(tickers + [BENCHMARK, "^VIX"], fetch_start, end, refresh)
    if raw.empty:
        return None

    all_dates = sorted(set(raw.index.date))
    bt_dates  = [d for d in all_dates if start <= d <= end]
    if not bt_dates:
        return None

    end_cap, trades, blocked = simulate(raw, tickers, all_dates, bt_dates)

    n     = len(trades)
    wins  = sum(1 for t in trades if t["pnl_pct"] > 0)
    ret   = (end_cap / CAPITAL_INIT - 1) * 100

    spy = raw["Close"][BENCHMARK].dropna()
    spy_w = spy[(spy.index.date >= bt_dates[0]) & (spy.index.date <= bt_dates[-1])]
    bench = (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * 100 if len(spy_w) >= 2 else 0.0
    alpha = ret - bench
    delta = alpha - baseline_alpha

    return {
        "name": name, "trades": n,
        "win_rate": (wins / n if n else 0),
        "return": ret, "bench": bench, "alpha": alpha,
        "baseline": baseline_alpha, "delta": delta,
        "blocked": blocked,
    }


def main():
    refresh = "--refresh" in sys.argv

    cfg = (f"RS_FILTER_ENABLED={sg.RS_FILTER_ENABLED} | "
           f"3M={sg.RS_LOOKBACK_3M} 6M={sg.RS_LOOKBACK_6M} "
           f"TOP_PCT={sg.RS_TOP_PCT}")
    print(f"\n{cfg}\n")

    rows = []
    t0 = time.time()
    for name, start, end, baseline in WINDOWS:
        print(f"  running {name:<22s} {start} → {end} ...", flush=True)
        r = run_window(name, start, end, baseline, refresh)
        if r is None:
            print(f"    SKIP (no data)")
            continue
        rows.append(r)
        flag = "✓" if r["delta"] >= -10.0 else "✗"
        print(f"    α {r['alpha']:+6.1f}pp  base {r['baseline']:+6.1f}  "
              f"Δ {r['delta']:+6.1f}  trades {r['trades']:3d}  "
              f"win {r['win_rate']:.0%}  blocked {r['blocked']}d  {flag}")

    print(f"\n  Done in {time.time()-t0:.0f}s.\n")
    print("="*86)
    print(f"  {'window':<20s}  {'alpha':>8} {'base':>8} {'Δ':>8} "
          f"{'trades':>7} {'win':>6} {'flag':>5}")
    print("-"*86)
    pos_2015 = False
    worst_drop = 0.0
    for r in rows:
        flag = "✓" if r["delta"] >= -10.0 else "✗"
        if "2015" in r["name"] and r["alpha"] > 0:
            pos_2015 = True
        if r["delta"] < worst_drop:
            worst_drop = r["delta"]
        print(f"  {r['name']:<20s}  {r['alpha']:>+7.1f}p {r['baseline']:>+7.1f}p "
              f"{r['delta']:>+7.1f}p {r['trades']:>7d} {r['win_rate']:>5.0%} "
              f"{flag:>5s}")
    print("="*86)
    print(f"  Ship rule: 2015 alpha > 0  AND  no window drops more than 10pp")
    print(f"    2015 positive: {pos_2015}")
    print(f"    Worst drop:    {worst_drop:+.1f}pp  "
          f"({'PASS' if worst_drop >= -10.0 else 'FAIL'})")


if __name__ == "__main__":
    main()
