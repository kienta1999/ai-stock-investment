#!/usr/bin/env python3
"""
Technical analysis scanner — long-only strategy.
  Rule: only consider LONG setups when price > SMA200.
        Price < SMA200 → no trade (do not short, do not fight the tape).
  Skips entries where ATR% > MAX_ATR_PCT (extreme-volatility guardrail).
  Scans all top-100 S&P 500 stocks.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import yfinance as yf
    from universe import load_universe
    from indicators import compute
    from signals import score, market_regime, quality
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  uv pip install yfinance pandas lxml requests --python .venv/bin/python3")
    sys.exit(1)

MAX_ATR_PCT = 4.0   # skip entries where ATR% exceeds this (vol-cap guardrail)


def run(force_refresh: bool = False) -> None:
    # ── Universe ──────────────────────────────────────────────────────────────
    universe = load_universe(force_refresh=force_refresh)
    tickers = universe["Ticker"].tolist()

    # ── Download full OHLCV (1 year daily) ───────────────────────────────────
    print(f"Downloading 1-year OHLCV for {len(tickers)} tickers...", flush=True)
    raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        print("ERROR: No data returned.")
        sys.exit(1)

    # ── Process each ticker ───────────────────────────────────────────────────
    long_rows, short_rows, triggered = [], [], []

    for ticker in tickers:
        try:
            df = pd.DataFrame({
                "Open":   raw["Open"][ticker],
                "High":   raw["High"][ticker],
                "Low":    raw["Low"][ticker],
                "Close":  raw["Close"][ticker],
                "Volume": raw["Volume"][ticker],
            }).dropna()
        except (KeyError, TypeError):
            continue

        if len(df) < 200:
            continue

        ind = compute(df)
        if not ind:
            continue

        price   = ind["price"]
        sma200  = ind["sma200"]
        sma50   = ind["sma50"]
        pct200  = ind["price_vs_sma200_pct"]
        trend   = "↑" if ind["sma50_above_sma200"] else "↓"

        if ind["price_above_sma200"]:
            long_rows.append({
                "Ticker": ticker, "Price": price,
                "SMA50": sma50, "SMA200": sma200,
                "% vs SMA200": pct200, "RSI": round(ind["rsi"], 1),
                "Trend": trend,
            })
        else:
            short_rows.append({
                "Ticker": ticker, "Price": price,
                "SMA50": sma50, "SMA200": sma200,
                "% vs SMA200": pct200, "RSI": round(ind["rsi"], 1),
                "Trend": trend,
            })

        if ind.get("atr_pct", 0) > MAX_ATR_PCT:
            continue
        for s in score(ind):
            row = {"Ticker": ticker, **ind, **s}
            row["quality"] = quality(row)
            triggered.append(row)

    long_count  = len(long_rows)
    short_count = len(short_rows)
    total       = long_count + short_count

    # ── Market breadth ────────────────────────────────────────────────────────
    regime = market_regime(long_count, total)
    print(f"\n{'='*66}")
    print(f"  MARKET BREADTH | {regime}")
    print(f"{'='*66}")

    # ── Long universe (price > SMA200) ────────────────────────────────────────
    if long_rows:
        ldf = pd.DataFrame(long_rows).sort_values("% vs SMA200", ascending=False).reset_index(drop=True)
        print(f"\n── LONG UNIVERSE  ({long_count} stocks — price > SMA200) ──")
        print(ldf.to_string(index=False))

    # ── Short universe (price < SMA200) ───────────────────────────────────────
    if short_rows:
        sdf = pd.DataFrame(short_rows).sort_values("% vs SMA200", ascending=True).reset_index(drop=True)
        print(f"\n── SHORT UNIVERSE  ({short_count} stocks — price < SMA200) ──")
        print(sdf.to_string(index=False))

    # ── Triggered setups ─────────────────────────────────────────────────────
    if not triggered:
        print("\nNo setups triggered today.")
        return

    setup_df = pd.DataFrame(triggered).sort_values("quality", ascending=False)
    longs  = setup_df[setup_df["direction"] == "LONG"]
    shorts = setup_df[setup_df["direction"] == "SHORT"]

    display_cols = ["Ticker", "setup", "quality", "entry", "sl", "tp", "rr",
                    "rsi", "vol_ratio", "atr_pct", "timeframe", "notes"]

    for label, subset in [("LONG", longs), ("SHORT", shorts)]:
        if subset.empty:
            continue
        cols = [c for c in display_cols if c in subset.columns]
        print(f"\n{'='*66}")
        print(f"  {label} SETUPS  ({len(subset)} found — sorted by quality desc)")
        print(f"{'='*66}")
        print(subset[cols].to_string(index=False))

    # ── Per-setup detail (highest quality first) ─────────────────────────────
    print(f"\n── Setup Detail ──")
    for _, row in setup_df.iterrows():
        arrow = "▲" if row["direction"] == "LONG" else "▼"
        print(f"\n  {arrow} {row['Ticker']} | {row['setup']} | {row['direction']} | Quality: {row['quality']}/100")
        print(f"    Entry: {row['entry']}  SL: {row['sl']}  TP: {row['tp']}  R:R {row['rr']}")
        print(f"    RSI: {row['rsi']:.0f}  |  Vol: {row['vol_ratio']:.1f}x MA  |  ATR: {row['atr_pct']:.1f}%")
        print(f"    MACD hist: {row['macd_hist']:+.3f}  |  BB pos: {row['above_mid_5d']}/5d above mid")
        print(f"    VWAP dist: {row['price_vs_vwap_pct']:+.1f}%  |  SMA50>SMA200: {row['sma50_above_sma200']}")
        print(f"    Timeframe: {row['timeframe']}")
        print(f"    Notes: {row['notes']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force refresh universe cache")
    args = parser.parse_args()
    run(force_refresh=args.refresh)
