#!/usr/bin/env python3
"""
Backtest: $10k, 1 trade at a time, March 20 → April 20 2026.
Rules:
  - Scan at EOD, enter next trading day at open
  - 1 trade at a time — only scan for next when current exits
  - Exit when TP or SL hit (daily High/Low check)
  - 20-day time stop if neither hit
  - Skip day if no clean setup (don't force trades)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from universe import load_universe
from indicators import compute
from signals import score as score_setups, quality, MIN_QUALITY_SCORE

START_DATE  = date(2024, 4, 20)
END_DATE    = date(2026, 4, 20)
CAPITAL_INIT = 10_000.0
TIME_STOP_DAYS = 40          # max days to hold if TP/SL not hit
MAX_ATR_PCT = 4.0            # skip entries where ATR% > this (vol-cap)
BENCHMARK   = "SPY"          # S&P 500 ETF for buy-and-hold comparison


# ─────────────────────────────────────────────────────────────────────────────
# Trade execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trade(ticker: str, direction: str,
                   entry: float, sl: float, tp: float,
                   entry_date: date, raw: pd.DataFrame,
                   all_dates: list[date]) -> dict:
    """
    Entry at `entry_date` open. Check subsequent days for TP/SL.
    Returns trade result dict.
    """
    try:
        ticker_close  = raw["Close"][ticker]
        ticker_open   = raw["Open"][ticker]
        ticker_high   = raw["High"][ticker]
        ticker_low    = raw["Low"][ticker]
    except KeyError:
        return None

    def ts(d):
        return pd.Timestamp(d)

    # Find entry bar: first date >= entry_date with data
    future_dates = [d for d in all_dates if d >= entry_date]
    if not future_dates:
        return None

    # Enter at open of entry_date
    entry_bar = future_dates[0]
    if ts(entry_bar) not in ticker_open.index:
        return None
    actual_entry = float(ticker_open.loc[ts(entry_bar)])

    outcome = "open"
    exit_price = actual_entry
    exit_date  = entry_bar
    days_held  = 0

    hold_dates = [d for d in future_dates[1:] if ts(d) in ticker_high.index]

    # Trailing-to-breakeven: once price moves halfway to TP, raise SL to entry.
    be_trigger = actual_entry + (tp - actual_entry) * 0.5
    moved_to_be = False

    for d in hold_dates:
        days_held += 1
        hi = float(ticker_high.loc[ts(d)])
        lo = float(ticker_low.loc[ts(d)])
        cl = float(ticker_close.loc[ts(d)])

        if direction == "LONG":
            if not moved_to_be and hi >= be_trigger:
                sl = max(sl, actual_entry)
                moved_to_be = True
            if lo <= sl:            # SL hit
                exit_price, outcome, exit_date = sl, "SL", d; break
            if hi >= tp:            # TP hit
                exit_price, outcome, exit_date = tp, "TP", d; break
        else:  # SHORT
            if not moved_to_be and lo <= be_trigger:
                sl = min(sl, actual_entry)
                moved_to_be = True
            if hi >= sl:            # SL hit
                exit_price, outcome, exit_date = sl, "SL", d; break
            if lo <= tp:            # TP hit
                exit_price, outcome, exit_date = tp, "TP", d; break

        if days_held >= TIME_STOP_DAYS:
            exit_price, outcome, exit_date = cl, "TIME", d; break
    else:
        # Reached end of data
        last = hold_dates[-1] if hold_dates else entry_bar
        exit_price = float(ticker_close.loc[ts(last)]) if ts(last) in ticker_close.index else actual_entry
        outcome, exit_date = "TIME", last

    if direction == "LONG":
        pnl_pct = (exit_price - actual_entry) / actual_entry * 100
    else:
        pnl_pct = (actual_entry - exit_price) / actual_entry * 100

    return {
        "ticker":       ticker,
        "direction":    direction,
        "entry_date":   entry_date,
        "entry":        round(actual_entry, 2),
        "sl":           round(sl, 2),
        "tp":           round(tp, 2),
        "exit_date":    exit_date,
        "exit_price":   round(exit_price, 2),
        "outcome":      outcome,
        "days_held":    days_held,
        "pnl_pct":      round(pnl_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run():
    universe = load_universe()
    tickers  = universe["Ticker"].tolist()
    dl_tickers = tickers + [BENCHMARK]

    # Download enough history: SMA200 needs 200 trading days before start
    fetch_start = date(2023, 7, 1)
    print(f"Downloading OHLCV from {fetch_start} to {END_DATE}...", flush=True)
    raw = yf.download(
        dl_tickers, start=fetch_start.isoformat(), end=END_DATE.isoformat(),
        interval="1d", auto_adjust=True, progress=False
    )
    if raw.empty:
        print("ERROR: No data."); sys.exit(1)

    # Build sorted list of trading dates in the backtest window
    all_dates   = sorted(set(raw.index.date))
    bt_dates    = [d for d in all_dates if START_DATE <= d <= END_DATE]

    print(f"Backtest: {START_DATE} → {END_DATE}  ({len(bt_dates)} trading days)\n")

    capital    = CAPITAL_INIT
    trades     = []
    active     = None     # active trade dict or None
    scan_from  = START_DATE

    for today in bt_dates:
        # ── Check if active trade exits today ────────────────────────────────
        if active:
            if today >= active["exit_date"]:
                pnl_pct = active["pnl_pct"]
                pnl_usd = capital * (pnl_pct / 100)
                capital += pnl_usd
                active["pnl_usd"] = round(pnl_usd, 2)
                active["capital_after"] = round(capital, 2)
                trades.append(active)
                print(f"  EXIT  {active['ticker']:6s} {active['direction']:5s} | "
                      f"{active['outcome']} @ {active['exit_price']}  "
                      f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.0f})  "
                      f"Capital: ${capital:,.0f}")
                active  = None
                scan_from = today  # scan for next setup tonight

        # ── Scan for next setup if idle ───────────────────────────────────────
        if active is None and today >= scan_from:
            # Build per-ticker DataFrames using data up to and including today
            today_ts   = pd.Timestamp(today)
            candidates = []

            for ticker in tickers:
                try:
                    df = pd.DataFrame({
                        "Open":   raw["Open"][ticker],
                        "High":   raw["High"][ticker],
                        "Low":    raw["Low"][ticker],
                        "Close":  raw["Close"][ticker],
                        "Volume": raw["Volume"][ticker],
                    }).loc[:today_ts].dropna()
                except (KeyError, TypeError):
                    continue
                if len(df) < 200:
                    continue

                ind = compute(df)
                if not ind:
                    continue

                if ind.get("atr_pct", 0) > MAX_ATR_PCT:
                    continue
                for s in score_setups(ind):
                    q = quality({**ind, **s})
                    candidates.append({
                        "ticker": ticker,
                        "quality": q,
                        **ind,
                        **s,
                    })

            if not candidates:
                continue

            # Pick single best setup
            best = max(candidates, key=lambda x: x["quality"])

            if best["quality"] < MIN_QUALITY_SCORE:
                continue  # No clean setup today — sit on hands

            # Enter next trading day
            future = [d for d in all_dates if d > today]
            if not future:
                continue
            entry_date = future[0]

            result = simulate_trade(
                ticker    = best["ticker"],
                direction = best["direction"],
                entry     = best["entry"],
                sl        = best["sl"],
                tp        = best["tp"],
                entry_date= entry_date,
                raw       = raw,
                all_dates = all_dates,
            )
            if not result:
                continue

            active = result
            print(f"  ENTER {result['ticker']:6s} {result['direction']:5s} | "
                  f"Setup: {best['setup']:<20s} Quality: {best['quality']}  "
                  f"Entry: {result['entry']}  SL: {result['sl']}  TP: {result['tp']}  "
                  f"RSI: {best['rsi']:.0f}  Vol: {best['vol_ratio']:.1f}x  "
                  f"Scan: {today}  Enter: {entry_date}")

    # Close any open trade at end
    if active:
        last_date = bt_dates[-1]
        try:
            ep = float(raw["Close"][active["ticker"]].loc[pd.Timestamp(last_date)])
        except Exception:
            ep = active["entry"]
        if active["direction"] == "LONG":
            pnl_pct = (ep - active["entry"]) / active["entry"] * 100
        else:
            pnl_pct = (active["entry"] - ep) / active["entry"] * 100
        pnl_usd = capital * (pnl_pct / 100)
        capital += pnl_usd
        active.update({"exit_price": round(ep,2), "outcome": "OPEN@END",
                        "pnl_pct": round(pnl_pct,2), "pnl_usd": round(pnl_usd,2),
                        "capital_after": round(capital,2)})
        trades.append(active)
        print(f"  OPEN  {active['ticker']:6s} still open — marked at ${ep:.2f}  "
              f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.0f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  BACKTEST RESULTS  {START_DATE} → {END_DATE}")
    print(f"{'='*66}")
    if not trades:
        print("  No trades executed.")
        return

    df = pd.DataFrame(trades)
    wins   = len(df[df["pnl_pct"] > 0])
    losses = len(df[df["pnl_pct"] <= 0])
    total  = len(df)

    strat_ret_pct = (capital / CAPITAL_INIT - 1) * 100

    # Benchmark: SPY buy-and-hold over backtest window
    bench_ret_pct = None
    bench_end_cap = None
    try:
        spy_close = raw["Close"][BENCHMARK].dropna()
        spy_window = spy_close[(spy_close.index.date >= bt_dates[0]) &
                               (spy_close.index.date <= bt_dates[-1])]
        if len(spy_window) >= 2:
            bench_ret_pct = (spy_window.iloc[-1] / spy_window.iloc[0] - 1) * 100
            bench_end_cap = CAPITAL_INIT * (1 + bench_ret_pct / 100)
    except Exception:
        pass

    print(f"  Trades:      {total}  ({wins}W / {losses}L)")
    print(f"  Win rate:    {wins/total:.0%}")
    print(f"  Start:       ${CAPITAL_INIT:,.0f}")
    print(f"  End:         ${capital:,.0f}")
    print(f"  Total P&L:   ${capital - CAPITAL_INIT:+,.0f}  ({strat_ret_pct:+.1f}%)")
    if bench_ret_pct is not None:
        alpha = strat_ret_pct - bench_ret_pct
        verdict = "BEAT ✓" if alpha > 0 else "LOST ✗"
        print(f"\n  Benchmark ({BENCHMARK} buy-and-hold):")
        print(f"    End:       ${bench_end_cap:,.0f}  ({bench_ret_pct:+.1f}%)")
        print(f"    Alpha:     {alpha:+.1f} pp   [{verdict}]")
    print(f"\n  Trade log:")

    display_cols = ["ticker","direction","entry_date","entry","sl","tp",
                    "exit_date","exit_price","outcome","days_held","pnl_pct","pnl_usd","capital_after"]
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    run()
