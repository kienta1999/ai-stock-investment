#!/usr/bin/env python3
"""
Backtest: $10k, MAX_SLOTS concurrent trades, 2024-04-20 → 2026-04-20.
Rules:
  - Scan at EOD, enter next trading day at open
  - Up to MAX_SLOTS concurrent positions; capital split equally per slot
    (each slot is an independent sub-account that compounds on its own)
  - No two slots in the same ticker at once
  - Exit when TP or SL hit (daily High/Low check)
  - TIME_STOP_DAYS time stop if neither hit
  - Skip day if no clean setup (don't force trades)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from universe import load_universe
from indicators import compute, ticker_frame
import signals as sg
from signals import (score as score_setups, quality, MIN_QUALITY_SCORE,
                     long_regime_ok, build_regime_series,
                     MAX_ATR_PCT, SPY_MA_PERIOD, VIX_MAX, BENCHMARK)

START_DATE  = date(2024, 4, 20)
END_DATE    = date(2026, 4, 20)
CAPITAL_INIT = 10_000.0
TIME_STOP_DAYS = 40          # max days to hold if TP/SL not hit


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
# Core simulation engine — single SOT for backtest.py and tune.py
# ─────────────────────────────────────────────────────────────────────────────

def simulate(raw: pd.DataFrame, tickers: list, all_dates: list, bt_dates: list, *,
             capital_init: float = CAPITAL_INIT,
             max_slots: int = None,
             spy_close=None, spy_ma=None, vix_close=None,
             verbose: bool = False) -> tuple:
    """Run the scan-pick-enter-exit loop over bt_dates with up to N concurrent slots.
    Each slot starts with capital_init/N and compounds independently. Returns
    (total_capital, trades, regime_blocked_days). If regime series are provided,
    applies the LONG entry gate from signals.long_regime_ok.

    max_slots=None reads sg.MAX_SLOTS at call time (so tune.py overrides apply)."""
    regime_ready = spy_close is not None and spy_ma is not None and vix_close is not None
    n_slots = max_slots if max_slots is not None else sg.MAX_SLOTS

    # Each slot: independent sub-account. trade["capital_after"] is slot capital.
    slots = [{"capital": capital_init / n_slots, "active": None, "idx": i}
             for i in range(n_slots)]
    trades = []
    regime_blocked_days = 0

    def held_tickers():
        return {s["active"]["ticker"] for s in slots if s["active"] is not None}

    for today in bt_dates:
        # ── 1. Exit any slot whose trade ended today ─────────────────────────
        for slot in slots:
            tr = slot["active"]
            if tr and today >= tr["exit_date"]:
                pnl_usd = slot["capital"] * (tr["pnl_pct"] / 100)
                slot["capital"] += pnl_usd
                tr["pnl_usd"] = round(pnl_usd, 2)
                tr["capital_after"] = round(slot["capital"], 2)
                tr["slot"] = slot["idx"]
                trades.append(tr)
                if verbose:
                    print(f"  EXIT  S{slot['idx']} {tr['ticker']:6s} {tr['direction']:5s} | "
                          f"{tr['outcome']} @ {tr['exit_price']}  "
                          f"P&L: {tr['pnl_pct']:+.1f}% (${pnl_usd:+.0f})  "
                          f"Slot cap: ${slot['capital']:,.0f}")
                slot["active"] = None

        free_slots = [s for s in slots if s["active"] is None]
        if not free_slots:
            continue

        # ── 2. Scan once, gather all candidates ──────────────────────────────
        today_ts = pd.Timestamp(today)
        long_allowed = (long_regime_ok(spy_close, spy_ma, vix_close, today_ts)
                        if regime_ready else True)

        candidates = []
        for ticker in tickers:
            df = ticker_frame(raw, ticker, up_to=today_ts)
            if df is None or len(df) < 200:
                continue
            ind = compute(df)
            if not ind:
                continue
            if ind.get("atr_pct", 0) > MAX_ATR_PCT:
                continue
            for s in score_setups(ind):
                if s.get("direction", "LONG") == "LONG" and not long_allowed:
                    continue
                q = quality({**ind, **s})
                candidates.append({"ticker": ticker, "quality": q, **ind, **s})

        if not candidates:
            if not long_allowed:
                regime_blocked_days += 1
            continue

        # ── 3. Pick top-N by quality, dedup tickers (incl. already held) ─────
        held = held_tickers()
        candidates.sort(key=lambda x: x["quality"], reverse=True)

        picks = []
        seen = set(held)
        for c in candidates:
            if c["quality"] < MIN_QUALITY_SCORE:
                break  # sorted desc — once below floor, all rest are too
            if c["ticker"] in seen:
                continue
            seen.add(c["ticker"])
            picks.append(c)
            if len(picks) >= len(free_slots):
                break

        if not picks:
            continue

        # ── 4. Enter trades, one pick per free slot ──────────────────────────
        future = [d for d in all_dates if d > today]
        if not future:
            continue
        entry_date = future[0]

        for slot, pick in zip(free_slots, picks):
            result = simulate_trade(
                ticker=pick["ticker"], direction=pick["direction"],
                entry=pick["entry"], sl=pick["sl"], tp=pick["tp"],
                entry_date=entry_date, raw=raw, all_dates=all_dates,
            )
            if not result:
                continue
            slot["active"] = result
            if verbose:
                print(f"  ENTER S{slot['idx']} {result['ticker']:6s} {result['direction']:5s} | "
                      f"Setup: {pick['setup']:<20s} Quality: {pick['quality']}  "
                      f"Entry: {result['entry']}  SL: {result['sl']}  TP: {result['tp']}  "
                      f"RSI: {pick['rsi']:.0f}  Vol: {pick['vol_ratio']:.1f}x  "
                      f"Scan: {today}  Enter: {entry_date}")

    # ── Close any still-open trades at end of window ─────────────────────────
    last_date = bt_dates[-1]
    for slot in slots:
        tr = slot["active"]
        if not tr:
            continue
        try:
            ep = float(raw["Close"][tr["ticker"]].loc[pd.Timestamp(last_date)])
        except Exception:
            ep = tr["entry"]
        pnl_pct = ((ep - tr["entry"]) / tr["entry"] * 100
                   if tr["direction"] == "LONG"
                   else (tr["entry"] - ep) / tr["entry"] * 100)
        pnl_usd = slot["capital"] * (pnl_pct / 100)
        slot["capital"] += pnl_usd
        tr.update({"exit_price": round(ep, 2), "outcome": "OPEN@END",
                   "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2),
                   "capital_after": round(slot["capital"], 2),
                   "slot": slot["idx"]})
        trades.append(tr)
        if verbose:
            print(f"  OPEN  S{slot['idx']} {tr['ticker']:6s} still open — marked at ${ep:.2f}  "
                  f"P&L: {pnl_pct:+.1f}% (${pnl_usd:+.0f})")

    total_capital = sum(s["capital"] for s in slots)
    return total_capital, trades, regime_blocked_days


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run():
    universe = load_universe()
    tickers  = universe["Ticker"].tolist()
    dl_tickers = tickers + [BENCHMARK, "^VIX"]

    # SMA200 needs 200 trading days (~290 calendar days) of warm-up.
    fetch_start = START_DATE - timedelta(days=300)
    print(f"Downloading OHLCV from {fetch_start} to {END_DATE}...", flush=True)
    raw = yf.download(
        dl_tickers, start=fetch_start.isoformat(), end=END_DATE.isoformat(),
        interval="1d", auto_adjust=True, progress=False,
    )
    if raw.empty:
        print("ERROR: No data."); sys.exit(1)

    all_dates = sorted(set(raw.index.date))
    bt_dates  = [d for d in all_dates if START_DATE <= d <= END_DATE]
    print(f"Backtest: {START_DATE} → {END_DATE}  ({len(bt_dates)} trading days)\n")

    spy_close, spy_ma, vix_close = build_regime_series(raw)
    if spy_close is None:
        print("WARN: regime data unavailable; gate disabled")

    capital, trades, regime_blocked_days = simulate(
        raw, tickers, all_dates, bt_dates,
        spy_close=spy_close, spy_ma=spy_ma, vix_close=vix_close,
        verbose=True,
    )

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

    bench_ret_pct = None
    bench_end_cap = None
    try:
        spy = raw["Close"][BENCHMARK].dropna()
        spy_window = spy[(spy.index.date >= bt_dates[0]) & (spy.index.date <= bt_dates[-1])]
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
    print(f"  Regime gate: blocked LONG scan on {regime_blocked_days} day(s) "
          f"(SPY>{SPY_MA_PERIOD}DMA & VIX<{VIX_MAX:.0f})")
    if bench_ret_pct is not None:
        alpha = strat_ret_pct - bench_ret_pct
        verdict = "BEAT ✓" if alpha > 0 else "LOST ✗"
        print(f"\n  Benchmark ({BENCHMARK} buy-and-hold):")
        print(f"    End:       ${bench_end_cap:,.0f}  ({bench_ret_pct:+.1f}%)")
        print(f"    Alpha:     {alpha:+.1f} pp   [{verdict}]")
    print(f"\n  Trade log:")

    display_cols = ["slot","ticker","direction","entry_date","entry","sl","tp",
                    "exit_date","exit_price","outcome","days_held","pnl_pct","pnl_usd","capital_after"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print(f"\n  Slots: {sg.MAX_SLOTS}  (each started at ${CAPITAL_INIT/sg.MAX_SLOTS:,.0f})")


if __name__ == "__main__":
    run()
