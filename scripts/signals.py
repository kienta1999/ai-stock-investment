#!/usr/bin/env python3
"""
Score each stock against the trading playbook strategies.

Rule: only LONG when price > SMA200. Price < SMA200 → no trade (do not short,
do not fight the tape).

All thresholds and multipliers below are tunable parameters — change here,
then re-run scripts/backtest.py to see the impact on the 2-year return.
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Algo-level filters — applied BEFORE per-setup evaluation. Consumed by the
# live scanner (sma200_filter.py) and the backtest engine (backtest.simulate).
# ─────────────────────────────────────────────────────────────────────────────

MAX_ATR_PCT    = 4.0    # skip entries where ATR% exceeds this (vol-cap guardrail)
SPY_MA_PERIOD  = 200    # SPY trend filter for long-entry regime gate
VIX_MAX        = 30.0   # block LONG entries when VIX >= this
BENCHMARK      = "SPY"  # market benchmark (regime gate + buy-and-hold comparison)


# ─────────────────────────────────────────────────────────────────────────────
# Tunable parameters
# ─────────────────────────────────────────────────────────────────────────────

# L1: Ride Uptrend — pullback to SMA50 in an established uptrend
L1_MIN_ABOVE_MID_5D   = 3     # of last 5 closes, how many must be above BB mid
L1_MIN_RSI            = 50    # momentum floor (>50 = bullish bias)
L1_MIN_VOL_RATIO      = 1.3   # today's volume must beat 20-day MA by 30%+
L1_SL_ATR             = 2.0   # stop-loss distance = N × ATR
L1_TP_ATR             = 4.0   # take-profit distance = N × ATR

# L2: MACD Bullish Cross — histogram just turned positive
L2_RSI_MIN            = 50    # RSI sweet spot floor
L2_RSI_MAX            = 70    # RSI sweet spot ceiling (avoid overbought)
L2_BB_UPPER_HEADROOM  = 0.99  # price must be < bb_upper × this (avoid extension)
L2_MIN_VOL_RATIO      = 1.3
L2_SL_ATR             = 2.5
L2_TP_ATR             = 5.0

# L3: VWAP Support — price dipping toward VWAP as support
L3_VWAP_DIST_MIN      = 0.0   # min % above VWAP (must be above, not below)
L3_VWAP_DIST_MAX      = 1.5   # max % above VWAP (close enough to be a touch)
L3_MIN_RSI            = 45    # looser than L1/L2 — pullbacks dent RSI
L3_MIN_ABOVE_MID_5D   = 1     # at least one of last 5 above BB mid
L3_MIN_VOL_RATIO      = 1.0
L3_SL_ATR             = 2.5
L3_TP_ATR             = 2.5   # tighter target — VWAP bounces are shorter moves

# L4: Pre-Golden Cross — SMA50 about to cross SMA200 from below
L4_MIN_RSI            = 45
L4_MIN_VOL_RATIO      = 1.2   # stricter — need conviction for early entry
L4_SL_ATR             = 2.0
L4_TP_ATR             = 4.0

# ─────────────────────────────────────────────────────────────────────────────
# Quality scoring — rates a TRIGGERED setup on 0–100. Used by the backtest to
# pick the single best candidate per day, and surfaced in the live scanner so
# the operator can prefer high-quality triggers over weak ones.
# ─────────────────────────────────────────────────────────────────────────────

MIN_QUALITY_SCORE     = 25    # backtest sits in cash if best setup scores below this

# Volume conviction — linear: vol_ratio 1.0 = 0 pts, 2.0 = full pts
Q_VOL_MAX_PTS         = 35    # heaviest weight — volume = real-money confirmation

# RSI sweet spot
Q_RSI_SWEET_PTS       = 30    # full credit when in tight band (LONG: 55–65)
Q_RSI_OK_PTS          = 15    # half credit for wider band (LONG: 50–70)
Q_RSI_LONG_SWEET      = (55, 65)
Q_RSI_LONG_OK         = (50, 70)

# ATR Goldilocks — too low = no movement, too high = blowup risk
Q_ATR_SWEET_PTS       = 20
Q_ATR_OK_PTS          = 10
Q_ATR_SWEET_RANGE     = (1.5, 3.5)
Q_ATR_OK_RANGE        = (1.0, 5.0)

# Confirmation bonuses
Q_TREND_ALIGN_PTS     = 10    # LONG + SMA50 > SMA200
Q_MACD_ALIGN_PTS      = 5     # LONG + MACD hist > 0

# ─────────────────────────────────────────────────────────────────────────────


def score(ind: dict) -> list[dict]:
    """
    Evaluate all long setups for a single ticker's indicator dict.
    Returns a list of triggered setup dicts (may be empty).
    """
    setups = []
    p = ind["price"]
    atr = ind["atr"]

    if ind["price_above_sma200"]:
        setups += _long_setups(ind, p, atr)
    return setups


# ─────────────────────────────────────────────────────────────────────────────
# LONG SETUPS  (price > SMA200)
# ─────────────────────────────────────────────────────────────────────────────

def _long_setups(ind: dict, p: float, atr: float) -> list[dict]:
    setups = []

    # ── L1: Ride the Uptrend ─────────────────────────────────────────────────
    # Consistent above BB mid → pullback to SMA50 → green vol spike
    if (
        ind["sma50_above_sma200"]                           # SMA50 > SMA200
        and ind["above_mid_5d"] >= L1_MIN_ABOVE_MID_5D      # price mostly above BB mid
        and ind["rsi"] > L1_MIN_RSI
        and ind["near_sma50_recently"]                      # recent pullback to SMA50
        and ind["is_green"]
        and ind["vol_ratio"] > L1_MIN_VOL_RATIO
    ):
        entry = p
        sl = round(p - L1_SL_ATR * atr, 2)
        tp = round(p + L1_TP_ATR * atr, 2)
        setups.append(_setup("Ride Uptrend", "LONG", entry, sl, tp,
                             f"1:{L1_TP_ATR / L1_SL_ATR:.1f}",
                             "1D → 3-6 wk options",
                             f"RSI {ind['rsi']:.0f}, vol {ind['vol_ratio']:.1f}x MA"))

    # ── L2: MACD Bullish Cross ───────────────────────────────────────────────
    # Hist just crossed above zero, RSI in sweet spot, not at upper BB
    if (
        ind["macd_crossed_up"]
        and L2_RSI_MIN <= ind["rsi"] <= L2_RSI_MAX
        and p > ind["sma50"]
        and p < ind["bb_upper"] * L2_BB_UPPER_HEADROOM      # not touching upper BB
        and ind["is_green"]
        and ind["vol_ratio"] > L2_MIN_VOL_RATIO
    ):
        entry = p
        sl = round(p - L2_SL_ATR * atr, 2)
        tp = round(p + L2_TP_ATR * atr, 2)
        setups.append(_setup("MACD Cross", "LONG", entry, sl, tp,
                             f"1:{L2_TP_ATR / L2_SL_ATR:.1f}",
                             "1D → 3-6 wk options",
                             f"MACD hist {ind['macd_hist']:+.3f}, RSI {ind['rsi']:.0f}"))

    # ── L3: VWAP Support Long ────────────────────────────────────────────────
    # Price just above VWAP (dipping toward it as support)
    vd = ind["price_vs_vwap_pct"]
    if (
        ind["sma50_above_sma200"]
        and L3_VWAP_DIST_MIN < vd < L3_VWAP_DIST_MAX
        and ind["rsi"] > L3_MIN_RSI
        and ind["above_mid_5d"] >= L3_MIN_ABOVE_MID_5D
        and ind["is_green"]
        and ind["vol_ratio"] > L3_MIN_VOL_RATIO
    ):
        entry = p
        sl = round(p - L3_SL_ATR * atr, 2)
        tp = round(p + L3_TP_ATR * atr, 2)
        setups.append(_setup("VWAP Support", "LONG", entry, sl, tp,
                             f"1:{L3_TP_ATR / L3_SL_ATR:.1f}",
                             "1H/4H → 2-3 wk options",
                             f"Price {vd:+.1f}% vs VWAP, RSI {ind['rsi']:.0f}"))

    # ── L4: Pre-Golden Cross Reversal ────────────────────────────────────────
    # SMA50 approaching SMA200 from below, price already above SMA200
    if (
        not ind["sma50_above_sma200"]
        and ind["cross_recent"]                             # SMA50/200 gap < 2%
        and ind["rsi"] > L4_MIN_RSI
        and ind["is_green"]
        and ind["vol_ratio"] > L4_MIN_VOL_RATIO
    ):
        entry = p
        sl = round(p - L4_SL_ATR * atr, 2)
        tp = round(p + L4_TP_ATR * atr, 2)
        setups.append(_setup("Pre-Golden Cross", "LONG", entry, sl, tp,
                             f"1:{L4_TP_ATR / L4_SL_ATR:.1f}",
                             "1D → 3-6 wk options",
                             f"SMA50/200 gap <2%, vol {ind['vol_ratio']:.1f}x"))

    return setups


# ─────────────────────────────────────────────────────────────────────────────

def _setup(name, direction, entry, sl, tp, rr, timeframe, notes) -> dict:
    return {
        "setup": name,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "timeframe": timeframe,
        "notes": notes,
    }


def quality(row: dict) -> float:
    """
    Score a triggered setup on 0–100. Higher = cleaner. Used by:
      - backtest.py to pick the single best candidate per day
      - sma200_filter.py to surface ranking alongside binary triggers
    The five factors and their weights are defined as Q_* constants above.
    """
    q = 0.0
    rsi       = row.get("rsi", 50)
    vol_ratio = row.get("vol_ratio", 1.0)
    atr_pct   = row.get("atr_pct", 2.5)
    direction = row.get("direction", "LONG")
    uptrend   = row.get("sma50_above_sma200", False)
    macd_hist = row.get("macd_hist", 0)

    # Volume conviction (linear, capped)
    q += min((vol_ratio - 1.0) * Q_VOL_MAX_PTS, Q_VOL_MAX_PTS)

    # RSI sweet spot
    if direction == "LONG":
        if Q_RSI_LONG_SWEET[0] <= rsi <= Q_RSI_LONG_SWEET[1]:
            q += Q_RSI_SWEET_PTS
        elif Q_RSI_LONG_OK[0] <= rsi <= Q_RSI_LONG_OK[1]:
            q += Q_RSI_OK_PTS

    # ATR Goldilocks
    if Q_ATR_SWEET_RANGE[0] <= atr_pct <= Q_ATR_SWEET_RANGE[1]:
        q += Q_ATR_SWEET_PTS
    elif Q_ATR_OK_RANGE[0] <= atr_pct <= Q_ATR_OK_RANGE[1]:
        q += Q_ATR_OK_PTS

    # Trend alignment
    if direction == "LONG" and uptrend:
        q += Q_TREND_ALIGN_PTS

    # MACD alignment
    if direction == "LONG" and macd_hist > 0:
        q += Q_MACD_ALIGN_PTS

    return round(q, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Regime helpers — algo-level gates that sit above per-setup triggers
# ─────────────────────────────────────────────────────────────────────────────

def market_regime(long_count: int, total: int) -> str:
    """Breadth-based regime label from count of tickers above their SMA200."""
    rate = long_count / total
    short_count = total - long_count
    base = f"{long_count}L / {short_count}S of {total}"
    if rate >= 0.70:
        return f"STRONG BULL ({base}) — broad uptrend, favour longs"
    elif rate >= 0.50:
        return f"MIXED ({base}) — selective longs, tighter stops"
    elif rate >= 0.30:
        return f"WEAKENING ({base}) — more shorts than longs, reduce long size"
    else:
        return f"BEAR ({base}) — majority below SMA200, favour shorts"


def long_regime_ok(spy_close: "pd.Series", spy_ma: "pd.Series",
                   vix_close: "pd.Series", today_ts: "pd.Timestamp",
                   vix_max: float = VIX_MAX) -> bool:
    """Gate for LONG entries: SPY > 200DMA AND VIX < vix_max. Fail-open on missing data."""
    try:
        spy_px = float(spy_close.asof(today_ts))
        spy_m  = float(spy_ma.asof(today_ts))
        vix_px = float(vix_close.asof(today_ts))
    except Exception:
        return True
    return (spy_px > spy_m) and (vix_px < vix_max)


def build_regime_series(raw, benchmark: str = BENCHMARK):
    """Precompute regime series from raw OHLCV. Returns (spy_close, spy_ma, vix_close)
    or (None, None, None) if either benchmark or ^VIX is missing."""
    try:
        spy_close = raw["Close"][benchmark].ffill()
        spy_ma    = spy_close.rolling(SPY_MA_PERIOD).mean()
        vix_close = raw["Close"]["^VIX"].ffill()
        return spy_close, spy_ma, vix_close
    except (KeyError, Exception):
        return None, None, None
