#!/usr/bin/env python3
"""Compute all technical indicators for a single ticker's OHLCV DataFrame."""

import pandas as pd


def ticker_frame(raw: pd.DataFrame, ticker: str, up_to: pd.Timestamp = None) -> pd.DataFrame:
    """Slice a yfinance multi-ticker download into a per-ticker OHLCV frame.
    Pass `up_to` to truncate at a given timestamp (backtest uses this to avoid look-ahead)."""
    try:
        df = pd.DataFrame({
            "Open":   raw["Open"][ticker],
            "High":   raw["High"][ticker],
            "Low":    raw["Low"][ticker],
            "Close":  raw["Close"][ticker],
            "Volume": raw["Volume"][ticker],
        })
    except (KeyError, TypeError):
        return None
    if up_to is not None:
        df = df.loc[:up_to]
    return df.dropna()


def compute(df: pd.DataFrame) -> dict:
    """
    df must have columns: Open, High, Low, Close, Volume
    Returns a flat dict of indicator values (last bar unless noted).
    """
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    n = len(c)
    if n < 200:
        return {}

    # ── Trend ────────────────────────────────────────────────────────────────
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # normalised band width

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    # ── ATR (14) ─────────────────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_ma20 = v.rolling(20).mean()
    is_green = c > o  # True = green candle (close > open)

    # ── Rolling VWAP (20-day proxy for anchored VWAP) ────────────────────────
    typical = (h + l + c) / 3
    vwap = (typical * v).rolling(20).sum() / v.rolling(20).sum()

    # ── Context windows ──────────────────────────────────────────────────────
    # Price position relative to BB mid over last 5 days
    above_mid_5d = int((c.tail(5) > bb_mid.tail(5)).sum())
    below_mid_5d = int((c.tail(5) < bb_mid.tail(5)).sum())

    # MACD histogram sign changes in last 5 bars
    hist_last5 = macd_hist.tail(5)
    macd_crossed_up = bool((hist_last5.iloc[-1] > 0) and any(hist_last5.iloc[:-1] <= 0))
    macd_crossed_down = bool((hist_last5.iloc[-1] < 0) and any(hist_last5.iloc[:-1] >= 0))

    # Pullback check: price within 3% of SMA50 in last 3 bars
    near_sma50_recently = bool(((c.tail(3) - sma50.tail(3)).abs() / sma50.tail(3) < 0.03).any())

    # Trend state (SMA50 vs SMA200) — current relationship, not the cross event
    sma50_above_sma200 = bool(sma50.iloc[-1] > sma200.iloc[-1])
    cross_recent = abs(sma50.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1] < 0.02  # within 2%

    return {
        # ── Last-bar values ───────────────────────────────────────────
        "price":        round(float(c.iloc[-1]), 2),
        "open":         round(float(o.iloc[-1]), 2),
        "sma50":        round(float(sma50.iloc[-1]), 2),
        "sma200":       round(float(sma200.iloc[-1]), 2),
        "bb_upper":     round(float(bb_upper.iloc[-1]), 2),
        "bb_mid":       round(float(bb_mid.iloc[-1]), 2),
        "bb_lower":     round(float(bb_lower.iloc[-1]), 2),
        "bb_width":     round(float(bb_width.iloc[-1]), 4),
        "rsi":          round(float(rsi.iloc[-1]), 1),
        "atr":          round(float(atr.iloc[-1]), 2),
        "atr_pct":      round(float(atr.iloc[-1] / c.iloc[-1]) * 100, 2),
        "macd":         round(float(macd_line.iloc[-1]), 4),
        "macd_signal":  round(float(macd_signal.iloc[-1]), 4),
        "macd_hist":    round(float(macd_hist.iloc[-1]), 4),
        "macd_hist_prev": round(float(macd_hist.iloc[-2]), 4),
        "volume":       int(v.iloc[-1]),
        "vol_ma20":     round(float(vol_ma20.iloc[-1]), 0),
        "vol_ratio":    round(float(v.iloc[-1] / vol_ma20.iloc[-1]), 2),
        "is_green":     bool(is_green.iloc[-1]),
        "vwap":         round(float(vwap.iloc[-1]), 2),
        # ── Context ───────────────────────────────────────────────────
        "sma50_above_sma200": sma50_above_sma200,
        "cross_recent":       bool(cross_recent),
        "above_mid_5d":       above_mid_5d,
        "below_mid_5d":       below_mid_5d,
        "near_sma50_recently": near_sma50_recently,
        "macd_crossed_up":    macd_crossed_up,
        "macd_crossed_down":  macd_crossed_down,
        "price_vs_sma200_pct": round(float((c.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1]) * 100, 1),
        "price_vs_vwap_pct":  round(float((c.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) * 100, 1),
        "price_above_sma200": bool(c.iloc[-1] > sma200.iloc[-1]),
    }
