# ai-stock-investment

Long-only, rules-based swing-trading strategy for the top 100 S&P 500 stocks by market cap. Designed to be run through Claude Code — the `/technical-analysis` slash command executes the scanner and interprets the output.

Backtested 2024-04-20 → 2026-04-20 (2 years):

```
$10,000 → $19,809   (+98.1%)
SPY B&H: +45.6%
Alpha:   +52.5 pp    [BEAT ✓]
30 trades, 14W / 16L, 47% win rate
```

---

## The full rulebook

### 1. Universe

- Top 100 S&P 500 companies by market cap, re-queried from Wikipedia + yfinance, cached locally in `data/universe_top100.csv`.
- Refreshed automatically if the cache is older than 7 days, or on demand with `--refresh`.

### 2. Direction rule (the one that makes everything work)

- **price > SMA200 → consider LONG setups only.**
- **price < SMA200 → no trade.** Never short, never fight the dominant trend.
- Do not "all in on the cross" in a bear regime. Market breadth <30% = prefer cash over forced longs.

### 3. The four long setups

All setups require a **green candle** AND **volume > 20-day volume MA** on the trigger bar. If volume is weak, skip and wait.

| #   | Setup            | Key conditions                                                                                 | SL           | TP         |
| --- | ---------------- | ---------------------------------------------------------------------------------------------- | ------------ | ---------- |
| L1  | Ride Uptrend     | SMA50 > SMA200, price ≥3/5d above BB mid, RSI > 50, recent pullback to SMA50                   | **2× ATR**   | **4× ATR** |
| L2  | MACD Cross       | MACD histogram just crossed zero up, RSI 50–70, price > SMA50 but not touching upper BB        | **2.5× ATR** | **5× ATR** |
| L3  | VWAP Support     | SMA50 > SMA200, price 0–1.5% above VWAP (dipping toward it as support), RSI > 45               | **2× ATR**   | **3× ATR** |
| L4  | Pre-Golden Cross | SMA50 approaching SMA200 from below (<2% gap), price already > SMA200, RSI > 45, vol > 1.2× MA | **2× ATR**   | **4× ATR** |

### 4. Volatility guardrail

**Skip any entry where ATR% > 4.0.** Extreme-vol names (high-beta tech on news days, etc.) blow through wide stops before the setup works. Enforced by both the live scanner and the backtest.

### 5. Trade management — this is where most of the alpha comes from

1. **Initial stop** at the SL shown in the table (2× or 2.5× ATR below entry).
2. **Trailing-to-breakeven**: once price travels **50% of the way from entry to TP**, raise SL to entry. One-way ratchet — SL only moves up, never down, and only once.
    - Worked example: entry $130.52, TP $144.48. Midpoint = $137.50. When price tags $137.50, SL moves from $123.54 → $130.52. From then on, worst case is $0, best case is still +10.7%.
    - This rule alone converts several would-be losers into $0 scratches (AVGO, NEE ×2 in the 2-year backtest).
3. **Time stop**: exit at the close after **40 trading days** if neither TP nor SL has been hit — free up capital for fresh setups.
4. **One trade at a time.** Only scan for a new setup after the current trade exits. If the best setup on a given day scores below quality threshold (25), sit on hands.

### 6. Setup quality score (0–100)

Defined in `scripts/signals.py` (`quality()`). Used by **both** the backtest (picks the highest-scoring trigger each day) and the live scanner (surfaced as a `Quality` column to help rank when multiple setups trigger).

| Factor            | Max pts | Full credit                  | Half credit              |
| ----------------- | ------- | ---------------------------- | ------------------------ |
| Volume conviction | 35      | vol_ratio ≥ 2.0× MA (linear) | 1.5× = 17 pts            |
| RSI sweet spot    | 30      | 55–65 for longs              | 50–70 for longs (15 pts) |
| ATR Goldilocks    | 20      | 1.5–3.5%                     | 1.0–5.0% (10 pts)        |
| Trend alignment   | 10      | LONG + SMA50 > SMA200        | —                        |
| MACD alignment    | 5       | LONG + macd_hist > 0         | —                        |

`MIN_QUALITY_SCORE = 25` is the floor — below this the backtest sits in cash and the operator should skip the trade.

Score buckets to keep in mind:

- **80–100**: premium setup (BA's Pre-Golden Cross today scores 90 with 2.4× volume)
- **60–79**: workable but watch the weak factor (usually volume)
- **40–59**: marginal — skip unless multiple setups confirm same ticker
- **<40**: weak — backtest skips automatically below 25

### 7. Indicators computed per ticker

- SMA 50, SMA 200 (trend direction)
- Bollinger Bands (20, 2σ): upper / mid / lower
- RSI (14, EWM)
- ATR (14, EWM) — drives SL/TP sizing
- MACD (12, 26, 9): line, signal, histogram
- Volume + 20-day volume MA + vol ratio
- Rolling VWAP (20-day proxy)

---

## How to use it

### Run the scanner (live)

```bash
.venv/bin/python3 scripts/sma200_filter.py
```

Or in Claude Code: `/technical-analysis`.

Output: market-breadth regime, LONG universe table, and any triggered setups with **Quality (0–100)**, exact Entry / SL / TP / breakeven-trigger prices. Setups are sorted by Quality descending.

### Run the backtest

```bash
.venv/bin/python3 scripts/backtest.py
```

Configurable at the top of `scripts/backtest.py`:

```python
START_DATE        = date(2024, 4, 20)   # 2-year window by default
END_DATE          = date(2026, 4, 20)
CAPITAL_INIT      = 10_000.0
TIME_STOP_DAYS    = 40
MAX_ATR_PCT       = 4.0
BENCHMARK         = "SPY"
```

Strategy tunables (setup thresholds, SL/TP ATR multipliers, quality-score weights, `MIN_QUALITY_SCORE`) live in `scripts/signals.py` so they're shared by the backtest and the live scanner. Edit there and re-run the backtest to A/B a change.

### First-time setup

```bash
uv venv .venv -q
uv pip install yfinance pandas lxml requests --python .venv/bin/python3 -q
```

---

## Code layout

```
scripts/
  universe.py       # Fetches + caches top 100 S&P 500 by market cap
  indicators.py     # Computes SMA/BB/RSI/ATR/MACD/VWAP per ticker
  signals.py        # Four long setups L1-L4 + quality() scorer + all tunables
  sma200_filter.py  # Live scanner — prints breadth + triggered setups w/ Quality
  backtest.py       # 1-trade-at-a-time engine with trailing-BE stop + SPY benchmark
.claude/
  commands/
    technical-analysis.md   # Slash command for Claude Code
  settings.json             # Project default model: claude-opus-4-7
```

---

## Why these specific rules — a short history

The first version of this strategy had **both long and short setups**, **1×/1.5× ATR stops**, and a **20-day time stop**. Over the 1-year window 2025-04-20 → 2026-04-20 it lost **-25.6% against SPY +39.8%** (alpha -65.4 pp). Four iterations fixed it:

| Change                                | Rationale                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------- |
| Disable SHORT setups                  | Shorts kept getting clipped by the dominant bull tape before any move developed |
| SL 2× ATR, TP 4× ATR (from 1× / 1.5×) | Tight stops get eaten by daily noise before the setup resolves                  |
| Time stop 20 → 40 days                | Winners like AVGO (+17.8% in 40 days) need room to run                          |
| Trailing-to-breakeven at 50% of TP    | Protects profits without capping upside — the single biggest alpha source       |

Result: flipped from losing -65 pp to beating +50 pp over 2 years.

Things that were tried and reverted (didn't help):

- Earlier BE trigger (40% of TP): scratches too many trades before they reach target
- Later BE trigger (60% of TP): more trades revert to full SL before lock

## Tuning session (2026-04-23) — L1 volume threshold raised to 1.3×

### What we did

Ran `scripts/tune.py` — 108-variation grid search across `MIN_QUALITY_SCORE`, `L1_SL_ATR`, `L1_TP_ATR`, `L1_MIN_VOL_RATIO`, `L4_TP_ATR`. Used proper train/test split: Year 1 (2024-04-20 → 2025-04-20) for tuning, Year 2 for held-out validation. A variation was only flagged as "robust" if it beat baseline by ≥10pp alpha on **both** windows.

### Headline finding

**24 of 108 variations cleared the bar — and every single one shares the same change: `L1_MIN_VOL_RATIO: 1.0 → 1.3`.** The other knobs (SL/TP/MQ) only mattered at the margins. One knob, one direction, validated across both years and many parameter combinations.

**Mechanism:** L1 (Ride Uptrend) was triggering on stocks with `vol_ratio` as low as 1.03 — barely above the 20-day average, essentially no real-money conviction. Quality scorer already weights volume heaviest (35 pts), but the trigger gate was letting marginal-volume names through. Raising the gate to 1.3× filters them out before they enter the candidate pool.

### Magnitude lesson — per-window tuning misled us

The tuner showed two attractive picks:

| Pick                                       | Y1 alpha (per-window) | Y2 alpha (per-window) |
| ------------------------------------------ | --------------------- | --------------------- |
| `L1_MIN_VOL_RATIO=1.3` + `L1_TP_ATR=5.0`   | +24.4pp               | +21.8pp               |
| Just `L1_MIN_VOL_RATIO=1.3`                | +20–30pp              | +16–18pp              |

We tried the first pick first (volume + wider TP). Continuous backtest result: **-10.5pp alpha — LOST to SPY**.

Reverted to just the volume change. Continuous backtest: **+52.5pp alpha (vs baseline +50.1pp)**.

So the volume-only change is a real win, but only **+2.4pp better than baseline** — not the +20pp the tuner predicted. The wider TP actively *hurt* in continuous mode (longer holds → more time-stop exits → fewer trades cycle through → less compounding) even though per-window analysis liked it.

### Lesson

**`tune.py`'s per-window-with-fresh-capital framework is useful for *ranking* parameter directions but wildly unreliable for predicting *continuous* P&L magnitude.** It's a screening tool, not a final verdict. Always re-validate any tuner finding by running the full continuous `backtest.py` before shipping.

### What's next

Today's tuning only varied L1 and L4 parameters (the most-triggered setups). L2 (MACD Cross) and L3 (VWAP Support) might have over-tight thresholds that block useful triggers — that's the next area to probe with another grid search round.

L2/L3 grid is wired up in `scripts/tune.py` (288 variations, ~4 hours overnight). Run with:

```bash
.venv/bin/python3 scripts/tune.py 2>&1 | tee /tmp/tune-l2l3.log
```

Output is saved to `/tmp/tune-l2l3.log` — read that file the next morning if the terminal session is gone. (Note: `/tmp` is wiped on reboot — copy somewhere durable if you don't analyze it the same day.)

## MCP Servers (optional tooling)

---

## MCP Servers (optional tooling)

### Prerequisites

- [uv](https://github.com/astral-sh/uv) — for `uvx` (Python MCP servers)
- [Node.js](https://nodejs.org) — for `npx` (Node MCP servers)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### yfinance-mcp

Fetches stock market data — prices, financials, historical OHLCV.

```bash
uvx yfinance-mcp
claude mcp add yfinance-mcp uvx yfinance-mcp
```

### playwright

Browser automation — scraping news, financial sites, charts.

```bash
npx -y @playwright/mcp --browser chromium
claude mcp add playwright -- npx -y @playwright/mcp --browser chromium
```

### Verify MCP servers are active

```bash
claude mcp list
```

---

## Disclaimer

This is backtested on historical data across a 2-year window. Past performance does not guarantee future results. The strategy was tuned against its own evaluation window, so the +50pp alpha number has some degree of in-sample fit — treat it as a reasoned starting point, not a promise.

## Claude session

claude --resume 42f69484-bc88-40bb-b701-fd0e2a251dd0 --dangerously-skip-permissions

claude --resume c4c65570-757b-44ec-9f03-5aa0b85044cd --dangerously-skip-permissions
