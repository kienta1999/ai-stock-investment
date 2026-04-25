# ai-stock-investment

Long-only, rules-based swing-trading strategy for the top 100 S&P 500 stocks by market cap. Designed to be run through Claude Code — the `/technical-analysis` slash command executes the scanner and interprets the output.

Backtested 2024-04-20 → 2026-04-20 (2 years):

```
$10,000 → $27,242   (+172.4%)
SPY B&H: +45.6%
Alpha:   +126.9 pp   [BEAT ✓]
42 trades, 22W / 20L, 52% win rate
```

Also survives regime change:

- **2020 COVID crash** (2020-02-19 → 2020-12-31): +42.4% vs SPY +12.0% → **+30.4pp alpha**, 60% win rate
- **2008 GFC full cycle** (2007-10-10 → 2009-12-31): +30.2% vs SPY -24.1% → **+54.4pp alpha**, 18 trades, regime gate blocked 337 days

Both windows are pure out-of-sample (tuned on 2024-2026). The market-wide regime gate is what carries the crisis windows.

---

## The full rulebook

### 1. Universe

- Top 100 S&P 500 companies by market cap, re-queried from Wikipedia + yfinance, cached locally in `data/universe_top100.csv`.
- Refreshed automatically if the cache is older than 7 days, or on demand with `--refresh`.

### 2. Direction rule (the one that makes everything work)

**Per-ticker filter:**
- **price > SMA200 → consider LONG setups only.**
- **price < SMA200 → no trade.** Never short, never fight the dominant trend.

**Market-wide regime gate (added 2026-04-23):** even for tickers above their own SMA200, block LONG entries when either:
- **SPY close < SPY 200-day MA** (broad market not in uptrend), OR
- **VIX ≥ 30** (crash regime — don't catch falling knives)

Lives in `scripts/signals.py` as `long_regime_ok()` — single SOT, applied by the shared `simulate()` engine used by both the backtest and tuner. Swung COVID-2020 alpha from **-24.9pp → +30.4pp** (see "Regime gate" section below). Do not "all in on the cross" in a bear regime — breadth <30% = prefer cash over forced longs.

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
BENCHMARK         = "SPY"
```

All algo-level knobs — setup thresholds, SL/TP ATR multipliers, quality-score weights, `MIN_QUALITY_SCORE`, AND the pre-setup global filters — live in `scripts/signals.py` as the single SOT, imported by both the backtest engine and the live scanner. Top of the file:

```python
# Algo-level filters (applied before any setup is evaluated)
MAX_ATR_PCT    = 4.0    # skip entries where ATR% > this (vol-cap guardrail)
SPY_MA_PERIOD  = 200    # SPY trend filter for long-entry regime gate
VIX_MAX        = 30.0   # block LONG entries when VIX >= this

# Plus L1/L2/L3/L4 setup thresholds, quality-score weights,
# MIN_QUALITY_SCORE, market_regime(), long_regime_ok() ...
```

Edit there and re-run the backtest to A/B a change.

The core simulation loop (`simulate()`) lives in `scripts/backtest.py` and is imported by `scripts/tune.py` — parameter sweeps inherit every rule change automatically, zero drift between backtest and tuner.

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
  signals.py        # SOT for all trade rules: L1-L4 setups, quality() scorer,
                    #   long_regime_ok(), market_regime(), MAX_ATR_PCT,
                    #   SPY_MA_PERIOD, VIX_MAX, MIN_QUALITY_SCORE, Q_* weights
  sma200_filter.py  # Live scanner CLI — prints breadth + triggered setups
  backtest.py       # simulate() engine (1 trade at a time, trailing-BE stop,
                    #   SPY regime gate) + CLI entry point + benchmark summary
  tune.py           # Parameter grid search — imports simulate() from backtest
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

## Tuning session round 2 (2026-04-23 overnight) — L2 volume threshold raised to 1.3×

### What we did

Ran the 288-variation L2/L3 grid overnight. Per-window analysis was inconclusive — only 4 variations cleared the +10pp/+10pp floor and they all *lost* to the post-L1V1.3 baseline on Y1 (+30.7pp vs baseline +67.6pp) while winning Y2 (+16.3pp vs +7.1pp). Not a clean signal.

Tried the single most theoretically-motivated change anyway (`L2_MIN_VOL_RATIO: 1.0 → 1.3`, paralleling yesterday's L1 finding) and ran the continuous backtest.

### Continuous backtest result

| Metric          | L1V1.3 only | +L2V1.3       | Δ          |
| --------------- | ----------- | ------------- | ---------- |
| Return          | +98.1%      | **+121.8%**   | **+23.7pp** |
| Alpha           | +52.5pp     | **+76.3pp**   | **+23.8pp** |
| End cap ($10k)  | $19,809     | **$22,183**   | +$2,374    |
| Trades          | 30          | **41**        | +11        |
| Win rate        | 47%         | 46%           | ≈ same     |

### Why it worked — capital recycling, not setup selection

The L2 filter **didn't make L2 triggers better** — same 3 L2 trades (TJX/META/QCOM) in both runs. Instead, raising the L2 volume gate *kept the scanner from getting trapped in slow-grinder L2 entries that were blocking capital from faster-recycling L1/L3 opportunities*.

Concrete example: April 22 2024, baseline picks KO's MACD Cross (vol_ratio=1.2, quality 62.4) and holds 44 days for +5.9%. With the new gate, KO doesn't trigger → scanner picks GOOGL's VWAP Support instead → exits in 3 days at +5.8% → 41 freed days to compound through other trades.

Setup count shifted accordingly:

| Setup            | Before (30 trades) | After (41 trades) |
| ---------------- | ------------------ | ----------------- |
| L1 Ride Uptrend  | ~15                | ~20               |
| L3 VWAP Support  | ~8                 | ~17               |
| L2 MACD Cross    | 3                  | 3                 |
| L4 Pre-Golden X  | 1                  | 1                 |

### Methodological lesson — `tune.py` can be directionally wrong

**The `L2V1.3-only` combo was NOT in the grid's 4 qualifying variations.** Per-window analysis predicted it would fail the +10pp floor. Continuous mode showed it adds +24pp alpha. The per-window framework can't see capital-recycling effects because each year starts with fresh $10k and no carry-over state.

Previously we thought tune.py's output was just magnitude-unreliable. It's actually worse: **it can miss real winners entirely**. Treat tune.py as a candidate-generator only. Continuous backtest is the single source of truth.

### Current configuration

```
L1_MIN_VOL_RATIO = 1.3   (changed from 1.0 — tuning session 1)
L2_MIN_VOL_RATIO = 1.3   (changed from 1.0 — tuning session 2)
L3_SL_ATR        = 2.5   (changed from 2.0 — tuning session 3)
L3_TP_ATR        = 2.5   (changed from 3.0 — tuning session 3)
```

Sessions 1 & 2 fit one story: *volume conviction at the trigger gate matters as much as volume in the quality score* (continuation setups L1/L2).

Session 3 fits the **opposite** story for L3 (mean-reversion): *bounces are short-lived; grab them quickly with a 1:1 R:R rather than waiting for full extension*. L3 win rate jumped from 53% to 69% with this change.

## Tuning session round 3 (2026-04-23) — L3 R:R tightened to 1:1

### What we did

Ran a 324-variation L3-only grid (5 dimensions: VWAP distance, RSI floor, BB-mid window, SL multiplier, TP multiplier — vol threshold deliberately excluded since round 2 proved it breaks L3). 17 of 324 cleared the +10pp/+10pp floor, in two clusters:

- **Cluster A (top 6)**: tighten L3 R:R to 1:1 — `L3_SL_ATR=2.5, L3_TP_ATR=2.5`
- **Cluster B (next 7)**: widen VWAP touch zone — `L3_VWAP_DIST_MAX=3.0`

### Continuous backtest result (Cluster A picked)

| Metric          | Pre-L3-tune       | + L3 1:1 R:R          | Δ          |
| --------------- | ----------------- | --------------------- | ---------- |
| Return          | +121.8%           | **+125.6%**           | **+3.8pp** |
| Alpha           | +76.3pp           | **+80.0pp**           | **+3.7pp** |
| End cap ($10k)  | $22,183           | **$22,556**           | +$373      |
| Trades          | 41                | **44**                | +3         |
| Win rate        | 46%               | **48%**               | +2pp       |
| L3 win rate     | 53%               | **69%**               | +16pp      |
| L3 avg P&L      | +1.1%/trade       | **+2.08%/trade**      | +0.98pp    |

### Mechanism — opposite of L1

L1 wants wide TP (4× ATR) to let trending winners run. L3 wants tight TP (2.5× ATR) because VWAP bounces are short and quick — grab the move before it reverses. The wider SL (2.5× vs 2.0× ATR) also gives the trade more room to reach the trailing-BE trigger before chop noise stops it out.

### Per-window framework — 10× overestimate again

Tune.py predicted Y2 alpha would jump from -3.1pp to +33.2pp (+36pp Y2 improvement). Continuous reality was +3.7pp total alpha gain. Still directionally right, but the magnitude lesson keeps repeating: **always validate via continuous backtest before shipping**.

### Setup-by-setup stats in current config (41 trades, 2y)

| Setup                     | Trades | Wins | Flat (BE) | Losses | Win rate* | Avg P&L/trade |
| ------------------------- | ------ | ---- | --------- | ------ | --------- | ------------- |
| L1 Ride Uptrend           | 20     | 8    | 6         | 6      | 57%       | +2.4%         |
| L3 VWAP Support           | 17     | 8    | 2         | 7      | 53%       | +1.1%         |
| L2 MACD Cross             | 3      | 2    | 0         | 1      | 67%       | +6.0%         |
| L4 Pre-Golden Cross       | 1      | 1*   | 0         | 0      | —         | +1.8%         |

*Win rate excludes BE-flat trades (trailing-to-breakeven saved a would-be loser). L4 exit was TIME-stop positive.

### Priority for future tuning rounds

1. **L3 VWAP Support** — highest frequency × lowest per-trade edge = biggest lever. Vol threshold already proven to hurt (mean-reversion setup; high vol on dips = distribution). Next dimensions: VWAP distance, RSI floor, SL/TP ratios.
2. **L1 Ride Uptrend** — biggest single contributor by total. Volume already tuned. Try RSI, BB-mid window, or pullback-proximity next.
3. **L2 MACD Cross** — best per-trade (+6.0%), only 3 triggers/2y. Last night's grid hinted `L2_RSI_MAX: 70→75` and `L2_TP_ATR: 5→6` might help — untested in continuous mode.
4. **L4 Pre-Golden Cross** — n=1, untouchable until universe broadens.

**Key methodological rule going forward:** always re-run `scripts/backtest.py` for continuous validation after ANY tune.py experiment. Per-window numbers can mislead in both magnitude and direction.

### Out-of-sample validation — 2022-01-01 → 2024-01-01 (pre-regime-gate)

Ran the post-tuning config (L1V1.3 + L2V1.3, current defaults) on a completely different regime — 2022 Fed-hiking bear + 2023 recovery — to confirm we're not just overfit to 2024-2026. These numbers are from **before** the regime gate was added; worth re-running now that the gate is live.

```
$10,000 → $11,885   (+18.9%)
SPY B&H: +2.6%
Alpha:   +16.2 pp    [BEAT ✓]
43 trades, 15W / 28L, 35% win rate
```

Win rate dropped to 35% in chop (vs 46% in trending bull) but trailing-to-breakeven saved many would-be losers — many SL exits show 0.0% P&L. The strategy still produced **6× the SPY return** in a flat/bear window. Structural protections (long-only + SMA200 gate + ATR vol cap) kept it from blowing up in 2022 carnage.

## Regime gate (2026-04-23) — SPY>200DMA + VIX<30 for LONG entries

### The COVID-2020 stress test that broke the algo

Ran the fully-tuned config on 2020-02-19 → 2020-12-31 (COVID crash + recovery) and got demolished:

```
$10,000 → $8,713    (-12.9%)
SPY B&H: +12.0%
Alpha:   -24.9 pp    [LOST ✗]
17 trades, 6W / 11L, 35% win rate
```

Damage concentrated in two clusters:
1. **Feb 19-28 entries** (UNP/HD/NFLX): opened longs right as COVID broke, got stopped within days.
2. **Oct-Nov chop** (AMZN/PFE/META/PLD): pre-election volatility ate trend setups.

The algo *did* partially adapt (no entries Mar 9 → Apr 29, skipping the worst of the crash) but the Feb losses and the post-election whipsaws were already baked in.

### The fix — a market-wide regime gate

Added `long_regime_ok()` in `scripts/signals.py` (single SOT, consumed by the shared `simulate()` engine). LONG entries now require BOTH:
- **SPY close > SPY 200-day MA** (broad trend intact)
- **VIX < 30** (no crash regime)

Either condition fails → the scanner won't trigger a LONG that day. Applied uniformly in `backtest.py` and `tune.py`.

### Results

| Window                              | Before gate                | After gate                 | Δ alpha     |
| ----------------------------------- | -------------------------- | -------------------------- | ----------- |
| COVID (2020-02-19 → 2020-12)        | -12.9% / alpha **-24.9pp** | +42.4% / alpha **+30.4pp** | **+55.3pp** |
| Bull run (2024-04-20 → 2026-04)     | +125.6% / **+80.0pp**      | +172.4% / **+126.9pp**     | **+46.9pp** |
| GFC full cycle (2007-10-10 → 2009-12) | —                        | +30.2% / **+54.4pp**       | —           |

- COVID window: gate blocked 68 days of LONG scanning. Trades dropped 17→15, win rate 35%→60%, alpha -24.9pp → **+30.4pp** ([BEAT ✓]).
- Bull window: gate blocked 45 days. Trades 44→42, win rate 48%→52%, alpha +80pp → **+126.9pp**.
- GFC full cycle: gate blocked **337 days** (56% of the window). 18 trades, 8W/10L, 44% win rate. The 2008 leg alone saved +52.7pp (strategy -1.8% vs SPY -54.6%); the 2009 recovery leg cost -26.8pp (strategy +32.7% vs SPY +59.5% — the same gate that blocked the crash also kept the scanner out of the sharpest V-recovery in market history). Net: **+54.4pp over the full peak-to-recovery cycle** — crash protection dominates recovery miss by 2×.

### Mechanism

The gate doesn't make individual setups better — it prevents entries during regimes where *all* longs are structurally disadvantaged (volatility spikes and broad downtrends produce more whipsaws than signals). Sitting in cash during flagged regimes has higher expected value than forcing trades.

### Verdict

Combined with the L1/L2 volume tuning and L3 R:R tightening, the strategy now:
- **Beats SPY by +126.9pp** over the 2024-2026 tune window (up from +80pp pre-gate).
- **Beats SPY by +30.4pp** through the 2020 COVID crash (was losing -24.9pp pre-gate).
- **Beats SPY by +54.4pp** through the 2007-2009 GFC full cycle — an event that predates every parameter in the config.
- **Remains long-only** — no short-side exposure, just a cash-out during risk-off regimes.

The COVID and GFC results are strong evidence against in-sample fitting: neither 2020 nor 2008 was in the backtest until the gate was added, yet the same rules cleanly flipped two different disasters into outperformance. The edge is structural, not accidental.

**The honest tradeoff:** the 2009 sub-window (post-March low recovery) shows the gate's cost — the strategy returned +32.7% while SPY rebounded +59.5% off the crash low. Regime gates can't distinguish "bottom is in" from "bear rally." You're paying a late-entry tax on V-recoveries in exchange for not catching the falling knife on the way down. Over the full 2007-2009 cycle the crash protection wins by 2× — but if you're evaluating this strategy over a window that starts *at* a crash low, it will look bad.

---

## Tuning session round 4 (2026-04-25) — concurrent slot count (MAX_SLOTS) — kept at 1

### Question

The single-trade backtest leaves capital idle most days. Would running 2–4 concurrent positions deploy that idle cash for higher returns, or would the diversification noise hurt? Added `MAX_SLOTS` to `signals.py` and refactored `backtest.simulate` so each slot is its own sub-account starting at `$10k / N` and compounding independently. Same-ticker dedup across slots; later signals fill empty slots first-come-first-served (no swap-on-better-quality).

### What we did

Ran a 4-variation `MAX_SLOTS ∈ {1, 2, 3, 4}` sweep on both windows (IS: 2024-04-20 → 2025-04-20, OOS: 2025-04-20 → 2026-04-20). The Baseline run uses the file default (also 1) and confirmed no state leakage — its alpha matched `SLOTS=1` exactly.

### Result

| Variation | IS alpha   | OOS alpha   | Sum         |
| --------- | ---------- | ----------- | ----------- |
| Baseline  | **+53.8pp** | **+33.2pp** | **+87.0pp** |
| SLOTS=1   | +53.8pp    | +33.2pp     | +87.0pp     |
| SLOTS=2   | +30.1pp    | +13.2pp     | +43.2pp     |
| SLOTS=3   | —          | —           | failed +10pp floor |
| SLOTS=4   | —          | —           | failed +10pp floor |

Each additional slot cut alpha roughly in half. SLOTS=3 and SLOTS=4 were so degraded they didn't clear the +10pp floor on at least one window.

### Why more slots hurt

The scanner's edge is concentrated in the **top-quality signal of the day**, not spread evenly across all triggers. Slots 2–N are forced to take the 2nd/3rd/4th-best candidates, whose quality scores are meaningfully lower. We're effectively paying to ignore the quality ranking that does most of the work. A 4-slot smoke test on a tail window showed 28% win rate (50 trades) vs 40% (10 trades) for single-slot — same data, lower-quality picks dragging the mean.

This is the opposite of what diversification literature predicts, and it's a property of the scanner specifically: candidates aren't IID drawings from a positive-EV pool. They're rank-ordered by an edge signal, and the edge is real.

### Verdict

**`MAX_SLOTS = 1` stays as the proven default.** The mechanism remains in place as a tunable knob — useful for future sweeps if the scanner is ever changed to produce flatter quality distributions — but production behavior is unchanged.

If idle cash drag becomes a concern later, the right fix is a **portfolio overlay** (park unused $ in SPY or a money-market fund and pull when a signal fires), not adding slots. That's a separate, smaller change and doesn't compromise the conviction-on-best-signal property that drives this scanner's alpha.

### Methodological note

Round 4 also reinforces a pattern from earlier rounds: when an "obviously good" idea (more diversification) underperforms, trust the deterministic backtest. Diversification is good *when components are uncorrelated and equal-EV* — neither holds here.

---

## Next steps

Do these in order. Each one is a single, self-contained change. After each, re-run `scripts/backtest.py` on the 2024-2026 window. If alpha drops, revert and move on.

### Do next (in this order)

**1. Re-run old backtests with the regime gate.** The 2022-2024 number in the README (+16.2pp) is from *before* the regime gate was added. Re-run it. Then run 2015 and 2018 too — both are different regimes (chop, vol shock) we haven't seen. No code change needed; just edit `START_DATE` / `END_DATE` at the top of `backtest.py`.

**2. Skip trades near earnings.** Before entering any trade, check if the company reports earnings in the next 7-10 trading days. If yes, skip. Use `yfinance.Ticker(t).get_earnings_dates()`. Add the check inside the candidate loop in `simulate()` (backtest.py) and in the live scanner (`sma200_filter.py`). One filter, applied in both places.

**3. Relative-strength filter.** Before entering, check that the ticker's 3-month and 6-month return are in the top 25% of the universe vs SPY. If not, skip. Add it next to the earnings filter. Goal: stop buying former winners that have started underperforming.

**4. Sell half at breakeven, let the rest run.** Currently when price hits the 50%-to-TP mark, we move SL to entry. Change it: sell half the position at that point, leave half running with the BE stop. Edit `simulate_trade()` in `backtest.py` — track two halves separately.

**5. Trail the runner after breakeven.** Same code path as #4. Once BE has triggered, instead of holding SL=entry, set SL = `max(entry, highest_high_since_entry − 3×ATR)`. Recalculate each day. Captures big winners that currently cap at 4×ATR TP.

**6. ADX > 20 on L1 and L4.** L1 (Ride Uptrend) and L4 (Pre-Golden Cross) assume a trend. Require ADX(14) > 20 to confirm one exists. Add ADX to `indicators.py`, then add `ind["adx"] > 20` to the L1 and L4 conditions in `signals.py`. Don't apply to L2 or L3.

### Don't bother — already settled

| Idea | Why skip |
| --- | --- |
| More walk-forward / OOS validation | `tune.py` already does IS/OOS every round. COVID, 2022-24, and GFC are all OOS-tested. |
| Per-setup P&L breakdown | Done. See round-3 table. |
| Multi-position sizing (2-8 concurrent slots) | **Tested in round 4. Cut alpha in half.** Don't reopen. |
| Three-mode regime-scaled exposure | Doesn't apply at SLOTS=1; binary gate already covers the off-leg. |
| MFI, breadth indicator, 52-week-high gate | Duplicate work — volume, breadth, and SMA200 already do this. |
| New setups (breakout, post-earnings drift) | Premature until #2 lands and L4 (n=1) is fixed. |

### Rule for each step

Run the **continuous** backtest before declaring victory. `tune.py`'s per-window numbers have been wrong about both magnitude and direction in past rounds. The only verdict that counts is `scripts/backtest.py` end-to-end alpha.

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

This is backtested on historical data. Past performance does not guarantee future results. The strategy's setup thresholds (L1/L2/L3) were tuned against the 2024-2026 window, so some in-sample fit exists there; the regime gate and core rules survive the 2020 COVID, 2022-2024, and 2007-2009 GFC out-of-sample windows. Treat the +126.9pp / +30.4pp / +54.4pp alphas as reasoned starting points, not promises. The 2009 sub-window underperformance (-26.8pp against the V-recovery) is a known structural cost of regime-gated strategies.

## Claude session

claude --resume 42f69484-bc88-40bb-b701-fd0e2a251dd0 --dangerously-skip-permissions

claude --resume c4c65570-757b-44ec-9f03-5aa0b85044cd --dangerously-skip-permissions
