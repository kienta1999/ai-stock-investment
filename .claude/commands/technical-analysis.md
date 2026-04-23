# Technical Analysis Scanner

Run the long-only technical analysis scanner across the top 100 S&P 500 stocks by market cap. Backtested over 2 years (2024-04-20 → 2026-04-20): **+95.7% vs SPY +45.6% (alpha +50.1pp, 16W/17L, 48% win rate)**.

## Step 1 — Run scanner

```bash
cd /home/talekien1710/personal_project/ai-stock-investment && .venv/bin/python3 scripts/sma200_filter.py
```

Force-refresh universe (if it's been >7 days or you want latest market cap ranking):

```bash
cd /home/talekien1710/personal_project/ai-stock-investment && .venv/bin/python3 scripts/sma200_filter.py --refresh
```

First-time setup (if .venv missing):

```bash
cd /home/talekien1710/personal_project/ai-stock-investment && uv venv .venv -q && uv pip install yfinance pandas lxml requests --python .venv/bin/python3 -q
```

## Step 2 — Interpret output

The scanner outputs three sections. Interpret each:

### Market breadth
- **>70%**: Strong bull — favour longs broadly, full size
- **50–70%**: Mixed market — be selective, reduce size 10–20%
- **30–50%**: Weakening — high caution, smaller positions, tighter stops
- **<30%**: Bear regime — majority below SMA200, avoid new longs, consider hedges or cash

### Triggered setups table

Each setup includes: Ticker | Setup type | **Quality** | Direction | Entry | SL | TP | R:R | RSI | Vol ratio | ATR% | Timeframe. Rows are sorted by Quality descending.

**Two-stage funnel:**
1. **L1–L4 setups (binary triggers)** — each setup is independent. A stock appears in the table if it triggers ANY one of L1, L2, L3, or L4. Within a single setup, ALL of *that setup's* conditions must pass. One stock can trigger multiple setups → it appears as multiple rows.
2. **Quality score (0–100)** — rates how clean the underlying conditions are. Surfaced so you can prefer high-quality triggers over weak ones.

**Strategy rule — long-only.** When price < SMA200: **no trade.** Never short, never fight the dominant trend. When price > SMA200: scan for one of the four long setups below.

**Setup types and their logic:**

| Setup | Key conditions | SL | TP | Timeframe |
|---|---|---|---|---|
| Ride Uptrend | SMA50>SMA200, price ≥3/5d above BB mid, RSI>50, pullback to SMA50, green vol>MA | 2× ATR | 4× ATR | 1D → 3–6 wk options |
| MACD Cross | MACD hist just crossed zero↑, RSI 50–70, price above SMA50 not touching upper BB, green vol>MA | 2.5× ATR | 5× ATR | 1D → 3–6 wk options |
| VWAP Support | SMA50>SMA200, price 0–1.5% above VWAP (dipping toward it), RSI>45, green vol>MA | 2× ATR | 3× ATR | 1H/4H → 2–3 wk options |
| Pre-Golden Cross | SMA50 approaching SMA200 (<2% gap), price already>SMA200, RSI>45, strong vol>1.2× | 2× ATR | 4× ATR | 1D → 3–6 wk options |

**Volume + ATR rules (always apply):**
- Green candle AND volume > vol MA20 — else skip, wait for confirmation
- ATR% ≤ 4.0 (scanner enforces) — extreme-vol names are filtered out as a guardrail against blowup losses

### Quality score (0–100) — how to read it

Defined in `scripts/signals.py` (`quality()` function). Five factors, weighted by how predictive they are:

| Factor | Max pts | Full credit when… | Half credit when… | Why this weight |
|---|---|---|---|---|
| **Volume conviction** | 35 | vol_ratio ≥ 2.0× MA | linear scaling: 1.5× = 17 pts | Real-money buyers showing up = strongest single signal |
| **RSI sweet spot** | 30 | RSI 55–65 | RSI 50–70 | "Trending but not overbought" zone |
| **ATR Goldilocks** | 20 | ATR% 1.5–3.5 | ATR% 1.0–5.0 | Enough movement to reach TP, not crazy-volatile |
| **Trend alignment** | 10 | LONG + SMA50>SMA200 | — | Tiebreaker bonus for trading with dominant trend |
| **MACD alignment** | 5 | LONG + macd_hist > 0 | — | Smallest weight — confirmatory only |

**Score interpretation:**
- **80–100**: Premium setup — full conviction. The backtest's most profitable trades cluster here.
- **60–79**: Good setup — workable, but watch one of the factors is weak (usually volume or RSI extension).
- **40–59**: Marginal — typically barely past one or two thresholds. Skip unless multiple setups confirm the same ticker.
- **<40**: Weak — backtest treats below-25 as "no trade." Skip.

**The backtest picks `max(quality)` automatically.** You should not — bring your own judgment using BOTH the trigger type AND the score:
- Prefer a 90-quality Pre-Golden Cross over a 50-quality Ride Uptrend, even though L1 is the "premier" setup
- Two setups firing on the same ticker (e.g. Ride Uptrend + MACD Cross) is a strong confluence signal even if individual scores are mid-tier
- A high-quality score + setup type that historically suits current market regime > raw highest score
- If quality is high but you spot context the scorer can't (earnings tomorrow, sector rotation, macro headline) — override and skip

### Trade management rules

These are part of the strategy, not the scanner output — apply them when you're in a position:

1. **Initial stop**: at the SL shown in the table (2× or 2.5× ATR).
2. **Trailing-to-breakeven**: once price travels **50% of the way from entry to TP**, raise SL to entry. One-way ratchet — never lower it. This is the single biggest alpha source: turns would-be losers into flat scratches.
3. **Time stop**: exit after 40 trading days if neither TP nor SL has been hit.
4. **One trade at a time** in the baseline playbook. Only scan for a new entry after the current trade exits.

### Setup detail section
For each triggered setup, show the full indicator dashboard and cross-check against the playbook:

1. **Trend direction**: Is SMA50 > SMA200? (golden cross = confirmed uptrend)
2. **BB position**: Price vs upper/mid/lower BB — not extended above upper BB?
3. **RSI**: In the healthy range for the setup (50–70 for longs)?
4. **MACD**: Histogram direction and zero-line position?
5. **Volume**: Green bar AND above vol MA20?
6. **VWAP**: Price relative to rolling VWAP — consistent side?

## Step 3 — Present findings

Summarise clearly:
1. Market regime (breadth %)
2. Best 3–5 setups ranked using **both** the Quality score AND the trigger type — explain *why* you ranked them this way (e.g. "BA scores 90 with 2.4× volume vs AAPL 66 with barely 1.0× — BA wins on volume conviction even though L1 is normally a stronger setup type")
3. For each top setup: exact entry, SL, TP, breakeven-trigger price (entry + (TP−entry)×0.5), which option expiry to target, and what to watch for invalidation
4. Any setups to skip despite triggering — call out the specific failing factor:
   - Quality < 40 → "weak score, skip"
   - Volume just barely above 1.0× → "no real conviction"
   - RSI > 70 → "extended, mean reversion risk"
   - VWAP distance > 3% → "not a real pullback, just a continuation entry"
   - MACD hist negative → "momentum already fading"

## Playbook reference

### Indicators computed per ticker
- SMA 50, SMA 200 (trend direction)
- Bollinger Bands (20, 2σ): upper / mid / lower
- RSI (14, EWM)
- ATR (14, EWM) — used for SL/TP sizing
- MACD (12, 26, 9): line, signal, histogram
- Volume + 20-day volume MA + vol ratio
- Rolling VWAP (20-day proxy — for anchored VWAP, anchor to a major turning point manually)

### Bear market note
When market breadth drops below 30% (majority of top 100 below SMA200):
- Do NOT "all in on cross" — crosses fail frequently in bear regimes
- Prefer cash over forced entries. The long-only rule already prevents shorting; breadth <30% should also shrink long exposure.
- When re-engaging: wait for golden cross (SMA50 > SMA200) CONFIRMED by price pulling back to SMA50 (now support) + RSI recovering above 50 + green volume > MA
- Scale in: 1/3 position on golden cross signal, add 1/3 on first confirmed pullback to SMA50, final 1/3 when RSI sustains above 50
- Choppy VWAP (price crossing VWAP repeatedly) = no trade

### Why long-only
The short side was originally included but **lost -25.6% over 2025-04-20 → 2026-04-20 vs SPY +39.8%** (alpha -65.4pp) — most shorts were clipped by the dominant bull tape before any move developed. Removing the short side and widening ATR stops flipped the strategy from losing to the +50pp/2y alpha it delivers today.
