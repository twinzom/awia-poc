# AWIA MVP proof of concept

Signal detectors from `wealth-agent-mvp-design.html`, starting with **A1 Price move**, built on free
end-of-day data from Yahoo Finance for the 200 stocks in `All_200_Stocks_POC.csv` (100 US, 50 HK, 50 UK).

> Yahoo data is unofficial and unlicensed: use it for internal testing only. Clients must not receive
> anything derived from it. Swap in a licensed feed before a client pilot (only `awia/data/yahoo.py` changes).

## Run

```sh
uv sync
uv run awia-a1 detect                 # fetch 2y of daily bars, evaluate each market's latest completed session
uv run awia-a1 detect --market HK     # one market
uv run awia-a1 detect --no-fetch --date 2026-11-05   # re-evaluate a past session from the cache
uv run awia-a1 replay --sessions 250  # how many signals the detector fires over the past year
uv run pytest
```

Outputs go to `data/`: raw Yahoo JSON (`raw/`), one JSONL file per market and session (`signals/`),
and `a1.sqlite` (tables `signals` and `quality_issues`). Re-running a session never stores a signal twice,
so `detect` can run on a schedule, e.g. hourly. Today's bar is ignored until 20 minutes after the closing auction:
HK 16:10, UK 16:35, US 16:00 local time.

## What it implements from the A1 spec

| Spec | Proof of concept |
|---|---|
| `price_move.close` | ✅ Close-to-close total return, bands 2.5σ / 4σ / 6σ → severity 7 / 4 / 2 |
| `price_move.gap` | ✅ Open vs previous close, severity 5 / 2. Only known after the close |
| `price_move.intraday`, `trading.halt`, `trading.resume` | ❌ Need an intraday feed |
| Normalise | Yahoo `adjclose`, which adjusts for splits and dividends. Every fetch re-downloads the full history, so adjusted values are never mixed across downloads |
| Baseline | EWMA variance λ = 0.94 (seeded with 20 returns, signals after 60). Beta from a 250-session rolling regression on the market's ETF: SPY, 2800.HK, ISF.L. Both use data up to the previous session only, so a replay has no look-ahead |
| Detect | `z = r ÷ (σ × √elapsed)`. `elapsed` counts sessions since the stock's last bar, so a resumption after a suspension is judged fairly. `residual_z = (r − β·r_bench) ÷ (σ_resid × √elapsed)`, where `σ_resid` is the EWMA volatility of residuals |
| Debounce | One observation per day, so one signal per instrument, window and session, at the highest band reached |
| Subscription filter | `awia/subscriptions.py`: `min_z` (floor 2.5), `min_move_pct`, `direction`, `residual_only` |

Data guards (the session is skipped and the problem logged to `quality_issues`): no bar while the benchmark
has one (`missing_bar`), zero volume (`no_trades`), close outside the high–low range, or |return| > 40% (`suspect_return`,
for example a pence/pounds mix-up on `.L` symbols). A suspect return also stays out of the volatility baseline.
An open outside the high–low range suppresses only the gap signal.

## Replay, 250 sessions to 2026-11-13

| Market | Close signals/session | Gap signals/session | Close signals passing `residual_only` |
|---|---|---|---|
| US (100) | 3.0 (median 2, max 29) | 0.8 | 80% |
| HK (50) | 1.5 | 0.15 | 65% |
| UK (50) | 1.4 | 0.36 | 72% |

About 3% of stock-days cross 2.5σ, against 1.2% if returns were normally distributed. Busy days cluster around
market-wide moves, which is why A6 and `residual_only` matter. Many 6σ moves fall on earnings days (overlap with B1).
