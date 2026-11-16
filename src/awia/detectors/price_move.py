"""A1 Price move detector, end-of-day edition.

Implements the design's A1 on daily bars only:
  price_move.close  close-to-close move crosses a band (2.5σ / 4σ / 6σ)
  price_move.gap    open vs previous close crosses a band (known only after the close)
Not implemented without an intraday feed: price_move.intraday, trading.halt, trading.resume.

Pipeline per instrument (all baselines use data up to the previous session only, so a move never
dampens its own z-score and a replay has no look-ahead):
  Normalise  total-return series from Yahoo's adjclose (splits and dividends), quality guards
  Baseline   EWMA variance (λ = 0.94) of per-session returns; rolling beta to the market's benchmark ETF
  Detect     z = r ÷ (σ × √elapsed); residual z = (r − β·r_bench) ÷ (σ_resid × √elapsed)
  Debounce   one signal per instrument, window and session, at the highest band reached
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from awia.data.yahoo import Bars
from awia.universe import MARKETS, Instrument, Market

DETECTOR_ID = "price_move"
DETECTOR_VERSION = "0.1.0-eod"
TRUST = "unlicensed_free_feed"

# severity_hint by band (2.5σ, 4σ, 6σ), from the design's signal-kind table
SEVERITY = {
    "price_move.close": (7, 4, 2),
    "price_move.gap": (5, 2, 2),
}
DEFAULT_KINDS = ("price_move.close", "price_move.gap")

# Row-level data issues that stop a session from being evaluated.
BLOCKING_ISSUES = ("no_trades", "close_outside_range", "suspect_return")


@dataclass(frozen=True)
class Config:
    lam: float = 0.94  # EWMA decay (RiskMetrics daily); weight half-life ≈ 11 sessions
    vol_seed: int = 20  # returns used to seed the EWMA variance
    warmup: int = 60  # returns needed before signals are emitted
    beta_window: int = 250  # sessions in the rolling beta regression
    beta_min: int = 120  # minimum paired returns before a beta exists
    bands: tuple[float, ...] = (2.5, 4.0, 6.0)
    vol_floor: float = 0.003  # 0.3% daily: stops stale prices producing huge z
    max_abs_return: float = 0.40  # bigger moves are held back as suspect data
    volume_lookback: int = 20


def _ewma_var(x: np.ndarray, lam: float, seed: int) -> np.ndarray:
    """Zero-mean EWMA variance after each observation; NaN observations carry the estimate forward."""
    out = np.full(len(x), np.nan)
    v = np.nan
    buf: list[float] = []
    for i, xi in enumerate(x):
        if not np.isnan(xi):
            if np.isnan(v):
                buf.append(xi)
                if len(buf) >= seed:
                    v = float(np.mean(np.square(buf)))
            else:
                v = lam * v + (1.0 - lam) * xi * xi
        out[i] = v
    return out


def _rolling_beta(x: np.ndarray, xb: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    valid = ~np.isnan(x) & ~np.isnan(xb)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        m = valid[lo : i + 1]
        if m.sum() < min_obs:
            continue
        a, b = x[lo : i + 1][m], xb[lo : i + 1][m]
        vb = b.var()
        if vb > 0:
            out[i] = float(np.mean((a - a.mean()) * (b - b.mean())) / vb)
    return out


def compute_features(stock: pd.DataFrame, bench: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """One row per stock bar with returns, baselines, z-scores and a data-quality issue (or None)."""
    s = stock.sort_index()
    idx = s.index

    # Sessions elapsed since the stock's previous bar, on the market calendar (benchmark ∪ stock dates),
    # so a stock resuming after a suspension is judged against σ × √(sessions missed).
    calendar = bench.index.union(idx)
    pos = pd.Series(calendar.get_indexer(idx), index=idx, dtype=float)
    elapsed = pos.diff()
    sqrt_el = np.sqrt(elapsed)

    prev_adj = s["adjclose"].shift(1)
    r = s["adjclose"] / prev_adj - 1.0
    price_chg = s["close"] / s["close"].shift(1) - 1.0
    adj_open = s["open"] * s["adjclose"] / s["close"]
    gap = adj_open / prev_adj - 1.0

    b_adj = bench["adjclose"].reindex(idx)
    b_adj_open = (bench["open"] * bench["adjclose"] / bench["close"]).reindex(idx)
    rb = b_adj / b_adj.shift(1) - 1.0
    rb_gap = b_adj_open / b_adj.shift(1) - 1.0

    # --- quality guards -------------------------------------------------------------------------
    tol = 1e-6
    no_trades = s["volume"].fillna(0) <= 0
    close_bad = (s["close"] > s["high"] * (1 + tol)) | (s["close"] < s["low"] * (1 - tol))
    suspect = r.abs() > cfg.max_abs_return
    open_ok = (
        (s["open"] > 0)
        & (s["open"] <= s["high"] * (1 + tol))
        & (s["open"] >= s["low"] * (1 - tol))
    )
    gap = gap.where(open_ok)

    issue = pd.Series(None, index=idx, dtype=object)
    issue[suspect.fillna(False)] = "suspect_return"
    issue[close_bad.fillna(False)] = "close_outside_range"
    issue[no_trades] = "no_trades"

    usable = r.notna() & elapsed.notna() & issue.isna()

    # --- baseline (each value includes the session itself; shifted below) ------------------------
    x = (r / sqrt_el).where(usable)
    var = pd.Series(_ewma_var(x.to_numpy(), cfg.lam, cfg.vol_seed), index=idx)
    xb = (rb / sqrt_el).where(usable)
    beta = pd.Series(_rolling_beta(x.to_numpy(), xb.to_numpy(), cfg.beta_window, cfg.beta_min), index=idx)

    sigma_prev = np.sqrt(var).shift(1)
    beta_prev = beta.shift(1)
    n_obs = usable.astype(int).cumsum().shift(1).fillna(0).astype(int)

    resid = r - beta_prev * rb
    xe = (resid / sqrt_el).where(usable)
    var_e = pd.Series(_ewma_var(xe.to_numpy(), cfg.lam, cfg.vol_seed), index=idx)
    sigma_e_prev = np.sqrt(var_e).shift(1)

    sig = sigma_prev.clip(lower=cfg.vol_floor) * sqrt_el
    sig_e = sigma_e_prev.clip(lower=cfg.vol_floor) * sqrt_el

    traded_vol = s["volume"].where(~no_trades)
    vol_median = traded_vol.rolling(cfg.volume_lookback, min_periods=cfg.volume_lookback // 2).median().shift(1)

    return pd.DataFrame(
        {
            "close": s["close"],
            "r": r,
            "price_chg": price_chg,
            "gap": gap,
            "r_bench": rb,
            "elapsed": elapsed,
            "sigma": sigma_prev,
            "beta": beta_prev,
            "sigma_resid": sigma_e_prev,
            "z": r / sig,
            "residual_z": resid / sig_e,
            "gap_z": gap / sig,
            "gap_residual_z": (gap - beta_prev * rb_gap) / sig_e,
            "volume_ratio": s["volume"] / vol_median,
            "n_obs": n_obs,
            "split_ratio": s["split_ratio"],
            "dividend": s["dividend"],
            "issue": issue,
        },
        index=idx,
    )


def band_of(z: float, bands: tuple[float, ...]) -> int:
    """0 = below the arm level, 1 = ≥ bands[0], 2 = ≥ bands[1], ..."""
    if z is None or np.isnan(z):
        return 0
    return sum(abs(z) >= b for b in bands)


def _num(v, nd: int = 2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


def _session_close_utc(market: Market, date: pd.Timestamp) -> datetime:
    local = datetime.combine(date.date(), market.close, ZoneInfo(market.tz))
    return local.astimezone(timezone.utc)


def build_signal(
    inst: Instrument, market: Market, date: pd.Timestamp, kind: str, band: int, band_sigma: float, magnitude: dict
) -> dict:
    window = kind.split(".", 1)[1]
    move = magnitude["gap_pct" if window == "gap" else "move_pct"]
    direction = "up" if move >= 0 else "down"
    day = date.date().isoformat()
    observed = _session_close_utc(market, date)
    return {
        "signal_id": f"{DETECTOR_ID}:{inst.ticker}:{day}:{window}:{direction}:b{band}",
        "detector": DETECTOR_ID,
        "detector_version": DETECTOR_VERSION,
        "kind": kind,
        "scope": "shared",
        "topics": [f"ticker:{inst.ticker}", inst.sector_topic],
        "magnitude": magnitude,
        "severity_hint": SEVERITY[kind][band - 1],
        "story_key": f"{inst.ticker}:price_move:{day}",
        "observed_at": observed.isoformat().replace("+00:00", "Z"),
        "valid_until": (observed + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        "deadline": None,
        "market_session": "regular",
        "trust": TRUST,
        "must_acknowledge": False,
        "evidence": [{"type": "daily_bar", "ref": f"yahoo:{inst.ticker}@{day}"}],
        # convenience fields (not part of the shared envelope)
        "session_date": day,
        "market": market.code,
        "instrument_name": inst.name,
        "band_sigma": band_sigma,
    }


def _pct(v) -> float | None:
    return None if v is None or np.isnan(v) else _num(v * 100)


def signals_for_row(inst: Instrument, market: Market, date: pd.Timestamp, row: pd.Series, cfg: Config) -> list[dict]:
    out: list[dict] = []
    pct = _pct

    band = band_of(row["z"], cfg.bands)
    if band:
        out.append(
            build_signal(
                inst, market, date, "price_move.close", band, cfg.bands[band - 1],
                {
                    "move_pct": pct(row["r"]),
                    "z": _num(row["z"]),
                    "residual_z": _num(row["residual_z"]),
                    "price_change_pct": pct(row["price_chg"]),
                    "gap_pct": pct(row["gap"]),
                    "benchmark_move_pct": pct(row["r_bench"]),
                    "beta": _num(row["beta"]),
                    "sigma_daily_pct": pct(row["sigma"]),
                    "volume_ratio": _num(row["volume_ratio"]),
                    "sessions_elapsed": int(row["elapsed"]),
                },
            )
        )
    band = band_of(row["gap_z"], cfg.bands)
    if band:
        out.append(
            build_signal(
                inst, market, date, "price_move.gap", band, cfg.bands[band - 1],
                {
                    "gap_pct": pct(row["gap"]),
                    "z": _num(row["gap_z"]),
                    "residual_z": _num(row["gap_residual_z"]),
                    "sigma_daily_pct": pct(row["sigma"]),
                    "sessions_elapsed": int(row["elapsed"]),
                },
            )
        )
    return out


@dataclass
class SessionResult:
    market: str
    session_date: pd.Timestamp
    signals: list[dict]
    issues: list[dict]  # {ticker, issue, detail}
    evaluated: int


class PriceMoveDetector:
    def __init__(self, universe: list[Instrument], bars: dict[str, Bars | None], cfg: Config = Config()):
        self.universe = universe
        self.bars = bars
        self.cfg = cfg
        self._features: dict[str, pd.DataFrame] = {}

    def benchmark_frame(self, market_code: str) -> pd.DataFrame | None:
        b = self.bars.get(MARKETS[market_code].benchmark)
        return b.frame if b is not None and len(b.frame) else None

    def sessions(self, market_code: str) -> pd.DatetimeIndex:
        bench = self.benchmark_frame(market_code)
        return bench.index if bench is not None else pd.DatetimeIndex([])

    def latest_session(self, market_code: str) -> pd.Timestamp | None:
        s = self.sessions(market_code)
        return s[-1] if len(s) else None

    def features(self, inst: Instrument) -> pd.DataFrame | None:
        if inst.ticker not in self._features:
            b = self.bars.get(inst.ticker)
            bench = self.benchmark_frame(inst.market)
            if b is None or bench is None or len(b.frame) < 2:
                return None
            self._features[inst.ticker] = compute_features(b.frame, bench, self.cfg)
        return self._features[inst.ticker]

    def run_session(self, market_code: str, date: pd.Timestamp) -> SessionResult:
        market = MARKETS[market_code]
        signals: list[dict] = []
        issues: list[dict] = []
        evaluated = 0
        for inst in (i for i in self.universe if i.market == market_code):
            f = self.features(inst)
            if f is None:
                issues.append({"ticker": inst.ticker, "issue": "no_data", "detail": "no usable bars"})
                continue
            if date not in f.index:
                last = f.index[-1].date().isoformat() if len(f) else "never"
                issues.append({"ticker": inst.ticker, "issue": "missing_bar", "detail": f"last bar {last}"})
                continue
            row = f.loc[date]
            if row["issue"] in BLOCKING_ISSUES:
                detail = f"move {row['r'] * 100:+.1f}%" if row["issue"] == "suspect_return" else ""
                if row["split_ratio"] != 1:
                    detail += f" (split {row['split_ratio']:g} that day)"
                issues.append({"ticker": inst.ticker, "issue": row["issue"], "detail": detail.strip()})
                continue
            if row["n_obs"] < self.cfg.warmup or np.isnan(row["z"]):
                issues.append({"ticker": inst.ticker, "issue": "warmup", "detail": f"{row['n_obs']} returns"})
                continue
            evaluated += 1
            signals.extend(signals_for_row(inst, market, date, row, self.cfg))
        return SessionResult(market_code, date, signals, issues, evaluated)
