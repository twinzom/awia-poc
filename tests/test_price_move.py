from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from awia.data.yahoo import Bars, parse_chart
from awia.detectors.price_move import Config, PriceMoveDetector, band_of, compute_features
from awia.subscriptions import matches, params_pass
from awia.universe import MARKETS, Instrument

N = 300
DATES = pd.bdate_range("2025-01-01", periods=N)


def make_frame(returns: np.ndarray, dates=DATES, start=100.0, volume=1e6) -> pd.DataFrame:
    close = start * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "adjclose": close,
            "volume": np.full(len(close), volume),
            "split_ratio": 1.0,
            "dividend": 0.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def market_and_stock():
    rng = np.random.default_rng(7)
    rb = rng.normal(0, 0.01, N)
    rb[0] = 0
    rs = 1.2 * rb + rng.normal(0, 0.01, N)
    rs[0] = 0
    return rb, rs


def detector_for(stock: pd.DataFrame, bench: pd.DataFrame) -> tuple[PriceMoveDetector, Instrument]:
    inst = Instrument("TEST", "Test Co", "US", "USD", "Information Technology")
    bars = {"TEST": Bars("TEST", "USD", stock), "SPY": Bars("SPY", "USD", bench)}
    return PriceMoveDetector([inst], bars, Config()), inst


def test_band_of():
    assert band_of(2.4, (2.5, 4, 6)) == 0
    assert band_of(-2.5, (2.5, 4, 6)) == 1
    assert band_of(5.0, (2.5, 4, 6)) == 2
    assert band_of(-7.0, (2.5, 4, 6)) == 3
    assert band_of(float("nan"), (2.5, 4, 6)) == 0


def test_idiosyncratic_drop_fires_close_signal(market_and_stock):
    rb, rs = market_and_stock
    rs[-1] = -0.12  # stock-specific crash, market flat
    rb[-1] = 0.0
    det, _ = detector_for(make_frame(rs), make_frame(rb))
    res = det.run_session("US", DATES[-1])
    close = [s for s in res.signals if s["kind"] == "price_move.close"]
    assert len(close) == 1
    s = close[0]
    assert s["magnitude"]["z"] < -6 and s["magnitude"]["residual_z"] < -6
    assert s["band_sigma"] == 6.0 and s["severity_hint"] == 2
    assert s["signal_id"] == f"price_move:TEST:{DATES[-1].date()}:close:down:b3"
    assert s["topics"] == ["ticker:TEST", "sector:information_technology"]
    assert abs(s["magnitude"]["beta"] - 1.2) < 0.25


def test_baseline_excludes_the_session_itself(market_and_stock):
    rb, rs = market_and_stock
    calm, crash = rs.copy(), rs.copy()
    calm[-1], crash[-1] = 0.0, -0.12
    f_calm = compute_features(make_frame(calm), make_frame(rb))
    f_crash = compute_features(make_frame(crash), make_frame(rb))
    for col in ("sigma", "beta", "sigma_resid"):
        assert f_crash[col].iloc[-1] == pytest.approx(f_calm[col].iloc[-1])


def test_market_wide_move_has_low_residual(market_and_stock):
    rb, rs = market_and_stock
    rb[-1] = -0.05
    rs[-1] = 1.2 * -0.05
    det, _ = detector_for(make_frame(rs), make_frame(rb))
    sig = [s for s in det.run_session("US", DATES[-1]).signals if s["kind"] == "price_move.close"][0]
    assert abs(sig["magnitude"]["z"]) >= 2.5
    assert abs(sig["magnitude"]["residual_z"]) < 2.5
    assert params_pass({}, sig)
    assert not params_pass({"residual_only": True}, sig)


def test_quiet_day_no_signal(market_and_stock):
    rb, rs = market_and_stock
    rs[-1] = 0.001
    rb[-1] = 0.0
    det, _ = detector_for(make_frame(rs), make_frame(rb))
    res = det.run_session("US", DATES[-1])
    assert res.signals == [] and res.evaluated == 1


def test_warmup_suppresses_signals():
    rng = np.random.default_rng(1)
    rb = rng.normal(0, 0.01, 40)
    rs = rng.normal(0, 0.01, 40)
    rs[-1] = -0.2
    det, _ = detector_for(make_frame(rs, DATES[:40]), make_frame(rb, DATES[:40]))
    res = det.run_session("US", DATES[39])
    assert res.signals == [] and res.issues[0]["issue"] == "warmup"


def test_suspect_return_is_held_back_and_kept_out_of_baseline(market_and_stock):
    rb, rs = market_and_stock
    rs[-2] = 99.0  # 100x print (e.g. pence/pounds mix-up)
    rs[-1] = 1 / 100 - 1
    det, _ = detector_for(make_frame(rs), make_frame(rb))
    for d in DATES[-2:]:
        res = det.run_session("US", d)
        assert res.signals == []
        assert res.issues[0]["issue"] == "suspect_return"
    f = det.features(det.universe[0])
    assert f["sigma"].iloc[-1] < 0.05


def test_no_trades_and_missing_bar(market_and_stock):
    rb, rs = market_and_stock
    stock = make_frame(rs)
    stock.loc[DATES[-1], "volume"] = 0
    det, _ = detector_for(stock, make_frame(rb))
    assert det.run_session("US", DATES[-1]).issues[0]["issue"] == "no_trades"

    det, _ = detector_for(make_frame(rs).iloc[:-1], make_frame(rb))
    assert det.run_session("US", DATES[-1]).issues[0]["issue"] == "missing_bar"


def test_suspension_scales_by_sessions_elapsed(market_and_stock):
    rb, rs = market_and_stock
    stock = make_frame(rs)
    # suspended for 3 sessions; resumes 4 sessions after its last bar, down 5% in total
    stock = stock.drop(DATES[-4:-1])
    stock.loc[DATES[-1], ["open", "high", "low", "close", "adjclose"]] = stock["close"].iloc[-2] * 0.95
    f = compute_features(stock, make_frame(rb))
    row = f.loc[DATES[-1]]
    assert row["elapsed"] == 4
    assert row["z"] == pytest.approx(row["r"] / (row["sigma"] * 2))


def test_gap_signal(market_and_stock):
    rb, rs = market_and_stock
    rs[-1] = -0.01
    rb[-1] = 0.0
    stock = make_frame(rs)
    prev = stock["close"].iloc[-2]
    stock.loc[DATES[-1], ["open", "low"]] = prev * 0.92  # opened -8%, recovered to -1%
    det, _ = detector_for(stock, make_frame(rb))
    gaps = [s for s in det.run_session("US", DATES[-1]).signals if s["kind"] == "price_move.gap"]
    assert len(gaps) == 1
    assert gaps[0]["magnitude"]["gap_pct"] == pytest.approx(-8.0, abs=0.01)
    assert gaps[0]["signal_id"].endswith(":gap:down:b3")


def test_dividend_day_uses_adjusted_close(market_and_stock):
    rb, rs = market_and_stock
    rs[-1] = 0.0
    rb[-1] = 0.0
    stock = make_frame(rs)
    # 6% special dividend: price drops, total return is flat; Yahoo scales earlier adjcloses down
    stock.loc[DATES[-1], ["open", "high", "low", "close"]] = stock["close"].iloc[-2] * 0.94
    stock.loc[stock.index[:-1], "adjclose"] = stock["close"].iloc[:-1] * 0.94
    stock.loc[DATES[-1], "adjclose"] = stock.loc[DATES[-1], "close"]
    det, _ = detector_for(stock, make_frame(rb))
    assert [s for s in det.run_session("US", DATES[-1]).signals if s["kind"] == "price_move.close"] == []


def test_subscription_params():
    sig = {
        "kind": "price_move.close",
        "topics": ["ticker:NVDA", "sector:information_technology"],
        "magnitude": {"move_pct": -2.8, "z": -3.1, "residual_z": -1.0},
    }
    assert matches({"topic": "ticker:NVDA"}, sig)
    assert not matches({"topic": "ticker:AAPL"}, sig)
    assert not matches({"topic": "ticker:NVDA", "params": {"min_move_pct": 3}}, sig)
    assert not matches({"topic": "ticker:NVDA", "params": {"direction": "up"}}, sig)
    assert not matches({"topic": "ticker:NVDA", "params": {"min_z": 3.5}}, sig)
    assert matches({"topic": "ticker:NVDA", "params": {"min_z": 1.0}}, sig)  # floor is 2.5
    assert not matches({"topic": "ticker:NVDA", "kinds": ["price_move.gap"]}, sig)
    assert matches({"topic": "ticker:NVDA", "kinds": ["price_move.*"]}, sig)
    assert not params_pass({"residual_only": True}, sig)


def _payload(dates_ts, closes, tz="America/New_York"):
    n = len(closes)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "X", "currency": "USD", "exchangeTimezoneName": tz},
                    "timestamp": dates_ts,
                    "events": {"splits": {str(dates_ts[1]): {"date": dates_ts[1], "numerator": 2, "denominator": 1}}},
                    "indicators": {
                        "quote": [{"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1] * n}],
                        "adjclose": [{"adjclose": closes}],
                    },
                }
            ],
            "error": None,
        }
    }


def test_parse_chart_drops_unsettled_bar_and_nulls():
    # session opens 09:30 New York on 2026-11-12/13/16 (14:30 UTC)
    ts = [1794583800 - 86400, 1794583800, 1794583800 + 3 * 86400]
    payload = _payload(ts, [10.0, None, 11.0])
    us = MARKETS["US"]
    during = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)  # 13:00 New York
    after = datetime(2026, 11, 16, 22, 0, tzinfo=timezone.utc)  # 17:00 New York
    b = parse_chart(payload, us, during)
    assert b.dropped_partial and list(b.frame.index.date.astype(str)) == ["2026-11-12"]
    b = parse_chart(payload, us, after)
    assert not b.dropped_partial and list(b.frame.index.date.astype(str)) == ["2026-11-12", "2026-11-16"]
