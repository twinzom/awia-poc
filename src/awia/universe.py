"""The instrument universe and the markets it trades in."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import time
from pathlib import Path


@dataclass(frozen=True)
class Market:
    code: str
    benchmark: str  # Yahoo symbol of an ETF tracking the market's main index
    benchmark_name: str
    tz: str
    close: time  # end of the closing auction, exchange local time


MARKETS: dict[str, Market] = {
    "US": Market("US", "SPY", "SPDR S&P 500 ETF", "America/New_York", time(16, 0)),
    "HK": Market("HK", "2800.HK", "Tracker Fund of Hong Kong (HSI)", "Asia/Hong_Kong", time(16, 10)),
    "UK": Market("UK", "ISF.L", "iShares Core FTSE 100 ETF", "Europe/London", time(16, 35)),
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+([.-][A-Z0-9]+)*$")


@dataclass(frozen=True)
class Instrument:
    ticker: str  # Yahoo symbol, e.g. AAPL, 0700.HK, BT-A.L
    name: str
    market: str
    currency: str
    sector: str

    @property
    def sector_topic(self) -> str:
        return "sector:" + re.sub(r"[^a-z0-9]+", "_", self.sector.lower()).strip("_")


def valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol))


def load_universe(path: str | Path) -> list[Instrument]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: list[Instrument] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):
        ticker = (row.get("Ticker") or "").strip().upper()
        market = (row.get("Market") or "").strip().upper()
        if not valid_symbol(ticker):
            raise ValueError(f"{path}:{i}: bad ticker {ticker!r}")
        if market not in MARKETS:
            raise ValueError(f"{path}:{i}: unknown market {market!r} (known: {', '.join(MARKETS)})")
        if ticker in seen:
            raise ValueError(f"{path}:{i}: duplicate ticker {ticker}")
        seen.add(ticker)
        out.append(
            Instrument(
                ticker=ticker,
                name=(row.get("Company_Name") or "").strip(),
                market=market,
                currency=(row.get("Currency") or "").strip(),
                sector=(row.get("GICS_Sector") or "").strip(),
            )
        )
    return out
