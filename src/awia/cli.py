"""Command line for the A1 price move detector (end-of-day, Yahoo data).

  awia-a1 fetch                    download ~2 years of daily bars for the universe and benchmarks
  awia-a1 detect [--market US]     evaluate each market's latest completed session, store and print signals
  awia-a1 replay [--sessions 250]  run the detector over past sessions to see how many signals it fires
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from awia.data import yahoo
from awia.detectors.price_move import Config, PriceMoveDetector
from awia.store import Store
from awia.subscriptions import params_pass
from awia.universe import MARKETS, Instrument, load_universe


def _symbols(universe: list[Instrument], markets: list[str]) -> list[str]:
    syms = [MARKETS[m].benchmark for m in markets]
    return syms + [i.ticker for i in universe if i.market in markets]


def _fetch(universe, markets, cache_dir: Path, range_: str) -> None:
    symbols = _symbols(universe, markets)
    print(f"Fetching {len(symbols)} symbols from Yahoo ({range_} daily)...", file=sys.stderr)
    errors = {s: e for s, e in yahoo.fetch_to_cache(symbols, cache_dir, range_).items() if e}
    print(f"  ok: {len(symbols) - len(errors)}  failed: {len(errors)}", file=sys.stderr)
    for s, e in sorted(errors.items()):
        print(f"  ! {e}", file=sys.stderr)


def _load(universe, markets, cache_dir: Path) -> dict:
    now = datetime.now(timezone.utc)
    by_symbol = {i.ticker: i for i in universe}
    bars = {}
    for s in _symbols(universe, markets):
        market = MARKETS[by_symbol[s].market] if s in by_symbol else next(m for m in MARKETS.values() if m.benchmark == s)
        bars[s] = yahoo.load_cached(cache_dir, s, market, now)
    return bars


def _move(signal: dict):
    m = signal["magnitude"]
    return m["gap_pct"] if signal["kind"] == "price_move.gap" else m["move_pct"]


def _fmt(v, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def _print_signals(signals: list[dict], new_ids: set[str] | None = None) -> None:
    if not signals:
        print("  no signals")
        return
    print(f"  {'ticker':9} {'kind':18} {'move%':>7} {'z':>6} {'res_z':>6} {'band':>5} {'sev':>3}  name")
    for s in sorted(signals, key=lambda s: (s["severity_hint"], -abs(s["magnitude"]["z"]))):
        m = s["magnitude"]
        move = _move(s)
        mark = "" if new_ids is None or s["signal_id"] in new_ids else "  (already sent)"
        print(
            f"  {s['topics'][0][7:]:9} {s['kind']:18} {_fmt(move, '+7.2f')} {_fmt(m['z'], '+6.2f')} "
            f"{_fmt(m.get('residual_z'), '+6.2f')} {s['band_sigma']:>4}σ {s['severity_hint']:>3}  {s['instrument_name']}{mark}"
        )


def cmd_fetch(args, universe) -> int:
    _fetch(universe, args.market, args.data_dir / "raw", args.range)
    return 0


def cmd_detect(args, universe) -> int:
    cache = args.data_dir / "raw"
    if not args.no_fetch:
        _fetch(universe, args.market, cache, args.range)
    bars = _load(universe, args.market, cache)
    det = PriceMoveDetector(universe, bars, Config())
    store = Store(args.data_dir / "a1.sqlite")
    out_dir = args.data_dir / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    for code in args.market:
        market = MARKETS[code]
        date = pd.Timestamp(args.date) if args.date else det.latest_session(code)
        if date is None:
            print(f"\n[{code}] benchmark {market.benchmark} has no data; skipped")
            continue
        if date not in det.sessions(code):
            print(f"\n[{code}] {date.date()} is not a session in the data; skipped")
            continue
        res = det.run_session(code, date)
        day = date.date().isoformat()
        new = store.save_signals(res.signals)
        store.save_issues(code, day, res.issues)
        path = out_dir / f"{code}_{day}.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in res.signals), encoding="utf-8")

        b = det.benchmark_frame(code)
        bpos = b.index.get_loc(date)
        bmove = (b["adjclose"].iloc[bpos] / b["adjclose"].iloc[bpos - 1] - 1) * 100 if bpos else float("nan")
        print(
            f"\n[{code}] session {day}  benchmark {market.benchmark} {bmove:+.2f}%  "
            f"evaluated {res.evaluated}/{sum(i.market == code for i in universe)}  "
            f"signals {len(res.signals)} ({len(new)} new)  -> {path}"
        )
        _print_signals(res.signals, {s["signal_id"] for s in new})
        problems = [i for i in res.issues if i["issue"] != "warmup"]
        if problems:
            print("  data issues (not evaluated):")
            for i in problems:
                print(f"    {i['ticker']:9} {i['issue']:20} {i.get('detail') or ''}")
        warm = [i["ticker"] for i in res.issues if i["issue"] == "warmup"]
        if warm:
            print(f"  warming up (not enough history): {', '.join(warm)}")
    return 0


def cmd_replay(args, universe) -> int:
    bars = _load(universe, args.market, args.data_dir / "raw")
    det = PriceMoveDetector(universe, bars, Config())
    rows = []
    for code in args.market:
        sessions = det.sessions(code)[-args.sessions :]
        b = det.benchmark_frame(code)
        bench_r = (b["adjclose"] / b["adjclose"].shift(1) - 1) * 100
        per_day = []
        for d in sessions:
            res = det.run_session(code, d)
            per_day.append((d, res.evaluated, res.signals, bench_r.get(d)))
            for s in res.signals:
                m = s["magnitude"]
                rows.append(
                    {
                        "market": code, "session_date": s["session_date"], "ticker": s["topics"][0][7:],
                        "kind": s["kind"], "move_pct": _move(s),
                        "z": m["z"], "residual_z": m.get("residual_z"), "band_sigma": s["band_sigma"],
                        "severity_hint": s["severity_hint"], "benchmark_move_pct": round(bench_r.get(d), 2),
                        "passes_residual_only": params_pass({"residual_only": True}, s),
                    }
                )
        _print_replay(code, per_day)

    if rows:
        path = args.data_dir / "replay_signals.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\nAll replayed signals written to {path}")
    return 0


def _print_replay(code: str, per_day: list) -> None:
    if not per_day:
        print(f"\n[{code}] no sessions")
        return
    n = len(per_day)
    first, last = per_day[0][0].date(), per_day[-1][0].date()
    print(f"\n[{code}] {n} sessions {first} → {last}, ~{per_day[-1][1]} stocks evaluated per session")
    for kind in ("price_move.close", "price_move.gap"):
        counts = pd.Series([sum(s["kind"] == kind for s in sig) for _, _, sig, _ in per_day])
        allsig = [s for _, _, sig, _ in per_day for s in sig if s["kind"] == kind]
        resid = [s for s in allsig if params_pass({"residual_only": True}, s)]
        bands = pd.Series([s["band_sigma"] for s in allsig]).value_counts().sort_index()
        top = counts.sort_values(ascending=False)
        top5 = top.head(max(1, n // 20)).sum()
        print(
            f"  {kind:18} total {len(allsig):4}  per session: mean {counts.mean():.2f}, median {counts.median():.0f}, "
            f"max {counts.max()}  | zero-signal sessions {int((counts == 0).sum())}/{n}"
        )
        print(
            f"  {'':18} by band: " + ", ".join(f"{b}σ: {c}" for b, c in bands.items())
            + f"  | busiest 5% of sessions hold {top5 / max(1, len(allsig)):.0%}"
            + f"  | pass residual_only: {len(resid)} ({len(resid) / max(1, len(allsig)):.0%})"
        )
    busiest = sorted(per_day, key=lambda x: -sum(s["kind"] == "price_move.close" for s in x[2]))[:5]
    print("  busiest sessions (close signals, benchmark move):")
    for d, _, sig, br in busiest:
        k = sum(s["kind"] == "price_move.close" for s in sig)
        print(f"    {d.date()}  {k:3} signals  benchmark {br:+.2f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="awia-a1", description="A1 price move detector (EOD, Yahoo data)")
    p.add_argument("--universe", type=Path, default=Path("All_200_Stocks_POC.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--market", nargs="+", choices=list(MARKETS), default=list(MARKETS))
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily bars into the cache")
    f.add_argument("--range", default="2y")

    d = sub.add_parser("detect", help="evaluate the latest completed session of each market")
    d.add_argument("--date", help="session date YYYY-MM-DD (default: latest completed)")
    d.add_argument("--no-fetch", action="store_true", help="use cached bars only")
    d.add_argument("--range", default="2y")

    r = sub.add_parser("replay", help="run over past sessions from the cache and summarise")
    r.add_argument("--sessions", type=int, default=250)

    args = p.parse_args(argv)
    universe = load_universe(args.universe)
    return {"fetch": cmd_fetch, "detect": cmd_detect, "replay": cmd_replay}[args.cmd](args, universe)


if __name__ == "__main__":
    raise SystemExit(main())
