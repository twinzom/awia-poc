"""Subscription matching for price_move signals (the design's kind_matches / params_pass)."""

from __future__ import annotations

from fnmatch import fnmatchcase

from awia.detectors.price_move import DEFAULT_KINDS

MIN_Z_FLOOR = 2.5


def kind_matches(patterns: list[str] | None, kind: str) -> bool:
    return any(fnmatchcase(kind, p) for p in (patterns or DEFAULT_KINDS))


def params_pass(params: dict, signal: dict) -> bool:
    """Apply min_z, min_move_pct, direction and residual_only to a price_move.* signal."""
    if not signal["kind"].startswith("price_move."):
        return True
    m = signal["magnitude"]
    move = m["gap_pct"] if signal["kind"] == "price_move.gap" else m["move_pct"]
    if move is None:
        return False

    min_z = max(float(params.get("min_z", MIN_Z_FLOOR)), MIN_Z_FLOOR)
    z = m["z"]
    if params.get("residual_only") and m.get("residual_z") is not None:
        # Without a residual (beta still warming up) the total z is used instead.
        z = m["residual_z"]
    if z is None or abs(z) < min_z:
        return False

    if "min_move_pct" in params and abs(move) < float(params["min_move_pct"]):
        return False

    direction = params.get("direction", "both")
    if direction == "up" and move < 0:
        return False
    if direction == "down" and move >= 0:
        return False
    return True


def matches(subscription: dict, signal: dict) -> bool:
    """subscription: {topic, kinds?, params?} as in the design's subscriptions table."""
    return (
        subscription["topic"] in signal["topics"]
        and kind_matches(subscription.get("kinds"), signal["kind"])
        and params_pass(subscription.get("params") or {}, signal)
    )
