# D:\Dev\kha\api\state\health_rules.py
from __future__ import annotations

def evaluate(stats: dict, dc_thresh: float = 0.05, e_inc_streak: int = 5) -> dict:
    if not stats:
        return {"status": "unknown", "reasons": []}
    reasons = []
    if stats.get("max_abs_dC", 0.0) > dc_thresh:
        reasons.append(f"|Delta C|>{dc_thresh}")
    if stats.get("inc_streak", 0) >= e_inc_streak:
        reasons.append(f"E increased {stats['inc_streak']} steps")
    status = "degraded" if reasons else "ok"
    return {"status": status, "reasons": reasons}
