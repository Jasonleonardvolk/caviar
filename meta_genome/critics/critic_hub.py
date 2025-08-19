"""
CriticHub – central dispatcher & weighted-consensus engine
Path: str(PROJECT_ROOT / "meta_genome\\critics\\critic_hub.py
"""

from __future__ import annotations
import importlib
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Callable

CRITIC_REGISTRY: Dict[str, Callable[[dict], Tuple[float, bool]]] = {}
RELIABILITY: Dict[str, Tuple[int, int]] = defaultdict(lambda: (2, 2))   # (α, β)

########################
#  Registry decorators #
########################
def critic(name: str):
    def decorator(fn: Callable[[dict], Tuple[float, bool]]):
        CRITIC_REGISTRY[name] = fn
        return fn
    return decorator

#############################
#  Built-in reference calls #
#############################
@critic("safety")
def safety_critic(report: dict) -> Tuple[float, bool]:
    if report["safety_pass"]:
        return 1.0, True
    return 0.0, False            # hard veto

@critic("tests")
def tests_critic(report: dict) -> Tuple[float, bool]:
    score = report["tests_passed"] / max(report["tests_total"], 1)
    return score, score == 1.0   # veto if any test fails

# Add more via import hook (e.g. performance, stability, novelty)  
def _load_dynamic():
    for mod in ("performance_critic", "stability_critic", "coherence_critic",
                "energy_critic", "novelty_critic", "alignment_critic"):
        try:
            importlib.import_module(f"kha.meta_genome.critics.{mod}")
        except ModuleNotFoundError:
            pass

_load_dynamic()

########################
#  Consensus function  #
########################
def _riemannian_mean(scores: Dict[str, float]) -> float:
    from math import acosh, cosh
    total_rel = sum((α+β) for α, β in RELIABILITY.values())
    Ω = 0.0
    for name, s in scores.items():
        α, β = RELIABILITY[name]
        w   = (α+β) / total_rel
        Ω  += w * acosh(max(s, 1e-9))
    return float(min(1.0, cosh(Ω) - 1e-6))

###################################
#  Public API for sandbox outcome #
###################################
def evaluate(report: dict) -> Tuple[bool, float, Dict[str, float]]:
    """
    report = {
      "safety_pass": bool,
      "tests_passed": int, "tests_total": int,
      ... extra keys for perf/energy critics ...
    }
    """
    scores, votes = {}, {}
    for name, fn in CRITIC_REGISTRY.items():
        score, accept = fn(report)
        scores[name], votes[name] = score, accept

    # Hard veto check
    if not all(votes.values()):
        _update_reliability(scores, False)
        return False, 0.0, scores

    consensus = _riemannian_mean(scores)
    accepted  = consensus >= 0.75
    _update_reliability(scores, accepted)
    return accepted, consensus, scores

#############################
#  Reliability update logic #
#############################
def _update_reliability(scores: Dict[str, float], accepted: bool):
    """Bayesian update of critic reliabilities."""
    for name, s in scores.items():
        α, β = RELIABILITY[name]
        if accepted and s >= 0.75:
            α += 1           # true-positive
        elif not accepted and s < 0.75:
            β += 1           # true-negative
        RELIABILITY[name] = (α, β)
    _dump()

#####################
#  Persist to disk  #
#####################
_DB = Path(__file__).parent.parent / "audit" / "critic_stats.json"

def _dump():
    _DB.parent.mkdir(parents=True, exist_ok=True)
    _DB.write_text(
        json.dumps(
            {k: {"α": v[0], "β": v[1]} for k, v in RELIABILITY.items()},
            indent=2
        )
    )

if not _DB.exists():
    _dump()
