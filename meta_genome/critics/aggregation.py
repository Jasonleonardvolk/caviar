# /meta_genome/critics/aggregation.py
import math
from typing import Dict, List

THETA = 0.7  # acceptance threshold

def logodds(p: float) -> float:
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p / (1-p))

def aggregate(scores: Dict[str, float], reliabilities: Dict[str, float]) -> bool:
    numer = 0.0
    denom = 0.0
    for cid, s in scores.items():
        w = logodds(reliabilities.get(cid, 0.5))
        numer += w * s
        denom += w
    S = numer / denom if denom else 0
    return S >= THETA, S
