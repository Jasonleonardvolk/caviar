"""Ψ-state (sentiment / arousal) estimator."""
import math
import re

__all__ = ["compute_psi_state"]

positive = re.compile(r"\b(love|great|fantastic|excellent|happy|amazing|wonderful|brilliant|perfect|beautiful)\b", re.I)
negative = re.compile(r"\b(hate|terrible|awful|bad|sad|horrible|disgusting|nasty|ugly|worst)\b", re.I)

def compute_psi_state(text: str) -> float:
    """
    Returns a scalar in [-1, 1]:
      • +1 → strongly positive
      •  0 → neutral
      • -1 → strongly negative
    Simple keyword score for now; replace with ML later.
    """
    if not text:
        return 0.0
    pos = len(positive.findall(text))
    neg = len(negative.findall(text))
    if pos + neg == 0:
        return 0.0
    score = (pos - neg) / (pos + neg)
    # squash to [-1, 1] smoothly
    return math.tanh(score)
