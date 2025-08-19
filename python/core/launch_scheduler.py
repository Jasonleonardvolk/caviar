# File: C:/Users/jason/Desktop/tori/kha/python/core/launch_scheduler.py

"""
Launch Scheduler
──────────────────────────────────────────────
Background task loop manager for TORI Phase 8 reinforcement.
Now includes auto-adaptive scheduling based on phase resonance entropy.
"""

import asyncio
import logging
import math
from python.core.phase_8_lattice_feedback import Phase8LatticeFeedback
from python.core.fractal_soliton_memory import FractalSolitonMemory

logger = logging.getLogger("launch_scheduler")

# Bounds for adaptive timing
MIN_INTERVAL = 120   # 2 minutes
MAX_INTERVAL = 1800  # 30 minutes
BASE_INTERVAL = 600  # starting default


def compute_entropy(waves) -> float:
    """
    Compute resonance entropy based on coherence distribution.
    Lower entropy = more synchronized (less randomness)
    Higher entropy = less organized
    """
    if not waves:
        return 1.0

    coherence_values = [w.coherence for w in waves if hasattr(w, "coherence")]
    if not coherence_values:
        return 1.0

    # Normalize
    total = sum(coherence_values)
    probs = [c / total for c in coherence_values if total > 0]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy / math.log2(len(probs))  # normalize to [0,1]


async def lattice_resonance_loop():
    """
    Continuously runs Phase 8 lattice-mesh feedback loop
    with auto-adaptive interval based on resonance entropy
    """
    feedback_engine = Phase8LatticeFeedback()
    soliton = FractalSolitonMemory.get_instance()

    current_interval = BASE_INTERVAL
    logger.info("🚀 Starting adaptive Phase 8 resonance loop")

    while True:
        try:
            feedback_engine.run_once()

            # Adapt interval based on wave coherence
            entropy = compute_entropy(soliton.waves.values())
            current_interval = int(MAX_INTERVAL * entropy + MIN_INTERVAL * (1 - entropy))

            logger.info(f"🧠 Resonance entropy: {entropy:.3f} → next run in {current_interval} sec")

        except Exception as e:
            logger.error(f"⚠️ Phase 8 loop error: {str(e)}")
            current_interval = BASE_INTERVAL

        await asyncio.sleep(current_interval)

# Usage:
# In enhanced_launcher.py:
# from python.core.launch_scheduler import lattice_resonance_loop
# asyncio.create_task(lattice_resonance_loop())
