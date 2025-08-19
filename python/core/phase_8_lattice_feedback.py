"""
Phase 8: Soliton ↔ ConceptMesh Feedback Reinforcement

Monitors the soliton lattice for phase-coherent interactions and injects 
reinforcement signals into ConceptMesh relation weights (edge boosting) 
if memory wave packets resonate through shared embedding-space alignment 
and coherent phase evolution.

Author: GPT + TORI
"""

import time
import numpy as np
import logging
from typing import List, Tuple, Dict

# Import TORI's core modules
from python.core.fractal_soliton_memory import FractalSolitonMemory
from python.core.concept_mesh import ConceptMesh

logger = logging.getLogger(__name__)


class Phase8LatticeFeedback:
    def __init__(self):
        self.soliton = FractalSolitonMemory.get_instance()
        self.mesh = ConceptMesh.instance()
        logger.info("🎯 Phase 8 Reinforcement Engine Initialized")

    def run_once(self, coherence_threshold: float = 0.85, similarity_threshold: float = 0.75):
        """
        Perform one pass of coherence-based edge reinforcement

        Args:
            coherence_threshold: Minimum average coherence to trigger boost
            similarity_threshold: Minimum embedding similarity to consider concepts related
        """
        resonant_pairs = self._get_resonant_wave_pairs(coherence_threshold, similarity_threshold)

        for (id1, id2, strength) in resonant_pairs:
            concept1 = self._extract_concept_id(id1)
            concept2 = self._extract_concept_id(id2)

            if concept1 and concept2:
                # Boost or create the edge between them
                success = self.mesh.add_relation(
                    source_id=concept1,
                    target_id=concept2,
                    relation_type="resonates_with",
                    strength=strength,
                    bidirectional=True,
                    metadata={"source": "phase_8_feedback", "phase_resonant": True}
                )
                if success:
                    logger.info(f"🌐 Reinforced relation: {concept1} ↔ {concept2} (strength={strength:.3f})")

        logger.info(f"✅ Phase 8 pass complete ({len(resonant_pairs)} reinforced)")

    def _get_resonant_wave_pairs(self, coherence_threshold: float, similarity_threshold: float) -> List[Tuple[str, str, float]]:
        """
        Detect pairs of soliton waves that are both coherent and similar in embedding space

        Returns:
            List of (memory_id_1, memory_id_2, similarity_score)
        """
        waves = list(self.soliton.waves.values())
        pairs = []

        for i in range(len(waves)):
            for j in range(i + 1, len(waves)):
                w1 = waves[i]
                w2 = waves[j]

                # Only consider sufficiently coherent waves
                if w1.coherence < coherence_threshold or w2.coherence < coherence_threshold:
                    continue

                if w1.embedding is None or w2.embedding is None:
                    continue

                # Cosine similarity
                sim = np.dot(w1.embedding, w2.embedding) / (
                    np.linalg.norm(w1.embedding) * np.linalg.norm(w2.embedding) + 1e-8
                )

                if sim >= similarity_threshold:
                    strength = sim * (w1.coherence + w2.coherence) / 2
                    pairs.append((w1.id, w2.id, strength))

        return pairs

    def _extract_concept_id(self, memory_id: str) -> str:
        """
        Extract Concept ID from memory ID prefix
        Assumes memory_id format like 'concept_<id>' or 'sculpted_<concept_id>_*'
        """
        if memory_id.startswith("concept_"):
            return memory_id
        elif memory_id.startswith("sculpted_"):
            parts = memory_id.split("_")
            if len(parts) >= 2:
                return f"concept_{parts[1]}"
        return None


# CLI interface for manual execution
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Phase 8 Reinforcement once")
    parser.add_argument("--coherence", type=float, default=0.85, help="Minimum coherence threshold")
    parser.add_argument("--similarity", type=float, default=0.75, help="Minimum embedding similarity")

    args = parser.parse_args()

    runner = Phase8LatticeFeedback()
    runner.run_once(coherence_threshold=args.coherence, similarity_threshold=args.similarity)
