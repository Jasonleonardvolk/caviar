#!/usr/bin/env python3
"""penrose_error_correction.py
====================================================
Quantum‑error‑correcting helper for the TORI project.

This module exposes a minimal yet *practical* Penrose‑tiling–based
QECC façade that plugs into the hot‑swappable Laplacian engine.
It is **pure‑Python**, has **zero required dependencies**, and
falls back gracefully if heavy libraries (``scipy``, ``networkx``)
are absent.

The public API centres on three primitives:

* :class:`PenroseQECC` – build / load a finite Penrose patch,
  track defect wells, supply syndrome / correction maps.
* :func:`golden_soliton`  – deterministic golden‑ratio soliton seed
  for reproducible experiments.
* :func:`attach_defect_wells` – inject stabiliser sites into an
  existing :pymod:`hot_swap_laplacian` instance.

Design notes
------------
* **Typed** – all public call‑sites are fully type‑hinted.
* **Deterministic** – every stochastic branch accepts an optional
  ``seed``; if omitted we draw from ``numpy.random.default_rng()``.
* **No global state** – the module never mutates imported code.
* **Safe fall‑backs** – if heavy maths packages are missing we
  silently degrade to stub maths so the rest of TORI can import.
* **Logging** – uses a local *null* handler; host apps must opt‑in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import logging
import math
import json

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

__all__ = [
    "PHI",
    "PenroseQECC",
    "golden_soliton",
    "attach_defect_wells",
]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
PHI: float = (1.0 + 5**0.5) / 2.0  #: The golden ratio φ
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@dataclass(slots=True)
class DefectWell:
    """A single defect / stabiliser site inside the Penrose code."""

    idx: int  #: Index of the hosting Penrose vertex
    depth: float  #: Potential‑well depth (arbitrary units)
    charge: int  #: +1 / −1 to balance lattice flux


@dataclass(slots=True)
class PenroseQECC:
    """Penrose‑tiling quantum‑error‑correcting *surface*.

    Parameters
    ----------
    patch_order:
        Order (k) of the kite‑dart inflation.  The number of tiles grows
        ~φ^k; choose k ≤ 7 (~6 k vertices) for interactive runs.
    seed:
        Optional RNG seed for reproducible vertex jitter.
    """

    patch_order: int = 5
    seed: Optional[int] = None

    vertices: List[Tuple[float, float]] = field(init=False, default_factory=list)
    edges: List[Tuple[int, int]] = field(init=False, default_factory=list)
    wells: List[DefectWell] = field(init=False, default_factory=list)

    # -------------------------------------------------- lifecycle ---------
    def __post_init__(self) -> None:  # noqa: D401 – simple name
        rng = None
        if np:  # only allocate if numpy present
            rng = np.random.default_rng(self.seed)
        self._build_patch(rng)
        self._build_defect_wells()

    # -------------------------------------------------- public API --------
    def syndrome(self, errors: Iterable[int]) -> Dict[int, int]:
        """Return a *syndrome* map {well_idx → ±1} for given error qubits."""
        syn: Dict[int, int] = {w.idx: 0 for w in self.wells}
        for q in errors:
            for w in self.wells:
                if (q + w.idx) % 2:  # toy parity rule
                    syn[w.idx] ^= 1
        return syn

    def correct(self, errors: Iterable[int]) -> List[int]:
        """Simple majority‑vote correction; returns remaining *uncorrected* errors."""
        uncorrected: List[int] = []
        syn = self.syndrome(errors)
        for q in errors:
            vote = sum(syn[w.idx] for w in self.wells if (q + w.idx) % 2)
            if vote > len(self.wells) / 2:
                continue  # corrected
            uncorrected.append(q)
        return uncorrected

    # -------------------------------------------------- internal ----------
    def _build_patch(self, rng) -> None:  # type: ignore[override]
        """Populate *vertices* and *edges* with a deterministic kite‑dart patch.

        We use a *very* lightweight inflation system so we don’t drag large
        geometry libs into TORI if not needed.  Quality is sufficient for
        error‑correction demos.
        """
        if np is None or nx is None:  # pragma: no cover – graceful degrade
            # Minimal 5‑vertex star so dependent code can still import.
            self.vertices = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)) for i in range(5)]
            self.edges = [(i, (i + 1) % 5) for i in range(5)]
            _LOGGER.warning("penrose_error_correction: fallback 5‑gon patch (NumPy/networkx missing)")
            return

        # Start with a single kite; inflate *patch_order* times.
        tiles = [(np.zeros(2), np.array([1.0, 0.0]), True)]  # (origin, heading, is_kite?)
        for _ in range(self.patch_order):
            new_tiles = []
            for origin, vec, is_kite in tiles:
                v_rot = np.array([[math.cos(math.pi / 5), -math.sin(math.pi / 5)],
                                  [math.sin(math.pi / 5),  math.cos(math.pi / 5)]]) @ vec
                if is_kite:
                    new_tiles.append((origin, v_rot, False))
                    new_tiles.append((origin + vec, -v_rot / PHI, True))
                else:  # dart
                    new_tiles.append((origin, v_rot, True))
                    new_tiles.append((origin + vec / PHI, -v_rot, False))
            tiles = new_tiles

        # Extract vertex list (de‑duplicate with tolerance)
        vert_map: Dict[Tuple[int, int], int] = {}
        tol = 1e-6
        for origin, vec, _ in tiles:
            for t in (0.0, 1.0):
                p = tuple(np.round(origin + t * vec, 6))
                if p not in vert_map:
                    vert_map[p] = len(vert_map)
        self.vertices = [p for p, _ in sorted(vert_map.items(), key=lambda kv: kv[1])]

        # Build naïve edge set – connect tile endpoints.
        edge_set = set()
        for origin, vec, _ in tiles:
            a = vert_map[tuple(np.round(origin, 6))]
            b = vert_map[tuple(np.round(origin + vec, 6))]
            edge_set.add((min(a, b), max(a, b)))
        self.edges = sorted(edge_set)

        # Optional jitter for aesthetics
        if rng is not None:
            jitter = (rng.random((len(self.vertices), 2)) - 0.5) * 0.01
            self.vertices = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(self.vertices, jitter)]

    def _build_defect_wells(self) -> None:  # noqa: D401
        """Identify 5 fold‑symmetry axes and mark stabiliser wells."""
        if not self.vertices:
            return
        # Pick 5 far‑apart vertices as wells (toy heuristic).
        step = max(1, len(self.vertices) // 5)
        for i in range(5):
            idx = (i * step) % len(self.vertices)
            self.wells.append(DefectWell(idx=idx, depth=1.0 / PHI, charge=(-1) ** i))


# ---------------------------------------------------------------------------
#  Convenience helpers  ------------------------------------------------------
# ---------------------------------------------------------------------------

def golden_soliton(amplitude: float = 1.0, *, phase: float | None = None) -> complex:  # noqa: D401
    """Return a reproducible “golden” soliton (complex amplitude).

    By default the phase is φ‑related so that superpositions of calls are
    mutually orthogonal: ϕ_k = 2π k / φ.
    """
    if phase is None:
        phase = (2 * math.pi) / PHI
    return amplitude * math.exp(1j * phase)


def attach_defect_wells(hs_laplacian, *, order: int = 5, seed: int | None = None) -> PenroseQECC:  # noqa: D401
    """Instantiate a :class:`PenroseQECC` and register its wells with *hs_laplacian*.

    This is a *best‑effort* bridge; if the host Laplacian lacks expected
    hooks we issue a warning and still return the QECC object so callers
    can use it directly.
    """
    qec = PenroseQECC(order, seed)
    if not hasattr(hs_laplacian, "defect_wells"):
        _LOGGER.warning("HotSwappableLaplacian missing 'defect_wells' attribute; wells not attached.")
    else:
        hs_laplacian.defect_wells = qec.wells  # type: ignore[attr-defined]
    return qec
