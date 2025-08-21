"""
Fractal Soliton Memory (FSM) â€” Canonical Ricci Burn Rewrite (v1.0)
===================================================================
File: python/core/fractal_soliton_memory.py

Purpose
-------
Canonical, productionâ€‘grade FSM core for the Ricci burn. This module provides:

1) Deterministic steppers (RK4 builtâ€‘in; Stormerâ€‘Verlet splitâ€‘step) with optional JIT lanes
2) Curvatureâ†’phase/amplitude coupling (Ricci / Kretschmann encoders) with tunable gain
3) Invariants: mass, energy, and phase coherence monitors with drift accounting
4) Laplacian validation (symmetry & PSD gate) and spectral radius retuning for stability
5) Hotâ€‘swap Laplacian with Î» rescale, checkpoint ringâ€‘buffer, and NPZ persistence
6) Resonance finder and lightweight embedding cache for /api/soliton/queryâ€‘style lookups

This file is **selfâ€‘contained** with conservative fallbacks.
If `python/core/graph_ops.py` exists, we will prefer its `symmetric_psd_sanity` and
`spectral_radius` functions automatically.

Author: ChatGPT (GPTâ€‘5 Thinking) with requirements from Jason (TORI)
Version: 1.0 (2025â€‘08â€‘21)

Dependencies
------------
- numpy (required)
- scipy (optional, for sparse Laplacian support)
- numba (optional, for JIT step lanes)

Directory Expectations
----------------------
- Canonical location: D:\\Dev\\kha\\python\\core\\fractal_soliton_memory.py
- Optional companion: D:\\Dev\\kha\\python\\core\\graph_ops.py
- Concept mesh assets (if any): D:\\Dev\\kha\\data\\concept_mesh\\

Usage (Smoke Test)
------------------
>>> from python.core.fractal_soliton_memory import FractalSolitonMemory, load_laplacian_from_npz
>>> L = np.eye(256)  # or load_laplacian_from_npz("data/concept_mesh/L_norm.npz")
>>> fsm = FractalSolitonMemory.from_random(n=256, laplacian=L, seed=123)
>>> stats0 = fsm.status()
>>> fsm.step(steps=10_000, dt=1e-2, method="auto", conserve_mass=True)
>>> stats1 = fsm.status()
>>> print({k: (stats1[k]-stats0[k]) for k in ("mass","energy","coherence")})

License: Proprietary â€” TORI Project
"""
from __future__ import annotations

import os
import io
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, Callable, Dict, Any

import numpy as np

# ------------------------------- Optional deps -------------------------------
HAS_SCIPY = False
HAS_NUMBA = False
try:
    import scipy.sparse as sp  # type: ignore
    import scipy.sparse.linalg as spla  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import numba as nb  # type: ignore
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

# --------------------------- External graph_ops hook --------------------------
# If available, prefer projectâ€‘specific implementations for sanity & spectra.
_spsd_ext = None
_spr_ext = None
try:  # local module in same package
    from .graph_ops import symmetric_psd_sanity as _spsd_ext  # type: ignore
    from .graph_ops import spectral_radius as _spr_ext  # type: ignore
except Exception:
    try:
        # Flat import path fallback
        from graph_ops import symmetric_psd_sanity as _spsd_ext  # type: ignore
        from graph_ops import spectral_radius as _spr_ext  # type: ignore
    except Exception:
        _spsd_ext, _spr_ext = None, None

ArrayLike = np.ndarray

# ==============================================================================
# Utility: Laplacian validation & spectral radius (fallbacks)
# ==============================================================================

def _as_dense(L: Any) -> np.ndarray:
    if HAS_SCIPY and sp.issparse(L):
        return L.toarray()
    return np.asarray(L)


def symmetric_psd_sanity(L: Any, atol: float = 1e-9) -> Tuple[bool, float, float]:
    """Return (ok, min_eig, max_eig) for symmetry + PSD check.
    Uses project hook if available; otherwise conservative dense check.
    """
    if _spsd_ext is not None:
        try:
            return _spsd_ext(L, atol=atol)  # type: ignore
        except Exception:
            pass
    A = _as_dense(L)
    if A.shape[0] != A.shape[1]:
        return (False, float("nan"), float("nan"))
    # Symmetry
    if not np.allclose(A, A.T, atol=atol):
        return (False, float("nan"), float("nan"))
    # PSD (use eigh on small/medium, otherwise partial power method)
    n = A.shape[0]
    if n <= 1024:
        w = np.linalg.eigvalsh(A)
        return (np.min(w) >= -atol, float(np.min(w)), float(np.max(w)))
    # Large: Gershgorin lower bound + power method upper bound
    # Lower bound (very loose)
    row_abs = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    lower = np.min(np.diag(A) - row_abs)
    # Power method for largest eigenvalue magnitude
    x = np.random.default_rng(0).normal(size=n)
    x = x / np.linalg.norm(x)
    for _ in range(100):
        x = A @ x
        x_norm = np.linalg.norm(x)
        if x_norm == 0:
            break
        x = x / x_norm
    upper = float(x.T @ (A @ x))
    return (lower >= -atol, float(lower), upper)


def spectral_radius(L: Any) -> float:
    """Spectral radius Ï(L). Prefer external implementation if present."""
    if _spr_ext is not None:
        try:
            return float(_spr_ext(L))  # type: ignore
        except Exception:
            pass
    if HAS_SCIPY and sp.issparse(L):
        # Use largest magnitude eigenvalue via eigsh
        try:
            vals = spla.eigs(L, k=1, which="LM", return_eigenvectors=False)
            return float(np.abs(vals[0]))
        except Exception:
            pass
    A = _as_dense(L)
    # Power iteration
    n = A.shape[0]
    x = np.random.default_rng(1).normal(size=n)
    x = x / (np.linalg.norm(x) + 1e-12)
    last = 0.0
    for _ in range(200):
        y = A @ x
        lam = float(np.dot(x, y))
        if abs(lam - last) < 1e-9:
            break
        last = lam
        x = y / (np.linalg.norm(y) + 1e-12)
    return float(abs(last))


# ==============================================================================
# Checkpoint Ring Buffer
# ==============================================================================
@dataclass
class _CheckpointRing:
    capacity: int = 8
    _buf: list = field(default_factory=list)
    _idx: int = 0

    def push(self, payload: Dict[str, Any]) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(payload)
        else:
            self._buf[self._idx] = payload
        self._idx = (self._idx + 1) % self.capacity

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self._buf:
            return None
        return self._buf[(self._idx - 1) % len(self._buf)]

    def get(self, k: int = 0) -> Optional[Dict[str, Any]]:
        # k=0 latest, k=1 previous, etc.
        if not self._buf:
            return None
        k = k % len(self._buf)
        return self._buf[(self._idx - 1 - k) % len(self._buf)]


# ==============================================================================
# Curvature Field + Encoders
# ==============================================================================
@dataclass
class CurvatureField:
    """Holds geometric fields and encodes them into phase/amplitude.

    Members are 1D arrays aligned to lattice nodes.
    """
    n: int
    ricci_scalar: Optional[ArrayLike] = None  # R
    kretschmann: Optional[ArrayLike] = None   # K = R_{abcd} R^{abcd}
    mean_curvature: Optional[ArrayLike] = None

    def ensure_shapes(self) -> None:
        for arr_name in ("ricci_scalar", "kretschmann", "mean_curvature"):
            arr = getattr(self, arr_name)
            if arr is not None:
                a = np.asarray(arr)
                if a.shape != (self.n,):
                    raise ValueError(f"CurvatureField.{arr_name} shape {a.shape} != ({self.n},)")

    def encode_curvature_to_phase(
        self,
        mode: Literal["log_tanh", "linear", "none"] = "log_tanh",
        gain: float = 1.0,
        clip_pi: bool = True,
        kcrit: Optional[float] = None,
        prefer: Literal["kretschmann", "ricci", "mean"] = "kretschmann",
    ) -> np.ndarray:
        """Produce a phase field Ï† in radians from curvature quantities.
        - log_tanh: Ï† = gain * tanh(log(1 + |C|/e)) * sign(C)
        - linear:   Ï† = gain * C (normalized)
        - none:     Ï† = 0
        Where C is the chosen curvature signal.
        """
        self.ensure_shapes()
        if mode == "none":
            return np.zeros(self.n, dtype=np.float64)

        if prefer == "kretschmann" and self.kretschmann is not None:
            C = np.asarray(self.kretschmann, dtype=np.float64)
        elif prefer == "ricci" and self.ricci_scalar is not None:
            C = np.asarray(self.ricci_scalar, dtype=np.float64)
        elif prefer == "mean" and self.mean_curvature is not None:
            C = np.asarray(self.mean_curvature, dtype=np.float64)
        else:
            # fallback: all zeros
            return np.zeros(self.n, dtype=np.float64)

        # normalize C into sensible numeric scale
        c = C - np.nanmean(C)
        denom = np.nanstd(c) + 1e-12
        c = c / denom

        if kcrit is not None and kcrit > 0:
            c = np.clip(c, -kcrit, kcrit)

        if mode == "linear":
            phi = gain * c
        else:  # log_tanh
            x = np.log1p(np.abs(c)) * np.sign(c)
            phi = gain * np.tanh(x)

        if clip_pi:
            phi = np.clip(phi, -math.pi, math.pi)
        return phi


# ==============================================================================
# Numbaâ€‘JIT optional kernels (vectorized complex RK4)
# ==============================================================================
if HAS_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _rk4_step_jit(psi: np.ndarray, dt: float, gamma: float, Hpsi: np.ndarray) -> np.ndarray:
        # psi' = -i * Hpsi - gamma * psi ; here Hpsi precomputed for k1 only
        i_const = 1j
        k1 = (-i_const * Hpsi) - (gamma * psi)
        y2 = psi + 0.5 * dt * k1
        k2 = (-i_const * Hpsi) - (gamma * y2)
        y3 = psi + 0.5 * dt * k2
        k3 = (-i_const * Hpsi) - (gamma * y3)
        y4 = psi + dt * k3
        k4 = (-i_const * Hpsi) - (gamma * y4)
        return psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
else:
    def _rk4_step_jit(psi: np.ndarray, dt: float, gamma: float, Hpsi: np.ndarray) -> np.ndarray:
        # Pureâ€‘python fallback (same signature)
        i_const = 1j
        k1 = (-i_const * Hpsi) - (gamma * psi)
        y2 = psi + 0.5 * dt * k1
        k2 = (-i_const * Hpsi) - (gamma * y2)
        y3 = psi + 0.5 * dt * k2
        k3 = (-i_const * Hpsi) - (gamma * y3)
        y4 = psi + dt * k3
        k4 = (-i_const * Hpsi) - (gamma * y4)
        return psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ==============================================================================
# FSM Core
# ==============================================================================
@dataclass
class FractalSolitonMemory:
    """Canonical FSM core with curvature coupling, invariants, and steppers.

    Dynamics (SchrÃ¶dingerâ€‘like with nonlinear/curvature terms):
        i dÏˆ/dt = -Î» L Ïˆ + Î± |Ïˆ|^2 Ïˆ + V âŠ™ Ïˆ  - i Î³ Ïˆ

    Where:
      - L: (normalized) graph Laplacian (symmetric PSD)
      - Î»: diffusion/dispersion gain, retuned by spectral radius Ï(L)
      - Î±: cubic nonlinearity strength
      - V: potential (phase/amplitude offsets)
      - Î³: damping (nonâ€‘Hamiltonian)
    """
    # Lattice size and primary state
    n: int
    psi: np.ndarray  # complex128, shape (n,)

    # Operators & parameters
    laplacian: Any
    lambda_: float = 0.25
    alpha: float = 0.0
    gamma: float = 0.0
    dt_default: float = 1e-2

    # Potential and curvature
    V0: Optional[np.ndarray] = None  # base potential (real), shape (n,)
    curvature: Optional[CurvatureField] = None

    # Integrator preferences
    prefer_external_stepper: bool = False  # route to python/core/fsm_lattice_integration if present

    # Checkpoint & metadata
    checkpoints: _CheckpointRing = field(default_factory=lambda: _CheckpointRing(capacity=8))
    meta: Dict[str, Any] = field(default_factory=dict)

    # Internal work buffers (not serialized)
    _last_energy: float = 0.0
    _last_mass: float = 0.0
    _last_coherence: float = 0.0
    _embeddings: Optional[np.ndarray] = None  # lightweight resonance cache

    # --------------------------- Construction helpers --------------------------
    @classmethod
    def from_random(
        cls,
        n: int,
        laplacian: Optional[Any] = None,
        seed: Optional[int] = None,
        lambda_: float = 0.25,
        alpha: float = 0.0,
        gamma: float = 0.0,
    ) -> "FractalSolitonMemory":
        rng = np.random.default_rng(seed)
        psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        psi0 = psi0.astype(np.complex128)
        psi0 /= np.linalg.norm(psi0) + 1e-12
        if laplacian is None:
            laplacian = np.eye(n, dtype=np.float64)
        return cls(n=n, psi=psi0, laplacian=laplacian, lambda_=lambda_, alpha=alpha, gamma=gamma)

    # -------------------------------- Lifecycle --------------------------------
    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=np.complex128)
        if self.psi.shape != (self.n,):
            raise ValueError(f"psi shape {self.psi.shape} != ({self.n},)")
        self.set_laplacian(self.laplacian, normalize=True, sanity=True, rescale_lambda=True)
        if self.V0 is None:
            self.V0 = np.zeros(self.n, dtype=np.float64)
        # Initialize invariants
        self._update_invariants()
        self.save_checkpoint(reason="init")

    # ------------------------------ Laplacian gate ------------------------------
    def set_laplacian(
        self,
        L: Any,
        *,
        normalize: bool = True,
        sanity: bool = True,
        rescale_lambda: bool = True,
        atol: float = 1e-9,
    ) -> None:
        """Install new Laplacian, validate PSD, optionally normalize & retune Î»."""
        if HAS_SCIPY and sp.issparse(L):
            self.laplacian = L.tocsr().astype(np.float64)
        else:
            A = np.asarray(L, dtype=np.float64)
            if A.shape != (self.n, self.n):
                raise ValueError(f"Laplacian shape {A.shape} != ({self.n},{self.n})")
            self.laplacian = A

        if sanity:
            ok, mineig, maxeig = symmetric_psd_sanity(self.laplacian, atol=atol)
            if not ok:
                raise ValueError(
                    f"Laplacian failed symmetry/PSD check (minEig={mineig:.3e}, maxEig={maxeig:.3e})"
                )

        if normalize:
            rho = spectral_radius(self.laplacian)
            if rho <= 0 or not np.isfinite(rho):
                raise ValueError("Invalid spectral radius for Laplacian")
            scale = 1.0 / rho
            if HAS_SCIPY and sp.issparse(self.laplacian):
                self.laplacian = self.laplacian.multiply(scale)
            else:
                self.laplacian = np.asarray(self.laplacian) * scale

        if rescale_lambda:
            self.tuned_lambda()

    def tuned_lambda(self, target: float = 0.25) -> None:
        """Retune Î» so that ||L|| â‰ˆ 1 implies stable step sizes at dt_default.
        target is an empirical dispersion gain; keep within [1e-6, 10].
        """
        self.lambda_ = float(np.clip(target, 1e-6, 10.0))

    def hot_swap_laplacian(self, L_new: Any, renorm_lambda: bool = True) -> None:
        self.set_laplacian(L_new, normalize=True, sanity=True, rescale_lambda=renorm_lambda)
        self.save_checkpoint(reason="hot_swap_laplacian")

    # ------------------------------- Invariants --------------------------------
    def mass(self) -> float:
        return float(np.vdot(self.psi, self.psi).real)

    def energy(self) -> float:
        # H = Î» Ïˆ* L Ïˆ + 0.5 Î± |Ïˆ|^4 + V |Ïˆ|^2
        psi = self.psi
        if HAS_SCIPY and sp.issparse(self.laplacian):
            Lpsi = self.laplacian.dot(psi)
        else:
            Lpsi = self.laplacian @ psi
        term_L = self.lambda_ * float(np.vdot(psi, Lpsi).real)
        term_nl = 0.5 * self.alpha * float(np.sum(np.abs(psi) ** 4))
        term_V = float(np.sum(self.V0 * (np.abs(psi) ** 2)))
        return term_L + term_nl + term_V

    def coherence(self) -> float:
        # |sum Ïˆ| / sum |Ïˆ|
        s = np.abs(np.sum(self.psi))
        d = np.sum(np.abs(self.psi)) + 1e-12
        return float(s / d)

    def _update_invariants(self) -> None:
        self._last_mass = self.mass()
        self._last_energy = self.energy()
        self._last_coherence = self.coherence()

    # ---------------------------- Hamiltonian pieces ---------------------------
    def _nonlinear_term(self, psi: np.ndarray) -> np.ndarray:
        if self.alpha == 0.0:
            return np.zeros_like(psi)
        return self.alpha * (np.abs(psi) ** 2) * psi

    def _laplacian_apply(self, psi: np.ndarray) -> np.ndarray:
        if HAS_SCIPY and sp.issparse(self.laplacian):
            return self.laplacian.dot(psi)
        return self.laplacian @ psi

    def _hamiltonian_matvec(self, psi: np.ndarray) -> np.ndarray:
        # HÏˆ = -Î» L Ïˆ + Î± |Ïˆ|^2 Ïˆ + V0 âŠ™ Ïˆ
        Lpsi = self._laplacian_apply(psi)
        Hpsi = (-self.lambda_) * Lpsi + self._nonlinear_term(psi)
        if self.V0 is not None:
            Hpsi = Hpsi + (self.V0.astype(np.float64) * psi)
        return Hpsi

    # ------------------------------- Curvature I/O -----------------------------
    def inject_curvature_field(
        self,
        *,
        phase_mode: Literal["log_tanh", "linear", "none"] = "log_tanh",
        amp_mode: Literal["none", "linear"] = "none",
        phase_gain: float = 0.15,
        amp_gain: float = 0.0,
        kcrit: Optional[float] = 6.0,
        prefer: Literal["kretschmann", "ricci", "mean"] = "kretschmann",
        inplace: bool = True,
    ) -> np.ndarray:
        """Map curvature â†’ {phase, amplitude}. Returns the new Ïˆ (or preview if inplace=False)."""
        if self.curvature is None:
            return self.psi.copy() if not inplace else self.psi
        phi = self.curvature.encode_curvature_to_phase(
            mode=phase_mode, gain=phase_gain, clip_pi=True, kcrit=kcrit, prefer=prefer
        )
        rot = np.exp(1j * phi)
        amp = 1.0
        if amp_mode != "none" and amp_gain != 0.0:
            # reuse the same field for amplitude modulation
            c = self.curvature.encode_curvature_to_phase(
                mode="linear", gain=amp_gain, clip_pi=False, kcrit=kcrit, prefer=prefer
            )
            amp = 1.0 + c
        new_psi = (self.psi * rot) * amp
        if inplace:
            self.psi = new_psi
        return new_psi

    # ------------------------------ Resonance API ------------------------------
    def find_resonant_memories(self, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of topâ€‘k amplitude sites and their complex values."""
        mag = np.abs(self.psi)
        if k >= self.n:
            idx = np.argsort(mag)[::-1]
        else:
            idx = np.argpartition(mag, -k)[-k:]
            idx = idx[np.argsort(mag[idx])[::-1]]
        return idx, self.psi[idx]

    def _ensure_embeddings(self) -> np.ndarray:
        # Lightweight embedding: [Re(Ïˆ), Im(Ïˆ), |Ïˆ|, phase]
        if self._embeddings is None or self._embeddings.shape != (self.n, 4):
            p = self.psi
            ph = np.angle(p)
            self._embeddings = np.stack([p.real, p.imag, np.abs(p), ph], axis=1)
        return self._embeddings

    # ------------------------------ Stepping core ------------------------------
    def step(
        self,
        *,
        steps: int = 1,
        dt: Optional[float] = None,
        method: Literal["auto", "rk4-builtin", "rk4-ext", "stormer", "jit"] = "auto",
        conserve_mass: bool = True,
        curvature_every: Optional[int] = None,
    ) -> None:
        """Advance Ïˆ for `steps` with integrator selection.

        method:
          - auto: prefer rk4-ext if python/core/fsm_lattice_integration is present; else rk4-builtin
          - rk4-builtin: pure Python RK4 in this file
          - rk4-ext: try to import python/core/fsm_lattice_integration.py (rk4_step)
          - stormer: splitâ€‘step leapfrog (Hamiltonian then damping)
          - jit: use numbaâ€‘accelerated RK4 lane
        """
        if steps <= 0:
            return
        if dt is None:
            dt = self.dt_default

        # Optionally inject curvature every N steps
        def maybe_curvature(s: int) -> None:
            if curvature_every is None:
                return
            if s % max(1, curvature_every) == 0:
                self.inject_curvature_field(inplace=True)

        # Integrator selection
        ext = None
        if method in ("auto", "rk4-ext"):
            try:
                # Lazy import to avoid hard dependency
                from .fsm_lattice_integration import rk4_step as ext_rk4_step  # type: ignore
                ext = ext_rk4_step
            except Exception:
                ext = None

        psi = self.psi
        for s in range(1, steps + 1):
            maybe_curvature(s)
            if method == "stormer":
                psi = self._stormer_verlet_step(psi, dt)
            elif method == "jit" and HAS_NUMBA:
                Hpsi = self._hamiltonian_matvec(psi)
                psi = _rk4_step_jit(psi, dt, self.gamma, Hpsi)
            elif (method == "rk4-ext" or method == "auto") and ext is not None:
                # Ext lane computes derivative via callback
                def deriv(y: np.ndarray) -> np.ndarray:
                    return (-1j) * self._hamiltonian_matvec(y) - (self.gamma * y)
                psi = ext(y=psi, dt=dt, deriv=deriv)
            else:  # rk4-builtin
                psi = self._rk4_step_builtin(psi, dt)

            if conserve_mass:
                # Renormalize to maintain mass â‰ˆ mass0
                m = np.linalg.norm(psi)
                if m > 0:
                    psi = psi / m

        self.psi = psi
        self._update_invariants()

    def step_hamiltonian(self, *, steps: int = 1, dt: Optional[float] = None, method: str = "auto") -> None:
        self.step(steps=steps, dt=dt, method=method, conserve_mass=True, curvature_every=None)

    # ---- Builtâ€‘in RK4 (uses _hamiltonian_matvec for the Hermitian part + damping)
    def _rk4_step_builtin(self, psi: np.ndarray, dt: float) -> np.ndarray:
        def f(y: np.ndarray) -> np.ndarray:
            return (-1j) * self._hamiltonian_matvec(y) - (self.gamma * y)
        k1 = f(psi)
        k2 = f(psi + 0.5 * dt * k1)
        k3 = f(psi + 0.5 * dt * k2)
        k4 = f(psi + dt * k3)
        return psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ---- Splitâ€‘step Stormerâ€‘Verlet (leapfrog). Not strictly symplectic here due to damping.
    def _stormer_verlet_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        half = 0.5 * dt
        # Half Hamiltonian kick
        psi_half = psi + half * ((-1j) * self._hamiltonian_matvec(psi))
        # Full damping drift
        psi_damped = psi_half * np.exp(-self.gamma * dt)
        # Half Hamiltonian kick (evaluate at damped state)
        psi_next = psi_damped + half * ((-1j) * self._hamiltonian_matvec(psi_damped))
        return psi_next

    # ------------------------------ Persistence I/O ----------------------------
    def to_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(
            n=self.n,
            psi=self.psi,
            lambda_=self.lambda_,
            alpha=self.alpha,
            gamma=self.gamma,
            V0=(self.V0 if self.V0 is not None else np.zeros(self.n)),
            meta=self.meta,
        )
        with open(path, "wb") as f:
            np.savez_compressed(f, **payload)

    @classmethod
    def from_npz(cls, path: str, laplacian: Optional[Any] = None) -> "FractalSolitonMemory":
        with np.load(path, allow_pickle=True) as data:
            n = int(data["n"])  # type: ignore
            psi = data["psi"].astype(np.complex128)
            lambda_ = float(data["lambda_"])  # type: ignore
            alpha = float(data["alpha"])  # type: ignore
            gamma = float(data["gamma"])  # type: ignore
            V0 = data.get("V0", np.zeros(n)).astype(np.float64)
            meta = data.get("meta", {}).item() if "meta" in data.files else {}
        if laplacian is None:
            laplacian = np.eye(n, dtype=np.float64)
        return cls(n=n, psi=psi, laplacian=laplacian, lambda_=lambda_, alpha=alpha, gamma=gamma, V0=V0, meta=meta)

    def save_checkpoint(self, reason: str = "manual") -> None:
        self.checkpoints.push({
            "t": time.time(),
            "reason": reason,
            "psi": self.psi.copy(),
            "lambda_": self.lambda_,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "V0": (self.V0.copy() if self.V0 is not None else None),
            "mass": self._last_mass,
            "energy": self._last_energy,
            "coherence": self._last_coherence,
        })

    def restore_checkpoint(self, k: int = 0) -> None:
        snap = self.checkpoints.get(k)
        if snap is None:
            return
        self.psi = snap["psi"].copy()
        self.lambda_ = float(snap["lambda_"])
        self.alpha = float(snap["alpha"])
        self.gamma = float(snap["gamma"])
        self.V0 = (snap["V0"].copy() if snap["V0"] is not None else None)
        self._last_mass = float(snap.get("mass", 0.0))
        self._last_energy = float(snap.get("energy", 0.0))
        self._last_coherence = float(snap.get("coherence", 0.0))

    # --------------------------------- Status ----------------------------------
    def status(self) -> Dict[str, float]:
        return {
            "mass": self._last_mass,
            "energy": self._last_energy,
            "coherence": self._last_coherence,
        }

    # ------------------------------ Ricci helpers ------------------------------
    def inject_curvature_as_potential(
        self,
        *,
        prefer: Literal["kretschmann", "ricci", "mean"] = "kretschmann",
        scale: float = 0.05,
        offset: float = 0.0,
    ) -> None:
        """Fold curvature field into V0 for longerâ€‘horizon biasing."""
        if self.curvature is None:
            return
        if prefer == "kretschmann" and self.curvature.kretschmann is not None:
            c = np.asarray(self.curvature.kretschmann, dtype=np.float64)
        elif prefer == "ricci" and self.curvature.ricci_scalar is not None:
            c = np.asarray(self.curvature.ricci_scalar, dtype=np.float64)
        elif prefer == "mean" and self.curvature.mean_curvature is not None:
            c = np.asarray(self.curvature.mean_curvature, dtype=np.float64)
        else:
            return
        c = (c - np.mean(c)) / (np.std(c) + 1e-12)
        self.V0 = (self.V0 if self.V0 is not None else np.zeros(self.n)) + (scale * c + offset)

    # ------------------------------ Long evolution -----------------------------
    def evolve_lattice(
        self,
        *,
        seconds: float = 1.0,
        dt: Optional[float] = None,
        method: str = "auto",
        conserve_mass: bool = True,
        checkpoint_every_s: float = 10.0,
        curvature_every: Optional[int] = None,
    ) -> None:
        t0 = time.time()
        dt = self.dt_default if dt is None else dt
        steps_per_sec = int(max(1, 1.0 / dt))
        next_ckpt = t0 + checkpoint_every_s
        total_steps = int(seconds / dt)
        done = 0
        while done < total_steps:
            batch = min(steps_per_sec, total_steps - done)
            self.step(steps=batch, dt=dt, method=method, conserve_mass=conserve_mass, curvature_every=curvature_every)
            done += batch
            if time.time() >= next_ckpt:
                self.save_checkpoint(reason="evolve_lattice")
                next_ckpt += checkpoint_every_s


# ==============================================================================
# Convenience I/O
# ==============================================================================

def load_laplacian_from_npz(path: str) -> Any:
    with np.load(path) as data:
        if "data" in data.files and "indices" in data.files and "indptr" in data.files and "shape" in data.files:
            # scipy CSR layout
            if not HAS_SCIPY:
                raise RuntimeError("scipy required to load sparse Laplacian")
            mat = sp.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
            return mat
        elif "L" in data.files:
            return np.asarray(data["L"])  # dense
        else:
            raise ValueError("Unsupported NPZ Laplacian schema")


# ==============================================================================
# Optional: minimal test harness when executed directly
# ==============================================================================
if __name__ == "__main__":
    n = 512
    L = np.eye(n, dtype=np.float64)
    fsm = FractalSolitonMemory.from_random(n=n, laplacian=L, seed=42)
    # Fake curvature for a sanity run
    x = np.linspace(-1, 1, n)
    curv = CurvatureField(n=n, ricci_scalar=0.1 * np.sin(4 * np.pi * x), kretschmann=0.05 * np.cos(6 * np.pi * x))
    fsm.curvature = curv

    print("[FSM] Initial:", fsm.status())
    fsm.evolve_lattice(seconds=0.1, dt=1e-3, method="rk4-builtin", conserve_mass=True, curvature_every=50)
    print("[FSM] After evolve:", fsm.status())
    out = os.path.join(os.getcwd(), "fsm_state_test.npz")
    fsm.to_npz(out)
    print(f"[FSM] Saved â†’ {out}")
