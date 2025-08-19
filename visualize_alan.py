import numpy as np
import matplotlib.pyplot as plt
from alan_core import AlanKernel

# --- 1. Run simulation or load history ---
brain = AlanKernel()
concepts = ["filter", "map", "reduce", "loop", "aggregate", "transform"]
brain.add_concepts(concepts)
for i in range(len(concepts)):
    brain.link(concepts[i], concepts[(i+1)%len(concepts)], weight=0.5)
brain.activate("map", phase=0.5)
brain.noise = 0.05
koopman_eigvals = []
lyapunov_energy = []
phases = []
spectral_entropy = []
active_modes = []
attractor_labels = []

# Geometry parameters
geometry_modes = [(0, "Euclidean"), (1, "Spherical"), (-1, "Hyperbolic")]
geom_idx = [0]
alpha_val = [1.0]

# Simulation parameters
num_steps = 100
koopman_window = 20
ENTROPY_EPS = 1e-12
MODE_THRESH = 0.95

# For interactive morphing
selected_mode = [0]  # mutable wrapper for closure
morph_epsilon = [0.2]

# --- 2. Spectral entropy & attractor ---
def compute_entropy(eigvals):
    mags = np.abs(eigvals)
    if mags.sum() == 0:
        return 0.0, 0
    p = mags / mags.sum()
    S = -np.sum(p * np.log(p + ENTROPY_EPS))
    n_active = np.sum(mags > MODE_THRESH)
    return S, n_active

def attractor_label(S, n_active):
    if n_active == 0:
        return "Quiescent"
    elif S < 1.0:
        return "Synchronous"
    elif S > 1.5:
        return "Chaotic"
    else:
        return "Resonant"

# --- 3. Simulation loop ---
def run_sim():
    koopman_eigvals.clear()
    lyapunov_energy.clear()
    phases.clear()
    spectral_entropy.clear()
    active_modes.clear()
    attractor_labels.clear()
    brain.history.clear()
    for i, c in enumerate(concepts):
        brain.concepts[c].phase = 0.0
    brain.activate("map", phase=0.5)
    for t in range(num_steps):
        brain.step_phase()
        lyapunov_energy.append(brain.lyapunov_energy())
        phases.append([brain.concepts[n].phase for n in concepts])
        if (t+1) % koopman_window == 0:
            X_full = np.array(brain.history[-koopman_window:])
            if X_full.shape[0] >= 3:
                X = X_full[:-1]
                Xp = X_full[1:]
                try:
                    K, _, _, _ = np.linalg.lstsq(X, Xp, rcond=None)
                    eigvals, _ = np.linalg.eig(K)
                    koopman_eigvals.append(eigvals)
                    S, n_active = compute_entropy(eigvals)
                    spectral_entropy.append(S)
                    active_modes.append(n_active)
                    attractor_labels.append(attractor_label(S, n_active))
                except Exception:
                    koopman_eigvals.append([])
                    spectral_entropy.append(0.0)
                    active_modes.append(0)
                    attractor_labels.append("Error")
            else:
                koopman_eigvals.append([])
                spectral_entropy.append(0.0)
                active_modes.append(0)
                attractor_labels.append("No Koopman")

# --- Geometry/coupling update ---
def set_geometry_and_rerun():
    kappa, name = geometry_modes[geom_idx[0] % len(geometry_modes)]
    brain.set_geometry(kappa=kappa, alpha=alpha_val[0])
    print(f"[UI] Geometry set to {name}, α={alpha_val[0]:.2f}")
    run_sim()
    global phases_np
    phases_np = np.array(phases)
    update_plots()

# --- Initial run ---
set_geometry_and_rerun()

# --- Visualization setup ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax_spec = axes[0, 0]
ax_lyap = axes[0, 1]
ax_phase = axes[1, 0]
ax_entropy = axes[1, 1]
ax_coords = axes[0, 2]

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

# --- Plotting function ---
def update_plots():
    # Koopman Spectrum
    ax_spec.clear()
    ax_spec.set_title("Koopman Spectrum (Complex Plane)")
    ax_spec.set_xlabel("Re")
    ax_spec.set_ylabel("Im")
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax_spec.add_artist(unit_circle)
    for i, eigs in enumerate(koopman_eigvals):
        if len(eigs) > 0:
            ax_spec.scatter(eigs.real, eigs.imag, color=colors[i % len(colors)], label=f"t={koopman_window*(i+1)}")
    ax_spec.legend()
    ax_spec.set_aspect('equal')
    ax_spec.set_xlim(-1.2, 1.2)
    ax_spec.set_ylim(-1.2, 1.2)

    # Lyapunov Energy
    ax_lyap.clear()
    ax_lyap.plot(lyapunov_energy, label="Lyapunov Energy", color='C1')
    ax_lyap.set_title("Lyapunov Energy Over Time")
    ax_lyap.set_xlabel("Step")
    ax_lyap.set_ylabel("V(θ)")
    ax_lyap.legend()

    # Phase Trajectories
    ax_phase.clear()
    for i, name in enumerate(concepts):
        ax_phase.plot(phases_np[:, i], label=name)
    ax_phase.set_title("Phase Trajectories (Concepts)")
    ax_phase.set_xlabel("Step")
    ax_phase.set_ylabel("Phase θ")
    ax_phase.legend()

    # Spectral Entropy & Attractor
    ax_entropy.clear()
    ax_entropy.plot(spectral_entropy, label="Spectral Entropy", color='C3')
    ax_entropy.set_title("Spectral Entropy & Attractor State")
    ax_entropy.set_xlabel("Window")
    ax_entropy.set_ylabel("Spectral Entropy S")
    for i, label in enumerate(attractor_labels):
        ax_entropy.annotate(label, (i, spectral_entropy[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black', rotation=30)
    ax_entropy.legend()

    # Coordinate Plot Inset
    ax_coords.clear()
    ax_coords.set_title("Concept Coordinates (Phase Color, Coupling Lines)")
    # Plot edges/coupling
    for i, ni in enumerate(concepts):
        xi = brain.concepts[ni].coord
        for j, nj in enumerate(concepts):
            if i != j:
                xj = brain.concepts[nj].coord
                K = brain.concepts[ni].neighbors.get(nj, 0)
                if xi is not None and xj is not None and K > 0.01:
                    ax_coords.plot([xi[0], xj[0]], [xi[1], xj[1]], color='gray', alpha=min(0.8, K), linewidth=2*K)
    # Plot nodes (color by phase)
    phases_now = [brain.concepts[n].phase for n in concepts]
    norm = plt.Normalize(0, 2*np.pi)
    for i, n in enumerate(concepts):
        c = brain.concepts[n]
        ax_coords.scatter(c.coord[0], c.coord[1], color=plt.cm.hsv(norm(c.phase)), s=150, edgecolor='k', zorder=3)
        ax_coords.text(c.coord[0], c.coord[1], n, fontsize=10, ha='center', va='center', color='white', zorder=4)
    ax_coords.set_aspect('equal')
    ax_coords.set_xlim(-1.2, 1.2)
    ax_coords.set_ylim(-1.2, 1.2)
    ax_coords.axis('off')

    kappa, name = geometry_modes[geom_idx[0] % len(geometry_modes)]
    fig.suptitle(f"κ-Geometry: {name} | α={alpha_val[0]:.2f} | 'g'=cycle geometry, 'a/z'=α+/-, 'r'=regen coords | 'm'=morph", fontsize=13)
    plt.draw()

# --- Key handler ---
def on_key(event):
    if event.key == 'm':
        print(f"[UI] Morphing along Koopman mode {selected_mode[0]} (ε={morph_epsilon[0]})")
        brain.koopman_decompose(window=koopman_window)
        brain.morph_concept_field(mode_index=selected_mode[0], epsilon=morph_epsilon[0])
        run_sim()
        global phases_np
        phases_np = np.array(phases)
        update_plots()
        plt.draw()
    elif event.key == 'left':
        selected_mode[0] = max(0, selected_mode[0] - 1)
        print(f"[UI] Selected Koopman mode: {selected_mode[0]}")
        update_plots()
    elif event.key == 'right':
        selected_mode[0] += 1
        print(f"[UI] Selected Koopman mode: {selected_mode[0]}")
        update_plots()
    elif event.key == '+':
        morph_epsilon[0] += 0.05
        print(f"[UI] Morph epsilon: {morph_epsilon[0]:.2f}")
        update_plots()
    elif event.key == '-':
        morph_epsilon[0] = max(0.01, morph_epsilon[0] - 0.05)
        print(f"[UI] Morph epsilon: {morph_epsilon[0]:.2f}")
        update_plots()
    elif event.key == 'g':
        geom_idx[0] = (geom_idx[0] + 1) % len(geometry_modes)
        set_geometry_and_rerun()
    elif event.key == 'a':
        alpha_val[0] += 0.1
        set_geometry_and_rerun()
    elif event.key == 'z':
        alpha_val[0] = max(0.1, alpha_val[0] - 0.1)
        set_geometry_and_rerun()
    elif event.key == 'r':
        # Regenerate coordinates in current geometry
        kappa, name = geometry_modes[geom_idx[0] % len(geometry_modes)]
        brain.assign_random_coords()
        brain.update_coupling_matrix()
        run_sim()
        print(f"[UI] Regenerated coordinates for {name}")
        global phases_np
        phases_np = np.array(phases)
        update_plots()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()
