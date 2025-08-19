"""
Time-Reversal Symmetric ODE Controller for ALAN.

This module implements a velocity-Verlet (symplectic, order 2) integrator
for time-reversible dynamics, which enables training with Time-Reversal
Symmetry (TRS) loss as described in Huh et al.

The controller provides:
1. Forward integration of hidden state h(t) with adjoint momentum p(t)
2. Reverse integration from t=T→0 for TRS loss computation
3. Interfaces for gradient-based training with PyTorch
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)


class TRSConfig:
    """Configuration for the TRS-ODE controller."""
    
    def __init__(
        self, 
        dt: float = 0.01, 
        trs_weight: float = 0.1,
        train_steps: int = 100,
        eval_steps: int = 200,
    ):
        """Initialize TRS configuration.
        
        Args:
            dt: Time step for integration
            trs_weight: Weight for the TRS loss component (λ_trs)
            train_steps: Number of integration steps during training
            eval_steps: Number of integration steps during evaluation
        """
        self.dt = dt
        self.trs_weight = trs_weight
        self.train_steps = train_steps
        self.eval_steps = eval_steps


class State:
    """State container for position (h) and momentum (p) vectors."""
    
    def __init__(self, h: np.ndarray, p: Optional[np.ndarray] = None):
        """Initialize state with position and optional momentum.
        
        Args:
            h: Position/hidden state vector
            p: Momentum vector (if None, initialized to zeros)
        """
        self.h = h.copy()
        
        if p is None:
            self.p = np.zeros_like(h)
        else:
            if p.shape != h.shape:
                raise ValueError(f"Position and momentum must have same shape, got {h.shape} and {p.shape}")
            self.p = p.copy()
    
    @property
    def dim(self) -> int:
        """Get the dimension of the state vectors."""
        return len(self.h)
    
    def copy(self) -> 'State':
        """Create a deep copy of this state."""
        return State(self.h.copy(), self.p.copy())
    
    def reverse_momentum(self) -> 'State':
        """Return a state with reversed momentum (for time reversal)."""
        return State(self.h.copy(), -self.p.copy())
    
    def __repr__(self) -> str:
        """String representation of the state."""
        return f"State(h={self.h}, p={self.p})"


class VectorField:
    """Abstract base class for vector fields used in TRS-ODE."""
    
    def evaluate(self, state: State) -> np.ndarray:
        """Evaluate the vector field at the given state.
        
        Args:
            state: Current state (h, p)
            
        Returns:
            dh_dt: Time derivative of the hidden state
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def gradient(self, state: State) -> np.ndarray:
        """Compute the gradient of potential energy w.r.t. position.
        
        This drives the momentum update in Hamiltonian systems.
        
        Args:
            state: Current state (h, p)
            
        Returns:
            grad_V: Gradient of potential energy V(h)
        """
        raise NotImplementedError("Subclasses must implement gradient()")


class ODEIntegrator:
    """Abstract base class for ODE integrators."""
    
    def step(self, state: State, vector_field: VectorField, dt: float) -> State:
        """Take a single time step.
        
        Args:
            state: Current state
            vector_field: Vector field defining the dynamics
            dt: Time step
            
        Returns:
            next_state: State after integration step
        """
        raise NotImplementedError("Subclasses must implement step()")
    
    def integrate(
        self, 
        initial_state: State, 
        vector_field: VectorField, 
        dt: float, 
        n_steps: int,
        store_trajectory: bool = False,
    ) -> Union[State, Tuple[State, List[State]]]:
        """Integrate the system forward in time.
        
        Args:
            initial_state: Starting state
            vector_field: Vector field defining the dynamics
            dt: Time step
            n_steps: Number of steps to take
            store_trajectory: Whether to store and return the full trajectory
            
        Returns:
            If store_trajectory is False:
                final_state: State after integration
            If store_trajectory is True:
                (final_state, trajectory): Final state and list of states at each step
        """
        state = initial_state.copy()
        trajectory = [state.copy()] if store_trajectory else None
        
        for _ in range(n_steps):
            state = self.step(state, vector_field, dt)
            if store_trajectory:
                trajectory.append(state.copy())
        
        if store_trajectory:
            return state, trajectory
        return state


class VerletIntegrator(ODEIntegrator):
    """Velocity-Verlet symplectic integrator (2nd order).
    
    This integrator preserves the symplectic structure of Hamiltonian systems,
    and is time-reversible, making it ideal for TRS applications.
    
    Attributes:
        inv_mass: Inverse mass matrix for position updates. If None, identity is used.
        direction: Integration direction (1 for forward, -1 for backward).
    """
    
    def __init__(self, inv_mass: Optional[np.ndarray] = None, direction: int = 1):
        """Initialize the Verlet integrator.
        
        Args:
            inv_mass: Inverse mass matrix M^{-1} for position updates. If None, identity is used.
            direction: Integration direction (1 for forward, -1 for backward).
        """
        self.inv_mass = inv_mass  # Inverse mass matrix (M^{-1})
        self.direction = direction  # Integration direction (1 for forward, -1 for backward)
    
    def set_direction(self, direction: int) -> None:
        """Set the integration direction.
        
        Args:
            direction: 1 for forward, -1 for backward
            
        Raises:
            ValueError: If direction is not 1 or -1
        """
        if direction not in [1, -1]:
            raise ValueError("Direction must be 1 (forward) or -1 (backward)")
        self.direction = direction
    
    def step(self, state: State, vector_field: VectorField, dt: float) -> State:
        """Take a single time step using the velocity-Verlet method.
        
        Implements the velocity-Verlet scheme:
        p_{n+½} = p_n − (Δt/2) ∇_x H(x_n)
        x_{n+1} = x_n + Δt M^{-1}p_{n+½}
        p_{n+1} = p_{n+½} − (Δt/2) ∇_x H(x_{n+1})
        
        Args:
            state: Current state
            vector_field: Vector field defining the dynamics
            dt: Time step
            
        Returns:
            next_state: State after integration step
        """
        # Adjust dt based on direction
        effective_dt = self.direction * dt
        
        # 1. Half-step momentum update
        p_half = state.p - 0.5 * effective_dt * vector_field.gradient(state)
        
        # 2. Full-step position update using half-updated momentum and inverse mass
        if self.inv_mass is not None:
            # Apply inverse mass matrix: h_next = h + dt * M^{-1} * p_half
            h_next = state.h + effective_dt * np.dot(self.inv_mass, p_half)
        else:
            # No mass matrix (equivalent to identity): h_next = h + dt * p_half
            h_next = state.h + effective_dt * p_half
        
        next_state = State(h_next, p_half)
        
        # 3. Half-step momentum update with new position
        p_next = p_half - 0.5 * effective_dt * vector_field.gradient(next_state)
        next_state.p = p_next
        
        return next_state


class TRSController:
    """Time-Reversal Symmetric ODE Controller.
    
    This controller implements the Time-Reversal Symmetry (TRS) approach
    for robust sequence modeling, based on Hamiltonian dynamics.
    """
    
    def __init__(
        self, 
        state_dim: int,
        vector_field: Optional[VectorField] = None,
        config: Optional[TRSConfig] = None,
        integrator: Optional[ODEIntegrator] = None,
    ):
        """Initialize the TRS controller.
        
        Args:
            state_dim: Dimension of the hidden state
            vector_field: Vector field defining the dynamics (can be set later)
            config: TRS configuration
            integrator: ODE integrator (defaults to VerletIntegrator)
        """
        self.state_dim = state_dim
        self.vector_field = vector_field
        self.config = config or TRSConfig()
        self.integrator = integrator or VerletIntegrator()
    
    def forward_integrate(
        self, 
        initial_state: Union[State, np.ndarray], 
        n_steps: Optional[int] = None,
        store_trajectory: bool = False,
    ) -> Union[State, Tuple[State, List[State]]]:
        """Integrate the system forward in time.
        
        Args:
            initial_state: Starting state (or just h vector, p will be zeros)
            n_steps: Number of steps (defaults to config.train_steps)
            store_trajectory: Whether to store and return the full trajectory
            
        Returns:
            If store_trajectory is False:
                final_state: State after forward integration
            If store_trajectory is True:
                (final_state, trajectory): Final state and list of states at each step
                
        Raises:
            ValueError: If no vector field has been set
        """
        if self.vector_field is None:
            raise ValueError("No vector field has been set")
        
        # Convert to State if needed
        if not isinstance(initial_state, State):
            initial_state = State(initial_state)
        
        if n_steps is None:
            n_steps = self.config.train_steps
        
        return self.integrator.integrate(
            initial_state, 
            self.vector_field, 
            self.config.dt, 
            n_steps,
            store_trajectory,
        )
    
    def reverse_integrate(
        self, 
        final_state: State, 
        n_steps: Optional[int] = None,
        store_trajectory: bool = False,
    ) -> Union[State, Tuple[State, List[State]]]:
        """Integrate the system backward in time.
        
        This is done by reversing the momentum and integrating forward,
        then reversing the momentum again in the final state.
        
        Args:
            final_state: State to reverse from
            n_steps: Number of steps (defaults to config.train_steps)
            store_trajectory: Whether to store and return the full trajectory
            
        Returns:
            If store_trajectory is False:
                reversed_initial_state: State after reverse integration
            If store_trajectory is True:
                (reversed_initial_state, trajectory): Final state and list of states
                
        Raises:
            ValueError: If no vector field has been set
        """
        if self.vector_field is None:
            raise ValueError("No vector field has been set")
        
        if n_steps is None:
            n_steps = self.config.train_steps
        
        # Reverse momentum for backward integration
        reversed_state = final_state.reverse_momentum()
        
        # Integrate forward with reversed momentum
        if store_trajectory:
            result, trajectory = self.integrator.integrate(
                reversed_state,
                self.vector_field,
                self.config.dt,
                n_steps,
                True,
            )
            # Reverse momentum in all trajectory states
            reversed_trajectory = [s.reverse_momentum() for s in trajectory]
            # Return the final state with reversed momentum and the reversed trajectory
            return result.reverse_momentum(), reversed_trajectory
        else:
            result = self.integrator.integrate(
                reversed_state,
                self.vector_field,
                self.config.dt,
                n_steps,
                False,
            )
            # Reverse momentum in the final state
            return result.reverse_momentum()
    
    def compute_trs_loss(self, initial_state: State, final_state: State, reversed_initial_state: State) -> float:
        """Compute the dimensionless TRS loss between initial and reversed initial states.
        
        The TRS loss measures how well the system preserves time-reversal symmetry:
        L_trs = (‖ĥ(0) – h(0)‖² + ‖p̂(0) + p(0)‖²) / dim
        
        This uses mean-square error normalization to ensure the loss is comparable
        across different state dimensions, preventing large-dimensional states
        from artificially appearing better.
        
        Args:
            initial_state: Original initial state
            final_state: State after forward integration
            reversed_initial_state: State after reverse integration
            
        Returns:
            trs_loss: The dimensionless TRS loss value
        """
        h_diff = reversed_initial_state.h - initial_state.h
        p_diff = reversed_initial_state.p + initial_state.p  # Note the + sign here
        
        # Calculate L2 squared norms
        h_diff_sq = np.sum(h_diff ** 2)
        p_diff_sq = np.sum(p_diff ** 2)
        
        # Get dimension for proper scaling
        dim = initial_state.dim
        
        # Compute mean-square error (properly scaled by dimension)
        trs_loss = (h_diff_sq + p_diff_sq) / dim
        
        return trs_loss
    
    def simulate_with_trs(self, initial_state: Union[State, np.ndarray]) -> Dict[str, Any]:
        """Run a complete simulation with forward and backward integration.
        
        This computes both the forward dynamics and the TRS loss.
        
        Args:
            initial_state: Starting state (or just h vector, p will be zeros)
            
        Returns:
            results: Dictionary containing:
                - forward_final: Final state after forward integration
                - backward_final: State after reverse integration
                - trs_loss: The TRS loss value
                - forward_trajectory: List of states during forward integration
                - backward_trajectory: List of states during backward integration
        """
        # Convert to State if needed
        if not isinstance(initial_state, State):
            initial_state = State(initial_state)
        
        # Forward integration
        forward_final, forward_trajectory = self.forward_integrate(
            initial_state, 
            store_trajectory=True,
        )
        
        # Backward integration
        backward_final, backward_trajectory = self.reverse_integrate(
            forward_final,
            store_trajectory=True,
        )
        
        # Compute TRS loss
        trs_loss = self.compute_trs_loss(initial_state, forward_final, backward_final)
        
        return {
            'forward_final': forward_final,
            'backward_final': backward_final,
            'trs_loss': trs_loss,
            'forward_trajectory': forward_trajectory,
            'backward_trajectory': backward_trajectory,
        }


# Example vector field implementations

class HarmonicOscillator(VectorField):
    """A simple harmonic oscillator vector field.
    
    This implements V(h) = k/2 * h^2, which results in simple harmonic motion.
    """
    
    def __init__(self, k: float = 1.0):
        """Initialize the harmonic oscillator.
        
        Args:
            k: Spring constant
        """
        self.k = k
    
    def evaluate(self, state: State) -> np.ndarray:
        """Evaluate dh/dt = p."""
        return state.p
    
    def gradient(self, state: State) -> np.ndarray:
        """Compute gradient of V(h) = k/2 * h^2, which is k*h."""
        return self.k * state.h


class DuffingOscillator(VectorField):
    """A Duffing oscillator vector field (nonlinear).
    
    This implements V(h) = -a/2 * h^2 + b/4 * h^4, which results in
    a double-well potential with chaotic dynamics.
    """
    
    def __init__(self, a: float = 1.0, b: float = 1.0, delta: float = 0.2):
        """Initialize the Duffing oscillator.
        
        Args:
            a: Coefficient of quadratic term (negative for double well)
            b: Coefficient of quartic term (must be positive)
            delta: Damping coefficient
        """
        self.a = a
        self.b = b
        self.delta = delta
    
    def evaluate(self, state: State) -> np.ndarray:
        """Evaluate dh/dt = p."""
        return state.p
    
    def gradient(self, state: State) -> np.ndarray:
        """Compute gradient of V(h) = -a/2 * h^2 + b/4 * h^4."""
        # Force = -∇V = a*h - b*h^3
        force = self.a * state.h - self.b * state.h**3
        
        # Add damping term (-delta * p)
        damping = self.delta * state.p
        
        # Return -F - damping (negative because gradient is -F)
        return -force + damping


class CoupledOscillator(VectorField):
    """A system of coupled oscillators.
    
    This implements a network of oscillators with pairwise coupling.
    """
    
    def __init__(self, coupling_matrix: np.ndarray, k: float = 1.0):
        """Initialize the coupled oscillator system.
        
        Args:
            coupling_matrix: Matrix of coupling strengths between oscillators
            k: Base spring constant
        """
        self.coupling = coupling_matrix
        self.k = k
        
        if coupling_matrix.shape[0] != coupling_matrix.shape[1]:
            raise ValueError("Coupling matrix must be square")
    
    def evaluate(self, state: State) -> np.ndarray:
        """Evaluate dh/dt = p."""
        return state.p
    
    def gradient(self, state: State) -> np.ndarray:
        """Compute gradient for the coupled oscillator potential."""
        # Individual oscillator forces
        individual = self.k * state.h
        
        # Coupling forces
        coupling = np.zeros_like(state.h)
        for i in range(len(state.h)):
            for j in range(len(state.h)):
                if i != j:
                    coupling[i] += self.coupling[i, j] * (state.h[i] - state.h[j])
        
        return individual + coupling


if __name__ == "__main__":
    # Simple demonstration of the TRS controller
    import matplotlib.pyplot as plt
    
    # Create a double-well Duffing oscillator
    vector_field = DuffingOscillator(a=1.0, b=0.3, delta=0.01)
    
    # Create a TRS controller
    config = TRSConfig(dt=0.05, train_steps=400)
    controller = TRSController(
        state_dim=1, 
        vector_field=vector_field,
        config=config,
    )
    
    # Define an initial state slightly off-center
    initial_state = State(np.array([0.5]), np.array([0.0]))
    
    # Run simulation
    results = controller.simulate_with_trs(initial_state)
    
    # Extract trajectories
    forward_traj = results['forward_trajectory']
    backward_traj = results['backward_trajectory']
    
    # Convert to numpy arrays for plotting
    forward_h = np.array([s.h[0] for s in forward_traj])
    forward_p = np.array([s.p[0] for s in forward_traj])
    backward_h = np.array([s.h[0] for s in backward_traj])
    backward_p = np.array([s.p[0] for s in backward_traj])
    time = np.arange(len(forward_traj)) * config.dt
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Position vs time
    plt.subplot(2, 2, 1)
    plt.plot(time, forward_h, 'b-', label='Forward h(t)')
    plt.plot(time, backward_h, 'r--', label='Backward h(t)')
    plt.xlabel('Time')
    plt.ylabel('Position h')
    plt.legend()
    plt.title('Position over Time')
    
    # Momentum vs time
    plt.subplot(2, 2, 2)
    plt.plot(time, forward_p, 'b-', label='Forward p(t)')
    plt.plot(time, backward_p, 'r--', label='Backward p(t)')
    plt.xlabel('Time')
    plt.ylabel('Momentum p')
    plt.legend()
    plt.title('Momentum over Time')
    
    # Phase space
    plt.subplot(2, 2, 3)
    plt.plot(forward_h, forward_p, 'b-', label='Forward')
    plt.plot(backward_h, backward_p, 'r--', label='Backward')
    plt.scatter(forward_h[0], forward_p[0], c='g', s=100, marker='o', label='Start')
    plt.scatter(forward_h[-1], forward_p[-1], c='m', s=100, marker='x', label='End')
    plt.xlabel('Position h')
    plt.ylabel('Momentum p')
    plt.legend()
    plt.title('Phase Space Portrait')
    
    # Error between forward and backward trajectories
    plt.subplot(2, 2, 4)
    h_error = np.abs(backward_h - forward_h[::-1])
    p_error = np.abs(backward_p - forward_p[::-1])
    plt.semilogy(time, h_error, 'g-', label='|h_back - h_forward|')
    plt.semilogy(time, p_error, 'm-', label='|p_back - p_forward|')
    plt.xlabel('Time')
    plt.ylabel('Error (log scale)')
    plt.legend()
    plt.title(f'TRS Error (Loss = {results["trs_loss"]:.6f})')
    
    plt.tight_layout()
    plt.show()
    
    print(f"TRS Loss: {results['trs_loss']:.6f}")
