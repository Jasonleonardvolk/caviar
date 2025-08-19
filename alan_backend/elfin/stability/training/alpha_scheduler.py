"""
Alpha scheduler for Lyapunov Network training.

This module provides schedulers for the alpha parameter in LyapunovNet,
which controls the weight of the norm term in the Lyapunov function.
Starting with a larger alpha value provides better regularization in early
training, while gradually decreasing it allows for tighter level sets and
a more accurate representation of the region of attraction.
"""

import logging
import math
from typing import Optional, Dict, List, Callable, Any

# Configure logging
logger = logging.getLogger(__name__)

class AlphaScheduler:
    """
    Base class for alpha schedulers.
    
    Alpha schedulers control the alpha parameter in LyapunovNet over the
    course of training. Alpha is the weight of the norm term in the Lyapunov
    function: V(x) = |phi(x) - phi(0)| + alpha*||x||.
    """
    
    def __init__(self, initial_alpha: float = 1e-2, min_alpha: float = 1e-3):
        """
        Initialize the AlphaScheduler.
        
        Args:
            initial_alpha: Initial value of alpha
            min_alpha: Minimum value of alpha (to prevent degeneration)
        """
        if initial_alpha <= 0 or min_alpha <= 0:
            raise ValueError("Alpha values must be positive")
            
        if min_alpha > initial_alpha:
            raise ValueError("min_alpha must be <= initial_alpha")
            
        self.initial_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.current_alpha = initial_alpha
        self.step_count = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with "
                   f"initial_alpha={initial_alpha}, min_alpha={min_alpha}")
    
    def step(self) -> float:
        """
        Update alpha value based on the schedule.
        
        Returns:
            The new alpha value
        """
        self.step_count += 1
        return self.current_alpha
    
    def get_alpha(self) -> float:
        """
        Get the current alpha value without updating.
        
        Returns:
            The current alpha value
        """
        return self.current_alpha
    
    def reset(self) -> None:
        """Reset the scheduler to its initial state."""
        self.current_alpha = self.initial_alpha
        self.step_count = 0
        
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary for checkpoint.
        
        Returns:
            State dictionary with scheduler state
        """
        return {
            'initial_alpha': self.initial_alpha,
            'min_alpha': self.min_alpha,
            'current_alpha': self.current_alpha,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from a state dictionary.
        
        Args:
            state_dict: State dictionary with scheduler state
        """
        self.initial_alpha = state_dict.get('initial_alpha', self.initial_alpha)
        self.min_alpha = state_dict.get('min_alpha', self.min_alpha)
        self.current_alpha = state_dict.get('current_alpha', self.initial_alpha)
        self.step_count = state_dict.get('step_count', 0)


class ExponentialAlphaScheduler(AlphaScheduler):
    """
    Exponential decay scheduler for alpha.
    
    Decays alpha exponentially from initial_alpha to min_alpha over
    a specified number of steps or with a given decay rate.
    """
    
    def __init__(
        self, 
        initial_alpha: float = 1e-2, 
        min_alpha: float = 1e-3,
        decay_steps: Optional[int] = None,
        decay_rate: float = 0.95,
        step_size: int = 100
    ):
        """
        Initialize the exponential decay scheduler.
        
        Args:
            initial_alpha: Initial value of alpha
            min_alpha: Minimum value of alpha
            decay_steps: Number of steps to decay from initial to min alpha.
                        If None, will use decay_rate instead.
            decay_rate: Rate of exponential decay to apply every step_size steps.
                        Only used if decay_steps is None.
            step_size: Number of steps between decay applications
        """
        super().__init__(initial_alpha, min_alpha)
        
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_size = step_size
        
        # Calculate decay rate if decay_steps is provided
        if decay_steps is not None:
            # Solve for decay_rate: min_alpha = initial_alpha * decay_rate^(decay_steps/step_size)
            # decay_rate = (min_alpha / initial_alpha)^(step_size/decay_steps)
            power = step_size / decay_steps
            self.decay_rate = (min_alpha / initial_alpha) ** power
        
        logger.info(f"Exponential alpha scheduler with decay_rate={self.decay_rate}, "
                   f"step_size={step_size}")
    
    def step(self) -> float:
        """
        Update alpha value with exponential decay.
        
        Returns:
            The new alpha value
        """
        self.step_count += 1
        
        # Apply decay every step_size steps
        if self.step_count % self.step_size == 0:
            self.current_alpha = max(
                self.min_alpha,
                self.current_alpha * self.decay_rate
            )
            
            logger.debug(f"Alpha updated to {self.current_alpha:.6f} at step {self.step_count}")
        
        return self.current_alpha


class WarmRestartAlphaScheduler(AlphaScheduler):
    """
    Alpha scheduler with cosine annealing and warm restarts.
    
    Implements a cosine annealing schedule with warm restarts,
    similar to SGDR (Stochastic Gradient Descent with Warm Restarts).
    """
    
    def __init__(
        self, 
        initial_alpha: float = 1e-2, 
        min_alpha: float = 1e-3,
        cycle_length: int = 1000,
        cycle_mult: float = 2.0
    ):
        """
        Initialize the warm restart scheduler.
        
        Args:
            initial_alpha: Initial and maximum value of alpha
            min_alpha: Minimum value of alpha
            cycle_length: Initial length of each cycle (in steps)
            cycle_mult: Multiplier for cycle length after each restart
        """
        super().__init__(initial_alpha, min_alpha)
        
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.cycle_count = 0
        self.cycle_step = 0
        
        logger.info(f"Warm restart alpha scheduler with cycle_length={cycle_length}, "
                   f"cycle_mult={cycle_mult}")
    
    def step(self) -> float:
        """
        Update alpha value with cosine annealing and warm restarts.
        
        Returns:
            The new alpha value
        """
        self.step_count += 1
        self.cycle_step += 1
        
        # Check if we need to restart
        current_cycle_length = self.cycle_length * (self.cycle_mult ** self.cycle_count)
        if self.cycle_step > current_cycle_length:
            # Start a new cycle
            self.cycle_count += 1
            self.cycle_step = 1
            current_cycle_length = self.cycle_length * (self.cycle_mult ** self.cycle_count)
            logger.debug(f"Alpha warm restart: cycle {self.cycle_count}, "
                        f"length {current_cycle_length}")
        
        # Calculate alpha with cosine annealing
        # alpha = min_alpha + 0.5 * (initial_alpha - min_alpha) * (1 + cos(pi * t / T))
        # where t is the current step in the cycle, T is the cycle length
        cosine_term = math.cos(math.pi * self.cycle_step / current_cycle_length)
        self.current_alpha = self.min_alpha + 0.5 * (self.initial_alpha - self.min_alpha) * (1 + cosine_term)
        
        return self.current_alpha
    
    def reset(self) -> None:
        """Reset the scheduler to its initial state."""
        super().reset()
        self.cycle_count = 0
        self.cycle_step = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary with scheduler state."""
        state = super().state_dict()
        state.update({
            'cycle_length': self.cycle_length,
            'cycle_mult': self.cycle_mult,
            'cycle_count': self.cycle_count,
            'cycle_step': self.cycle_step
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from a state dictionary."""
        super().load_state_dict(state_dict)
        self.cycle_length = state_dict.get('cycle_length', self.cycle_length)
        self.cycle_mult = state_dict.get('cycle_mult', self.cycle_mult)
        self.cycle_count = state_dict.get('cycle_count', 0)
        self.cycle_step = state_dict.get('cycle_step', 0)


class StepAlphaScheduler(AlphaScheduler):
    """
    Step-based alpha scheduler.
    
    Decays alpha by a factor at specified milestone steps.
    """
    
    def __init__(
        self, 
        initial_alpha: float = 1e-2, 
        min_alpha: float = 1e-3,
        milestones: List[int] = None,
        gamma: float = 0.1
    ):
        """
        Initialize the step scheduler.
        
        Args:
            initial_alpha: Initial value of alpha
            min_alpha: Minimum value of alpha
            milestones: List of step indices at which to decay alpha
            gamma: Multiplicative factor by which to decay alpha at each milestone
        """
        super().__init__(initial_alpha, min_alpha)
        
        self.milestones = milestones or [500, 1000, 2000]
        self.gamma = gamma
        
        # Internal state for milestone tracking
        self._milestones_passed = 0
        
        logger.info(f"Step alpha scheduler with milestones={self.milestones}, gamma={gamma}")
    
    def step(self) -> float:
        """
        Update alpha value at specified milestones.
        
        Returns:
            The new alpha value
        """
        self.step_count += 1
        
        # Check if we've reached a milestone
        if self._milestones_passed < len(self.milestones) and self.step_count >= self.milestones[self._milestones_passed]:
            # Apply decay
            self.current_alpha = max(
                self.min_alpha,
                self.current_alpha * self.gamma
            )
            
            logger.debug(f"Alpha decayed to {self.current_alpha:.6f} at milestone "
                        f"{self.milestones[self._milestones_passed]}")
            
            self._milestones_passed += 1
        
        return self.current_alpha
    
    def reset(self) -> None:
        """Reset the scheduler to its initial state."""
        super().reset()
        self._milestones_passed = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary with scheduler state."""
        state = super().state_dict()
        state.update({
            'milestones': self.milestones,
            'gamma': self.gamma,
            '_milestones_passed': self._milestones_passed
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from a state dictionary."""
        super().load_state_dict(state_dict)
        self.milestones = state_dict.get('milestones', self.milestones)
        self.gamma = state_dict.get('gamma', self.gamma)
        self._milestones_passed = state_dict.get('_milestones_passed', 0)
