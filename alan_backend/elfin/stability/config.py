"""
Configuration module for ELFIN Stability Framework.

This module provides configuration classes and loading utilities for the
stability framework, supporting YAML files and environment variable overrides.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
import yaml

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    import dataclasses
    BaseModel = object
    Field = lambda *args, **kwargs: None

# Configure logging
logger = logging.getLogger(__name__)

# Define configuration classes (using Pydantic if available, otherwise dataclass)
if PYDANTIC_AVAILABLE:
    class TrainerConfig(BaseModel):
        """Configuration for the Lyapunov function trainer."""
        
        lr: float = Field(1e-3, description="Learning rate for optimizer")
        gamma: float = Field(0.0, description="Margin for decrease condition: V̇(x) ≤ -gamma*||x||")
        weight_decay: float = Field(0.0, description="L2 regularization coefficient")
        max_norm: float = Field(1.0, description="Maximum norm for gradient clipping")
        use_amp: bool = Field(True, description="Whether to use Automatic Mixed Precision training")
        
    class AlphaSchedulerConfig(BaseModel):
        """Configuration for the alpha parameter scheduler."""
        
        type: Literal["exponential", "step", "warm_restart"] = Field(
            "exponential", 
            description="Type of alpha scheduler to use"
        )
        alpha0: float = Field(1e-2, description="Initial alpha value")
        min_alpha: float = Field(1e-4, description="Minimum alpha value")
        decay_rate: float = Field(0.63, description="Decay rate for exponential scheduler")
        step_size: int = Field(100, description="Steps between alpha updates for exponential scheduler")
        decay_steps: Optional[int] = Field(2000, description="Number of steps to decay from initial to min alpha")
        milestones: Optional[List[int]] = Field(None, description="Milestones for step scheduler")
        gamma: float = Field(0.1, description="Multiplicative factor for step scheduler")
        cycle_length: int = Field(1000, description="Cycle length for warm restart scheduler")
        cycle_mult: float = Field(2.0, description="Cycle length multiplier for warm restart scheduler")
        
    class SamplerConfig(BaseModel):
        """Configuration for the trajectory sampler."""
        
        batch_size: int = Field(1024, description="Batch size for sampling")
        counterexample_buffer_size: int = Field(1000, description="Maximum number of counterexamples to store")
        
    class VerifierConfig(BaseModel):
        """Configuration for the MILP verifier."""
        
        time_limit: float = Field(600.0, description="Time limit for MILP solver in seconds")
        verbose: bool = Field(False, description="Whether to show solver output")
        big_m_scale: float = Field(1.1, description="Scaling factor for big-M constants")
        epsilon: float = Field(1e-6, description="Small constant for numerical stability")
        
    class DomainConfig(BaseModel):
        """Configuration for the verification domain."""
        
        dim: int = Field(2, description="State space dimension")
        bounds: List[float] = Field([-3.0, 3.0], description="Domain bounds [low, high]")
        
    class StabilityConfig(BaseModel):
        """Root configuration for the stability framework."""
        
        trainer: TrainerConfig = Field(default_factory=TrainerConfig)
        alpha_scheduler: AlphaSchedulerConfig = Field(default_factory=AlphaSchedulerConfig)
        sampler: SamplerConfig = Field(default_factory=SamplerConfig)
        verifier: VerifierConfig = Field(default_factory=VerifierConfig)
        domain: DomainConfig = Field(default_factory=DomainConfig)
        
else:
    # Fallback to dataclasses if Pydantic not available
    @dataclasses.dataclass
    class TrainerConfig:
        """Configuration for the Lyapunov function trainer."""
        
        lr: float = 1e-3
        gamma: float = 0.0
        weight_decay: float = 0.0
        max_norm: float = 1.0
        use_amp: bool = True
        
    @dataclasses.dataclass
    class AlphaSchedulerConfig:
        """Configuration for the alpha parameter scheduler."""
        
        type: str = "exponential"
        alpha0: float = 1e-2
        min_alpha: float = 1e-4
        decay_rate: float = 0.63
        step_size: int = 100
        decay_steps: Optional[int] = 2000
        milestones: Optional[List[int]] = None
        gamma: float = 0.1
        cycle_length: int = 1000
        cycle_mult: float = 2.0
        
    @dataclasses.dataclass
    class SamplerConfig:
        """Configuration for the trajectory sampler."""
        
        batch_size: int = 1024
        counterexample_buffer_size: int = 1000
        
    @dataclasses.dataclass
    class VerifierConfig:
        """Configuration for the MILP verifier."""
        
        time_limit: float = 600.0
        verbose: bool = False
        big_m_scale: float = 1.1
        epsilon: float = 1e-6
        
    @dataclasses.dataclass
    class DomainConfig:
        """Configuration for the verification domain."""
        
        dim: int = 2
        bounds: List[float] = dataclasses.field(default_factory=lambda: [-3.0, 3.0])
        
    @dataclasses.dataclass
    class StabilityConfig:
        """Root configuration for the stability framework."""
        
        trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
        alpha_scheduler: AlphaSchedulerConfig = dataclasses.field(default_factory=AlphaSchedulerConfig)
        sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig)
        verifier: VerifierConfig = dataclasses.field(default_factory=VerifierConfig)
        domain: DomainConfig = dataclasses.field(default_factory=DomainConfig)


def load_yaml(path: Union[str, Path]) -> StabilityConfig:
    """
    Load configuration from a YAML file with environment variable overrides.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        StabilityConfig object with loaded configuration
    """
    # Check if file exists
    path = Path(path)
    if not path.exists():
        logger.warning(f"Configuration file {path} not found, using defaults")
        return StabilityConfig()
    
    # Read YAML file
    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}
    
    # Apply environment variable overrides
    apply_env_overrides(raw_config)
    
    # Create config object
    if PYDANTIC_AVAILABLE:
        return StabilityConfig(**raw_config)
    else:
        # Manual construction with dataclasses
        return construct_config(raw_config)


def apply_env_overrides(config: Dict[str, Any]) -> None:
    """
    Apply environment variable overrides to configuration.
    
    The format for environment variables is:
    ELFIN_STAB_{section}_{key}
    
    For example:
    ELFIN_STAB_TRAINER_LR=0.01
    
    Args:
        config: Configuration dictionary to modify
    """
    prefix = "ELFIN_STAB_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Get section and parameter
            parts = key[len(prefix):].lower().split('_', 1)
            if len(parts) != 2:
                continue
                
            section, param = parts
            
            # Initialize section if needed
            if section not in config:
                config[section] = {}
            
            # Convert value to appropriate type
            if value.lower() in ('true', 'yes', '1'):
                typed_value = True
            elif value.lower() in ('false', 'no', '0'):
                typed_value = False
            else:
                try:
                    # Try as int, then float
                    typed_value = int(value)
                except ValueError:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        # Keep as string
                        typed_value = value
            
            # Set value in config
            config[section][param] = typed_value
            logger.info(f"Applied environment override {key}={typed_value}")


def construct_config(config_dict: Dict[str, Any]) -> StabilityConfig:
    """
    Manually construct a config object from a dictionary when not using Pydantic.
    
    Args:
        config_dict: Dictionary containing configuration values
        
    Returns:
        StabilityConfig object
    """
    # Create sub-configs
    trainer = TrainerConfig()
    if 'trainer' in config_dict:
        for k, v in config_dict['trainer'].items():
            if hasattr(trainer, k):
                setattr(trainer, k, v)
                
    alpha_scheduler = AlphaSchedulerConfig()
    if 'alpha_scheduler' in config_dict:
        for k, v in config_dict['alpha_scheduler'].items():
            if hasattr(alpha_scheduler, k):
                setattr(alpha_scheduler, k, v)
                
    sampler = SamplerConfig()
    if 'sampler' in config_dict:
        for k, v in config_dict['sampler'].items():
            if hasattr(sampler, k):
                setattr(sampler, k, v)
                
    verifier = VerifierConfig()
    if 'verifier' in config_dict:
        for k, v in config_dict['verifier'].items():
            if hasattr(verifier, k):
                setattr(verifier, k, v)
                
    domain = DomainConfig()
    if 'domain' in config_dict:
        for k, v in config_dict['domain'].items():
            if hasattr(domain, k):
                setattr(domain, k, v)
    
    # Combine into StabilityConfig
    return StabilityConfig(
        trainer=trainer,
        alpha_scheduler=alpha_scheduler,
        sampler=sampler,
        verifier=verifier,
        domain=domain
    )


def from_global(global_config: Dict[str, Any]) -> StabilityConfig:
    """
    Create a StabilityConfig from a global config dictionary.
    
    This allows integration with a future global ELFIN config system.
    
    Args:
        global_config: Global configuration dictionary
        
    Returns:
        StabilityConfig object
    """
    # Extract stability section, or use empty dict if not present
    stability_config = global_config.get('stability', {})
    
    # Create config
    if PYDANTIC_AVAILABLE:
        return StabilityConfig(**stability_config)
    else:
        return construct_config(stability_config)
