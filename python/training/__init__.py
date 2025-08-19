"""
TORI/Saigon Training Module
============================
Adapter training, validation, and data generation.
"""

from .train_lora_adapter_v5 import (
    AdapterTrainer,
    train_lora_adapter
)

from .synthetic_data_generator import (
    SyntheticDataGenerator,
    generate_synthetic_data
)

from .validate_adapter import (
    AdapterValidator,
    validate_adapter
)

from .rollback_adapter import (
    RollbackManager,
    rollback_adapter,
    emergency_rollback_all_users
)

__version__ = "5.0.0"

__all__ = [
    # Training
    "AdapterTrainer",
    "train_lora_adapter",
    
    # Data Generation
    "SyntheticDataGenerator",
    "generate_synthetic_data",
    
    # Validation
    "AdapterValidator",
    "validate_adapter",
    
    # Rollback
    "RollbackManager",
    "rollback_adapter",
    "emergency_rollback_all_users"
]
