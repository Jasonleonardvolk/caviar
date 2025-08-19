"""
ELFIN Learning Tools Integration

This module integrates machine learning tools with ELFIN for data-driven barrier synthesis
and verification.
"""

from .models import (
    NeuralBarrierNetwork,
    NeuralLyapunovNetwork,
    TorchBarrierNetwork,
    JAXBarrierNetwork,
)

from .training import (
    BarrierTrainer,
    LyapunovTrainer,
    DataGenerator,
    BarrierLoss,
    LyapunovLoss,
)

from .integration import (
    BenchmarkIntegration,
    export_to_elfin,
    import_from_elfin,
    verify_learned_barrier,
)

__all__ = [
    # Neural network models
    'NeuralBarrierNetwork',
    'NeuralLyapunovNetwork',
    'TorchBarrierNetwork',
    'JAXBarrierNetwork',
    
    # Training utilities
    'BarrierTrainer',
    'LyapunovTrainer',
    'DataGenerator',
    'BarrierLoss',
    'LyapunovLoss',
    
    # Integration utilities
    'BenchmarkIntegration',
    'export_to_elfin',
    'import_from_elfin',
    'verify_learned_barrier',
]
