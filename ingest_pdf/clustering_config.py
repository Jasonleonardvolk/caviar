"""
TORI Clustering Configuration Management
Centralized configuration for clustering parameters, thresholds, and system settings.
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class OscillatorConfig:
    """Configuration for oscillator clustering."""
    steps: int = 60
    tolerance: float = 1e-3
    cohesion_threshold: float = 0.15
    phase_bins: int = 25  # Number of phase bins (2π / 0.25)
    singleton_merge_threshold: float = 0.4
    orphan_reassign_threshold: float = 0.3
    max_iterations: int = 100

@dataclass
class KMeansConfig:
    """Configuration for K-means clustering."""
    auto_k: bool = True
    k: Optional[int] = None
    max_iterations: int = 100
    n_init: int = 10
    tol: float = 1e-4
    
@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering."""
    min_cluster_size: int = 2
    min_samples: Optional[int] = None
    metric: str = 'cosine'
    alpha: float = 1.0
    cluster_selection_epsilon: float = 0.0

@dataclass
class AffinityPropagationConfig:
    """Configuration for Affinity Propagation clustering."""
    preference: Optional[float] = None
    convergence_iter: int = 15
    max_iter: int = 200
    damping: float = 0.5

@dataclass
class BenchmarkConfig:
    """Configuration for clustering benchmarking."""
    enabled_methods: List[str] = field(default_factory=lambda: ['oscillator', 'kmeans', 'hdbscan'])
    compute_silhouette: bool = True
    compute_ground_truth_metrics: bool = False
    save_results: bool = True
    enable_logging: bool = True
    parallel_execution: bool = True

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    batch_size: int = 1000
    max_memory_mb: int = 1000
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    use_multiprocessing: bool = True
    max_workers: int = 4

@dataclass
class QualityConfig:
    """Configuration for clustering quality assessment."""
    min_cohesion: float = 0.25
    min_silhouette: float = 0.2
    max_singleton_ratio: float = 0.4
    min_cluster_size: int = 2
    quality_weight_cohesion: float = 0.6
    quality_weight_silhouette: float = 0.4

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    enable_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_cohesion': 0.25,
        'min_silhouette': 0.2,
        'max_runtime_seconds': 10.0,
        'max_memory_mb': 500.0,
        'min_convergence_efficiency': 0.5,
        'max_singleton_ratio': 0.4,
        'min_quality_score': 0.6
    })
    history_retention_days: int = 30
    log_level: str = 'INFO'
    enable_alerts: bool = True

@dataclass
class TORIClusteringConfig:
    """Complete TORI clustering system configuration."""
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    affinity_propagation: AffinityPropagationConfig = field(default_factory=AffinityPropagationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    default_method: str = 'oscillator'
    fallback_method: str = 'kmeans'
    enable_method_selection: bool = True
    method_selection_criteria: str = 'quality'  # 'quality', 'speed', 'balanced'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TORIClusteringConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TORIClusteringConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class ConfigManager:
    """Manages TORI clustering configurations with environment-specific overrides."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations for different environments
        self.environments = {
            'development': self._create_development_config(),
            'testing': self._create_testing_config(),
            'production': self._create_production_config()
        }
        
        self.current_env = os.getenv('TORI_ENV', 'development')
        self.current_config = self.environments[self.current_env]
    
    def _create_development_config(self) -> TORIClusteringConfig:
        """Create development configuration."""
        config = TORIClusteringConfig()
        
        # Development-specific overrides
        config.benchmark.enabled_methods = ['oscillator', 'kmeans']  # Faster for dev
        config.benchmark.enable_logging = True
        config.performance.batch_size = 100
        config.performance.max_memory_mb = 200
        config.monitoring.enable_monitoring = True
        config.monitoring.log_level = 'DEBUG'
        
        return config
    
    def _create_testing_config(self) -> TORIClusteringConfig:
        """Create testing configuration."""
        config = TORIClusteringConfig()
        
        # Testing-specific overrides
        config.oscillator.steps = 30  # Faster convergence for tests
        config.benchmark.enabled_methods = ['oscillator', 'kmeans', 'hdbscan']
        config.benchmark.save_results = False
        config.performance.batch_size = 50
        config.performance.enable_caching = False
        config.monitoring.enable_alerts = False
        
        return config
    
    def _create_production_config(self) -> TORIClusteringConfig:
        """Create production configuration."""
        config = TORIClusteringConfig()
        
        # Production-specific overrides
        config.oscillator.steps = 100  # More thorough convergence
        config.benchmark.enabled_methods = ['oscillator', 'kmeans', 'hdbscan', 'affinity_propagation']
        config.benchmark.parallel_execution = True
        config.performance.batch_size = 2000
        config.performance.max_memory_mb = 2000
        config.performance.use_multiprocessing = True
        config.performance.max_workers = 8
        config.monitoring.enable_monitoring = True
        config.monitoring.enable_alerts = True
        config.monitoring.log_level = 'INFO'
        
        return config
    
    def get_config(self, environment: Optional[str] = None) -> TORIClusteringConfig:
        """Get configuration for specified environment."""
        env = environment or self.current_env
        return self.environments.get(env, self.current_config)
    
    def set_environment(self, environment: str):
        """Set current environment."""
        if environment in self.environments:
            self.current_env = environment
            self.current_config = self.environments[environment]
        else:
            raise ValueError(f"Unknown environment: {environment}")
    
    def save_config(self, environment: str, config: TORIClusteringConfig):
        """Save configuration for specific environment."""
        filepath = self.config_dir / f"{environment}_config.json"
        config.save(str(filepath))
        self.environments[environment] = config
    
    def load_config(self, environment: str) -> TORIClusteringConfig:
        """Load configuration from file."""
        filepath = self.config_dir / f"{environment}_config.json"
        if filepath.exists():
            config = TORIClusteringConfig.load(str(filepath))
            self.environments[environment] = config
            return config
        else:
            return self.environments.get(environment, TORIClusteringConfig())
    
    def create_custom_config(self, name: str, base_env: str = 'production') -> TORIClusteringConfig:
        """Create a custom configuration based on existing environment."""
        base_config = self.get_config(base_env)
        custom_config = TORIClusteringConfig.from_dict(base_config.to_dict())
        self.environments[name] = custom_config
        return custom_config
    
    def get_method_config(self, method: str) -> Dict[str, Any]:
        """Get configuration for specific clustering method."""
        config = self.current_config
        method_configs = {
            'oscillator': asdict(config.oscillator),
            'kmeans': asdict(config.kmeans),
            'hdbscan': asdict(config.hdbscan),
            'affinity_propagation': asdict(config.affinity_propagation)
        }
        return method_configs.get(method, {})
    
    def validate_config(self, config: TORIClusteringConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate oscillator config
        if config.oscillator.steps < 10:
            issues.append("Oscillator steps too low (minimum 10)")
        if config.oscillator.cohesion_threshold < 0 or config.oscillator.cohesion_threshold > 1:
            issues.append("Oscillator cohesion threshold must be between 0 and 1")
        
        # Validate performance config
        if config.performance.batch_size < 10:
            issues.append("Batch size too small (minimum 10)")
        if config.performance.max_memory_mb < 50:
            issues.append("Max memory too low (minimum 50MB)")
        
        # Validate quality config
        if config.quality.min_cohesion < 0 or config.quality.min_cohesion > 1:
            issues.append("Min cohesion must be between 0 and 1")
        
        # Validate benchmark config
        if not config.benchmark.enabled_methods:
            issues.append("At least one clustering method must be enabled")
        
        return issues

# Utility functions for configuration management

def get_optimal_config_for_dataset(n_concepts: int, n_dimensions: int, 
                                 expected_clusters: Optional[int] = None) -> TORIClusteringConfig:
    """Generate optimal configuration based on dataset characteristics."""
    config = TORIClusteringConfig()
    
    # Adjust based on dataset size
    if n_concepts < 100:
        # Small dataset
        config.oscillator.steps = 30
        config.performance.batch_size = n_concepts
        config.benchmark.enabled_methods = ['oscillator', 'kmeans']
    elif n_concepts < 1000:
        # Medium dataset
        config.oscillator.steps = 60
        config.performance.batch_size = 500
        config.benchmark.enabled_methods = ['oscillator', 'kmeans', 'hdbscan']
    else:
        # Large dataset
        config.oscillator.steps = 100
        config.performance.batch_size = 1000
        config.performance.use_multiprocessing = True
        config.benchmark.enabled_methods = ['oscillator', 'kmeans']
        config.benchmark.parallel_execution = True
    
    # Adjust based on dimensionality
    if n_dimensions > 500:
        config.hdbscan.metric = 'euclidean'  # Better for high dimensions
        config.performance.max_memory_mb = min(2000, config.performance.max_memory_mb * 2)
    
    # Adjust based on expected clusters
    if expected_clusters:
        config.kmeans.k = expected_clusters
        config.kmeans.auto_k = False
        config.hdbscan.min_cluster_size = max(2, n_concepts // (expected_clusters * 3))
    
    return config

def create_config_from_template(template_name: str) -> TORIClusteringConfig:
    """Create configuration from predefined templates."""
    templates = {
        'fast_prototype': {
            'oscillator': {'steps': 20, 'tolerance': 1e-2},
            'benchmark': {'enabled_methods': ['oscillator', 'kmeans'], 'enable_logging': False},
            'performance': {'batch_size': 100, 'use_multiprocessing': False}
        },
        'high_quality': {
            'oscillator': {'steps': 150, 'tolerance': 1e-4, 'cohesion_threshold': 0.2},
            'benchmark': {'enabled_methods': ['oscillator', 'kmeans', 'hdbscan', 'affinity_propagation']},
            'quality': {'min_cohesion': 0.3, 'min_silhouette': 0.25}
        },
        'memory_efficient': {
            'performance': {'batch_size': 200, 'max_memory_mb': 100, 'enable_caching': False},
            'benchmark': {'enabled_methods': ['oscillator', 'kmeans'], 'parallel_execution': False}
        },
        'research_analysis': {
            'oscillator': {'steps': 200, 'tolerance': 1e-5},
            'benchmark': {'enabled_methods': ['oscillator', 'kmeans', 'hdbscan', 'affinity_propagation'],
                         'compute_silhouette': True, 'compute_ground_truth_metrics': True},
            'monitoring': {'enable_monitoring': True, 'log_level': 'DEBUG'}
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    config = TORIClusteringConfig()
    template_data = templates[template_name]
    
    # Apply template overrides
    for section, overrides in template_data.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, value in overrides.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
    
    return config

# Example usage and testing
if __name__ == "__main__":
    print("🔧 TORI Clustering Configuration Demo")
    print("=====================================")
    
    # Create config manager
    manager = ConfigManager()
    
    # Show different environment configs
    for env in ['development', 'testing', 'production']:
        config = manager.get_config(env)
        print(f"\n📋 {env.upper()} Configuration:")
        print(f"   Oscillator steps: {config.oscillator.steps}")
        print(f"   Batch size: {config.performance.batch_size}")
        print(f"   Enabled methods: {config.benchmark.enabled_methods}")
        print(f"   Monitoring: {config.monitoring.enable_monitoring}")
    
    # Create custom config
    print(f"\n🎯 Creating custom configuration...")
    custom_config = manager.create_custom_config('custom_research', 'production')
    custom_config.oscillator.steps = 200
    custom_config.benchmark.compute_ground_truth_metrics = True
    print(f"   Custom oscillator steps: {custom_config.oscillator.steps}")
    
    # Validate configuration
    issues = manager.validate_config(custom_config)
    print(f"   Validation issues: {len(issues)}")
    
    # Generate optimal config for dataset
    print(f"\n📊 Optimal config for dataset (500 concepts, 384 dimensions):")
    optimal_config = get_optimal_config_for_dataset(500, 384, 8)
    print(f"   Oscillator steps: {optimal_config.oscillator.steps}")
    print(f"   Batch size: {optimal_config.performance.batch_size}")
    print(f"   K-means k: {optimal_config.kmeans.k}")
    
    # Template examples
    print(f"\n🎨 Template configurations:")
    for template in ['fast_prototype', 'high_quality', 'memory_efficient']:
        template_config = create_config_from_template(template)
        print(f"   {template}: {template_config.oscillator.steps} steps, "
              f"{len(template_config.benchmark.enabled_methods)} methods")
    
    print(f"\n✅ Configuration system ready for production use!")
