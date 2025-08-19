"""
Configuration management for TORI soliton backend
Handles environment variables, config files, and runtime overrides
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import tomli for reading config files
try:
    import tomli
    HAS_TOMLI = True
except ImportError:
    HAS_TOMLI = False
    logger.debug("tomli not available, config files disabled")

class SolitonConfig:
    """
    Centralized configuration for soliton backend parameters
    Priority: runtime kwargs > env vars > config file > defaults
    """
    
    # Default values
    DEFAULTS = {
        # Physics parameters
        'gamma_eff': 2000.0,
        'ring_boost_db': 20.0,
        'snr_threshold': 10.0,
        
        # Phase locking
        'phase_lock': True,
        'pilot_tone_interval': 100,
        'phase_tolerance_rad': 0.01,
        
        # Thermal management
        'duty_cycle': 0.85,
        'thermal_limit_c': 85.0,
        'thermal_backoff_factor': 0.5,
        
        # Scaling/sharding
        'max_tile_count': 65536,
        'shard_size': 256,
        'parallel_shards': 4,
    }
    
    def __init__(self):
        self._config = self.DEFAULTS.copy()
        self._load_config_file()
        self._load_env_vars()
    
    def _load_config_file(self):
        """Load from ~/.tori/config.toml if it exists"""
        if not HAS_TOMLI:
            return
            
        config_path = Path.home() / '.tori' / 'config.toml'
        if config_path.exists():
            try:
                with open(config_path, 'rb') as f:
                    toml_data = tomli.load(f)
                    
                # Extract soliton section
                if 'soliton' in toml_data:
                    self._config.update(toml_data['soliton'])
                    logger.info(f"Loaded config from {config_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
    
    def _load_env_vars(self):
        """Override with TORI_* environment variables"""
        env_mapping = {
            'TORI_GAMMA_EFF': ('gamma_eff', float),
            'TORI_RING_BOOST_DB': ('ring_boost_db', float),
            'TORI_SNR_THRESHOLD': ('snr_threshold', float),
            'TORI_PHASE_LOCK': ('phase_lock', lambda x: x.lower() in ('true', '1', 'yes')),
            'TORI_PILOT_TONE_INTERVAL': ('pilot_tone_interval', int),
            'TORI_PHASE_TOLERANCE_RAD': ('phase_tolerance_rad', float),
            'TORI_DUTY_CYCLE': ('duty_cycle', float),
            'TORI_THERMAL_LIMIT_C': ('thermal_limit_c', float),
            'TORI_THERMAL_BACKOFF_FACTOR': ('thermal_backoff_factor', float),
            'TORI_MAX_TILE_COUNT': ('max_tile_count', int),
            'TORI_SHARD_SIZE': ('shard_size', int),
            'TORI_PARALLEL_SHARDS': ('parallel_shards', int),
        }
        
        for env_key, (config_key, converter) in env_mapping.items():
            if env_key in os.environ:
                try:
                    self._config[config_key] = converter(os.environ[env_key])
                    logger.debug(f"Set {config_key} from {env_key}")
                except ValueError as e:
                    logger.warning(f"Invalid {env_key}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def update(self, **kwargs):
        """Update configuration with runtime values"""
        self._config.update(kwargs)
    
    def as_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()

# Global instance
_config = SolitonConfig()

def get_config() -> SolitonConfig:
    """Get global configuration instance"""
    return _config

def reset_config():
    """Reset configuration (mainly for testing)"""
    global _config
    _config = SolitonConfig()

__all__ = ['SolitonConfig', 'get_config', 'reset_config']
