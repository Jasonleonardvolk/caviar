import yaml
from pathlib import Path
import os

def load_cfg():
    """Load configuration from YAML file with fallback defaults"""
    
    # Try multiple possible locations for config.yaml
    possible_paths = [
        Path(__file__).parent / "banksy" / "config.yaml",  # Original location
        Path(__file__).parent.parent / "banksy" / "config.yaml",  # One level up
        Path(__file__).parent / "config.yaml",  # Same directory
        Path(__file__).parent.parent / "config.yaml"  # Parent directory
    ]
    
    config = None
    config_path = None
    
    for path in possible_paths:
        if path.exists():
            config_path = path
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                print(f"✅ Loaded config from: {path}")
                break
            except Exception as e:
                print(f"⚠️ Failed to load config from {path}: {e}")
                continue
    
    if config is None:
        print("⚠️ No config.yaml found, using default configuration")
        # Fallback default configuration
        config = {
            "extraction": {
                "enabled": True,
                "threshold": 0.0,
                "max_concepts": 100,
                "use_enhanced_pipeline": True
            },
            "koopman_enabled": False,
            "koopman_mode": False,
            "spectral_analysis": True,
            "semantic_boost": True,
            "methods": {
                "keybert": {"enabled": True, "top_k": 20},
                "yake": {"enabled": True, "num_keywords": 20},
                "ner": {"enabled": True}
            },
            "scoring": {
                "keybert_weight": 0.4,
                "yake_weight": 0.3,
                "ner_weight": 0.2,
                "frequency_weight": 0.1
            },
            "output": {
                "include_scores": True,
                "include_methods": True,
                "max_concepts_returned": 50
            },
            "logging": {
                "level": "INFO",
                "enable_debug": True
            }
        }
    
    return config

# Load configuration on import
try:
    cfg = load_cfg()
    print(f"📋 Configuration loaded successfully")
except Exception as e:
    print(f"❌ Failed to load configuration: {e}")
    # Minimal fallback config to prevent crashes
    cfg = {
        "extraction": {"enabled": True, "threshold": 0.0},
        "koopman_enabled": False,
        "methods": {"keybert": {"enabled": True}, "yake": {"enabled": True}},
        "output": {"max_concepts_returned": 50}
    }
    print("📋 Using minimal fallback configuration")
