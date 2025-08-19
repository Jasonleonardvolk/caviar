"""
Prajna Configuration System
===========================

Centralized configuration for all Prajna components.
Supports environment variables, config files, and runtime overrides.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("prajna.config")

@dataclass
class PrajnaConfig:
    """
    Complete configuration for Prajna system
    """
    
    # Model Configuration
    model_type: str = "rwkv"  # "rwkv", "llama", "gpt", "custom", "demo"
    model_path: str = "./models/prajna_v1.pth"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    max_context_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Memory System Configuration
    soliton_rest_endpoint: str = "http://localhost:8002"
    soliton_ffi_enabled: bool = False
    soliton_ffi_lib_path: Optional[str] = None
    soliton_timeout: float = 10.0
    
    # Concept Mesh Configuration
    concept_mesh_in_memory: bool = True
    concept_mesh_snapshot_path: str = "./data/concept_mesh_snapshot.pkl"
    concept_mesh_max_nodes: int = 100000
    concept_mesh_embedding_dim: int = 384
    
    # Context Building Configuration
    max_context_snippets: int = 10
    context_relevance_threshold: float = 0.3
    context_temporal_decay: float = 0.1
    
    # Audit System Configuration
    audit_trust_threshold: float = 0.7
    audit_similarity_threshold: float = 0.3
    audit_phase_drift_threshold: float = 0.5
    audit_max_reasoning_gap: int = 3
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_cors_origins: list = field(default_factory=lambda: ["*"])
    api_timeout: float = 30.0
    
    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_audit_logging: bool = True
    enable_performance_logging: bool = True
    
    # Development Configuration
    debug_mode: bool = False
    enable_demo_mode: bool = False
    demo_responses: bool = False
    
    # Security Configuration
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100  # per minute
    enable_input_validation: bool = True
    max_query_length: int = 10000
    
    # Storage Configuration
    data_directory: str = "./data"
    models_directory: str = "./models"
    logs_directory: str = "./logs"
    snapshots_directory: str = "./snapshots"
    
    # Runtime state
    start_time: float = field(default_factory=lambda: 0.0)
    config_source: str = "default"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        import time
        self.start_time = time.time()
        
        # Create directories if they don't exist
        for directory in [
            self.data_directory,
            self.models_directory,
            self.logs_directory,
            self.snapshots_directory
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate model configuration
        if self.model_type not in ["rwkv", "llama", "gpt", "custom", "demo"]:
            logger.warning(f"Unknown model type: {self.model_type}, using 'demo'")
            self.model_type = "demo"
        
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            logger.warning(f"Invalid temperature: {self.temperature}, using 0.7")
            self.temperature = 0.7
        
        # Validate context length
        if self.max_context_length < 128:
            logger.warning(f"Context length too small: {self.max_context_length}, using 512")
            self.max_context_length = 512
        
        # Validate thresholds
        if not 0.0 <= self.audit_trust_threshold <= 1.0:
            logger.warning(f"Invalid trust threshold: {self.audit_trust_threshold}, using 0.7")
            self.audit_trust_threshold = 0.7
        
        logger.info("✅ Prajna configuration validated")
    
    @classmethod
    def from_env(cls) -> 'PrajnaConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Model configuration
        config.model_type = os.getenv("PRAJNA_MODEL_TYPE", config.model_type)
        config.model_path = os.getenv("PRAJNA_MODEL_PATH", config.model_path)
        config.device = os.getenv("PRAJNA_DEVICE", config.device)
        
        # Parse numeric values
        try:
            config.max_context_length = int(os.getenv("PRAJNA_MAX_CONTEXT_LENGTH", str(config.max_context_length)))
            config.temperature = float(os.getenv("PRAJNA_TEMPERATURE", str(config.temperature)))
            config.top_p = float(os.getenv("PRAJNA_TOP_P", str(config.top_p)))
            config.top_k = int(os.getenv("PRAJNA_TOP_K", str(config.top_k)))
        except ValueError as e:
            logger.warning(f"Error parsing numeric config from env: {e}")
        
        # Memory system configuration
        config.soliton_rest_endpoint = os.getenv("SOLITON_REST_ENDPOINT", config.soliton_rest_endpoint)
        config.soliton_ffi_enabled = os.getenv("SOLITON_FFI_ENABLED", "false").lower() == "true"
        config.soliton_ffi_lib_path = os.getenv("SOLITON_FFI_LIB_PATH", config.soliton_ffi_lib_path)
        
        # Concept mesh configuration
        config.concept_mesh_in_memory = os.getenv("CONCEPT_MESH_IN_MEMORY", "true").lower() == "true"
        config.concept_mesh_snapshot_path = os.getenv("CONCEPT_MESH_SNAPSHOT_PATH", config.concept_mesh_snapshot_path)
        
        # API configuration
        config.api_host = os.getenv("PRAJNA_API_HOST", config.api_host)
        config.api_port = int(os.getenv("PRAJNA_API_PORT", str(config.api_port)))
        
        # Parse CORS origins
        cors_origins = os.getenv("PRAJNA_CORS_ORIGINS")
        if cors_origins:
            config.api_cors_origins = [origin.strip() for origin in cors_origins.split(",")]
        
        # Debug and development
        config.debug_mode = os.getenv("PRAJNA_DEBUG", "false").lower() == "true"
        config.enable_demo_mode = os.getenv("PRAJNA_DEMO_MODE", "false").lower() == "true"
        
        # Logging
        config.log_level = os.getenv("PRAJNA_LOG_LEVEL", config.log_level)
        config.log_file = os.getenv("PRAJNA_LOG_FILE", config.log_file)
        
        config.config_source = "environment"
        logger.info("📋 Prajna configuration loaded from environment")
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PrajnaConfig':
        """Create configuration from JSON file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config = cls()
            
            # Update config with file data
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            config.config_source = f"file:{config_path}"
            logger.info(f"📋 Prajna configuration loaded from file: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from file: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                f.write(self.to_json())
            
            logger.info(f"💾 Prajna configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to file: {e}")
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"🔄 Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Re-validate after updates
        self._validate_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "max_context_length": self.max_context_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory system configuration"""
        return {
            "soliton_rest_endpoint": self.soliton_rest_endpoint,
            "soliton_ffi_enabled": self.soliton_ffi_enabled,
            "soliton_ffi_lib_path": self.soliton_ffi_lib_path,
            "soliton_timeout": self.soliton_timeout,
            "concept_mesh_in_memory": self.concept_mesh_in_memory,
            "concept_mesh_snapshot_path": self.concept_mesh_snapshot_path,
            "concept_mesh_max_nodes": self.concept_mesh_max_nodes,
            "concept_mesh_embedding_dim": self.concept_mesh_embedding_dim
        }
    
    def get_audit_config(self) -> Dict[str, Any]:
        """Get audit system configuration"""
        return {
            "audit_trust_threshold": self.audit_trust_threshold,
            "audit_similarity_threshold": self.audit_similarity_threshold,
            "audit_phase_drift_threshold": self.audit_phase_drift_threshold,
            "audit_max_reasoning_gap": self.audit_max_reasoning_gap
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_cors_origins": self.api_cors_origins,
            "api_timeout": self.api_timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout
        }
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode"""
        return not self.debug_mode and not self.enable_demo_mode
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.debug_mode or self.enable_demo_mode
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        # Configure log level
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Setup formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            log_file_path = Path(self.logs_directory) / self.log_file
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        logger.info(f"📝 Logging configured: level={self.log_level}, file={self.log_file}")

# Configuration factory functions
def load_config(
    config_file: Optional[str] = None,
    use_env: bool = True,
    **overrides
) -> PrajnaConfig:
    """
    Load Prajna configuration with precedence:
    1. Overrides (highest priority)
    2. Environment variables
    3. Config file
    4. Defaults (lowest priority)
    """
    
    # Start with defaults
    if config_file and Path(config_file).exists():
        config = PrajnaConfig.from_file(config_file)
    else:
        config = PrajnaConfig()
    
    # Apply environment variables if enabled
    if use_env:
        env_config = PrajnaConfig.from_env()
        # Merge environment config (only non-default values)
        for field_name in config.__dataclass_fields__:
            env_value = getattr(env_config, field_name)
            default_value = getattr(PrajnaConfig(), field_name)
            if env_value != default_value:
                setattr(config, field_name, env_value)
    
    # Apply overrides
    if overrides:
        config.update(**overrides)
    
    # Setup logging
    config.setup_logging()
    
    logger.info(f"🔧 Prajna configuration loaded from: {config.config_source}")
    return config

def create_default_config_file(config_path: str):
    """Create a default configuration file"""
    config = PrajnaConfig()
    config.save_to_file(config_path)
    logger.info(f"📝 Default configuration file created: {config_path}")

# Default configuration instance
default_config = PrajnaConfig()

if __name__ == "__main__":
    # Demo configuration system
    print("🔧 Prajna Configuration Demo")
    
    # Load configuration
    config = load_config()
    
    print(f"Model Type: {config.model_type}")
    print(f"Device: {config.device}")
    print(f"API Port: {config.api_port}")
    print(f"Debug Mode: {config.debug_mode}")
    print(f"Config Source: {config.config_source}")
    
    # Save example config
    config.save_to_file("./config/prajna_example.json")
    
    # Test environment loading
    os.environ["PRAJNA_MODEL_TYPE"] = "demo"
    os.environ["PRAJNA_DEBUG"] = "true"
    
    env_config = PrajnaConfig.from_env()
    print(f"\nEnvironment Config:")
    print(f"Model Type: {env_config.model_type}")
    print(f"Debug Mode: {env_config.debug_mode}")
