"""
Configuration management for TORI MCP Server
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ServerConfig(BaseModel):
    """Server configuration settings."""
    
    # Server identification
    server_name: str = Field(
        default="TORI-Metacognitive-Engine",
        description="MCP server name"
    )
    server_version: str = Field(
        default="0.1.0",
        description="MCP server version"
    )
    server_description: str = Field(
        default="MCP server for TORI cognitive framework",
        description="MCP server description"
    )
    
    # Cognitive configuration
    cognitive_dimension: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Cognitive dimension size"
    )
    manifold_metric: Literal["euclidean", "fisher_rao"] = Field(
        default="fisher_rao",
        description="Manifold metric type"
    )
    consciousness_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Consciousness threshold"
    )
    max_metacognitive_levels: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum metacognitive levels"
    )
    
    # Transport configuration
    transport_type: Literal["stdio", "sse", "streamable_http"] = Field(
        default="stdio",
        description="Transport type"
    )
    server_host: str = Field(
        default="localhost",
        description="Server host"
    )
    server_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path"
    )
    
    # Paths
    tori_base_path: Path = Field(
        default=Path("C:/Users/jason/Desktop/tori/kha"),
        description="TORI base path"
    )
    state_persistence_path: Path = Field(
        default=Path("./cognitive_states"),
        description="State persistence path"
    )
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Create config from environment variables."""
        return cls(
            server_name=os.getenv("MCP_SERVER_NAME", "TORI-Metacognitive-Engine"),
            server_version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
            server_description=os.getenv("MCP_SERVER_DESCRIPTION", "MCP server for TORI cognitive framework"),
            cognitive_dimension=int(os.getenv("COGNITIVE_DIMENSION", "10")),
            manifold_metric=os.getenv("MANIFOLD_METRIC", "fisher_rao"),
            consciousness_threshold=float(os.getenv("CONSCIOUSNESS_THRESHOLD", "0.3")),
            max_metacognitive_levels=int(os.getenv("MAX_METACOGNITIVE_LEVELS", "3")),
            transport_type=os.getenv("TRANSPORT_TYPE", "stdio"),
            server_host=os.getenv("SERVER_HOST", "localhost"),
            server_port=int(os.getenv("SERVER_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None,
            tori_base_path=Path(os.getenv("TORI_BASE_PATH", "C:/Users/jason/Desktop/tori/kha")),
            state_persistence_path=Path(os.getenv("STATE_PERSISTENCE_PATH", "./cognitive_states"))
        )
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Ensure paths exist
        self.state_persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Add TORI paths to Python path
        import sys
        sys.path.insert(0, str(self.tori_base_path))
        sys.path.insert(0, str(self.tori_base_path.parent))


# Global config instance
try:
    config = ServerConfig.from_env()
except Exception as e:
    print(f"Warning: Could not load config from environment: {e}")
    config = ServerConfig()