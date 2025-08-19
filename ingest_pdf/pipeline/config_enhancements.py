"""
Future enhancements for TORI Pipeline Configuration
Ready-to-use implementations for secrets, YAML, and per-request overrides
"""

import os
import json
import logging
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from pydantic import SecretStr Field validator
from pydantic_settings import BaseSettings
try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# Track if we've already warned about missing credentials
_warned_about_missing_vault = False
_warned_about_missing_aws = False

# ============================================
# 1. SECRETS MANAGEMENT
# ============================================

class SecureSettings(BaseSettings):
    """Settings with secure secret handling"""
    
    # Public settings (same as before)
    enable_entropy_pruning: bool = True
    max_parallel_workers: Optional[int] = None
    
    # Secure settings
    api_key: Optional[SecretStr] = Field(None, env="TORI_API_KEY")
    database_url: Optional[SecretStr] = Field(None, env="TORI_DATABASE_URL")
    embedding_service_token: Optional[SecretStr] = Field(None, env="TORI_EMBEDDING_TOKEN")
    aws_access_key: Optional[SecretStr] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_key: Optional[SecretStr] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
        # Custom settings sources for secrets
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings
        ):
            # Priority order (highest to lowest):
            # 1. Vault/AWS Secrets Manager
            # 2. Environment variables  
            # 3. .env file
            # 4. Defaults
            return (
                vault_settings_source,
                aws_secrets_manager_source,
                env_settings,
                file_secret_settings,
                init_settings,
            )
    
    def get_secret_value(self, key: str) -> Optional[str]:
        """Safely get secret value"""
        secret = getattr(self, key, None)
        if secret and hasattr(secret, 'get_secret_value'):
            return secret.get_secret_value()
        return None


@functools.lru_cache(maxsize=1)
def vault_settings_source(settings: BaseSettings = None) -> Dict[str, Any]:
    """Load settings from HashiCorp Vault (lazy, cached)"""
    global _warned_about_missing_vault
    
    vault_addr = os.getenv("VAULT_ADDR")
    vault_token = os.getenv("VAULT_TOKEN")
    vault_path = os.getenv("VAULT_SECRET_PATH", "secret/data/tori")
    
    if not (vault_addr and vault_token):
        if not _warned_about_missing_vault:
            logger.warning(
                "Vault credentials not found (VAULT_ADDR/VAULT_TOKEN). "
                "Falling back to environment variables for secrets."
            )
            _warned_about_missing_vault = True
        return {}
    
    try:
        import hvac
        client = hvac.Client(url=vault_addr, token=vault_token)
        
        if not client.is_authenticated():
            logger.warning("Vault authentication failed. Falling back to environment variables.")
            return {}
            
        response = client.secrets.kv.v2.read_secret_version(
            path=vault_path.replace("secret/data/", "")
        )
        
        # Map Vault keys to settings keys
        vault_data = response.get("data", {}).get("data", {})
        logger.info(f"Successfully loaded {len(vault_data)} secrets from Vault")
        
        return {
            "api_key": vault_data.get("api_key"),
            "database_url": vault_data.get("database_url"),
            "embedding_service_token": vault_data.get("embedding_token"),
        }
    except ImportError:
        logger.warning("hvac package not installed. Cannot use Vault integration.")
        return {}
    except Exception as e:
        logger.error(f"Failed to load from Vault: {e}")
        return {}


@functools.lru_cache(maxsize=1)
def aws_secrets_manager_source(settings: BaseSettings = None) -> Dict[str, Any]:
    """Load settings from AWS Secrets Manager (lazy, cached)"""
    global _warned_about_missing_aws
    
    secret_name = os.getenv("AWS_SECRET_NAME", "tori/production")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        if not _warned_about_missing_aws:
            logger.warning(
                "AWS credentials not found (AWS_ACCESS_KEY_ID). "
                "Falling back to environment variables for secrets."
            )
            _warned_about_missing_aws = True
        return {}
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region)
        
        response = client.get_secret_value(SecretId=secret_name)
        secret_data = json.loads(response['SecretString'])
        
        logger.info(f"Successfully loaded {len(secret_data)} secrets from AWS Secrets Manager")
        
        # Map AWS Secrets to settings keys
        return {
            "api_key": secret_data.get("TORI_API_KEY"),
            "database_url": secret_data.get("TORI_DATABASE_URL"),
            "embedding_service_token": secret_data.get("TORI_EMBEDDING_TOKEN"),
        }
    except ImportError:
        logger.warning("boto3 package not installed. Cannot use AWS Secrets Manager integration.")
        return {}
    except ClientError as e:
        logger.error(f"AWS Secrets Manager error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load from AWS Secrets Manager: {e}")
        return {}


# ============================================
# 2. YAML CONFIGURATION SUPPORT
# ============================================

def yaml_settings_source(yaml_path: str = None) -> Dict[str, Any]:
    """Load settings from YAML file with validation"""
    
    if yaml is None:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
        
    yaml_path = yaml_path or os.getenv("TORI_CONFIG_FILE", "config.yaml")
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        return {}
    
    try:
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        # Flatten nested structures if needed
        flattened = {}
        
        # Handle feature flags
        if "features" in yaml_data:
            for key, value in yaml_data["features"].items():
                flattened[f"enable_{key}"] = value
                
        # Handle limits
        if "limits" in yaml_data:
            limits = yaml_data["limits"]
            if "file_sizes" in limits:
                for size, values in limits["file_sizes"].items():
                    flattened[f"{size}_file_mb"] = values.get("mb")
                    flattened[f"{size}_chunks"] = values.get("chunks")
                    flattened[f"{size}_concepts"] = values.get("concepts")
                    
        # Handle direct mappings
        direct_mappings = [
            "max_parallel_workers", "entropy_threshold", 
            "similarity_threshold", "section_weights"
        ]
        for key in direct_mappings:
            if key in yaml_data:
                flattened[key] = yaml_data[key]
                
        # Validate the flattened data using Pydantic
        from ingest_pdf.pipeline.config import Settings
        try:
            # This will validate all values and raise if invalid
            validated_settings = Settings(**flattened)
            # Return the validated dict
            return validated_settings.dict()
        except Exception as validation_error:
            raise ValueError(f"Invalid YAML configuration: {validation_error}")
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")
    except Exception as e:
        print(f"Warning: Could not load YAML config: {e}")
        return {}


class YamlSettings(BaseSettings):
    """Settings with YAML file support"""
    
    # ... all your normal settings ...
    
    class Config:
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings
        ):
            return (
                env_settings,  # Env vars highest priority
                yaml_settings_source,  # Then YAML
                file_secret_settings,  # Then .env
                init_settings,  # Finally defaults
            )


# Example YAML configuration file
EXAMPLE_YAML_CONFIG = """
# TORI Pipeline Configuration
# This file is loaded if TORI_CONFIG_FILE env var is set

# Feature flags
features:
  context_extraction: true
  frequency_tracking: true
  smart_filtering: true
  entropy_pruning: false  # Disabled for this environment
  ocr_fallback: true
  parallel_processing: true

# Performance settings
max_parallel_workers: 16
entropy_threshold: 0.00005
similarity_threshold: 0.88

# File size limits
limits:
  file_sizes:
    small:
      mb: 2
      chunks: 400
      concepts: 300
    medium:
      mb: 10
      chunks: 800 
      concepts: 1000
    large:
      mb: 50
      chunks: 2000
      concepts: 2500

# Section weights
section_weights:
  title: 2.5
  abstract: 2.0
  introduction: 1.5
  methodology: 1.3
  conclusion: 1.5
  discussion: 1.1
  body: 1.0
  references: 0.5
"""


# ============================================
# 3. PER-REQUEST CONFIGURATION OVERRIDES
# ============================================

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

class ConfigOverride(BaseModel):
    """Request-specific configuration overrides"""
    enable_ocr_fallback: Optional[bool] = None
    max_parallel_workers: Optional[int] = None
    entropy_threshold: Optional[float] = None
    section_weights: Optional[Dict[str, float]] = None
    

@app.post("/ingest")
async def ingest_with_overrides(
    file: UploadFile = File(...),
    config: Optional[ConfigOverride] = None
):
    """Ingest endpoint with per-request configuration"""
    
    from ingest_pdf.pipeline.config import settings
    from ingest_pdf.pipeline import ingest_pdf_clean
    
    # Start with global settings - DEEP COPY to avoid mutation
    request_settings = settings.copy(deep=True)
    
    # Apply overrides if provided
    if config:
        overrides = config.dict(exclude_unset=True)
        request_settings = request_settings.copy(update=overrides)
    
    # Process with custom settings
    # Note: You'd need to modify ingest_pdf_clean to accept settings
    result = await ingest_pdf_clean(
        file.filename,
        settings=request_settings
    )
    
    return {
        "file": file.filename,
        "settings_used": {
            "ocr_enabled": request_settings.enable_ocr_fallback,
            "workers": request_settings.max_parallel_workers,
            "entropy_threshold": request_settings.entropy_threshold
        },
        "result": result
    }


@app.post("/batch_ingest")
async def batch_ingest_with_tenant_config(
    tenant_id: str,
    files: List[UploadFile] = File(...)
):
    """Batch processing with tenant-specific configuration"""
    
    # Load tenant-specific config
    tenant_config = load_tenant_config(tenant_id)
    
    results = []
    for file in files:
        # Each file gets tenant-specific settings
        result = await ingest_with_overrides(
            file=file,
            config=tenant_config
        )
        results.append(result)
        
    return {"tenant": tenant_id, "results": results}


def load_tenant_config(tenant_id: str) -> ConfigOverride:
    """Load configuration for specific tenant"""
    
    # Could load from database, config service, etc.
    tenant_configs = {
        "research_team": ConfigOverride(
            enable_ocr_fallback=True,
            entropy_threshold=0.0002,  # Less aggressive pruning
            section_weights={
                "methodology": 2.0,  # Research team cares about methods
                "results": 2.0
            }
        ),
        "legal_dept": ConfigOverride(
            enable_ocr_fallback=True,
            max_parallel_workers=4,  # Conservative resource usage
            entropy_threshold=0.0001
        ),
        "marketing": ConfigOverride(
            enable_ocr_fallback=False,  # They don't use scanned docs
            max_parallel_workers=32,  # Fast processing
            entropy_threshold=0.00005  # Aggressive pruning
        )
    }
    
    return tenant_configs.get(tenant_id, ConfigOverride())


# ============================================
# 4. CONFIGURATION VALIDATION & MONITORING
# ============================================

class ConfigValidator:
    """Validate configuration changes"""
    
    @staticmethod
    def validate_settings(settings: BaseSettings) -> Tuple[bool, List[str]]:
        """Validate settings and return (is_valid, errors)"""
        errors = []
        
        # Check worker limits
        if settings.max_parallel_workers:
            if settings.max_parallel_workers > 64:
                errors.append(f"max_parallel_workers too high: {settings.max_parallel_workers}")
            if settings.max_parallel_workers < 1:
                errors.append(f"max_parallel_workers too low: {settings.max_parallel_workers}")
                
        # Check thresholds
        if not (0.0 <= settings.entropy_threshold <= 1.0):
            errors.append(f"entropy_threshold out of range: {settings.entropy_threshold}")
            
        # Check file limits make sense
        if settings.small_file_mb >= settings.medium_file_mb:
            errors.append("small_file_mb must be less than medium_file_mb")
            
        return len(errors) == 0, errors


# Example: Save YAML config template
def create_yaml_template():
    with open("config.yaml.example", "w") as f:
        f.write(EXAMPLE_YAML_CONFIG)
    print("Created config.yaml.example")


if __name__ == "__main__":
    # Demo secure settings
    print("Testing secure settings...")
    secure = SecureSettings()
    print(f"API Key is set: {secure.api_key is not None}")
    
    # Demo YAML loading
    print("\nTesting YAML config...")
    create_yaml_template()
    os.environ["TORI_CONFIG_FILE"] = "config.yaml.example"
    yaml_settings = YamlSettings()
    print(f"Entropy pruning from YAML: {yaml_settings.enable_entropy_pruning}")
    
    # Validate settings
    print("\nValidating settings...")
    is_valid, errors = ConfigValidator.validate_settings(yaml_settings)
    print(f"Settings valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
