# Migration Guide: Static to Dynamic Configuration

## Overview

This guide helps you migrate from TORI's static configuration to the new dynamic Pydantic-based configuration system. The migration is designed to be **100% backward compatible** - your existing code will continue to work without modifications.

## What Changed?

### Before (Static Configuration)
```python
# config.py - Old style
ENABLE_ENTROPY_PRUNING = True
MAX_PARALLEL_WORKERS = 16
ENTROPY_THRESHOLD = 0.0001

# Hard-coded, requires code changes to modify
```

### After (Dynamic Configuration)
```python
# config.py - New style
class Settings(BaseSettings):
    enable_entropy_pruning: bool = True
    max_parallel_workers: int = 16
    entropy_threshold: float = 0.0001

settings = Settings()

# Can be overridden via environment without code changes
# MAX_PARALLEL_WORKERS=32 python process.py
```

## Migration Steps

### Step 1: Update Your Imports (Optional)

Your existing imports will continue to work:

```python
# This still works (backward compatibility)
from ingest_pdf.pipeline.config import ENABLE_ENTROPY_PRUNING

# But you can now also use the settings object
from ingest_pdf.pipeline.config import settings
if settings.enable_entropy_pruning:
    # do pruning
```

### Step 2: Create Environment Configuration

Instead of modifying `config.py`, create a `.env` file:

```bash
# .env
MAX_PARALLEL_WORKERS=32
ENTROPY_THRESHOLD=0.00005
ENABLE_OCR_FALLBACK=false
```

### Step 3: Update Deployment Scripts

Replace code modifications with environment variables:

#### Old Way (Modifying Code)
```bash
# deploy.sh - Old
sed -i 's/MAX_PARALLEL_WORKERS = 16/MAX_PARALLEL_WORKERS = 32/' config.py
python process_pdfs.py
```

#### New Way (Environment Variables)
```bash
# deploy.sh - New
export MAX_PARALLEL_WORKERS=32
python process_pdfs.py
```

### Step 4: Update Docker/Kubernetes Configs

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  tori:
    image: tori/pipeline
    environment:
      - MAX_PARALLEL_WORKERS=32
      - ENTROPY_THRESHOLD=0.00005
    # No need to rebuild image for config changes!
```

#### Kubernetes
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tori-config
data:
  MAX_PARALLEL_WORKERS: "32"
  ENTROPY_THRESHOLD: "0.00005"
```

## Common Migration Patterns

### 1. Feature Flags in CI/CD

**Before:**
```python
# test_config.py
ENABLE_ENTROPY_PRUNING = False  # Disabled for tests
ENABLE_OCR_FALLBACK = False
```

**After:**
```bash
# .env.test
ENABLE_ENTROPY_PRUNING=false
ENABLE_OCR_FALLBACK=false

# Run tests with test config
pytest
```

### 2. Environment-Specific Settings

**Before:**
```python
# config_dev.py, config_prod.py, config_staging.py
if os.environ.get('ENV') == 'production':
    MAX_PARALLEL_WORKERS = 32
else:
    MAX_PARALLEL_WORKERS = 4
```

**After:**
```bash
# .env.development
MAX_PARALLEL_WORKERS=4

# .env.production  
MAX_PARALLEL_WORKERS=32

# No code changes needed!
```

### 3. Dynamic Configuration Updates

**Before:**
```python
# Had to modify and redeploy
import config
config.ENTROPY_THRESHOLD = 0.0002  # Dangerous!
```

**After:**
```python
# Can tune without code changes
os.environ['ENTROPY_THRESHOLD'] = '0.0002'
from ingest_pdf.pipeline.config import Settings
settings = Settings()  # Picks up new value
```

### 4. Configuration Validation

**Before:**
```python
# No validation - could have type errors
MAX_PARALLEL_WORKERS = "32"  # String instead of int!
```

**After:**
```python
# Pydantic validates and converts automatically
MAX_PARALLEL_WORKERS="32"  # String in env
settings.max_parallel_workers  # Returns int(32)
```

## Gradual Migration Strategy

You don't need to migrate everything at once:

### Phase 1: Add .env Support (No Code Changes)
1. Keep existing code as-is
2. Create `.env` files for different environments
3. Values in `.env` override hardcoded defaults

### Phase 2: Update New Code
1. New code uses `settings` object
2. Old code continues using constants
3. Both work simultaneously

### Phase 3: Refactor When Convenient
1. Gradually update imports to use `settings`
2. Remove environment-specific Python files
3. Consolidate all config in `.env` files

## Configuration Precedence

Understanding precedence helps avoid confusion:

1. **Environment variables** (highest priority)
2. **`.env` file** 
3. **Default values in Settings class** (lowest priority)

Example:
```python
# In Settings class
max_parallel_workers: int = 16

# In .env
MAX_PARALLEL_WORKERS=24

# In environment
export MAX_PARALLEL_WORKERS=32

# Result: settings.max_parallel_workers = 32 (env wins)
```

## Testing Your Migration

Run the included test script:
```bash
python test_dynamic_config.py
```

This will verify:
- ✅ Configuration loads correctly
- ✅ Environment overrides work
- ✅ Backward compatibility maintained
- ✅ Pipeline imports successfully

## Troubleshooting

### Issue: Settings not updating
```python
# Wrong - imports are cached
from ingest_pdf.pipeline.config import MAX_PARALLEL_WORKERS
os.environ['MAX_PARALLEL_WORKERS'] = '32'
print(MAX_PARALLEL_WORKERS)  # Still shows old value!

# Correct - create new Settings instance
os.environ['MAX_PARALLEL_WORKERS'] = '32'
from ingest_pdf.pipeline.config import Settings
settings = Settings()
print(settings.max_parallel_workers)  # Shows 32
```

### Issue: Type errors
```bash
# Wrong - invalid type
ENTROPY_THRESHOLD=high  # Not a float!

# Correct
ENTROPY_THRESHOLD=0.0001
```

### Issue: Missing configuration
```python
# Check current configuration
from ingest_pdf.pipeline.config import settings, CONFIG
import json
print(json.dumps(CONFIG, indent=2))
```

## Best Practices

1. **Never commit .env files** - Add to `.gitignore`
2. **Document your settings** - Update `.env.example`
3. **Use descriptive names** - Clear variable names
4. **Validate in CI/CD** - Test config loads correctly
5. **Monitor changes** - Log config values at startup

## Benefits After Migration

- ✨ **Zero-downtime config changes** - No redeploys
- 🚀 **Faster experimentation** - Try different settings easily  
- 🔧 **Ops-friendly** - DevOps can tune without code access
- 📊 **A/B testing** - Different configs per environment
- 🛡️ **Type safety** - Pydantic validates all inputs
- 🔄 **Easy rollback** - Just change env vars back

## Need Help?

- Check `DYNAMIC_CONFIG_README.md` for detailed configuration options
- Run `python dynamic_config_examples.py` for practical examples
- Review `.env.example` for all available settings

Remember: **Your existing code continues to work!** Migrate at your own pace.
