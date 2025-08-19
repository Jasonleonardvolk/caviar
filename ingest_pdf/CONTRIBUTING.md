# Contributing to TORI Pipeline

## Managing Exports

The TORI pipeline maintains strict control over its public API to ensure stability and backwards compatibility.

### 📋 Export Rules

1. **All public exports must be documented**
   - Listed in module `__all__`
   - Documented in README
   - Tested in test suite

2. **Follow the warning**
   ```python
   # ⚠️ Update test_exports() if you change this list
   __all__ = [...]
   ```

3. **Test before committing**
   ```bash
   python test_exports_simple.py
   pytest tests/test_exports_guard.py  # If using pytest
   ```

### 🔧 Adding a New Export

1. **Add to `__all__` in the source module**
   ```python
   # pipeline.py
   __all__ = [..., 'new_function']
   ```

2. **Import in `__init__.py`**
   ```python
   # pipeline/__init__.py
   from .pipeline import ..., new_function
   ```

3. **Add to package `__all__`**
   ```python
   # pipeline/__init__.py
   __all__ = [..., 'new_function']
   ```

4. **Update tests**
   - Add to `test_exports_simple.py`
   - Add to `test_enhanced_production_fixes.py` export section

5. **Document the change**
   - Update README with new function
   - Add to CHANGELOG

### 🚫 Breaking Changes

Avoid removing or renaming exports. If you must:

1. Deprecate first (minimum 2 releases)
   ```python
   def old_function():
       warnings.warn("old_function is deprecated, use new_function", 
                     DeprecationWarning, stacklevel=2)
       return new_function()
   ```

2. Document in CHANGELOG
3. Update migration guide

### 🧪 Export Testing

Run these checks before submitting PR:

```bash
# Quick export check
python test_exports_simple.py

# Full test suite
python test_enhanced_production_fixes.py

# Pytest guard (if available)
pytest tests/test_exports_guard.py

# Package sanity
python -m pip check
```

### 📦 Future: Single Source of Truth

We're moving towards a single source for exports:

```python
# _exports.py (future)
PUBLIC_API = ['ingest_pdf_clean', 'get_db', ...]

# In modules
from ._exports import PUBLIC_API as __all__
```

This will eliminate any possibility of export lists drifting out of sync.

---

Thank you for contributing to TORI! 🚀
