# TORI Pipeline - Future Polish Implementation Guide

Thank you for the excellent polish suggestions! Here's how to implement them:

## 🛡️ 1. Protect Export Lists in CI

### Add pytest guard (tests/test_exports_guard.py) ✅
```python
pytest tests/test_exports_guard.py
```

This test will:
- ✅ Fail if `__all__` drifts from expected exports
- ✅ Verify all exports are actually available
- ✅ Check critical functions are callable
- ✅ Ensure backwards compatibility

### Add to CI workflow:
```yaml
# .github/workflows/test.yml
- name: Export Guard Tests
  run: pytest tests/test_exports_guard.py -v
```

## 📦 2. Single Source of Truth

### Created `_exports.py` ✅
```python
# ingest_pdf/pipeline/_exports.py
PUBLIC_API = [
    'ingest_pdf_clean', 'ingest_pdf_async', 'handle', 'handle_pdf',
    'get_db', 'preload_concept_database', 'ProgressTracker',
]
```

### To migrate existing modules:
```python
# Instead of:
__all__ = ['ingest_pdf_clean', 'ingest_pdf_async', ...]

# Use:
from ._exports import PIPELINE_EXPORTS as __all__
```

## 🧹 3. Package Hygiene

### Run these checks:
```bash
# Install in editable mode
python -m pip install -e .

# Check dependencies
python -m pip check

# Verify imports work
python -c "from ingest_pdf.pipeline import get_db; print(get_db())"
```

## 📚 4. Documentation Updates

### Add to CONTRIBUTING.md:
```markdown
## Managing Exports

The TORI pipeline maintains strict control over its public API.

### Rules:
1. All public exports must be listed in `_exports.py`
2. Update `tests/test_exports_guard.py` when adding new exports
3. Run `pytest tests/test_exports_guard.py` before committing
4. Add comment: `# ⚠️ Update test_exports() if you change this list`

### Adding a new export:
1. Add to `pipeline/_exports.py`
2. Import in `pipeline/__init__.py`
3. Add test in `test_exports_guard.py`
4. Update documentation
```

## 🔄 Migration Plan

### Phase 1: Current State ✅
- Manual `__all__` lists with warning comments
- Basic export test

### Phase 2: Single Source (Future)
1. Update modules to use `from ._exports import`
2. Remove duplicate lists
3. Strengthen CI tests

### Phase 3: Full Automation
1. Pre-commit hooks for export validation
2. Auto-generate `__all__` from type stubs
3. Export compatibility matrix

## 🧪 Quick Validation

```bash
# Run all polish checks
pytest tests/test_exports_guard.py -v
python -m pip check
python -c "from ingest_pdf import pipeline; print(len(pipeline.__all__))"
```

## 📋 Checklist Before Next Release

- [ ] Run export guard tests
- [ ] Check pip dependencies
- [ ] Update CHANGELOG with API changes
- [ ] Bump version if API changed
- [ ] Tag release

## 🎯 Benefits

1. **No more export drift** - Single source of truth
2. **CI catches breaks early** - Automated testing
3. **Clean dependency tree** - pip check passes
4. **Clear contribution rules** - Documented process

The foundation is now in place for these improvements. Implement them gradually as needed!
