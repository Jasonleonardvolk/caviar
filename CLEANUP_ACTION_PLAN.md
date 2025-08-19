# Repository Cleanup Action Plan

Based on the second-pass audit, here are the actions needed to fix the repository issues:

## 🔴 Critical (Blocking) Issues

### 1. Remove Sensitive Data and Bloat
```bash
# Run the cleanup script
./cleanup_repository.sh  # or cleanup_repository.bat on Windows

# For sensitive data already in history:
pip install git-filter-repo
git filter-repo --path data/memory_vault/vault_live.jsonl --invert-paths
git push --force origin main
```

### 2. Production Code Location
✅ **Already Fixed**: The production soliton.py is already in `api/routes/soliton.py`
- The duplicate in `fixes/soliton_500_fixes/` will be removed by cleanup script

### 3. Fix Namespace Clashes
**Action Required**: Rename the concept_mesh package to avoid conflicts:
```bash
cd concept_mesh
# Edit setup.py to change name to "concept-mesh-rs"
# Update imports in code to use the new name
```

## 🟡 Medium Priority Issues

### 4. CI Workflow Visibility
The file exists at `.github/workflows/build-concept-mesh.yml` but may not be committed:
```bash
git add .github/workflows/build-concept-mesh.yml
git commit -m "ci: Add workflow to build concept_mesh wheel"
git push
```

### 5. Requirements Lock File
✅ **Already Fixed**: `requirements.lock` exists with pinned versions

### 6. Remove Duplicate Scripts
The cleanup script will remove:
- `GREMLIN_*.ps1`
- `fix_soliton_*.ps1`
- `test_soliton_*.ps1`
- `fixes/soliton_500_fixes/`

## 🟢 Good (But Could Be Better)

### 7. Add Tests
Create unit tests for the rate limiting in solitonMemory.ts:
```typescript
// tests/solitonMemory.test.ts
test('cooldown resets on successful response', async () => {
  // Test that lastStatsFailure = 0 after 2xx response
});
```

### 8. Add pytest-asyncio Test
Create test for the async pipeline:
```python
# tests/test_pipeline_async.py
@pytest.mark.asyncio
async def test_pipeline_no_asyncio_run():
    # Test that pipeline uses await in running loop
```

## Order of Operations

1. **First**: Run `cleanup_repository.bat` to remove all unwanted files
2. **Second**: Use git filter-repo if sensitive data exists in history
3. **Third**: Commit and push the cleaned repository
4. **Fourth**: Verify CI workflow is visible on GitHub
5. **Fifth**: Add the missing tests

## Verification

After cleanup, verify:
```bash
# Check no .pyc files
find . -name "*.pyc" | wc -l  # Should be 0

# Check no logs
ls logs/  # Should not exist

# Check no .egg-info
find . -name "*.egg-info" | wc -l  # Should be 0

# Check git status is clean
git status --porcelain  # Should be empty after cleanup
```

## What NOT to Delete

Keep these production files:
- `api/routes/soliton.py` (production fix)
- `tori_ui_svelte/src/lib/services/solitonMemory.ts` (with guards)
- `requirements.lock` (pinned dependencies)
- `.github/workflows/build-concept-mesh.yml` (CI workflow)
- `tests/test_soliton_api.py` (test suite)
- `.gitignore` (prevents future issues)

---

Generated: 2025-01-12
