# Detailed Sprint Prep Guide

## What the Script Does (Step by Step)

### Step 1: Repository Cleanup (10 minutes)
**What happens:**
- Removes all `*.pyc`, `*.log`, `*.pkl` files from git tracking
- Removes `logs/` directory
- Removes duplicate scripts (GREMLIN_*.ps1, fix_soliton_*.ps1)
- Removes egg-info directories
- Stages .gitignore to prevent future issues

**You'll see:**
```
Removing cached files that shouldn't be in git...
rm 'logs/session_12345.log'
rm 'data/concept_mesh/embeddings.pkl'
rm 'alan_core.egg-info/PKG-INFO'
...
```

### Step 2: Create Release Tag
**What happens:**
- Creates git tag `v0.11.0-hotfix`
- Marks this commit as the stable hotfix release

**You'll see:**
```
Creating release tag v0.11.0-hotfix...
Tag created: v0.11.0-hotfix
```

### Step 3: Optimize CI Configuration
**What happens:**
- Renames current CI to `build-concept-mesh-full.yml`
- Activates optimized CI that:
  - For PRs: Only builds on Ubuntu + Python 3.11 (fast)
  - For main/nightly: Full matrix (4 OS × 4 Python)

**You'll see:**
```
Optimizing CI for faster PR builds...
CI configuration optimized!
```

### Step 4: Manual README Update
**What you need to do:**
1. Open `README.md`
2. Find this line:
   ```markdown
   [![Build Concept Mesh](https://github.com/USERNAME/REPO/actions/workflows/build-concept-mesh.yml/badge.svg)]
   ```
3. Replace `USERNAME/REPO` with your actual GitHub path
   - Example: `jasonsmith/tori-kha`

### Step 5: Stage Changes
**What happens:**
- Runs `git add .` to stage all changes
- Specifically adds critical files:
  - `.gitignore`
  - `.github/workflows/`
  - `tests/`

**You'll see:**
```
Staging all changes...
```

### Step 6: Commit Everything
**What happens:**
- Creates a single commit with all cleanup changes
- Descriptive commit message for the release

**You'll see:**
```
Creating commit...
[main 1234567] chore: Final polish - CI optimization and cleanup for v0.11.0-hotfix
 42 files changed, 500 insertions(+), 2000 deletions(-)
```

## Expected File Changes

### Files Removed:
```
- logs/
- *.pyc files
- *.pkl files  
- GREMLIN_*.ps1
- fix_soliton_*.ps1
- test_soliton_*.ps1
- alan_core.egg-info/
- concept_mesh/concept_mesh.egg-info/
```

### Files Added/Modified:
```
+ .gitignore (comprehensive)
+ .github/workflows/build-concept-mesh.yml (optimized)
+ tests/test_pipeline_async.py
+ tests/test_soliton_api.py
+ tori_ui_svelte/src/lib/services/solitonMemory.test.ts
```

## After Running the Script

### 1. Check Git Status
```bash
git status
```
Should show:
- Clean working directory
- Ahead of origin by 2 commits

### 2. Check Tag
```bash
git tag
```
Should show: `v0.11.0-hotfix`

### 3. Push Everything
```bash
# If you cleaned history with git filter-repo:
git push --force origin main

# Otherwise:
git push origin main

# Always push the tag:
git push origin v0.11.0-hotfix
```

### 4. Verify on GitHub
- Go to Actions tab → Should see CI workflow starting
- Go to Releases → Create release from v0.11.0-hotfix tag
- Check README → Badge should show build status

## Troubleshooting

### "Permission denied" on Windows
Run as Administrator or use:
```cmd
quick_sprint_prep.bat
```

### Git commands fail
Make sure you're in the right directory:
```cmd
cd ${IRIS_ROOT}
```

### CI doesn't start
Check if `.github/workflows/build-concept-mesh.yml` was committed:
```bash
git ls-files .github/workflows/
```

### Can't push (rejected)
If you get "Updates were rejected", you need force push after cleanup:
```bash
git push --force origin main
```

## Time Breakdown

- **0-10 min**: Script runs cleanup
- **10-12 min**: Manual README edit  
- **12-15 min**: Review changes with `git status`
- **15-20 min**: Push to GitHub
- **20-30 min**: Monitor first CI run

## Success Checklist

✅ No more .pyc/.log/.pkl files in git  
✅ Tag v0.11.0-hotfix exists  
✅ CI badge in README (with your repo path)  
✅ All changes pushed  
✅ CI running on GitHub  

Then you're 100% ready for the Albert sprint! 🚀
