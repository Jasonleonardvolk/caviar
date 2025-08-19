# 🚦 Final Merge Gate Checklist for v0.12.0

## Quick Status
All critical items from the audit are addressed. Run `FINAL_MERGE_GATE.bat` to complete.

## ✅ Already Green (No Action Needed)
- **Clean history**: .gitignore comprehensive, no junk since 21b2441
- **CI wheel builder**: Workflow operational, 19-minute builds
- **Repro env**: requirements.lock with exact pins
- **Frontend guard**: 30-second cooldown on failed requests
- **Backend router**: Graceful 503/200 responses

## 🟡 Final 4 Items (15 minutes)

### 1. Concept Mesh Namespace (5 min)
**Current**: Both `concept-mesh/` and `concept_mesh/` exist  
**Fix**: Rename Rust package to `concept_mesh_rs`

**Action**:
- Update `concept-mesh/Cargo.toml`: `name = "concept_mesh_rs"`
- Update imports per `CONCEPT_MESH_IMPORT_ORDER.md`
- CI will build new wheel name automatically

### 2. Scripts Archive (3 min)
**Current**: 480KB of legacy scripts in `scripts_archive/`  
**Fix**: Archive to zip and clean directory

**Action**: Run in the script or manually:
```powershell
Compress-Archive -Path 'scripts_archive\*' -DestinationPath 'scripts_archive_backup.zip'
Remove-Item scripts_archive\* -Recurse -Force
echo "See scripts_archive_backup.zip for legacy scripts" > scripts_archive\README.md
```

### 3. README Badge (2 min)
**Current**: Placeholder `USERNAME/REPO`  
**Fix**: Update with actual GitHub path

**Action**: Run `python update_readme_badge.py` or manually edit

### 4. Final Tag (5 min)
**Current**: v0.11.0-hotfix exists  
**New**: v0.12.0-pre-albert

**Action**:
```bash
git add .
git commit -m "chore: Final cleanup for v0.12.0-pre-albert"
git tag -a v0.12.0-pre-albert -m "Pre-Albert Release"
git push origin main
git push origin v0.12.0-pre-albert
```

## 🎯 One Command Solution

```cmd
FINAL_MERGE_GATE.bat
```

This script:
1. ✅ Archives scripts_archive to zip
2. ✅ Prompts for README badge update  
3. ✅ Commits all changes
4. ✅ Creates v0.12.0-pre-albert tag
5. ✅ Shows push commands

## 📋 Post-Merge

After pushing:
1. Check CI badge shows green
2. Create Albert sprint issue:
   ```bash
   gh issue create --title "Sprint 0: Tensor Core + Kerr Metric" --body "Bootstrap Albert tensor operations"
   ```
3. Point team to `albert/` directory

## 🚀 Time Estimate

- 0-5 min: Run FINAL_MERGE_GATE.bat
- 5-10 min: Update README badge
- 10-15 min: Push and verify CI

**Total: 15 minutes to v0.12.0-pre-albert!**
