# IRIS Shipping Scripts Documentation

## Overview
Complete suite of validation and shipping scripts for IRIS/TORI project.

## Scripts

### 1. **ShipIt.ps1** (Master Orchestrator)
**Purpose:** One-command shipping process that runs everything in sequence
```powershell
# Full shipping process
.\tools\release\ShipIt.ps1

# Quick mode (skip some validations)
.\tools\release\ShipIt.ps1 -QuickMode

# Skip auto-fix step
.\tools\release\ShipIt.ps1 -SkipFix

# Skip build step
.\tools\release\ShipIt.ps1 -SkipBuild
```

### 2. **ShipReadyValidation.ps1** (Comprehensive Validation)
**Purpose:** Complete validation of all shipping requirements
```powershell
# Full validation with build
.\tools\release\ShipReadyValidation.ps1

# Validation without building
.\tools\release\ShipReadyValidation.ps1 -SkipBuild

# Verbose output for debugging
.\tools\release\ShipReadyValidation.ps1 -Verbose

# Stop on first error
.\tools\release\ShipReadyValidation.ps1 -StopOnFirstError
```

**Validates:**
- [x] Environment (Node.js, npm, Git)
- [x] Dependencies (package.json, node_modules)
- [x] TypeScript compilation
- [x] Critical files (QuiltGenerator, Svelte components)
- [x] Shader validation
- [x] API smoke tests
- [x] Build process
- [x] Release structure
- [x] Final ship-ready checks

### 3. **QuickShipCheck.ps1** (Rapid Validation)
**Purpose:** Fast validation of critical items only
```powershell
# Quick check
.\tools\release\QuickShipCheck.ps1

# With detailed output
.\tools\release\QuickShipCheck.ps1 -Detailed
```

**Checks:**
- Node.js/npm installed
- TypeScript compiles
- Package.json exists
- Dependencies installed
- Svelte files present
- Build script defined

### 4. **AutoFixForShipping.ps1** (Automatic Issue Fixer)
**Purpose:** Automatically fix common issues before shipping
```powershell
# Fix issues
.\tools\release\AutoFixForShipping.ps1

# Dry run (show what would be fixed)
.\tools\release\AutoFixForShipping.ps1 -DryRun

# Force fixes without prompts
.\tools\release\AutoFixForShipping.ps1 -Force
```

**Fixes:**
- QuiltGenerator import paths
- Missing QuiltGenerator.ts file
- Missing dependencies
- TypeScript configuration issues
- Old build artifacts
- Missing .env files
- Missing release directory

### 5. **Verify-EndToEnd.ps1** (Original E2E Verification)
**Purpose:** Original comprehensive end-to-end verification
```powershell
# Full verification
.\tools\release\Verify-EndToEnd.ps1

# Quick build mode
.\tools\release\Verify-EndToEnd.ps1 -QuickBuild

# Open report after completion
.\tools\release\Verify-EndToEnd.ps1 -OpenReport
```

### 6. **FixAndBuild.ps1** (Original Fix and Build)
**Purpose:** Original script for fixing import issues and building

## Recommended Workflow

### For Quick Development Checks:
```powershell
.\tools\release\QuickShipCheck.ps1
```

### For Pre-Commit Validation:
```powershell
.\tools\release\AutoFixForShipping.ps1
.\tools\release\QuickShipCheck.ps1
```

### For Full Shipping Process:
```powershell
# Option 1: Use master orchestrator
.\tools\release\ShipIt.ps1

# Option 2: Manual sequence
.\tools\release\AutoFixForShipping.ps1
.\tools\release\ShipReadyValidation.ps1
.\tools\release\Verify-EndToEnd.ps1
```

## Output Locations

- **Reports:** `tools\release\reports\` (Verify-EndToEnd)
- **Ship Reports:** `tools\release\ship-ready-reports\` (ShipReadyValidation)
- **Release Artifacts:** `releases\[version]\`
- **Build Output:** `tori_ui_svelte\build\` or `tori_ui_svelte\dist\`

## Exit Codes

All scripts return:
- `0` = Success, ready to ship
- `1` = Failure, issues found

## Troubleshooting

### TypeScript Errors
1. Run `.\tools\release\AutoFixForShipping.ps1`
2. Check `npx tsc --noEmit` output
3. Review previous fixes that worked

### Build Failures
1. Clean: Remove `dist`, `build`, `.svelte-kit` folders
2. Reinstall: `npm ci` or `npm install`
3. Check: `npm run build --verbose`

### Shader Issues
1. Check `frontend\shaders\` for .wgsl files
2. Verify `tools\shaders\validate-wgsl.js` exists
3. Run shader validator directly

### API Test Failures
1. Check `.env` or `.env.production` exists
2. Verify API endpoints in `tools\release\api-smoke.js`
3. Ensure backend services are configured

## Success Criteria

You're ready to ship when:
- [OK] ShipIt.ps1 shows "READY TO SHIP!"
- [OK] All critical checks pass
- [OK] Build artifacts exist in `releases\` folder
- [OK] No TypeScript errors
- [OK] manifest.json generated

## Notes

- Always run from repo root (`D:\Dev\kha`)
- Scripts use PowerShell 5.1+ features
- Requires Node.js and npm installed
- Git clean working tree recommended but not required
- Build process uses `npm run build` in tori_ui_svelte folder

## Quick Reference

```powershell
# Emergency shipping command (does everything)
.\tools\release\ShipIt.ps1 -Force

# I just want to know if it's ready
.\tools\release\QuickShipCheck.ps1

# Full validation without building
.\tools\release\ShipReadyValidation.ps1 -SkipBuild

# Fix everything automatically
.\tools\release\AutoFixForShipping.ps1
```

---
*Generated for IRIS/TORI Project Shipping*
