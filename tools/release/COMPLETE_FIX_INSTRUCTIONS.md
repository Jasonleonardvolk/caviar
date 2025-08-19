# IMMEDIATE FIX - Complete E2E Verification with Detailed Reports

## The Problem You're Experiencing

1. **Prompt Issue**: The "Open release folder?" prompt appears and the script exits without completing all 7 steps
2. **Missing Report**: No GO/NO-GO summary is shown
3. **Previous Failures**: You had failures in steps 06 and 07 but no detailed analysis

## Solution: Use the Fixed Script

### Option 1: Quick Test (Recommended First)

Run the new fixed script that handles prompts automatically:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Fixed.ps1 -QuickBuild
```

This script:
- **Automatically handles the prompt** (sends 'N' after timeout)
- **Completes all 7 steps** even if there are prompts
- **Generates detailed reports** including diagnostic analysis
- **Shows GO/NO-GO** at the end

### Option 2: Analyze Previous Failures

If you want to understand your previous failures:

```powershell
# Analyze the most recent failure
powershell -ExecutionPolicy Bypass -File .\tools\release\Analyze-BuildFailure.ps1

# Or analyze a specific report
powershell -ExecutionPolicy Bypass -File .\tools\release\Analyze-BuildFailure.ps1 -LogFile "D:\Dev\kha\tools\release\reports\iris_e2e_report_2025-08-14_21-31-49.json"
```

### Option 3: Fix IrisOneButton.ps1 Permanently

Add NonInteractive support to IrisOneButton.ps1:

```powershell
# At the top of IrisOneButton.ps1, add to params:
[switch]$NonInteractive = $false

# At the bottom, replace the prompt with:
if (-not $NonInteractive) {
    if ($Host.UI.PromptForChoice("Open release folder?", "", @("&Yes", "&No"), 1) -eq 0) {
        explorer.exe $releaseDir
    }
}
```

Then update Verify-EndToEnd.ps1 to pass -NonInteractive.

## What You'll See with the Fixed Script

### Successful Run:
```
PASS  01_Preflight
PASS  02_TypeScript
PASS  03_ShaderGate_AllProfiles_Strict
PASS  04_API_Smoke
PASS  05_DesktopLow_ShaderGate
PASS  05_Desktop_ShaderGate
PASS  06_Build_and_Package        <- No prompt!
PASS  07_Artifact_Verification

============================================================
RESULT: GO
Elapsed Time: 00:02:34
Pass/Fail/Skip: 7/0/0
Markdown: iris_e2e_report_2025-08-15_14-30-00.md
JSON: iris_e2e_report_2025-08-15_14-30-00.json
Diagnostic: iris_e2e_diagnostic_2025-08-15_14-30-00.txt
============================================================
```

### Failed Run (with details):
```
PASS  01_Preflight
FAIL  02_TypeScript
PASS  03_ShaderGate_AllProfiles_Strict
PASS  04_API_Smoke
PASS  05_DesktopLow_ShaderGate
PASS  05_Desktop_ShaderGate
FAIL  06_Build_and_Package
FAIL  07_Artifact_Verification

============================================================
RESULT: NO-GO
Elapsed Time: 00:01:45
Pass/Fail/Skip: 5/3/0
...
Failed Steps:
  - 02_TypeScript (Exit: 1)
  - 06_Build_and_Package (Exit: 3)
  - 07_Artifact_Verification (Exit: 1)

See diagnostic report for details: iris_e2e_diagnostic_2025-08-15_14-30-00.txt
============================================================
```

## Reports Generated

The fixed script generates THREE reports:

1. **Markdown Report** (iris_e2e_report_*.md)
   - Human-readable summary
   - Error summaries for failed steps
   - Action items

2. **JSON Report** (iris_e2e_report_*.json)
   - Machine-readable data
   - Used by analysis tools
   - CI/CD integration

3. **Diagnostic Report** (iris_e2e_diagnostic_*.txt) **NEW!**
   - Detailed failure analysis
   - Last 10 lines of error logs
   - Artifact details
   - Troubleshooting information

## Common Issues and Fixes

### TypeScript Errors
```powershell
# See all errors
npx tsc --noEmit --pretty

# Fix imports automatically
powershell -ExecutionPolicy Bypass -File .\tools\release\FixTypeScriptImports.ps1
```

### Shader Validation Failures
```powershell
# Check shader issues
node tools\shaders\validate-wgsl.js --dir=frontend --strict

# Fix shader bundles
powershell -ExecutionPolicy Bypass -File .\tools\release\fix_shader_bundle.ps1
```

### Build Failures
```powershell
# Clean and rebuild
Remove-Item -Recurse -Force node_modules
npm ci
npm run build

# Or quick build
powershell -ExecutionPolicy Bypass -File .\tools\release\IrisOneButton.ps1 -QuickBuild -SkipTypeCheck
```

### Artifact Verification Failures
Usually means the build didn't create output. Check:
- dist folder exists
- releases folder has write permissions
- Previous build step succeeded

## Next Steps

1. **Run the fixed verification now**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Fixed.ps1 -QuickBuild
   ```

2. **If it fails**, run the analyzer:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\release\Analyze-BuildFailure.ps1
   ```

3. **Apply suggested fixes** from the analyzer

4. **Re-run verification** until you get GO status

## Why the Release Folder Prompt?

The release folder contains your production-ready build:
- `releases\iris_v1_[timestamp]\dist\` - Built files
- `releases\iris_v1_[timestamp]\manifest.json` - Build metadata

The prompt asks if you want to browse these files in Windows Explorer. For automation, we need to skip this prompt, which the fixed script does automatically.
