# Quick Fix for "Open release folder?" Prompt

## The Problem
When running `Verify-EndToEnd.ps1`, you get an interactive prompt asking "Open release folder?" that blocks automation. This happens because `IrisOneButton.ps1` is designed for interactive use.

## What is a Release Folder?
A **release folder** is a timestamped directory containing your production-ready application:

```
D:\Dev\kha\releases\iris_v1_20250815_143022\
    dist\           <- Built application files (HTML, JS, CSS, etc.)
    manifest.json   <- Build metadata (version, date, validation status)
```

This folder contains everything needed to deploy your application.

## Quick Fix Options

### Option 1: Use the Updated Scripts (Recommended)

1. **Replace IrisOneButton.ps1** with the non-interactive version:
```powershell
# Backup original
Copy-Item .\tools\release\IrisOneButton.ps1 .\tools\release\IrisOneButton.ps1.original

# Copy the non-interactive version
Copy-Item .\tools\release\IrisOneButton_NonInteractive.ps1 .\tools\release\IrisOneButton.ps1
```

2. **Use the improved verification script** that's already configured:
```powershell
powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Improved.ps1 -QuickBuild
```

### Option 2: Add NonInteractive Flag to Original Script

Edit `IrisOneButton.ps1` and add this parameter at the top:
```powershell
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$SkipShaderCheck = $false,
  [switch]$SkipTypeCheck = $false,
  [switch]$QuickBuild = $false,
  [switch]$NonInteractive = $false  # ADD THIS LINE
)
```

Then wrap the prompt at the end:
```powershell
# Replace this:
if ($Host.UI.PromptForChoice("Open release folder?", "", @("&Yes", "&No"), 1) -eq 0) {
    explorer.exe $releaseDir
}

# With this:
if (-not $NonInteractive) {
    if ($Host.UI.PromptForChoice("Open release folder?", "", @("&Yes", "&No"), 1) -eq 0) {
        explorer.exe $releaseDir
    }
}
```

### Option 3: Quick Workaround (Temporary)

When the prompt appears, just press **N** or **Enter** (No is the default). The script will continue and complete successfully.

## Testing Your Fix

After applying the fix, run:
```powershell
powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Improved.ps1 -QuickBuild
```

You should see:
- All steps running without prompts
- "Non-interactive mode: Skipping folder prompt" message
- Clean GO/NO-GO result at the end

## Understanding the Build Process

The complete build process:
1. **Dependencies** - Installs required packages
2. **TypeScript** - Validates types (can have non-blocking errors)
3. **Shaders** - Validates WebGPU shaders against device profiles
4. **Build** - Compiles application with Vite/npm
5. **Package** - Creates timestamped release folder
6. **Verification** - Checks artifacts and generates checksums

## Release Folder Contents

After a successful build, your release folder contains:
- **dist/** - Production-ready application files
- **manifest.json** - Build metadata
- Ready for deployment to your web server

To manually inspect a release:
```powershell
# List recent releases
Get-ChildItem .\releases -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Open the latest release
$latest = Get-ChildItem .\releases -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
explorer.exe $latest.FullName
```

## Complete Automation

For fully automated CI/CD:
1. Use `Verify-EndToEnd-Improved.ps1` with the fixed IrisOneButton.ps1
2. Check exit code: 0 = success (GO), 1 = failure (NO-GO)
3. Deploy the release folder contents if successful

The release folder path is logged in the verification report for your deployment scripts to use.
