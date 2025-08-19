# Files Architecture - Release Verification System

## Complete File List and Relationships

### Main Verification Script
```
Verify-EndToEnd-Improved.ps1
    |
    +-- Calls these scripts/tools:
        |
        +-- tools\runtime\preflight.mjs              (Step 01)
        +-- npx tsc                                  (Step 02)
        +-- tools\shaders\run_shader_gate.ps1        (Step 03)
        +-- tools\release\api-smoke.js               (Step 04)
        +-- tools\shaders\validate-wgsl.js           (Step 05)
        +-- tools\release\IrisOneButton.ps1          (Step 06) <-- THE ISSUE
        +-- (internal artifact verification)         (Step 07)
```

## The Two Scenarios

### Scenario 1: Using Original Files (WILL HAVE PROMPT)
```
Verify-EndToEnd-Improved.ps1 
    --> IrisOneButton.ps1 (original - no -NonInteractive support)
        --> Shows "Open release folder?" prompt
```

### Scenario 2: Using Fixed Files (NO PROMPT)
```
Verify-EndToEnd-Improved.ps1 
    --> IrisOneButton_NonInteractive.ps1 (if exists)
        --> No prompts, fully automated
    OR
    --> IrisOneButton.ps1 (if replaced with fixed version)
        --> No prompts, fully automated
```

## Current Smart Detection

The improved script now:
1. Looks for `IrisOneButton_NonInteractive.ps1` first
2. Falls back to `IrisOneButton.ps1` if not found
3. Shows which one it's using: "Using: IrisOneButton_NonInteractive.ps1"

## Files You Have Now

### Created/Modified Files:
- **Verify-EndToEnd-Improved.ps1** - Smart verification that auto-detects which IrisOneButton to use
- **IrisOneButton_NonInteractive.ps1** - Fixed version with -NonInteractive support
- **APPLY_NONINTERACTIVE_FIX.bat** - Applies the fix by replacing files
- **FIX_RELEASE_PROMPT.md** - Documentation
- **RELEASE_VERIFICATION_REVIEW.md** - Overall review
- **IMPLEMENTATION_GUIDE.md** - Implementation guide

### Original Files (Unchanged):
- **Verify-EndToEnd.ps1** - Your original verification script
- **IrisOneButton.ps1** - Original build script (has the prompt)

## How to Run Without Prompts

### Option 1: Keep Both Files Separate (Recommended)
```powershell
# Just run the improved script - it will find and use IrisOneButton_NonInteractive.ps1
powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Improved.ps1 -QuickBuild
```

### Option 2: Replace Original
```powershell
# Run the fix batch file
.\tools\release\APPLY_NONINTERACTIVE_FIX.bat

# Then run verification
powershell -ExecutionPolicy Bypass -File .\tools\release\Verify-EndToEnd-Improved.ps1 -QuickBuild
```

## What Each File Does

| File | Purpose | Has Prompt? |
|------|---------|-------------|
| **IrisOneButton.ps1** | Original build/package script | YES - asks to open folder |
| **IrisOneButton_NonInteractive.ps1** | Fixed version with -NonInteractive flag | NO - skips all prompts |
| **Verify-EndToEnd.ps1** | Original verification | Depends on IrisOneButton.ps1 |
| **Verify-EndToEnd-Improved.ps1** | Smart verification | Auto-uses NonInteractive if available |

## The Fix Applied

The improved script now:
1. Checks if `IrisOneButton_NonInteractive.ps1` exists
2. Uses it if found (no prompts)
3. Falls back to original if not found (will have prompt)
4. Shows which version it's using in the output

## Testing

Run this to see which version will be used:
```powershell
$RepoRoot = "D:\Dev\kha"
$IrisOneButtonNonInt = Join-Path $RepoRoot "tools\release\IrisOneButton_NonInteractive.ps1"
$IrisOneButtonOrig = Join-Path $RepoRoot "tools\release\IrisOneButton.ps1"

if (Test-Path $IrisOneButtonNonInt) {
    Write-Host "Will use: IrisOneButton_NonInteractive.ps1 (No prompts)" -ForegroundColor Green
} else {
    Write-Host "Will use: IrisOneButton.ps1 (Has prompts)" -ForegroundColor Yellow
}
```

## Summary

- **Verify-EndToEnd-Improved.ps1** is smart and auto-detects which IrisOneButton to use
- **IrisOneButton_NonInteractive.ps1** is the fixed version without prompts
- Original **IrisOneButton.ps1** still has the prompt unless replaced
- The improved script will work either way, but only be prompt-free if the NonInteractive version exists
