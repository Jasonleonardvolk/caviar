# PID Variable Fix - Quick Reference

## Issue Fixed
**Error:** "Cannot overwrite variable PID because it is read-only or constant"
**Cause:** `$pid` is a reserved/automatic variable in PowerShell that's read-only
**Solution:** Use `$procId` instead of `$pid`

## Files Fixed

### 1. Bulletproof-Build-And-Ship.ps1
```powershell
# OLD (causes error):
$pid = $existingProcess.OwningProcess

# NEW (fixed):
$procId = $existingProcess.OwningProcess
```

### 2. Quick-Test-Mocks.ps1  
```powershell
# OLD (causes error):
$pid = $existingProcess.OwningProcess

# NEW (fixed):
$procId = $existingProcess.OwningProcess
```

## Files That Don't Need Fixing
- **Reset-And-Ship.ps1** - Uses `$__pid` (with underscores) which is not reserved

## Testing the Fix

Run any of these without the PID error:
```powershell
cd D:\Dev\kha\tori_ui_svelte

# Test the bulletproof script
.\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2

# Or test the quick mock script
.\Quick-Test-Mocks.ps1

# Or run the final runbook
.\Final-Runbook-Clean.ps1
```

## Reserved Variables to Avoid in PowerShell
- `$pid` - Process ID of current PowerShell session
- `$PSVersionTable` - PowerShell version info  
- `$Host` - PowerShell host information
- `$Home` - User's home directory
- `$Error` - Array of error objects

Always use custom names like:
- `$procId`
- `$processId`
- `$existingPid`
- `$targetPid`
