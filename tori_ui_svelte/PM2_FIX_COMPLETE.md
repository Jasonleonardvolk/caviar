# PM2 Installation Fix - Complete Summary

## Issue Fixed
**Error:** "The term 'pm2' is not recognized as the name of a cmdlet, function, script file, or operable program"
**Cause:** PM2 was not installed or not in PATH
**Solution:** Install PM2 locally and use `npx pm2` commands

## Changes Applied

### Scripts Updated

1. **Bulletproof-Build-And-Ship.ps1**
   - Changed to install PM2 locally (not globally)
   - All commands now use `npx pm2` instead of just `pm2`
   - Avoids global installation permission issues

2. **Reset-And-Ship.ps1**
   - Updated PM2 check to look for local installation
   - Installs locally if not found
   - Already used `npx pm2` for commands

3. **Final-Runbook-Clean.ps1**
   - Updated displayed commands to use `npx pm2`

### New Helper Script
**Install-PM2-Local.ps1** - Standalone PM2 installation helper

## How PM2 is Now Handled

```powershell
# 1. Check if PM2 exists globally
$pm2Exists = Get-Command pm2 -ErrorAction SilentlyContinue

# 2. If not, check locally
if (-not (Test-Path "node_modules\pm2")) {
    # 3. Install locally (no admin needed)
    & npm install pm2
}

# 4. Always use npx to run PM2
& npx pm2 start build/index.js --name iris
```

## Usage Instructions

### Quick Install PM2
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Install-PM2-Local.ps1
```

### Run Deployment with PM2
```powershell
# These will auto-install PM2 if needed:
.\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2
# OR
.\tools\release\Reset-And-Ship.ps1 -UsePM2
```

### Manual PM2 Commands
Always use `npx` prefix:
```powershell
npx pm2 start build/index.js --name iris
npx pm2 logs iris
npx pm2 status
npx pm2 restart iris
npx pm2 stop iris
npx pm2 delete iris
```

## Why Local Installation?

1. **No admin rights needed** - Global npm installs often require admin
2. **Project-specific** - PM2 version stays with project
3. **npx always works** - Whether PM2 is global or local
4. **Cleaner** - Doesn't pollute global namespace

## Testing the Fix

```powershell
cd D:\Dev\kha\tori_ui_svelte

# This should now work without errors:
.\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2

# Check PM2 is running:
npx pm2 status

# View logs:
npx pm2 logs iris-ui
```

## Alternative: Run Without PM2

If you prefer not to use PM2:
```powershell
# Use the scripts without -UsePM2 flag:
.\Bulletproof-Build-And-Ship.ps1 -Mode mock
# This will use a PowerShell background job instead
```

## Troubleshooting

If you still get PM2 errors:

1. **Clear npm cache:**
   ```powershell
   npm cache clean --force
   ```

2. **Remove and reinstall:**
   ```powershell
   Remove-Item node_modules\pm2 -Recurse -Force -ErrorAction SilentlyContinue
   npm install pm2
   ```

3. **Check npx works:**
   ```powershell
   npx pm2 --version
   ```

## Status: FIXED ✅

PM2 commands now work reliably through local installation and npx usage.
