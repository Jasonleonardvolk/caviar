# 🔧 Git Hook Skip Assets Feature

## Overview

Sometimes you need to push documentation, CI fixes, or other non-asset changes without having the dev server running or 3D models downloaded. The skip assets feature allows you to bypass asset validation while still running all other checks.

## Usage

### Windows PowerShell
```powershell
# Enable skip mode
$env:IRIS_SKIP_ASSETS = "1"

# Push your changes
git push

# Disable skip mode (important!)
$env:IRIS_SKIP_ASSETS = $null
```

### Windows Command Prompt
```batch
rem Enable skip mode
set IRIS_SKIP_ASSETS=1

rem Push your changes
git push

rem Disable skip mode
set IRIS_SKIP_ASSETS=
```

### Git Bash
```bash
# Enable skip mode
export IRIS_SKIP_ASSETS=1

# Push your changes
git push

# Disable skip mode
unset IRIS_SKIP_ASSETS
```

## What Gets Skipped

When `IRIS_SKIP_ASSETS=1`:
- ❌ Asset manifest validation
- ❌ Triangle count checks
- ❌ Texture format validation
- ❌ Alpha/normal map limits
- ❌ Total size validation

## What Still Runs

Even with skip mode:
- ✅ Project directory checks
- ✅ Environment file checks
- ✅ Key route file checks
- ✅ Node modules check
- ✅ Build directory check
- ✅ All non-asset validations

## When to Use

Use skip mode for:
- 📝 Documentation updates
- 🔧 CI/CD configuration changes
- 🐛 Bug fixes unrelated to assets
- 📦 Package updates
- 🎨 CSS/styling changes

## When NOT to Use

Don't use skip mode for:
- 🎮 3D model additions/changes
- 🖼️ Texture modifications
- 📊 Asset manifest updates
- 🔄 Any PR that touches `/assets` directory

## Quick Test

Test that skip mode works:
```powershell
# Test with skip enabled
$env:IRIS_SKIP_ASSETS = "1"
.\tools\release\verify-setup.ps1 -SkipAssets

# Clean up
$env:IRIS_SKIP_ASSETS = $null
```

## Important Notes

⚠️ **Always unset the variable after use!** Otherwise all future pushes will skip validation.

⚠️ **Don't commit with skip mode active** in CI/CD pipelines - it defeats the purpose of validation.

⚠️ **Team members should know** when you've used skip mode so they can verify assets separately if needed.

## Troubleshooting

If push still fails with skip mode:
1. Check other validation failures (non-asset)
2. Ensure variable is set correctly
3. Test with: `.\test-git-hook.bat`
4. Fix line endings if needed: `.\tools\git\Fix-GitHookLineEndings.ps1`

---

**Remember**: Skip mode is a convenience feature. Use it responsibly! 🚀