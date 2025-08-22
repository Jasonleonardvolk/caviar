# 🔧 Git Hook Quick Fix Reference

## 🚨 If Push Fails

### Quick Fix (Run This First)
```batch
.\fix-git-hook.bat
```

### Manual Steps
```powershell
# 1. Fix line endings
.\tools\git\Fix-GitHookLineEndings.ps1

# 2. Test hook
.\test-git-hook.bat

# 3. Try push again
git push
```

## 🎯 Skip Assets Mode

### PowerShell
```powershell
$env:IRIS_SKIP_ASSETS = "1"
git push
$env:IRIS_SKIP_ASSETS = $null
```

### CMD
```batch
set IRIS_SKIP_ASSETS=1
git push
set IRIS_SKIP_ASSETS=
```

### One-Liner (PowerShell)
```powershell
$env:IRIS_SKIP_ASSETS="1"; git push; $env:IRIS_SKIP_ASSETS=$null
```

## 📊 Test Commands

| Command | Purpose |
|---------|---------|
| `.\test-git-hook.bat` | Test hook without pushing |
| `.\tools\release\verify-setup.ps1` | Run verifier directly |
| `.\tools\release\verify-setup.ps1 -SkipAssets` | Test skip mode |

## ⚠️ Common Errors

| Error | Fix |
|-------|-----|
| "non-zero exit" | Run `.\fix-git-hook.bat` |
| "node not found" | Add Node to PATH or use skip mode |
| "asset validation failed" | Fix assets or use skip mode |
| "CRLF line endings" | Run line ending fixer |

## ✅ Success Indicators

Good push output:
```
pre-push: verify-setup.ps1 passed (exit 0)
```

Good test output:
```
[OK] Pre-push hook is working correctly!
```

---
**Quick Help**: Run `.\fix-git-hook.bat` to fix most issues!