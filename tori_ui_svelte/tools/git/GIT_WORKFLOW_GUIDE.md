# iRis Git Workflow Guide

## 🚀 Quick Start

### Initial Setup (one-time)
```powershell
# Configure Git to use versioned hooks
.\tools\git\Setup-GitHooks.ps1
```

## 📋 Common Commands

### Creating Feature Branches
```powershell
# New feature branch for renderer work
.\tools\git\Git-Workflow.ps1 branch feat renderer "holographic-core" -Push

# Bug fix branch for API
.\tools\git\Git-Workflow.ps1 branch fix api "memory-leak" -Push

# Documentation branch
.\tools\git\Git-Workflow.ps1 branch docs setup "installation-guide" -Push
```

### Creating Tags & Releases

#### Internal Demo/Testing
```powershell
# Tag internal version for team testing
.\tools\git\Git-Workflow.ps1 internal "0.1.1" -Message "Internal demo: mocks enabled" -Push
```

#### Release Candidates
```powershell
# Create RC for testing
.\tools\git\Git-Workflow.ps1 rc "1.0.0-rc.1" -Message "RC1: production ready" -Push
```

#### Production Releases
```powershell
# Full GA release (runs smoke tests automatically)
.\tools\git\Git-Workflow.ps1 release "1.0.0" -Message "iRis 1.0 GA release" -Push
```

### Status & Information
```powershell
# View current status, recent tags, and branches
.\tools\git\Git-Workflow.ps1 status
```

## 🏷️ Tag Schema

| Type | Format | Example | Usage |
|------|--------|---------|--------|
| Internal | `iris-internal-v{version}` | `iris-internal-v0.1.1` | Team testing, demos |
| RC | `iris-rc-v{version}` | `iris-rc-v1.0.0-rc.1` | Pre-release testing |
| Release | `iris-v{version}` | `iris-v1.0.0` | Production releases |

## 🔍 Pre-Push Verification

Every `git push` automatically runs verification checks:

### What's Checked
- ✅ Project structure integrity
- ✅ Environment files present
- ✅ API endpoints responding (if server running)
- ✅ Build directory status
- ✅ Upload directory exists
- ⚠️ Warns about uncommitted changes

### Manual Testing
```powershell
# Run verification manually
.\tools\release\verify-setup.ps1

# Test the pre-push hook directly
.\.githooks\pre-push
```

### Bypassing Checks (Emergency Only!)
```bash
# Skip pre-push verification
git push --no-verify
```

## 🌿 Branch Naming Convention

| Type | Pattern | Example | Purpose |
|------|---------|---------|---------|
| Feature | `feat/{scope}-{description}` | `feat/renderer-holographic-core` | New features |
| Fix | `fix/{scope}-{description}` | `fix/api-memory-leak` | Bug fixes |
| Docs | `docs/{scope}-{description}` | `docs/setup-installation-guide` | Documentation |
| Chore | `chore/{scope}-{description}` | `chore/deps-update-svelte` | Maintenance |

## 📊 Workflow Examples

### Complete Feature Development
```powershell
# 1. Create feature branch
.\tools\git\Git-Workflow.ps1 branch feat renderer "particle-system" -Push

# 2. Make changes and commit
git add .
git commit -m "feat(renderer): add particle system support"

# 3. Push changes (pre-push hook runs automatically)
git push

# 4. Create PR and merge to main
# ... GitHub PR process ...

# 5. Tag internal version for testing
.\tools\git\Git-Workflow.ps1 internal "0.2.0" -Message "Particle system beta" -Push
```

### Release Process
```powershell
# 1. Ensure on main branch
git checkout main
git pull origin main

# 2. Run full verification
.\tools\release\verify-setup.ps1

# 3. Create release candidate
.\tools\git\Git-Workflow.ps1 rc "1.0.0-rc.1" -Message "First RC" -Push

# 4. Test RC in staging
# ... testing ...

# 5. Create final release (updates package.json automatically)
.\tools\git\Git-Workflow.ps1 release "1.0.0" -Message "iRis 1.0 GA" -Push

# 6. Deploy to production
.\tools\release\Reset-And-Ship.ps1 -UsePM2
```

## ⚙️ Configuration

### Git Hooks Location
```bash
# Check current hooks path
git config core.hooksPath

# Reset to default (.git/hooks)
git config --unset core.hooksPath

# Use versioned hooks
git config core.hooksPath .githooks
```

### Environment Variables
The pre-push hook respects these environment variables:
- `SKIP_TESTS` - Skip verification if set
- `FORCE_PUSH` - Override failures (not recommended)

## 🛠️ Troubleshooting

### Hook Not Running
```powershell
# Verify hooks are configured
git config core.hooksPath
# Should output: .githooks

# Re-run setup
.\tools\git\Setup-GitHooks.ps1
```

### Server Not Running for Tests
The pre-push hook will:
- ✅ Skip endpoint tests if server isn't running
- ✅ Still check file structure and configuration
- ⚠️ Warn that full verification couldn't complete

### Permission Issues
```powershell
# Run as administrator if needed
Start-Process powershell -Verb RunAs -ArgumentList "-File .\tools\git\Setup-GitHooks.ps1"
```

## 📚 Additional Resources

- [Verify Setup Script](../release/verify-setup.ps1) - Full verification logic
- [Reset & Ship Script](../release/Reset-And-Ship.ps1) - Build and deployment
- [Ship Status](../../SHIP_STATUS.md) - Current release status

---

**Version**: 1.0.0  
**Last Updated**: 2025-08-16  
**Maintainer**: iRis Team