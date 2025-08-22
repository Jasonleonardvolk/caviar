# Git Conventions for iRis/TORI Project

## Repository Structure

- **Repository**: `tori_ui_svelte`
- **Default Branch**: `main`
- **Location**: `D:\Dev\kha\tori_ui_svelte\`

## Branch Naming Convention

Format: `type/scope-description`

### Branch Types
- `feat/` - New features
- `fix/` - Bug fixes
- `chore/` - Maintenance tasks
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `perf/` - Performance improvements
- `style/` - Code style changes

### Examples
```bash
feat/renderer-modes
fix/upload-mime-guard
chore/adapter-node-build
docs/ship-readme
refactor/services-storage-hub
```

## Commit Message Convention

Format: `type(scope): summary`

### Scopes (Directory-Based)

| Scope | Directory/Files |
|-------|----------------|
| `renderer` | `src\routes\renderer\+page.svelte`, `src\lib\components\HolographicDisplay.svelte` |
| `shaders` | `src\lib\shaders\**` |
| `upload` | `src\routes\upload\+server.ts`, `src\routes\upload\+page.svelte` |
| `api-list` | `src\routes\api\list\+server.ts` |
| `api-pdf-stats` | `src\routes\api\pdf\stats\+server.ts` |
| `api-memory-state` | `src\routes\api\memory\state\+server.ts` |
| `api-health` | `src\routes\api\health\+server.ts` |
| `auth` | `src\lib\server\auth.ts` |
| `fetch` | `src\lib\server\safeFetch.ts` |
| `services` | `src\lib\services\storage.ts`, `storage.local.ts`, `storage.s3.ts` |
| `persona` | `src\lib\stores\ghostPersona.ts` |
| `elfin` | `src\lib\elfin\interpreter.ts` |
| `config` | `.env.local`, `.env.production`, `svelte.config.js`, `vite.config.ts` |
| `repo` | `.gitignore`, `.gitattributes`, Git configuration |

### Commit Examples
```bash
feat(renderer): iRis-first redirect in src\routes\+page.server.ts to /renderer
fix(upload): enforce MIME/size guard in src\routes\upload\+server.ts
chore(services): wire local adapter in src\lib\services\storage.ts
feat(api-pdf-stats): add mock JSON with stable schema
refactor(shaders): emit shaderSources.wgsl.js under src\lib\shaders\out\
docs(config): add .env.production template for S3 + INTERNAL_API_KEY
```

## Tag Schema

All tags are annotated and immutable.

### Tag Formats

| Type | Format | Example | Usage |
|------|--------|---------|-------|
| Internal Demo | `iris-internal-vX.Y.Z` | `iris-internal-v0.1.0` | Internal testing builds |
| Release Candidate | `iris-rc-vX.Y.Z-rc.N` | `iris-rc-v1.0.0-rc.1` | Pre-release testing |
| GA Release | `iris-vX.Y.Z` | `iris-v1.0.0` | Production releases |
| Daily Cut | `iris-cut-YYYY.MM.DD` | `iris-cut-2025.08.16` | Daily snapshots (optional) |

### Creating Tags
```bash
# Internal demo
git tag -a iris-internal-v0.1.0 -m "Internal demo build: unauth + mocks + local uploads"

# Release candidate
git tag -a iris-rc-v1.0.0-rc.1 -m "RC1: mocks off except /upload"

# GA release
git tag -a iris-v1.0.0 -m "First GA release of iRis"

# Push tags
git push --follow-tags
```

## Long-Lived Branches

Protected branches that persist:

1. **`main`** - Stable, deployable code
2. **`release/iris-v1`** - Optional maintenance branch post-1.0
3. **`infra/ci`** - Optional for build infrastructure changes

## Git Hooks

### Pre-commit Hook
Location: `.git\hooks\pre-commit`

Runs:
- Svelte type checking
- Linting

### Pre-push Hook
Location: `.git\hooks\pre-push`

Runs:
- Full build
- End-to-end verification (`tools\release\Verify-EndToEnd.ps1`)

## Helper Scripts

### Initialize Repository
```powershell
.\Initialize-GitRepo.ps1
```

### Workflow Helper
```powershell
# Create branch
.\Git-Workflow.ps1 branch feat renderer "holographic-modes"

# Create commit
.\Git-Workflow.ps1 commit fix upload "mime-validation" -Message "Add stricter checks"

# Create tag
.\Git-Workflow.ps1 tag "iris-v1.0.0" -Message "First GA release"

# Check status
.\Git-Workflow.ps1 status
```

### End-to-End Verification
```powershell
.\tools\release\Verify-EndToEnd.ps1
```

## File Configuration

### .gitignore
- Node/SvelteKit artifacts
- Local runtime files
- Environment files
- Shader intermediates

### .gitattributes
- Windows scripts: CRLF
- POSIX scripts: LF
- Code files: LF

## Quick Commands Reference

```bash
# Switch to main branch
git switch main

# Create feature branch
git switch -c feat/renderer-modes

# Stage and commit
git add .
git commit -m "feat(renderer): add holographic display modes"

# Push with tags
git push --follow-tags

# List branches
git branch -a

# List tags
git tag --sort=-version:refname

# Merge feature branch
git switch main
git merge --no-ff feat/renderer-modes -m "merge: renderer modes feature"
```

## Release Process

1. Create release branch (optional)
   ```bash
   git switch -c release/iris-v1
   ```

2. Update version and changelog
   ```bash
   # Update package.json version
   # Create docs\releases\iris-v1.0.0.md
   ```

3. Commit changes
   ```bash
   git commit -m "chore(release): prep iris v1.0.0"
   ```

4. Create RC tag
   ```bash
   git tag -a iris-rc-v1.0.0-rc.1 -m "RC1: feature complete"
   ```

5. After testing, create GA tag
   ```bash
   git tag -a iris-v1.0.0 -m "GA Release: iRis v1.0.0"
   ```

6. Push everything
   ```bash
   git push origin release/iris-v1 --follow-tags
   ```

## Best Practices

1. **Always use descriptive branch names** that indicate the scope and purpose
2. **Keep commits atomic** - one logical change per commit
3. **Reference issue numbers** in commit messages when applicable
4. **Run hooks locally** before pushing to catch issues early
5. **Tag important milestones** for easy rollback and reference
6. **Use annotated tags** with meaningful messages
7. **Protect main branch** from direct pushes
8. **Squash merge** feature branches to keep history clean
9. **Delete merged branches** to keep repository tidy
10. **Document breaking changes** in commit messages and changelogs
