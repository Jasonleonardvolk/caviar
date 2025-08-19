# Runtime Path Resolution System

After refactoring **930 files** to remove absolute paths, this system ensures paths are resolved at runtime and prevents regressions.

## ✅ What Was Done

### Refactoring Complete
- **930 files modified** - Replaced `C:\Users\jason\Desktop\tori\kha` with:
  - `${IRIS_ROOT}` in Python files (with Path import header added)
  - `${IRIS_ROOT}` in TypeScript, JavaScript, JSON, and other files
- **11 files preserved** in `docs/conversations/` for historical reference

## 🔒 Runtime Resolution

### Node.js/TypeScript (Server-side only)
```typescript
import { resolveFS, replaceTokens } from "$lib/node/paths";

// Resolve paths relative to project root
const configPath = resolveFS("config", "settings.json");

// Replace ${IRIS_ROOT} tokens in strings
const resolved = replaceTokens("${IRIS_ROOT}/data/file.txt");
```

### Python Scripts
```python
from scripts.iris_paths import resolve, PROJECT_ROOT

# Use resolve() for path building
config_file = resolve("config", "settings.json")

# Or use PROJECT_ROOT directly (compatible with refactored files)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Auto-added by refactor
data_file = PROJECT_ROOT / "data" / "file.txt"
```

## 🚨 Preflight Checks

### Automatic (npm scripts)
The preflight check runs automatically before:
- `npm run dev` 
- `npm run build`
- Any script that has a `pre` version defined

If absolute paths are detected, the build/dev will fail with an error listing the offending files.

### Manual Checks

#### Quick Check (Windows)
```batch
tools\runtime\CHECK_PATHS.bat
```

#### PowerShell Check
```powershell
.\tools\runtime\Check-AbsolutePaths.ps1
```

#### Node.js Check
```bash
node tools/runtime/preflight.mjs
```

## 📁 File Structure

```
D:\Dev\kha\
├── src/
│   └── lib/
│       └── node/
│           └── paths.ts          # Node.js runtime resolver
├── scripts/
│   └── iris_paths.py            # Python runtime resolver
├── tools/
│   ├── runtime/
│   │   ├── preflight.mjs        # Preflight checker (Node)
│   │   ├── Check-AbsolutePaths.ps1  # PowerShell checker
│   │   └── CHECK_PATHS.bat      # Batch wrapper
│   └── refactor/                # Refactoring tools
│       ├── refactor_fast.py     # Main refactor script
│       ├── refactor_continue.py # Resume after interruption
│       └── [logs]                # Refactor logs
└── package.json                  # Wired with predev/prebuild checks
```

## 🚫 Excluded from Checks

These directories are intentionally excluded:
- `docs/conversations/` - Historical preservation (11 files with old paths)
- `node_modules/` - Third-party code
- `.venv/`, `venv/` - Python virtual environments
- `.git/` - Version control
- `dist/`, `build/` - Build outputs
- `__pycache__/`, `.cache/` - Cache directories
- `tools/dawn/` - Specific exclusion

## 🔄 If Paths Creep Back In

If the preflight check fails:

1. **See which files have issues:**
   ```bash
   node tools/runtime/preflight.mjs
   ```

2. **Fix them:**
   ```bash
   python tools/refactor/refactor_continue.py
   ```

3. **Or fix specific files manually:**
   - Python: Add `PROJECT_ROOT` header and use `${IRIS_ROOT}`
   - Others: Replace with `${IRIS_ROOT}`

## 🌍 Environment Variable

You can override the root directory:
```bash
# Windows
set IRIS_ROOT=D:\Dev\kha

# Linux/Mac
export IRIS_ROOT=/home/user/kha

# Then run normally
npm run dev
```

## 📊 Statistics

- **Total files scanned**: 19,530
- **Files modified**: 930 (4.8%)
- **Files preserved**: 11 (in docs/conversations)
- **Processing time**: ~5 minutes for 23GB repo
- **No regressions allowed**: Preflight ensures this

## 🎯 Benefits

1. **Portable** - Code works on any machine
2. **CI/CD Ready** - No hardcoded developer paths
3. **Self-Checking** - Can't accidentally commit absolute paths
4. **Fast** - Runtime resolution adds negligible overhead
5. **Complete** - Covers Python, TypeScript, JavaScript, JSON, etc.

---

The refactoring is complete and locked in. The system will now prevent any absolute paths from creeping back into the codebase.
