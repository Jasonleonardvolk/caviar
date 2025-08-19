# What Gets Auto-Pushed

## SOURCE CODE (All Programming Languages)
The auto-push includes ALL source code extensions:

### Web/Frontend
- `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs` - JavaScript/TypeScript
- `.svelte`, `.vue`, `.jsx` - Framework components
- `.html`, `.css`, `.scss`, `.sass`, `.less` - Markup/Styles
- `.wgsl`, `.glsl`, `.hlsl`, `.vert`, `.frag`, `.comp` - Shaders

### Backend/Systems
- `.py`, `.pyx`, `.pyi` - Python
- `.rs`, `.toml` - Rust
- `.go` - Go
- `.c`, `.cpp`, `.cc`, `.h`, `.hpp` - C/C++
- `.java`, `.kt` - Java/Kotlin
- `.cs`, `.fs`, `.vb` - .NET
- `.rb` - Ruby
- `.php` - PHP
- `.lua` - Lua
- `.jl` - Julia
- `.r` - R

### Mobile/Embedded
- `.swift`, `.m`, `.mm` - iOS
- `.dart` - Flutter
- `.kt`, `.java` - Android
- `.cu`, `.cuh` - CUDA
- `.cl` - OpenCL
- `.metal` - Metal

### Config/Data
- `.json`, `.jsonl`, `.yaml`, `.yml` - Data formats
- `.xml`, `.toml`, `.ini`, `.cfg`, `.conf` - Configs
- `.env`, `.env.*` - Environment (BE CAREFUL!)
- `.md`, `.mdx`, `.txt` - Documentation
- `.sql`, `.prisma`, `.graphql` - Schemas

### Scripts/Build
- `.sh`, `.bash` - Shell
- `.ps1`, `.psm1` - PowerShell
- `.bat`, `.cmd` - Windows batch
- `.makefile`, `.cmake` - Build systems
- `.dockerfile`, `.containerfile` - Containers
- `.gradle`, `.sbt` - JVM build

### Special
- `.proto` - Protocol Buffers
- `.tf`, `.tfvars` - Terraform
- `.nix` - Nix
- `.sol` - Solidity
- `.wasm`, `.wat` - WebAssembly
- `.asm`, `.s` - Assembly

## WHAT'S EXCLUDED (Never Pushed)
Per `.gitignore`:
- `node_modules/` - NPM packages
- `.venv/`, `venv/`, `venv_*/` - Python environments
- `__pycache__/`, `*.pyc` - Python bytecode
- `.svelte-kit/`, `build/`, `dist/`, `target/` - Build outputs
- `*.log`, `*.tmp`, `*.temp` - Temporary files
- Large binaries (`.exe`, `.dll`, `.so`)
- Media files > 100MB
- `_NONREPO_LARGE/` - Quarantined large files

## IMPORTANT NOTES
1. **Secrets**: `.env` files ARE tracked by default! Add `.env` to `.gitignore` if it has secrets
2. **Binary files**: Images, PDFs, videos push by default unless in `.gitignore`
3. **Generated files**: Add build outputs to `.gitignore` to keep repo clean

## Check What Will Push
```powershell
# See what git will track
git status

# See all tracked files
git ls-files

# See what would be added
git add -A --dry-run
```