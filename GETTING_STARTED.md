# Getting Started with TORI - One Command Setup

## 🚀 Quick Start (One Command!)

### Windows (PowerShell)
```powershell
.\scripts\dev_install.ps1
```

### Linux/macOS
```bash
./scripts/dev_install.sh
```

That's it! The script will:
- ✅ Create and activate a virtual environment
- ✅ Install all Python dependencies
- ✅ Build Rust components (Penrose similarity engine)
- ✅ Install frontend dependencies
- ✅ Configure MCP packages
- ✅ Launch TORI automatically

## 🎯 Alternative: Using Make

If you have `make` installed:
```bash
make dev   # One-command setup
make run   # Start TORI later
make test  # Run tests
```

## 🔍 What Happens During Setup

1. **Python Environment**: Creates `.venv` and installs all dependencies
2. **Rust Components**: Builds high-performance similarity engine
3. **Frontend**: Installs Node.js dependencies for the web UI
4. **Verification**: Runs health checks to ensure everything works

## 📋 Requirements

- Python 3.8+ 
- Git
- (Optional) Rust toolchain for building native extensions
- (Optional) Node.js for frontend development

## 🆘 Troubleshooting

If the script fails:

1. **Python not found**: Install Python 3.8+ from https://python.org
2. **Rust not found**: Install from https://rustup.rs (optional, for performance)
3. **npm not found**: Install Node.js from https://nodejs.org (optional, for frontend)

## 🎉 Next Steps

After successful installation:
- Frontend: http://localhost:5174
- API Docs: http://localhost:8003/docs
- Health Check: http://localhost:8003/api/health

Happy coding! 🚀
