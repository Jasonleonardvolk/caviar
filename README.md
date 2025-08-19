# caviar

Fast, clean bootstrap to finish **iRis** (demo), monetize via **Snapchat/TikTok**, then wrap **TORI**.

## Quickstart (Frontend - SvelteKit)
```powershell
cd tori_ui_svelte
pnpm install
pnpm dev   # http://localhost:5173/hologram?show=wow   (Hotkeys: 1-5 modes, 0 cycle, B boost, G ghost-fade)
```

## Backend (Penrose / FastAPI)
```powershell
cd services\penrose
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7401   # http://127.0.0.1:7401/docs
```

## Production SSR (Node adapter)
```powershell
cd tori_ui_svelte
pnpm install
pnpm run build
$env:PORT=3000; node .\build\index.js   # http://localhost:3000/hologram?show=wow
```

## Verify iRis (Smoke Tests)
```powershell
# Dev environment
.\tools\release\Verify-Iris.ps1 -Environment dev

# Production environment
.\tools\release\Verify-Iris.ps1 -Environment prod

# Both environments
.\tools\release\Verify-Iris.ps1 -Environment both
```

## Video Encoding (WowPack)
```powershell
# Encode to HEVC-HDR10 and AV1
.\tools\encode\Build-WowPack.ps1 -InputFile "video.mp4" -Codec both -HDR10

# Export 9:16 vertical for social media
.\tools\social\Export-Snaps.ps1 -In "video.mp4"
```

## Daily Workflow
```powershell
# End of day commit
.\tools\git\Commit-EOD.ps1 -Message "finished iRis features"

# Or just use default timestamped message
.\tools\git\Commit-EOD.ps1
```

## Project Structure
```
caviar/
├── tori_ui_svelte/        # Frontend (SvelteKit)
├── services/
│   └── penrose/           # Backend API (FastAPI)
├── tools/
│   ├── git/               # Version control utilities
│   ├── release/           # Verification & deployment
│   ├── encode/            # Video encoding pipeline
│   └── social/            # Social media exports
├── content/
│   └── wowpack/           # Media assets
└── docs/                  # Documentation
```

## Next Milestones
1. **iRis Demo** - Stable hologram experience at 60fps
2. **Monetize** - Deploy to Snapchat Lens Studio & TikTok Effect House
3. **TORI** - Complete core modules and wrap v1.0

## Repository Policy
- **Code-only**: Large binaries (≥100MB) stay in `_NONREPO_LARGE/` or Drive
- **Line endings**: Normalized via `.gitattributes`
- **Virtual environments**: Never committed (use `.venv/`, `venv/`, etc.)

---
🚀 **Live Demo**: https://github.com/Jasonleonardvolk/caviar