# caviar

Fast, clean bootstrap to finish **iRis** (demo), monetize via **Snapchat/TikTok**, then wrap **TORI**.

## Mobile (iOS 26 Beta 7+ strongly recommended)

**Mobile Status:** ✅ iOS 26 (Beta 7+) preferred – iPhone 13+/iPad Pro confirmed via `tools\release\Verify-Hologram-Route.ps1` and `/health`.

**Why iOS 26?**  
Our hologram stack prefers WebGPU. iOS 26 (Beta 7+) delivers the performance and API stability we need. On older iOS builds, we fall back to CPU/WebGL2 with reduced fidelity and lower FPS.

### Recommended Devices (internal test matrix)
- **iPhone:** iPhone 13 or newer (A15/M-class GPU or better)
- **iPad:** iPad Pro (M1/M2/M4), iPad Air (M2)

> Minimum supported (fail-closed): iPhone 13 class. See `frontend\src\lib\device\capabilities.ts` and the device matrix notes below.

### Enable WebGPU & run on device
1) **Start the dev server (host on LAN)**  
   ```powershell
   cd D:\Dev\kha\frontend
   pnpm install
   pnpm dev -- --host 0.0.0.0
   ```

   If needed, set a fixed host script in `frontend\package.json`:
   ```json
   { "scripts": { "dev:host": "vite --host 0.0.0.0" } }
   ```

2) **Open on iPhone/iPad (same Wi-Fi):**
   ```
   http://<YOUR-PC-LAN-IP>:5173/hologram
   ```

3) **If you see "navigator.gpu missing":**
   - Ensure you're on iOS 26 Beta 7+.
   - If your build still hides WebGPU behind flags, enable the WebGPU feature in Safari's Advanced/Feature Flags (varies by build).

4) **Record a clip:**
   - Tap Start in the recorder UI (plan-gated limits, watermark on Free).
   - MP4/WebM auto-downloads.

### What the app does on iOS 26+
- Prefers WebGPU automatically: `frontend\src\lib\device\capabilities.ts` → `prefersWebGPUHint(...)`
- Renders to `#hologram-canvas` on `/hologram`: `frontend\src\routes\hologram\+page.svelte`
- Records via MediaRecorder from `exportVideo.ts`: `frontend\src\lib\utils\exportVideo.ts`

### Fallback behavior
Older devices/OS: routes remain usable, but fallback visuals (CPU/WebGL2) may engage via `engineShim.ts`:
- `frontend\src\lib\hologram\engineShim.ts`

### Known constraints (mobile)
- Thermal throttling can reduce frame rate during long recordings.
- If your mobile capture looks dim: increase canvas brightness/contrast in your shader path; watermark opacity is adjustable in `exportVideo.ts`.

### Device matrix & gating (internal)
Minimum device class and "prefer WebGPU" logic live in:
- `frontend\src\lib\device\capabilities.ts`
- `frontend\src\routes\hologram\+page.svelte`

Update matrix if you expand support; run:
```powershell
powershell -ExecutionPolicy Bypass -File D:\Dev\kha\tools\release\Verify-Hologram-Route.ps1
```

---

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