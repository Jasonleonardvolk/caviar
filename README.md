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