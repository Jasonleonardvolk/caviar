# TORI Chat System Architecture

## Current Implementation: SvelteKit (Not React!)

Your TORI Chat is built with **SvelteKit**, not React. Here's the actual architecture:

### Frontend: SvelteKit (`tori_ui_svelte/`)
- **Framework**: SvelteKit 2.0 with Vite
- **UI**: Tailwind CSS
- **Features**: 
  - Full cognitive system integration
  - WebSocket real-time updates
  - Auto-scroll chat interface
  - Memory vault system
  - Multiple AI personas
  - Concept mesh visualization

### Backend Services:

1. **Banksy Backend** (`alan_backend/server/simulation_api.py`)
   - FastAPI server on port 8000
   - AI simulation and processing

2. **PDF Extraction Service** (`run_stable_server.py`)
   - FastAPI server on port 8002
   - PDF ingestion pipeline
   - MCP integration

3. **MCP Services** (`mcp-server-architecture/`)
   - TypeScript/Node.js services
   - Port 3000/3001

### Cognitive Systems:
- 🌊 **Soliton Memory System** - Phase-based memory storage
- 🧬 **Braid Memory** - Loop detection and compression
- 🔮 **Holographic Memory** - 3D spatial memory encoding
- 👻 **Ghost Collective** - Multiple AI personas
- 🧠 **Cognitive Engine** - Advanced processing

## Directory Structure:

```
${IRIS_ROOT}\
├── tori_ui_svelte/          # ✅ ACTIVE SvelteKit frontend
├── tori_chat_frontend/      # ❌ OLD React implementation (not used)
├── alan_backend/            # Banksy AI backend
├── ingest_pdf/              # PDF processing pipeline
├── mcp-server-architecture/ # MCP services
└── Various launch scripts
```

## How to Launch:

### Quick Start (Frontend Only):
```bash
cd ${IRIS_ROOT}\tori_ui_svelte
npm run dev
```

### Full System:
```bash
cd ${IRIS_ROOT}
LAUNCH_TORI_SVELTE.bat
# Choose option 2 or 3
```

### Manual Start:
1. **Frontend**: 
   ```bash
   cd tori_ui_svelte
   npm run dev
   ```

2. **Backend**:
   ```bash
   cd alan_backend/server
   python simulation_api.py
   ```

3. **PDF Service** (optional):
   ```bash
   python run_stable_server.py
   ```

## Why the Confusion?

The `deploy-tori-chat-with-mcp.bat` script you were trying to run is for the OLD React-based `tori_chat_frontend`, which appears to be deprecated. The active system uses SvelteKit in `tori_ui_svelte`.

## Key Differences:

| Feature | Old (React) | Current (SvelteKit) |
|---------|-------------|---------------------|
| Framework | React 18 | SvelteKit 2.0 |
| Directory | `tori_chat_frontend/` | `tori_ui_svelte/` |
| Build Tool | Vite | Vite (via SvelteKit) |
| Server | Express.js | SvelteKit + FastAPI |
| Port | 3000 | 5173 |

## Recommended Action:

Use the SvelteKit version (`tori_ui_svelte`) as it's the active implementation with all the advanced features. The React version appears to be legacy code.

To start immediately:
```bash
cd ${IRIS_ROOT}
LAUNCH_TORI_SVELTE.bat
```

This will give you options to start just the frontend or the full system with all cognitive features enabled.
