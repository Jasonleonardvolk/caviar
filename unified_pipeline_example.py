"""
UNIFIED MULTI-MODAL PIPELINE WITH HOLOGRAPHIC VISUALIZATION
==========================================================

A complete example showing how to use the unified ingestion pipeline
with real-time holographic visualization.
"""

import asyncio
from fastapi import FastAPI, UploadFile, WebSocket, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import tempfile
import shutil

# PYTHONPATH FIX: Add project root so imports work
import sys
sys.path.append(str(Path(__file__).resolve().parent))  # add project root

# Import the unified pipeline components
from ingest_pdf.pipeline.router import ingest_file, set_hologram_bus
from ingest_pdf.pipeline.holographic_bus import (
    get_event_bus, get_display_api, start_websocket_bridge
)

# Create FastAPI app
app = FastAPI(title="Unified Multi-Modal Ingestion API")

# === Startup Events ===
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Start WebSocket bridge for holographic display
    await start_websocket_bridge()
    
    # Connect router to hologram bus
    set_hologram_bus(get_event_bus())
    
    print("🚀 Unified Multi-Modal Pipeline Started")
    print("📊 Holographic Visualization Ready")
    print("🌐 WebSocket: ws://localhost:8000/ws/hologram")

# === File Upload Endpoint ===
@app.post("/api/ingest")
async def ingest_endpoint(file: UploadFile):
    """
    Ingest any supported file type with real-time visualization.
    
    Supported formats:
    - PDF documents (with OCR support)
    - Text files (TXT, HTML, Markdown)
    - Images (JPEG, PNG, GIF, WebP, TIFF, BMP)
    - Audio (coming soon: MP3, WAV, M4A)
    - Video (coming soon: MP4, AVI, MOV)
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Ingest file with holographic visualization
        result = await ingest_file(
            tmp_path,
            admin_mode=False,
            progress_callback=None  # WebSocket handles progress
        )
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

# === WebSocket Endpoint for Holographic Display ===
@app.websocket("/ws/hologram")
async def hologram_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time holographic visualization.
    Connect your HolographicDisplay.svelte component to this endpoint.
    """
    await websocket.accept()
    bridge = await start_websocket_bridge()
    
    try:
        await bridge.add_client(websocket)
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            message = await websocket.receive_text()
            
            # Handle client commands if needed
            if message == "ping":
                await websocket.send_text("pong")
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await bridge.remove_client(websocket)

# === Demo Endpoint ===
@app.post("/api/demo/{file_type}")
async def demo_ingestion(file_type: str):
    """
    Demo endpoint showing different file type processing.
    
    file_type: "pdf", "image", "text", "html", "markdown"
    """
    # Create demo content based on type
    demo_files = {
        "pdf": "demo_scientific_paper.pdf",
        "image": "demo_diagram.png",
        "text": "demo_notes.txt",
        "html": "demo_article.html",
        "markdown": "demo_readme.md"
    }
    
    if file_type not in demo_files:
        raise HTTPException(status_code=400, detail=f"Unknown demo type: {file_type}")
    
    # For demo purposes, create synthetic data
    demo_path = Path(f"demos/{demo_files[file_type]}")
    
    if not demo_path.exists():
        # Create synthetic demo file
        await create_demo_file(demo_path, file_type)
    
    # Process with visualization
    result = await ingest_file(str(demo_path))
    
    return JSONResponse(content={
        "demo_type": file_type,
        "result": result
    })

# === Health Check ===
@app.get("/health")
async def health_check():
    """Check system health"""
    event_bus = get_event_bus()
    
    return {
        "status": "healthy",
        "services": {
            "router": "active",
            "event_bus": "active",
            "websocket_bridge": "active",
            "holographic_display": "ready"
        },
        "waveform": event_bus.get_current_waveform(),
        "supported_formats": {
            "text": ["pdf", "txt", "html", "markdown"],
            "image": ["jpg", "png", "gif", "webp", "tiff", "bmp"],
            "audio": ["coming_soon"],
            "video": ["coming_soon"]
        }
    }

# === Client Example ===
async def example_client_usage():
    """Example of how to use the API from Python client"""
    import aiohttp
    
    # Upload and process a file
    async with aiohttp.ClientSession() as session:
        # Upload file
        with open("research_paper.pdf", "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="research_paper.pdf")
            
            async with session.post("http://localhost:8000/api/ingest", data=data) as resp:
                result = await resp.json()
                print(f"Extracted {result['concept_count']} concepts")
        
        # Connect to WebSocket for live updates
        async with session.ws_connect("ws://localhost:8000/ws/hologram") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = msg.json()
                    
                    if event["type"] == "progress":
                        print(f"Progress: {event['data']['stage']} - {event['data']['percent']}%")
                    elif event["type"] == "waveform":
                        # Update holographic display
                        update_hologram(event["data"])

# === HTML Demo Page ===
DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Modal Ingestion with Holographic Display</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #0a0a0a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .upload-zone { 
            border: 2px dashed #0ff; 
            padding: 40px; 
            text-align: center;
            background: rgba(0,255,255,0.1);
            margin: 20px 0;
        }
        #hologram-canvas { 
            width: 100%; 
            height: 400px; 
            background: #000;
            border: 1px solid #0ff;
        }
        .status { margin: 20px 0; }
        .concept { 
            display: inline-block; 
            padding: 5px 10px; 
            margin: 5px;
            background: rgba(0,255,255,0.2);
            border: 1px solid #0ff;
            border-radius: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌟 Unified Multi-Modal Ingestion Pipeline</h1>
        
        <div class="upload-zone" id="dropZone">
            <p>Drop any file here (PDF, Image, Text, HTML, Markdown)</p>
            <input type="file" id="fileInput" />
        </div>
        
        <canvas id="hologram-canvas"></canvas>
        
        <div class="status" id="status">Ready</div>
        
        <div id="concepts"></div>
    </div>
    
    <script>
        // Connect to WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws/hologram');
        const canvas = document.getElementById('hologram-canvas');
        const ctx = canvas.getContext('2d');
        
        // Holographic visualization
        let waveform = {
            amplitude: 0,
            frequency: 1,
            phase: 0,
            coherence: 0.8,
            interference_pattern: []
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'waveform') {
                waveform = data.data;
                renderHologram();
            } else if (data.type === 'progress') {
                document.getElementById('status').textContent = 
                    `${data.data.stage}: ${data.data.percent}% - ${data.data.message}`;
            } else if (data.type === 'concept') {
                addConcept(data.data);
            }
        };
        
        function renderHologram() {
            // Simple holographic effect
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw interference pattern
            ctx.strokeStyle = `rgba(0,255,255,${waveform.coherence})`;
            ctx.lineWidth = 2;
            
            for (let x = 0; x < canvas.width; x += 5) {
                const y = canvas.height/2 + 
                    Math.sin((x + waveform.phase) * waveform.frequency * 0.01) * 
                    waveform.amplitude * 100;
                
                ctx.beginPath();
                ctx.moveTo(x, canvas.height/2);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
            
            // Draw ripples
            waveform.interference_pattern.forEach(ripple => {
                ctx.beginPath();
                ctx.arc(
                    ripple.center[0] * canvas.width,
                    ripple.center[1] * canvas.height,
                    ripple.amplitude * 50,
                    0, 2 * Math.PI
                );
                ctx.stroke();
            });
        }
        
        function addConcept(concept) {
            const div = document.createElement('span');
            div.className = 'concept';
            div.textContent = `${concept.name} (${concept.score.toFixed(2)})`;
            document.getElementById('concepts').appendChild(div);
        }
        
        // File upload
        document.getElementById('fileInput').onchange = async (e) => {
            const file = e.target.files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('concepts').innerHTML = '';
            
            const response = await fetch('/api/ingest', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            console.log('Result:', result);
        };
        
        // Animation loop
        setInterval(renderHologram, 50);
    </script>
</body>
</html>
"""

@app.get("/")
async def demo_page():
    """Serve demo HTML page"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=DEMO_HTML)

# === Helper Functions ===
async def create_demo_file(path: Path, file_type: str):
    """Create synthetic demo files for testing"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_type == "text":
        path.write_text("""
        Unified Multi-Modal Pipeline Demo
        
        This revolutionary system combines advanced concept extraction
        with real-time holographic visualization. Key features include:
        
        - Support for multiple file formats
        - Real-time progress visualization
        - Holographic waveform display
        - Concept extraction with quality scoring
        - Entropy-based pruning
        - Thread-safe parallel processing
        """)
    elif file_type == "markdown":
        path.write_text("""
# Unified Pipeline Demo

## Features
- **Multi-format support**: PDF, images, text, audio, video
- **Real-time visualization**: Holographic waveform display
- **Advanced extraction**: Concept mining with purity analysis

## Architecture
```
Router → Handler → Extractor → Hologram
```
        """)
    elif file_type == "html":
        path.write_text("""
        <html>
        <body>
            <h1>Demo Article</h1>
            <p>This demonstrates HTML content extraction with concept mining.</p>
            <ul>
                <li>Quantum computing advances</li>
                <li>Neural interface development</li>
                <li>Holographic data storage</li>
            </ul>
        </body>
        </html>
        """)

def update_hologram(waveform_data):
    """Update holographic display (client-side implementation)"""
    # This would be implemented in your Svelte component
    pass

# === Run the server ===
if __name__ == "__main__":
    print("""
    🌌 UNIFIED MULTI-MODAL INGESTION PIPELINE 🌌
    ==========================================
    
    Starting server with:
    - FastAPI endpoints: http://localhost:8000
    - WebSocket hologram: ws://localhost:8000/ws/hologram
    - Demo page: http://localhost:8000/
    
    Supported formats:
    ✅ PDF (with OCR)
    ✅ Text files
    ✅ HTML documents
    ✅ Markdown
    ✅ Images (JPEG, PNG, etc.)
    🚧 Audio (coming soon)
    🚧 Video (coming soon)
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
