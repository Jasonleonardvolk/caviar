"""
MCP Metacognitive Web Interface
===============================

A web dashboard for monitoring and controlling the MCP metacognitive system,
including Kaizen insights, metrics, and system status.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import MCP components
try:
    # Try absolute imports first
    from kha.mcp_metacognitive.agents.kaizen import KaizenImprovementEngine
    from kha.mcp_metacognitive.agents.kaizen_metrics import create_metrics_app, metrics_router
    from kha.mcp_metacognitive.core.psi_archive import psi_archive
    from kha.mcp_metacognitive.core.agent_registry import agent_registry
    KAIZEN_AVAILABLE = True
except ImportError:
    try:
        # Try relative imports as fallback
        from ..agents.kaizen import KaizenImprovementEngine
        from ..agents.kaizen_metrics import create_metrics_app, metrics_router
        from ..core.psi_archive import psi_archive
        from ..core.agent_registry import agent_registry
        KAIZEN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import Kaizen components: {e}")
        KAIZEN_AVAILABLE = False

logger = logging.getLogger(__name__)

class MCPWebInterface:
    """Web interface for MCP metacognitive system"""
    
    def __init__(self, port: int = 8088):
        self.port = port
        self.app = FastAPI(title="MCP Metacognitive Dashboard", version="1.0.0")
        self.kaizen_engine = None
        self.websocket_clients = set()
        
        # Setup middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status"""
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "kaizen_available": KAIZEN_AVAILABLE,
                "kaizen_running": False,
                "agents": {},
                "metrics": {}
            }
            
            if KAIZEN_AVAILABLE and self.kaizen_engine:
                status["kaizen_running"] = self.kaizen_engine.is_running
                status["insights_count"] = len(self.kaizen_engine.insights)
                status["knowledge_base_size"] = len(self.kaizen_engine.knowledge_base)
                
            # Get agent status
            try:
                agents = agent_registry.get_all_agents()
                status["agents"] = {
                    name: {
                        "enabled": agent._metadata.get("enabled", False),
                        "version": agent._metadata.get("version", "unknown")
                    }
                    for name, agent in agents.items()
                }
            except:
                pass
                
            return JSONResponse(status)
        
        @self.app.get("/api/insights")
        async def get_insights(limit: int = 10):
            """Get recent Kaizen insights"""
            if not KAIZEN_AVAILABLE or not self.kaizen_engine:
                raise HTTPException(status_code=503, detail="Kaizen not available")
                
            result = await self.kaizen_engine.get_recent_insights(limit=limit)
            return JSONResponse(result)
        
        @self.app.post("/api/kaizen/start")
        async def start_kaizen():
            """Start Kaizen continuous improvement"""
            if not KAIZEN_AVAILABLE or not self.kaizen_engine:
                raise HTTPException(status_code=503, detail="Kaizen not available")
                
            result = await self.kaizen_engine.start_continuous_improvement()
            return JSONResponse(result)
        
        @self.app.post("/api/kaizen/stop")
        async def stop_kaizen():
            """Stop Kaizen continuous improvement"""
            if not KAIZEN_AVAILABLE or not self.kaizen_engine:
                raise HTTPException(status_code=503, detail="Kaizen not available")
                
            result = await self.kaizen_engine.stop_continuous_improvement()
            return JSONResponse(result)
        
        @self.app.post("/api/kaizen/analyze")
        async def trigger_analysis():
            """Trigger immediate Kaizen analysis"""
            if not KAIZEN_AVAILABLE or not self.kaizen_engine:
                raise HTTPException(status_code=503, detail="Kaizen not available")
                
            result = await self.kaizen_engine.run_analysis_cycle()
            return JSONResponse(result)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)
                    
                    if KAIZEN_AVAILABLE and self.kaizen_engine:
                        update = {
                            "type": "metrics_update",
                            "data": {
                                "avg_response_time": self.kaizen_engine.metrics.get_average_response_time(),
                                "error_rate": self.kaizen_engine.metrics.get_error_rate(),
                                "total_queries": self.kaizen_engine.metrics.total_queries,
                                "insights_count": len(self.kaizen_engine.insights)
                            }
                        }
                        await websocket.send_json(update)
                        
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
                
        # Include metrics router if available
        if KAIZEN_AVAILABLE:
            self.app.include_router(metrics_router)
            
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Metacognitive Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        .button:hover {
            background: #2980b9;
        }
        .button.danger {
            background: #e74c3c;
        }
        .button.danger:hover {
            background: #c0392b;
        }
        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status.running {
            background: #2ecc71;
        }
        .status.stopped {
            background: #e74c3c;
        }
        .insights-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .insight {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .insight:last-child {
            border-bottom: none;
        }
        .insight-type {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            background: #ecf0f1;
            margin-right: 10px;
        }
        .confidence {
            float: right;
            color: #7f8c8d;
        }
        #log {
            background: #2c3e50;
            color: #2ecc71;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 MCP Metacognitive Dashboard</h1>
            <p>Real-time monitoring and control for TORI's self-improvement system</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Kaizen Status</h3>
                <p>
                    <span id="kaizen-status" class="status stopped"></span>
                    <span id="kaizen-status-text">Stopped</span>
                </p>
                <button class="button" onclick="startKaizen()">Start</button>
                <button class="button danger" onclick="stopKaizen()">Stop</button>
                <button class="button" onclick="triggerAnalysis()">Analyze Now</button>
            </div>
            
            <div class="card">
                <div class="label">Average Response Time</div>
                <div id="avg-response-time" class="metric">-</div>
            </div>
            
            <div class="card">
                <div class="label">Error Rate</div>
                <div id="error-rate" class="metric">-</div>
            </div>
            
            <div class="card">
                <div class="label">Total Queries</div>
                <div id="total-queries" class="metric">-</div>
            </div>
            
            <div class="card">
                <div class="label">Active Insights</div>
                <div id="insights-count" class="metric">-</div>
            </div>
            
            <div class="card">
                <div class="label">Knowledge Base Size</div>
                <div id="kb-size" class="metric">-</div>
            </div>
        </div>
        
        <div class="grid" style="margin-top: 20px;">
            <div class="card" style="grid-column: span 2;">
                <h3>Recent Insights</h3>
                <div id="insights-list" class="insights-list">
                    <p style="color: #999;">Loading insights...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>System Log</h3>
                <div id="log"></div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 20px; color: #999;">
            <a href="/kaizen/metrics" target="_blank">Prometheus Metrics</a> |
            <a href="/docs" target="_blank">API Documentation</a>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                const kaizenStatus = document.getElementById('kaizen-status');
                const kaizenStatusText = document.getElementById('kaizen-status-text');
                
                if (status.kaizen_running) {
                    kaizenStatus.className = 'status running';
                    kaizenStatusText.textContent = 'Running';
                } else {
                    kaizenStatus.className = 'status stopped';
                    kaizenStatusText.textContent = 'Stopped';
                }
                
                document.getElementById('insights-count').textContent = status.insights_count || '-';
                document.getElementById('kb-size').textContent = status.knowledge_base_size || '-';
                
            } catch (error) {
                log('Error updating status: ' + error);
            }
        }
        
        async function loadInsights() {
            try {
                const response = await fetch('/api/insights?limit=10');
                const data = await response.json();
                
                const listDiv = document.getElementById('insights-list');
                if (data.insights && data.insights.length > 0) {
                    listDiv.innerHTML = data.insights.map(insight => `
                        <div class="insight">
                            <span class="insight-type">${insight.type}</span>
                            <span class="confidence">${(insight.confidence * 100).toFixed(0)}%</span>
                            <div>${insight.description}</div>
                            <small style="color: #999;">${new Date(insight.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                } else {
                    listDiv.innerHTML = '<p style="color: #999;">No insights available</p>';
                }
            } catch (error) {
                log('Error loading insights: ' + error);
            }
        }
        
        async function startKaizen() {
            try {
                const response = await fetch('/api/kaizen/start', { method: 'POST' });
                const result = await response.json();
                log('Kaizen: ' + result.message);
                updateStatus();
            } catch (error) {
                log('Error starting Kaizen: ' + error);
            }
        }
        
        async function stopKaizen() {
            try {
                const response = await fetch('/api/kaizen/stop', { method: 'POST' });
                const result = await response.json();
                log('Kaizen: ' + result.message);
                updateStatus();
            } catch (error) {
                log('Error stopping Kaizen: ' + error);
            }
        }
        
        async function triggerAnalysis() {
            try {
                log('Triggering analysis...');
                const response = await fetch('/api/kaizen/analyze', { method: 'POST' });
                const result = await response.json();
                log(`Analysis complete: ${result.insights_generated} insights generated`);
                loadInsights();
            } catch (error) {
                log('Error triggering analysis: ' + error);
            }
        }
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:' + window.location.port + '/ws');
            
            ws.onopen = () => {
                log('Connected to server');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'metrics_update') {
                    const metrics = data.data;
                    document.getElementById('avg-response-time').textContent = 
                        metrics.avg_response_time ? metrics.avg_response_time.toFixed(2) + 's' : '-';
                    document.getElementById('error-rate').textContent = 
                        metrics.error_rate ? (metrics.error_rate * 100).toFixed(1) + '%' : '-';
                    document.getElementById('total-queries').textContent = 
                        metrics.total_queries || '-';
                }
            };
            
            ws.onclose = () => {
                log('Disconnected from server');
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = (error) => {
                log('WebSocket error: ' + error);
            };
        }
        
        // Initialize
        updateStatus();
        loadInsights();
        connectWebSocket();
        
        // Periodic updates
        setInterval(updateStatus, 10000);
        setInterval(loadInsights, 30000);
        
        log('Dashboard initialized');
    </script>
</body>
</html>
"""
    
    async def initialize(self):
        """Initialize the web interface"""
        if KAIZEN_AVAILABLE:
            # Create Kaizen engine
            self.kaizen_engine = KaizenImprovementEngine()
            logger.info("Kaizen engine initialized")
            
            # Start Kaizen if configured
            if self.kaizen_engine.config.get("enable_auto_start", False):
                await self.kaizen_engine.start_continuous_improvement()
                logger.info("Kaizen auto-started")
        else:
            logger.warning("Kaizen not available - running in limited mode")
    
    async def run(self):
        """Run the web interface"""
        await self.initialize()
        
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Metacognitive Web Interface")
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to run the web server on (default: 8088)"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run web interface
    web_interface = MCPWebInterface(port=args.port)
    
    print(f"\n🚀 Starting MCP Metacognitive Dashboard on http://localhost:{args.port}")
    print(f"📊 Prometheus metrics available at http://localhost:{args.port}/kaizen/metrics")
    print(f"📚 API documentation available at http://localhost:{args.port}/docs\n")
    
    await web_interface.run()


if __name__ == "__main__":
    asyncio.run(main())
