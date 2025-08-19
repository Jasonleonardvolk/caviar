"""
Simplified MCP Dashboard - Standalone Version
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

class SimpleMCPDashboard:
    """Simplified dashboard that works without full MCP imports"""
    
    def __init__(self, port: int = 8088):
        self.port = port
        self.app = FastAPI(title="MCP Metacognitive Dashboard (Simplified)", version="1.0.0")
        self.mock_data = {
            "insights": [],
            "metrics": {
                "avg_response_time": 1.23,
                "error_rate": 0.02,
                "total_queries": 1337,
                "insights_count": 42,
                "kb_size": 256
            },
            "is_running": False
        }
        
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
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "kaizen_available": True,
                "kaizen_running": self.mock_data["is_running"],
                "insights_count": self.mock_data["metrics"]["insights_count"],
                "knowledge_base_size": self.mock_data["metrics"]["kb_size"],
                "agents": {
                    "kaizen": {"enabled": True, "version": "1.0.0"},
                    "daniel": {"enabled": True, "version": "1.0.0"}
                }
            }
        
        @self.app.get("/api/insights")
        async def get_insights(limit: int = 10):
            """Get recent insights (mock data)"""
            # Generate some mock insights
            if not self.mock_data["insights"]:
                self.mock_data["insights"] = [
                    {
                        "id": f"insight_{i}",
                        "type": ["performance", "error_pattern", "usage_pattern"][i % 3],
                        "description": f"Mock insight {i}: System performance could be improved",
                        "confidence": 0.7 + (i % 3) * 0.1,
                        "timestamp": datetime.utcnow().isoformat(),
                        "applied": i % 2 == 0
                    }
                    for i in range(10)
                ]
            
            return {
                "status": "success",
                "insights": self.mock_data["insights"][:limit]
            }
        
        @self.app.post("/api/kaizen/start")
        async def start_kaizen():
            """Start Kaizen (mock)"""
            self.mock_data["is_running"] = True
            return {"status": "started", "message": "Kaizen started (mock mode)"}
        
        @self.app.post("/api/kaizen/stop")
        async def stop_kaizen():
            """Stop Kaizen (mock)"""
            self.mock_data["is_running"] = False
            return {"status": "stopped", "message": "Kaizen stopped (mock mode)"}
        
        @self.app.post("/api/kaizen/analyze")
        async def trigger_analysis():
            """Trigger analysis (mock)"""
            # Add a new mock insight
            new_insight = {
                "id": f"insight_{len(self.mock_data['insights'])}",
                "type": "analysis_result",
                "description": "New insight from manual analysis",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat(),
                "applied": False
            }
            self.mock_data["insights"].insert(0, new_insight)
            self.mock_data["metrics"]["insights_count"] += 1
            
            return {
                "status": "completed",
                "events_analyzed": 100,
                "insights_generated": 1,
                "analysis_time": 2.5
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)
                    
                    # Simulate changing metrics
                    import random
                    self.mock_data["metrics"]["avg_response_time"] += random.uniform(-0.1, 0.1)
                    self.mock_data["metrics"]["total_queries"] += random.randint(0, 5)
                    
                    update = {
                        "type": "metrics_update",
                        "data": self.mock_data["metrics"]
                    }
                    await websocket.send_json(update)
                    
            except WebSocketDisconnect:
                pass
        
        @self.app.get("/kaizen/metrics", response_class=Response)
        async def prometheus_metrics():
            """Mock Prometheus metrics"""
            from fastapi.responses import Response
            
            metrics = """# HELP kaizen_insights_total Total number of insights generated
# TYPE kaizen_insights_total counter
kaizen_insights_total{insight_type="performance"} 15
kaizen_insights_total{insight_type="error_pattern"} 8
kaizen_insights_total{insight_type="usage_pattern"} 12

# HELP kaizen_avg_response_time_seconds Average response time
# TYPE kaizen_avg_response_time_seconds gauge
kaizen_avg_response_time_seconds 1.23

# HELP kaizen_error_rate Current error rate
# TYPE kaizen_error_rate gauge
kaizen_error_rate 2.0
"""
            return Response(content=metrics, media_type="text/plain")
    
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
        .notice {
            background: #f39c12;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
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
        
        <div class="notice">
            ⚠️ Running in simplified mode - Full Kaizen integration pending
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
                log('WebSocket error');
            };
        }
        
        // Initialize
        updateStatus();
        loadInsights();
        connectWebSocket();
        
        // Periodic updates
        setInterval(updateStatus, 10000);
        setInterval(loadInsights, 30000);
        
        log('Dashboard initialized (simplified mode)');
    </script>
</body>
</html>
"""
    
    async def run(self):
        """Run the web interface"""
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
    parser = argparse.ArgumentParser(description="MCP Metacognitive Dashboard (Simplified)")
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to run the web server on (default: 8088)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run web interface
    dashboard = SimpleMCPDashboard(port=args.port)
    
    print(f"\n🚀 Starting MCP Metacognitive Dashboard (Simplified) on http://localhost:{args.port}")
    print(f"📊 Mock Prometheus metrics available at http://localhost:{args.port}/kaizen/metrics")
    print(f"📚 API documentation available at http://localhost:{args.port}/docs")
    print(f"\n⚠️  Running in simplified mode with mock data - Full Kaizen integration pending\n")
    
    await dashboard.run()


if __name__ == "__main__":
    asyncio.run(main())
