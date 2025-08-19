#!/usr/bin/env python3
"""
TORI Isolated Component Startup
Starts each component with health checks and proper isolation
"""

import asyncio
import subprocess
import time
import sys
import os
import socket
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Dict, Optional, List

class ComponentStatus(Enum):
    NOT_STARTED = "not_started"
    STARTING = "starting"
    HEALTHY = "healthy"
    FAILED = "failed"
    DEGRADED = "degraded"

class IsolatedComponentLauncher:
    def __init__(self):
        self.components = {
            "redis": {
                "priority": 1,
                "dependencies": [],
                "start_command": "redis-server",
                "port": 6379,
                "health_check": self.check_redis,
                "critical": True
            },
            "api_server": {
                "priority": 2,
                "dependencies": ["redis"],
                "start_command": "python -m uvicorn enhanced_launcher:app --port 8002 --host 0.0.0.0",
                "port": 8002,
                "health_check": self.check_api,
                "critical": True
            },
            "mcp_server": {
                "priority": 3,
                "dependencies": ["api_server"],
                "start_command": "python mcp_metacognitive/server.py",
                "port": 8100,
                "health_check": self.check_mcp,
                "critical": False
            },
            "audio_bridge": {
                "priority": 4,
                "dependencies": ["api_server"],
                "start_command": "python audio_hologram_bridge.py",
                "port": 8765,
                "health_check": self.check_websocket,
                "critical": False
            },
            "concept_bridge": {
                "priority": 4,
                "dependencies": ["api_server"],
                "start_command": "python concept_mesh_hologram_bridge.py",
                "port": 8766,
                "health_check": self.check_websocket,
                "critical": False
            },
            "frontend": {
                "priority": 5,
                "dependencies": ["api_server"],
                "start_command": "cd tori_ui_svelte && npm run dev",
                "port": 5173,
                "health_check": self.check_frontend,
                "critical": True
            }
        }
        
        self.status = {name: ComponentStatus.NOT_STARTED for name in self.components}
        self.processes = {}
        
    def kill_port(self, port: int):
        """Kill process on a specific port"""
        if os.name == 'nt':  # Windows
            os.system(f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}\') do taskkill /PID %a /F 2>nul')
        else:  # Unix
            os.system(f'lsof -ti:{port} | xargs kill -9 2>/dev/null')
    
    async def check_redis(self, port: int) -> bool:
        """Check if Redis is healthy"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=port, socket_connect_timeout=1)
            return r.ping()
        except:
            return False
    
    async def check_api(self, port: int) -> bool:
        """Check if API server is healthy"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/api/health", timeout=5) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def check_mcp(self, port: int) -> bool:
        """Check if MCP server is healthy"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/api/system/status", timeout=5) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def check_websocket(self, port: int) -> bool:
        """Check if WebSocket is accepting connections"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    async def check_frontend(self, port: int) -> bool:
        """Check if frontend is serving"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}", timeout=10) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def wait_for_health(self, name: str, health_check, timeout: int = 30):
        """Wait for component to become healthy"""
        start_time = time.time()
        port = self.components[name]["port"]
        
        print(f"  ⏳ Waiting for {name} to become healthy...")
        
        while time.time() - start_time < timeout:
            if await health_check(port):
                return True
            await asyncio.sleep(1)
            print(f"    Still waiting for {name}... ({int(time.time() - start_time)}s)")
        
        return False
    
    async def start_component(self, name: str) -> bool:
        """Start a single component with retry logic"""
        component = self.components[name]
        
        # Check dependencies first
        for dep in component["dependencies"]:
            if self.status[dep] != ComponentStatus.HEALTHY:
                print(f"❌ Cannot start {name}: dependency {dep} is not healthy")
                return False
        
        print(f"\n🚀 Starting {name}...")
        self.status[name] = ComponentStatus.STARTING
        
        # Clean up port first
        self.kill_port(component["port"])
        await asyncio.sleep(2)  # Wait for port to be freed
        
        # Try to start with retry logic
        for attempt in range(3):
            try:
                print(f"  Attempt {attempt + 1}/3 to start {name}")
                
                # Start the process
                if os.name == 'nt':  # Windows
                    if name == "frontend":
                        # Special handling for frontend
                        process = subprocess.Popen(
                            component["start_command"],
                            shell=True,
                            cwd=os.getcwd(),
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                    else:
                        process = subprocess.Popen(
                            component["start_command"],
                            shell=True,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                else:
                    process = subprocess.Popen(
                        component["start_command"],
                        shell=True
                    )
                
                self.processes[name] = process
                
                # Wait for health check
                healthy = await self.wait_for_health(name, component["health_check"])
                
                if healthy:
                    self.status[name] = ComponentStatus.HEALTHY
                    print(f"✅ {name} started successfully on port {component['port']}")
                    return True
                else:
                    print(f"  ⚠️  {name} started but health check failed")
                    process.terminate()
                    
            except Exception as e:
                print(f"  ⚠️  Error starting {name}: {e}")
            
            if attempt < 2:
                print(f"  Retrying in 3 seconds...")
                await asyncio.sleep(3)
        
        self.status[name] = ComponentStatus.FAILED
        print(f"❌ Failed to start {name} after 3 attempts")
        
        # Check if it's critical
        if component.get("critical", False):
            print(f"⚠️  {name} is a critical component. System may not function properly.")
        
        return False
    
    async def launch_all(self):
        """Launch all components in dependency order"""
        print("🚀 TORI Isolated Component Launcher")
        print("=" * 60)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Sort by priority
        sorted_components = sorted(
            self.components.items(),
            key=lambda x: x[1]["priority"]
        )
        
        failed_critical = False
        
        for name, component in sorted_components:
            success = await self.start_component(name)
            
            if not success and component.get("critical", False):
                failed_critical = True
                print(f"\n❌ Critical component {name} failed. Stopping launch.")
                break
        
        # Generate status report
        self.generate_status_report()
        
        if failed_critical:
            print("\n❌ Launch failed due to critical component failure.")
            print("Run 'python tori_emergency_fix.py' to fix common issues.")
            return False
        
        return True
    
    def generate_status_report(self):
        """Generate component status report"""
        print("\n" + "="*60)
        print("TORI SYSTEM STATUS REPORT")
        print("="*60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        
        for name, status in self.status.items():
            icon = {
                ComponentStatus.HEALTHY: "✅",
                ComponentStatus.FAILED: "❌",
                ComponentStatus.DEGRADED: "⚠️",
                ComponentStatus.NOT_STARTED: "⏸️",
                ComponentStatus.STARTING: "🔄"
            }[status]
            
            port = self.components[name]["port"]
            critical = "CRITICAL" if self.components[name].get("critical", False) else "Optional"
            print(f"{icon} {name:20} - {status.value:15} (port {port}) [{critical}]")
        
        healthy_count = sum(1 for s in self.status.values() if s == ComponentStatus.HEALTHY)
        total_count = len(self.components)
        
        print("-"*60)
        print(f"Overall Health: {healthy_count}/{total_count} components healthy")
        
        if healthy_count == total_count:
            print("\n✅ All systems operational!")
            print("\n🌐 Access Points:")
            print("  - Frontend: http://localhost:5173")
            print("  - API Docs: http://localhost:8002/docs")
            print("  - Health Check: http://localhost:8002/api/health")
        elif healthy_count >= 3:  # At least core components
            print("\n⚠️  System running with reduced functionality")
            print("Core services are up, but some features may be unavailable.")
        else:
            print("\n❌ System is not operational")
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        print("\n🛑 Shutting down TORI components...")
        
        # Shutdown in reverse order
        sorted_components = sorted(
            self.components.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        for name, _ in sorted_components:
            if name in self.processes and self.processes[name].poll() is None:
                print(f"  Stopping {name}...")
                self.processes[name].terminate()
                
        # Wait a bit for graceful shutdown
        await asyncio.sleep(2)
        
        # Force kill any remaining
        for name, process in self.processes.items():
            if process.poll() is None:
                process.kill()
        
        print("✅ All components stopped")

async def main():
    launcher = IsolatedComponentLauncher()
    
    try:
        success = await launcher.launch_all()
        
        if success:
            print("\nPress Ctrl+C to stop all services")
            
            # Keep running until interrupted
            await asyncio.Event().wait()
            
    except KeyboardInterrupt:
        print("\n\nReceived shutdown signal...")
    finally:
        await launcher.shutdown()

if __name__ == "__main__":
    # Change to TORI directory
    os.chdir("C:\\Users\\jason\\Desktop\\tori\\kha")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)
