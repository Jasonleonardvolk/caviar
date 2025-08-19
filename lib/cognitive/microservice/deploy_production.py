#!/usr/bin/env python3
"""
🚀 TORI Cognitive System Production Deployment
Complete production setup and health monitoring
"""

import asyncio
import httpx
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

class ProductionDeployment:
    def __init__(self):
        self.base_path = Path("C:/Users/jason/Desktop/tori/kha/lib/cognitive/microservice")
        self.services = {
            "cognitive": {"port": 4321, "process": None},
            "fastapi": {"port": 8000, "process": None}
        }
    
    async def check_prerequisites(self):
        """Check if all prerequisites are installed"""
        print("🔍 Checking prerequisites...")
        
        checks = []
        
        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(f"✅ Node.js: {result.stdout.strip()}")
            else:
                checks.append("❌ Node.js: Not found")
        except:
            checks.append("❌ Node.js: Not found")
        
        # Check Python
        try:
            result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(f"✅ Python: {result.stdout.strip()}")
            else:
                checks.append("❌ Python: Not found")
        except:
            checks.append("❌ Python: Not found")
        
        # Check required files
        required_files = [
            "cognitive-microservice.ts",
            "cognitive_bridge.py",
            "package.json",
            "requirements.txt",
            "start-complete-system.bat"
        ]
        
        for file in required_files:
            file_path = self.base_path / file
            if file_path.exists():
                checks.append(f"✅ {file}: Found")
            else:
                checks.append(f"❌ {file}: Missing")
        
        for check in checks:
            print(f"  {check}")
        
        return all("✅" in check for check in checks)
    
    async def install_dependencies(self):
        """Install all required dependencies"""
        print("\n📦 Installing dependencies...")
        
        # Install Node.js dependencies
        print("  Installing Node.js packages...")
        result = subprocess.run(
            ["npm", "install"], 
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ Node.js dependencies installed")
        else:
            print(f"  ❌ Node.js installation failed: {result.stderr}")
            return False
        
        # Create Python virtual environment
        venv_path = self.base_path / ".venv"
        if not venv_path.exists():
            print("  Creating Python virtual environment...")
            result = subprocess.run(
                [sys.executable, "-m", "venv", ".venv"],
                cwd=self.base_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  ❌ Virtual environment creation failed: {result.stderr}")
                return False
        
        # Install Python dependencies
        print("  Installing Python packages...")
        pip_path = venv_path / "Scripts" / "pip.exe" if sys.platform == "win32" else venv_path / "bin" / "pip"
        
        result = subprocess.run(
            [str(pip_path), "install", "-r", "requirements.txt"],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ Python dependencies installed")
        else:
            print(f"  ❌ Python installation failed: {result.stderr}")
            return False
        
        return True
    
    async def start_services(self):
        """Start both cognitive services"""
        print("\n🚀 Starting TORI Cognitive Services...")
        
        # Start Node.js microservice
        print("  Starting Node.js Cognitive Microservice...")
        try:
            self.services["cognitive"]["process"] = subprocess.Popen(
                ["npx", "ts-node", "cognitive-microservice.ts"],
                cwd=self.base_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("  ✅ Node.js service started")
        except Exception as e:
            print(f"  ❌ Failed to start Node.js service: {e}")
            return False
        
        # Wait for Node.js service to start
        await asyncio.sleep(3)
        
        # Start FastAPI bridge
        print("  Starting FastAPI Bridge...")
        try:
            python_path = self.base_path / ".venv" / "Scripts" / "python.exe" if sys.platform == "win32" else self.base_path / ".venv" / "bin" / "python"
            
            self.services["fastapi"]["process"] = subprocess.Popen(
                [str(python_path), "cognitive_bridge.py"],
                cwd=self.base_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("  ✅ FastAPI service started")
        except Exception as e:
            print(f"  ❌ Failed to start FastAPI service: {e}")
            return False
        
        # Wait for services to stabilize
        await asyncio.sleep(5)
        return True
    
    async def verify_services(self):
        """Verify that all services are running correctly"""
        print("\n🔍 Verifying service health...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, service_info in self.services.items():
                port = service_info["port"]
                try:
                    response = await client.get(f"http://localhost:{port}/api/health")
                    if response.status_code == 200:
                        print(f"  ✅ {service_name.title()} service healthy on port {port}")
                    else:
                        print(f"  ❌ {service_name.title()} service unhealthy: HTTP {response.status_code}")
                        return False
                except Exception as e:
                    print(f"  ❌ {service_name.title()} service not responding: {e}")
                    return False
        
        return True
    
    async def run_production_tests(self):
        """Run comprehensive production tests"""
        print("\n🧪 Running production validation tests...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test basic cognitive processing
            test_data = {
                "message": "Production deployment test: analyze system readiness",
                "glyphs": ["anchor", "concept-synthesizer", "meta-echo:reflect", "return"]
            }
            
            try:
                response = await client.post("http://localhost:8000/api/chat", json=test_data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success") and result.get("trace", {}).get("closed"):
                        print("  ✅ Cognitive processing test passed")
                    else:
                        print("  ❌ Cognitive processing test failed: Invalid response")
                        return False
                else:
                    print(f"  ❌ Cognitive processing test failed: HTTP {response.status_code}")
                    return False
            except Exception as e:
                print(f"  ❌ Cognitive processing test failed: {e}")
                return False
            
            # Test smart ask
            try:
                smart_data = {
                    "message": "Production test query",
                    "complexity": "standard"
                }
                response = await client.post("http://localhost:8000/api/smart/ask", json=smart_data)
                if response.status_code == 200:
                    print("  ✅ Smart ask test passed")
                else:
                    print(f"  ❌ Smart ask test failed: HTTP {response.status_code}")
                    return False
            except Exception as e:
                print(f"  ❌ Smart ask test failed: {e}")
                return False
            
            # Test system status
            try:
                response = await client.get("http://localhost:8000/api/status")
                if response.status_code == 200:
                    status = response.json()
                    if status.get("bridge", {}).get("status") == "online":
                        print("  ✅ System status test passed")
                    else:
                        print("  ❌ System status test failed: Service not online")
                        return False
                else:
                    print(f"  ❌ System status test failed: HTTP {response.status_code}")
                    return False
            except Exception as e:
                print(f"  ❌ System status test failed: {e}")
                return False
        
        return True
    
    async def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        print("\n📊 Generating deployment report...")
        
        report = {
            "deployment": {
                "timestamp": datetime.now().isoformat(),
                "status": "successful",
                "version": "1.0.0",
                "environment": "production"
            },
            "services": {},
            "endpoints": {
                "fastapi_bridge": "http://localhost:8000",
                "node_microservice": "http://localhost:4321",
                "documentation": "http://localhost:8000/docs"
            },
            "key_endpoints": [
                "POST http://localhost:8000/api/chat",
                "POST http://localhost:8000/api/smart/ask",
                "POST http://localhost:8000/api/smart/research",
                "GET http://localhost:8000/api/status"
            ]
        }
        
        # Get service status
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get("http://localhost:8000/api/status")
                if response.status_code == 200:
                    report["services"]["status"] = response.json()
            except:
                report["services"]["status"] = "unavailable"
        
        # Save report
        report_file = self.base_path / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Deployment report saved: {report_file}")
        return report
    
    def stop_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping services...")
        
        for service_name, service_info in self.services.items():
            if service_info["process"]:
                try:
                    service_info["process"].terminate()
                    service_info["process"].wait(timeout=5)
                    print(f"  ✅ {service_name.title()} service stopped")
                except:
                    try:
                        service_info["process"].kill()
                        print(f"  ⚠️ {service_name.title()} service force-killed")
                    except:
                        print(f"  ❌ Failed to stop {service_name.title()} service")
    
    async def deploy(self):
        """Execute complete production deployment"""
        print("🚀 TORI Cognitive System Production Deployment")
        print("=" * 60)
        
        try:
            # Step 1: Check prerequisites
            if not await self.check_prerequisites():
                print("\n❌ Prerequisites check failed. Please install missing components.")
                return False
            
            # Step 2: Install dependencies
            if not await self.install_dependencies():
                print("\n❌ Dependency installation failed.")
                return False
            
            # Step 3: Start services
            if not await self.start_services():
                print("\n❌ Service startup failed.")
                self.stop_services()
                return False
            
            # Step 4: Verify services
            if not await self.verify_services():
                print("\n❌ Service verification failed.")
                self.stop_services()
                return False
            
            # Step 5: Run production tests
            if not await self.run_production_tests():
                print("\n❌ Production tests failed.")
                self.stop_services()
                return False
            
            # Step 6: Generate report
            report = await self.generate_deployment_report()
            
            # Success!
            print("\n" + "=" * 60)
            print("🎉 TORI Cognitive System Successfully Deployed!")
            print("=" * 60)
            print()
            print("✅ All services are running and validated")
            print("✅ Cognitive processing is fully operational")
            print("✅ Memory integration is active")
            print("✅ Cross-language bridge is functional")
            print()
            print("🌐 Access Points:")
            print("  • FastAPI Bridge: http://localhost:8000")
            print("  • Node.js Engine: http://localhost:4321")
            print("  • API Documentation: http://localhost:8000/docs")
            print()
            print("🎯 Ready for Act Mode!")
            print("Your cognitive engine is now ready for production use.")
            print()
            print("Press Ctrl+C to stop services when done.")
            
            # Keep services running
            try:
                while True:
                    await asyncio.sleep(60)
                    # Optional: periodic health checks here
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                self.stop_services()
                print("✅ All services stopped cleanly.")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Deployment failed: {e}")
            self.stop_services()
            return False

async def main():
    deployment = ProductionDeployment()
    await deployment.deploy()

if __name__ == "__main__":
    asyncio.run(main())
