#!/usr/bin/env python3
"""
🔬 SURGICAL DIAGNOSTICS - Precision Analysis of Prajna Startup Issue
Analyzes the exact point of failure in the startup sequence
"""

import subprocess
import sys
import os
import time
import socket
import json
import requests
from pathlib import Path
from datetime import datetime

class SurgicalDiagnostics:
    """Precise diagnosis of Prajna startup issues"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.prajna_dir = self.script_dir / "prajna"
        self.start_script = self.prajna_dir / "start_prajna.py"
        
    def diagnose_full_chain(self):
        """Complete diagnostic chain"""
        print("🔬 SURGICAL DIAGNOSTICS - PRAJNA STARTUP ANALYSIS")
        print("=" * 60)
        
        # Step 1: Environment Check
        self.check_environment()
        
        # Step 2: File System Check
        self.check_filesystem()
        
        # Step 3: Python Path Check
        self.check_python_paths()
        
        # Step 4: Direct Process Test
        self.test_direct_process()
        
        # Step 5: Port Binding Analysis
        self.analyze_port_binding()
        
        # Step 6: Launcher vs Direct Comparison
        self.compare_launcher_vs_direct()
        
        print("\n🎯 DIAGNOSTIC COMPLETE")
        
    def check_environment(self):
        """Check Python environment and dependencies"""
        print("\n🔍 STEP 1: ENVIRONMENT CHECK")
        print("-" * 30)
        
        print(f"Python Executable: {sys.executable}")
        print(f"Python Version: {sys.version}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        
        # Check if we can import key modules
        try:
            import uvicorn
            print("✅ uvicorn available")
        except ImportError as e:
            print(f"❌ uvicorn import failed: {e}")
        
        try:
            import fastapi
            print("✅ fastapi available")
        except ImportError as e:
            print(f"❌ fastapi import failed: {e}")
        
    def check_filesystem(self):
        """Check file system and Prajna structure"""
        print("\n🔍 STEP 2: FILESYSTEM CHECK")
        print("-" * 30)
        
        print(f"Script dir: {self.script_dir} ({'EXISTS' if self.script_dir.exists() else 'MISSING'})")
        print(f"Prajna dir: {self.prajna_dir} ({'EXISTS' if self.prajna_dir.exists() else 'MISSING'})")
        print(f"Start script: {self.start_script} ({'EXISTS' if self.start_script.exists() else 'MISSING'})")
        
        if self.prajna_dir.exists():
            print("\n📁 Prajna directory contents:")
            for item in self.prajna_dir.iterdir():
                print(f"  {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    
    def check_python_paths(self):
        """Check Python path configuration"""
        print("\n🔍 STEP 3: PYTHON PATH CHECK")
        print("-" * 30)
        
        # Test if prajna package can be imported
        parent_dir = self.prajna_dir.parent
        print(f"Parent directory: {parent_dir}")
        
        # Add to path temporarily
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
            print(f"✅ Added to sys.path: {parent_dir}")
        
        # Test imports
        try:
            import prajna
            print("✅ prajna package importable")
        except ImportError as e:
            print(f"❌ prajna import failed: {e}")
        
        try:
            from prajna.config.prajna_config import load_config
            print("✅ prajna.config importable")
        except ImportError as e:
            print(f"❌ prajna.config import failed: {e}")
        
        try:
            from prajna.api.prajna_api import app
            print("✅ prajna.api importable")
        except ImportError as e:
            print(f"❌ prajna.api import failed: {e}")
    
    def test_direct_process(self):
        """Test Prajna process directly with detailed monitoring"""
        print("\n🔍 STEP 4: DIRECT PROCESS TEST")
        print("-" * 30)
        
        if not self.start_script.exists():
            print("❌ Start script not found")
            return
        
        # Prepare command
        cmd = [
            sys.executable,
            str(self.start_script),
            "--port", "8001",
            "--host", "0.0.0.0",
            "--log-level", "DEBUG"
        ]
        
        # Environment setup
        env = os.environ.copy()
        parent_dir = self.prajna_dir.parent
        env['PYTHONPATH'] = str(parent_dir)
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Environment PYTHONPATH: {env['PYTHONPATH']}")
        print(f"Working directory: {self.prajna_dir}")
        
        try:
            # Start process WITHOUT console detachment
            process = subprocess.Popen(
                cmd,
                cwd=str(self.prajna_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"🚀 Process started with PID: {process.pid}")
            
            # Monitor for 15 seconds
            start_time = time.time()
            output_lines = []
            
            while time.time() - start_time < 15:
                # Check if process ended
                if process.poll() is not None:
                    print(f"📊 Process exited with code: {process.poll()}")
                    break
                
                # Read output
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        output_lines.append(line)
                        print(f"OUTPUT: {line}")
                        
                        # Check for key indicators
                        if "Uvicorn running" in line:
                            print("✅ Uvicorn server started!")
                        elif "Application startup complete" in line:
                            print("✅ Application startup complete!")
                        elif "error" in line.lower() or "exception" in line.lower():
                            print(f"🚨 ERROR DETECTED: {line}")
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error reading output: {e}")
                    break
            
            # Test port binding after 5 seconds
            if time.time() - start_time >= 5:
                self.test_port_connection(8001)
            
            # Cleanup
            if process.poll() is None:
                print("🧹 Terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # Analyze output
            self.analyze_process_output(output_lines)
            
        except Exception as e:
            print(f"❌ Failed to start process: {e}")
    
    def test_port_connection(self, port):
        """Test port connection"""
        print(f"\n🔌 Testing port {port} connection...")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    print(f"✅ Port {port} is bound and accepting connections!")
                    
                    # Test HTTP request
                    try:
                        response = requests.get(f'http://127.0.0.1:{port}/api/health', timeout=3)
                        print(f"✅ Health check: {response.status_code}")
                        if response.status_code == 200:
                            print(f"📊 Health response: {response.json()}")
                    except Exception as e:
                        print(f"❌ HTTP request failed: {e}")
                else:
                    print(f"❌ Port {port} connection failed (error: {result})")
        except Exception as e:
            print(f"❌ Socket test failed: {e}")
    
    def analyze_port_binding(self):
        """Analyze port binding behavior"""
        print("\n🔍 STEP 5: PORT BINDING ANALYSIS")
        print("-" * 30)
        
        # Check if 8001 is already in use
        ports_to_check = [8001, 8002, 8080, 3000, 5173]
        
        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    print(f"✅ Port {port} is available")
            except OSError as e:
                print(f"❌ Port {port} is in use: {e}")
                
                # Try to identify what's using it
                try:
                    result = subprocess.run(
                        ['netstat', '-ano'],
                        capture_output=True,
                        text=True
                    )
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTENING' in line:
                            print(f"   🔍 Process: {line.strip()}")
                except:
                    pass
    
    def analyze_process_output(self, output_lines):
        """Analyze process output for issues"""
        print(f"\n📊 PROCESS OUTPUT ANALYSIS ({len(output_lines)} lines)")
        print("-" * 30)
        
        startup_stages = {
            "config_loaded": False,
            "uvicorn_started": False,
            "app_startup": False,
            "server_running": False,
            "errors_found": []
        }
        
        for line in output_lines:
            line_lower = line.lower()
            
            if "configuration loaded" in line_lower:
                startup_stages["config_loaded"] = True
            elif "started server process" in line_lower:
                startup_stages["uvicorn_started"] = True
            elif "application startup complete" in line_lower:
                startup_stages["app_startup"] = True
            elif "uvicorn running" in line_lower:
                startup_stages["server_running"] = True
            elif any(error_word in line_lower for error_word in ["error", "exception", "failed", "traceback"]):
                startup_stages["errors_found"].append(line)
        
        print("Startup stages:")
        for stage, status in startup_stages.items():
            if stage != "errors_found":
                print(f"  {stage}: {'✅' if status else '❌'}")
        
        if startup_stages["errors_found"]:
            print(f"\n🚨 ERRORS DETECTED ({len(startup_stages['errors_found'])}):")
            for error in startup_stages["errors_found"]:
                print(f"  {error}")
        
        return startup_stages
    
    def compare_launcher_vs_direct(self):
        """Compare launcher behavior vs direct execution"""
        print("\n🔍 STEP 6: LAUNCHER VS DIRECT COMPARISON")
        print("-" * 30)
        
        print("🎯 KEY DIFFERENCES TO INVESTIGATE:")
        print("1. Console detachment: Launcher uses CREATE_NEW_CONSOLE")
        print("2. Process monitoring: Launcher may not see output properly")
        print("3. Health check timing: Launcher might check too early")
        print("4. Environment differences: Different working directories")
        
        print("\n💡 HYPOTHESIS:")
        print("- Prajna starts successfully but launcher can't detect it")
        print("- Console detachment prevents proper process monitoring")
        print("- Health check uses wrong timing or address")

def main():
    """Run surgical diagnostics"""
    diagnostics = SurgicalDiagnostics()
    diagnostics.diagnose_full_chain()

if __name__ == "__main__":
    main()
