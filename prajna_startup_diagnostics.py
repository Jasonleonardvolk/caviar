#!/usr/bin/env python3
"""
🔬 PRAJNA STARTUP DIAGNOSTICS - Find exactly why Prajna hangs
"""

import subprocess
import sys
import os
import time
import socket
from pathlib import Path
from datetime import datetime

class PrajnaStartupDiagnostics:
    """Surgical diagnosis of Prajna startup hanging"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.prajna_dir = self.script_dir / "prajna"
        self.start_script = self.prajna_dir / "start_prajna.py"
        
    def diagnose_prajna_startup(self):
        """Complete Prajna startup diagnostic"""
        print("🔬 SURGICAL DIAGNOSTICS - PRAJNA STARTUP ANALYSIS")
        print("=" * 60)
        
        # Step 1: Environment Check
        self.check_prajna_environment()
        
        # Step 2: Prajna Directory Structure
        self.check_prajna_structure()
        
        # Step 3: Test Prajna Imports
        self.test_prajna_imports()
        
        # Step 4: Test Direct Prajna Startup
        self.test_direct_prajna_startup()
        
        # Step 5: Port Conflict Analysis
        self.analyze_port_conflicts()
        
        print("\n🎯 PRAJNA DIAGNOSTIC COMPLETE")
        
    def check_prajna_environment(self):
        """Check Prajna environment"""
        print("\n🔍 STEP 1: PRAJNA ENVIRONMENT CHECK")
        print("-" * 40)
        
        print(f"Script directory: {self.script_dir}")
        print(f"Prajna directory: {self.prajna_dir}")
        print(f"Prajna exists: {'✅' if self.prajna_dir.exists() else '❌'}")
        print(f"Start script: {self.start_script}")
        print(f"Start script exists: {'✅' if self.start_script.exists() else '❌'}")
        
        # Check PYTHONPATH setup
        parent_dir = self.prajna_dir.parent
        print(f"Parent dir for PYTHONPATH: {parent_dir}")
        
        # Test if prajna package is importable
        sys.path.insert(0, str(parent_dir))
        try:
            import prajna
            print("✅ Prajna package importable")
        except ImportError as e:
            print(f"❌ Prajna package import failed: {e}")
    
    def check_prajna_structure(self):
        """Check Prajna directory structure"""
        print("\n🔍 STEP 2: PRAJNA STRUCTURE CHECK")
        print("-" * 40)
        
        if not self.prajna_dir.exists():
            print("❌ Prajna directory doesn't exist")
            return
        
        # Check critical files
        critical_files = [
            "__init__.py",
            "start_prajna.py", 
            "config/prajna_config.py",
            "api/prajna_api.py"
        ]
        
        for file_path in critical_files:
            full_path = self.prajna_dir / file_path
            exists = full_path.exists()
            print(f"{file_path}: {'✅' if exists else '❌'}")
            if exists:
                size = full_path.stat().st_size
                print(f"  Size: {size:,} bytes")
        
        # List actual contents
        print("\n📁 Actual Prajna directory contents:")
        try:
            for item in self.prajna_dir.iterdir():
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")
        except Exception as e:
            print(f"❌ Error listing directory: {e}")
    
    def test_prajna_imports(self):
        """Test Prajna imports that startup script needs"""
        print("\n🔍 STEP 3: PRAJNA IMPORTS TEST")
        print("-" * 40)
        
        # Add parent to path for prajna package import
        parent_dir = self.prajna_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        imports_to_test = [
            ("prajna", "Base package"),
            ("prajna.config.prajna_config", "Config module"), 
            ("prajna.api.prajna_api", "API module"),
        ]
        
        for module_name, description in imports_to_test:
            try:
                __import__(module_name)
                print(f"✅ {description}: {module_name}")
            except ImportError as e:
                print(f"❌ {description}: {module_name} - {e}")
            except Exception as e:
                print(f"⚠️ {description}: {module_name} - {e}")
        
        # Test specific imports that might cause issues
        try:
            from prajna.config.prajna_config import load_config
            print("✅ load_config function importable")
        except Exception as e:
            print(f"❌ load_config import failed: {e}")
        
        try:
            from prajna.api.prajna_api import app
            print("✅ FastAPI app importable")
        except Exception as e:
            print(f"❌ FastAPI app import failed: {e}")
    
    def test_direct_prajna_startup(self):
        """Test Prajna startup directly with detailed monitoring"""
        print("\n🔍 STEP 4: DIRECT PRAJNA STARTUP TEST")
        print("-" * 40)
        
        if not self.start_script.exists():
            print("❌ Start script not found")
            return
        
        print("🚀 Testing direct Prajna startup...")
        
        # Prepare command exactly like the launcher does
        prajna_cmd = [
            sys.executable,
            str(self.start_script),
            "--port", "8001",
            "--host", "0.0.0.0", 
            "--log-level", "DEBUG"  # More verbose for debugging
        ]
        
        # Environment setup exactly like launcher
        env = os.environ.copy()
        parent_dir = self.prajna_dir.parent
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{parent_dir}{os.pathsep}{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(parent_dir)
        
        print(f"Command: {' '.join(prajna_cmd)}")
        print(f"Working dir: {self.prajna_dir}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        
        try:
            process = subprocess.Popen(
                prajna_cmd,
                cwd=str(self.prajna_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"📊 Process started with PID: {process.pid}")
            
            # Monitor for 30 seconds with detailed output
            start_time = time.time()
            output_lines = []
            
            while time.time() - start_time < 30:
                # Check if process ended
                if process.poll() is not None:
                    print(f"📊 Process exited with code: {process.poll()}")
                    break
                
                # Read output line by line
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        output_lines.append(line)
                        print(f"PRAJNA: {line}")
                        
                        # Check for key indicators
                        if "Uvicorn running" in line:
                            print("✅ Uvicorn started!")
                        elif "Application startup complete" in line:
                            print("✅ Application startup complete!")
                        elif "error" in line.lower() or "exception" in line.lower():
                            print(f"🚨 ERROR DETECTED: {line}")
                        elif "port" in line.lower() and "8001" in line:
                            print(f"🔌 Port reference: {line}")
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error reading output: {e}")
                    break
            
            # Test port after some time
            if time.time() - start_time >= 10:
                print("🔌 Testing port 8001 after 10 seconds...")
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
            self.analyze_prajna_output(output_lines)
            
        except Exception as e:
            print(f"❌ Failed to start Prajna process: {e}")
    
    def test_port_connection(self, port):
        """Test port connection"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    print(f"✅ Port {port} is bound and accepting connections!")
                else:
                    print(f"❌ Port {port} connection failed (error: {result})")
        except Exception as e:
            print(f"❌ Socket test failed: {e}")
    
    def analyze_port_conflicts(self):
        """Check for port conflicts"""
        print("\n🔍 STEP 5: PORT CONFLICT ANALYSIS")
        print("-" * 40)
        
        ports_to_check = [8001, 8002, 8080, 3000]
        
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
    
    def analyze_prajna_output(self, output_lines):
        """Analyze Prajna output for issues"""
        print(f"\n📊 PRAJNA OUTPUT ANALYSIS ({len(output_lines)} lines)")
        print("-" * 40)
        
        startup_indicators = {
            "config_loaded": False,
            "uvicorn_started": False,
            "app_startup": False,
            "port_bound": False,
            "errors": []
        }
        
        for line in output_lines:
            line_lower = line.lower()
            
            if "configuration" in line_lower and "loaded" in line_lower:
                startup_indicators["config_loaded"] = True
            elif "uvicorn running" in line_lower:
                startup_indicators["uvicorn_started"] = True
            elif "application startup complete" in line_lower:
                startup_indicators["app_startup"] = True
            elif "port" in line_lower and "8001" in line:
                startup_indicators["port_bound"] = True
            elif any(error_word in line_lower for error_word in ["error", "exception", "failed", "traceback"]):
                startup_indicators["errors"].append(line)
        
        print("Startup progress:")
        for indicator, status in startup_indicators.items():
            if indicator != "errors":
                print(f"  {indicator}: {'✅' if status else '❌'}")
        
        if startup_indicators["errors"]:
            print(f"\n🚨 ERRORS DETECTED ({len(startup_indicators['errors'])}):")
            for error in startup_indicators["errors"]:
                print(f"  {error}")
        
        # Diagnosis
        print(f"\n🎯 DIAGNOSIS:")
        if not startup_indicators["config_loaded"]:
            print("❌ Configuration not loaded - check prajna_config.py")
        elif not startup_indicators["uvicorn_started"]:
            print("❌ Uvicorn not started - check API module imports")
        elif not startup_indicators["port_bound"]:
            print("❌ Port not bound - check for port conflicts or binding issues")
        else:
            print("✅ Startup appears successful based on output")

def main():
    """Run Prajna startup diagnostics"""
    diagnostics = PrajnaStartupDiagnostics()
    diagnostics.diagnose_prajna_startup()

if __name__ == "__main__":
    main()
