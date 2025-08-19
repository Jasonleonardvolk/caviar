#!/usr/bin/env python3
"""
Fix TORI UI Issues - Comprehensive Solution
Fixes:
1. Missing mathjs dependency in frontend
2. Backend API not running
3. Health check connection errors
"""

import os
import subprocess
import sys
import json
import time
import psutil
from pathlib import Path

def print_status(message, status="INFO"):
    """Print formatted status message"""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "ERROR": "\033[91m",
        "WARNING": "\033[93m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def check_port_in_use(port):
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return True
    return False

def kill_process_on_port(port):
    """Kill process using a specific port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print_status(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}", "WARNING")
                    proc.kill()
                    time.sleep(1)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def install_frontend_dependencies():
    """Install missing frontend dependencies"""
    print_status("Installing frontend dependencies...", "INFO")
    
    frontend_path = Path("C:/Users/jason/Desktop/tori/kha/tori_ui_svelte")
    if not frontend_path.exists():
        print_status("Frontend directory not found!", "ERROR")
        return False
    
    # Install dependencies
    try:
        subprocess.run("npm install", cwd=frontend_path, shell=True, check=True)
        print_status("Frontend dependencies installed successfully!", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "ERROR")
        return False

def check_backend_config():
    """Check and fix backend configuration"""
    print_status("Checking backend configuration...", "INFO")
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print_status("main.py not found!", "ERROR")
        return False
    
    # Check if .env exists
    if not Path(".env").exists():
        print_status(".env file not found, creating from example...", "WARNING")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print_status(".env file created", "SUCCESS")
        else:
            # Create minimal .env file
            with open(".env", "w", encoding='utf-8') as f:
                f.write("""# TORI Environment Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=True
ENABLE_CORS=True
CORS_ORIGINS=["http://localhost:5173", "http://192.168.1.115:5173", "http://172.29.96.1:5173"]
""")
            print_status("Created minimal .env file", "SUCCESS")
    
    return True

def start_backend():
    """Start the backend API server"""
    print_status("Starting backend API server...", "INFO")
    
    # Check if port 8000 is in use
    if check_port_in_use(8000):
        print_status("Port 8000 is already in use", "WARNING")
        kill_process_on_port(8000)
    
    # Check Python environment
    if Path(".venv").exists():
        if sys.platform == "win32":
            python_cmd = str(Path(".venv/Scripts/python.exe"))
        else:
            python_cmd = str(Path(".venv/bin/python"))
    else:
        python_cmd = sys.executable
    
    # Start the backend
    try:
        # Try to start with uvicorn first
        backend_cmd = [python_cmd, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        print_status(f"Starting backend with command: {' '.join(backend_cmd)}", "INFO")
        
        # Start backend in background
        if sys.platform == "win32":
            backend_process = subprocess.Popen(
                backend_cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            backend_process = subprocess.Popen(
                backend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Wait a bit for backend to start
        time.sleep(5)
        
        # Check if backend started successfully
        if backend_process.poll() is None:
            print_status("Backend API server started successfully!", "SUCCESS")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print_status(f"Backend failed to start: {stderr.decode()}", "ERROR")
            
            # Try alternative start method
            print_status("Trying alternative start method...", "INFO")
            backend_cmd = [python_cmd, "main.py"]
            backend_process = subprocess.Popen(
                backend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)
            
            if backend_process.poll() is None:
                print_status("Backend started with alternative method!", "SUCCESS")
                return backend_process
            else:
                print_status("Failed to start backend with alternative method", "ERROR")
                return None
            
    except Exception as e:
        print_status(f"Error starting backend: {e}", "ERROR")
        return None

def update_vite_config():
    """Update Vite configuration to properly proxy API calls"""
    print_status("Updating Vite configuration...", "INFO")
    
    vite_config_path = Path("tori_ui_svelte/vite.config.ts")
    if not vite_config_path.exists():
        vite_config_path = Path("tori_ui_svelte/vite.config.js")
    
    if vite_config_path.exists():
        # Read current config with UTF-8 encoding
        with open(vite_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check if proxy is already configured
        if "proxy" not in config_content:
            print_status("Adding proxy configuration to Vite...", "INFO")
            
            # Create new config with proxy
            new_config = """import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                secure: false,
                ws: true,
                configure: (proxy, _options) => {
                    proxy.on('error', (err, _req, _res) => {
                        console.log('proxy error', err);
                    });
                    proxy.on('proxyReq', (proxyReq, req, _res) => {
                        console.log('Sending Request:', req.method, req.url);
                    });
                    proxy.on('proxyRes', (proxyRes, req, _res) => {
                        console.log('Received Response:', proxyRes.statusCode, req.url);
                    });
                }
            }
        }
    }
});
"""
            with open(vite_config_path, 'w', encoding='utf-8') as f:
                f.write(new_config)
            
            print_status("Vite configuration updated!", "SUCCESS")
        else:
            print_status("Vite proxy already configured", "INFO")
    else:
        print_status("Vite config file not found!", "WARNING")

def create_launch_script():
    """Create a comprehensive launch script"""
    print_status("Creating launch script...", "INFO")
    
    if sys.platform == "win32":
        script_path = Path("LAUNCH_TORI_FIXED.bat")
        script_content = """@echo off
echo Starting TORI System...

REM Activate virtual environment
if exist .venv\\Scripts\\activate.bat (
    call .venv\\Scripts\\activate.bat
) else (
    echo Virtual environment not found!
    exit /b 1
)

REM Start backend
echo Starting backend API...
start /B python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Wait for backend to start
timeout /t 5 /nobreak > nul

REM Start frontend
echo Starting frontend...
cd tori_ui_svelte
start cmd /k "npm run dev -- --host"
cd ..

echo.
echo TORI System started!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo.
echo Press any key to stop all services...
pause > nul

REM Kill processes
taskkill /F /IM node.exe 2>nul
taskkill /F /IM python.exe 2>nul
"""
    else:
        script_path = Path("launch_tori_fixed.sh")
        script_content = """#!/bin/bash
echo "Starting TORI System..."

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found!"
    exit 1
fi

# Start backend
echo "Starting backend API..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd tori_ui_svelte
npm run dev -- --host &
FRONTEND_PID=$!
cd ..

echo ""
echo "TORI System started!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID" INT
wait
"""
        # Make script executable
        os.chmod(script_path, 0o755)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print_status(f"Launch script created: {script_path}", "SUCCESS")

def main():
    """Main function to fix all issues"""
    print_status("=== TORI Issue Fixer ===", "INFO")
    
    # Change to project directory
    project_dir = Path("C:/Users/jason/Desktop/tori/kha")
    if project_dir.exists():
        os.chdir(project_dir)
    
    # Step 1: Install frontend dependencies
    if not install_frontend_dependencies():
        print_status("Failed to install frontend dependencies", "ERROR")
        return
    
    # Step 2: Check backend configuration
    if not check_backend_config():
        print_status("Backend configuration check failed", "ERROR")
        return
    
    # Step 3: Update Vite configuration
    update_vite_config()
    
    # Step 4: Create launch script
    create_launch_script()
    
    # Step 5: Start services
    print_status("\n=== Starting Services ===", "INFO")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print_status("Failed to start backend", "ERROR")
        return
    
    # Start frontend
    print_status("Starting frontend development server...", "INFO")
    frontend_path = Path("C:/Users/jason/Desktop/tori/kha/tori_ui_svelte")
    
    try:
        if sys.platform == "win32":
            frontend_cmd = "npm run dev -- --host"
        else:
            frontend_cmd = "npm run dev -- --host"
        
        subprocess.run(frontend_cmd, cwd=frontend_path, shell=True)
    except KeyboardInterrupt:
        print_status("\nShutting down services...", "INFO")
        if backend_process:
            backend_process.terminate()
        print_status("Services stopped", "SUCCESS")
    except Exception as e:
        print_status(f"Error starting frontend: {e}", "ERROR")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\nOperation cancelled by user", "WARNING")
    except Exception as e:
        print_status(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)
