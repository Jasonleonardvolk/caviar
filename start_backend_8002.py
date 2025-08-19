#!/usr/bin/env python3
"""
Start TORI Backend API on Port 8002
This script starts the backend API server on the correct port that the frontend expects.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

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

def start_backend_api():
    """Start the backend API on port 8002"""
    print_status("Starting TORI Backend API on port 8002...", "INFO")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print_status("main.py not found! Please run this from the project root.", "ERROR")
        return False
    
    # Determine Python command
    if Path(".venv").exists():
        if sys.platform == "win32":
            python_cmd = str(Path(".venv/Scripts/python.exe"))
        else:
            python_cmd = str(Path(".venv/bin/python"))
    else:
        python_cmd = sys.executable
    
    # Start the backend on port 8002
    try:
        print_status("Starting backend with uvicorn on port 8002...", "INFO")
        cmd = [
            python_cmd, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8002",  # Important: Port 8002!
            "--reload"
        ]
        
        print_status(f"Command: {' '.join(cmd)}", "INFO")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print_status("\nBackend stopped by user", "WARNING")
    except Exception as e:
        print_status(f"Error starting backend: {e}", "ERROR")
        return False
    
    return True

if __name__ == "__main__":
    # Change to project directory if needed
    project_dir = Path("C:/Users/jason/Desktop/tori/kha")
    if project_dir.exists() and Path.cwd() != project_dir:
        os.chdir(project_dir)
        print_status(f"Changed to project directory: {project_dir}", "INFO")
    
    # Start the backend
    if not start_backend_api():
        sys.exit(1)
