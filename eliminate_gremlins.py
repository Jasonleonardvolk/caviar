#!/usr/bin/env python3
"""
TORI System Final Gremlin Elimination Script
Ensures all components are properly configured and running
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print("🔧 TORI FINAL GREMLIN ELIMINATION SCRIPT")
    print("="*60 + "\n")

def check_and_fix_soliton_frontend():
    """Ensure frontend soliton memory uses correct field names"""
    print("🔍 Checking soliton frontend field names...")
    
    soliton_file = Path("tori_ui_svelte/src/lib/services/solitonMemory.ts")
    if not soliton_file.exists():
        print("❌ solitonMemory.ts not found!")
        return False
    
    # Read the file
    content = soliton_file.read_text(encoding='utf-8')
    
    # Check if fix is needed
    if "user_id: uid" in content:
        print("❌ Found incorrect field name 'user_id', fixing...")
        content = content.replace("user_id: uid", "userId: uid")
        soliton_file.write_text(content, encoding='utf-8')
        print("✅ Fixed soliton field name mismatch")
    else:
        print("✅ Soliton field names are correct")
    
    return True

def update_env_file():
    """Update .env file with correct API port"""
    print("\n🔍 Checking frontend .env configuration...")
    
    env_file = Path("tori_ui_svelte/.env")
    if env_file.exists():
        lines = env_file.read_text(encoding='utf-8').splitlines()
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith("VITE_API_BASE_URL="):
                if "localhost:3001" in line:
                    lines[i] = "VITE_API_BASE_URL=http://localhost:8002"
                    updated = True
                    print("✅ Updated VITE_API_BASE_URL to port 8002")
        
        if updated:
            env_file.write_text('\n'.join(lines), encoding='utf-8')
        else:
            print("✅ .env file already configured correctly")
    
    return True

def verify_prajna_syntax():
    """Verify prajna_api.py has no syntax errors"""
    print("\n🔍 Verifying prajna_api.py syntax...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "prajna_api.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ prajna_api.py syntax is valid")
            return True
        else:
            print("❌ Syntax error in prajna_api.py:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"⚠️ Could not verify syntax: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n🔍 Checking Python dependencies...")
    
    required = [
        "fastapi",
        "uvicorn",
        "psutil",
        "requests",
        "numpy",
        "pydantic"
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: poetry install")
        return False
    else:
        print("✅ All core dependencies installed")
        return True

def check_ports():
    """Check if required ports are available"""
    print("\n🔍 Checking port availability...")
    
    import socket
    ports = {
        8002: "API Server",
        5173: "Frontend",
        8100: "MCP Metacognitive"
    }
    
    available = True
    for port, service in ports.items():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                print(f"✅ Port {port} available for {service}")
        except OSError:
            print(f"⚠️ Port {port} busy ({service}) - launcher will handle this")
            available = False
    
    return available

def create_data_directories():
    """Ensure all data directories exist"""
    print("\n🔍 Creating data directories...")
    
    dirs = [
        "data/cognitive",
        "data/memory_vault", 
        "data/concept_mesh",
        "data/eigenvalue_monitor",
        "data/lyapunov",
        "data/koopman",
        "logs",
        "tmp"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ All data directories created")
    return True

def check_concept_mesh_data():
    """Verify concept mesh data integrity"""
    print("\n🔍 Checking concept mesh data...")
    
    data_file = Path("concept_mesh/data.json")
    if data_file.exists():
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure it's a dict with proper structure
            if isinstance(data, list):
                print("⚠️ Converting concept mesh data from list to dict...")
                new_data = {
                    "concepts": data,
                    "metadata": {
                        "version": "1.0",
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                }
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2)
                print("✅ Fixed concept mesh data structure")
            else:
                print("✅ Concept mesh data structure is valid")
        except Exception as e:
            print(f"⚠️ Could not verify concept mesh data: {e}")
    else:
        print("ℹ️ No concept mesh data file found (will be created on first use)")
    
    return True

def main():
    print_banner()
    
    # Change to TORI directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    all_good = True
    
    # Run all checks
    checks = [
        check_and_fix_soliton_frontend,
        update_env_file,
        verify_prajna_syntax,
        check_dependencies,
        check_ports,
        create_data_directories,
        check_concept_mesh_data
    ]
    
    for check in checks:
        if not check():
            all_good = False
    
    # Final summary
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL GREMLINS ELIMINATED! TORI is ready to launch!")
        print("\nTo start TORI, run:")
        print("  poetry run python enhanced_launcher.py")
    else:
        print("⚠️ Some issues remain. Please address them before launching.")
        print("\nOnce fixed, run:")
        print("  poetry run python enhanced_launcher.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
