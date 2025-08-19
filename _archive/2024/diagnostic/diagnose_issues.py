#!/usr/bin/env python3
"""
Detailed diagnostic script to identify remaining TORI issues
"""

import os
import sys
import json
import subprocess
import socket
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def diagnose_syntax_errors():
    """Check for Python syntax errors in key files"""
    print_section("Python Syntax Check")
    
    files_to_check = [
        "prajna_api.py",
        "enhanced_launcher.py",
        "api/routes/soliton_production.py",
        "api/routes/soliton_router.py",
        "api/routes/concept_mesh.py"
    ]
    
    errors_found = []
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"❌ Syntax error in {file_path}:")
                    print(f"   {result.stderr}")
                    errors_found.append((file_path, result.stderr))
                else:
                    print(f"✅ {file_path} - OK")
            except Exception as e:
                print(f"⚠️ Could not check {file_path}: {e}")
                errors_found.append((file_path, str(e)))
        else:
            print(f"⚠️ {file_path} not found")
    
    return errors_found

def check_imports():
    """Check if all required imports work"""
    print_section("Import Check")
    
    import_tests = [
        ("FastAPI", "fastapi", "FastAPI"),
        ("Uvicorn", "uvicorn", None),
        ("Requests", "requests", None),
        ("NumPy", "numpy", None),
        ("Pydantic", "pydantic", None),
        ("PSUtil", "psutil", None),
        ("Concept Mesh", "concept_mesh", None),
        ("Prajna", "prajna.api.prajna_api", "app"),
    ]
    
    failed_imports = []
    
    for name, module, attr in import_tests:
        try:
            imported = __import__(module, fromlist=[attr] if attr else [])
            if attr and not hasattr(imported, attr):
                print(f"❌ {name}: Module imported but missing '{attr}'")
                failed_imports.append((name, f"Missing attribute {attr}"))
            else:
                print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append((name, str(e)))
    
    return failed_imports

def check_port_conflicts_detailed():
    """Detailed port conflict check"""
    print_section("Port Availability")
    
    ports = {
        8002: ("API Server", True),
        5173: ("Frontend", True),
        8100: ("MCP Metacognitive", False),
        8001: ("Alternative API", False),
        3001: ("Old API Port", False)
    }
    
    conflicts = []
    
    for port, (service, required) in ports.items():
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                print(f"✅ Port {port} ({service}) - Available")
        except OSError as e:
            if required:
                print(f"❌ Port {port} ({service}) - BUSY (REQUIRED)")
                # Try to find what's using it
                try:
                    result = subprocess.run(
                        f'netstat -ano | findstr :{port}',
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.stdout:
                        print(f"   Process using port: {result.stdout.strip()}")
                except:
                    pass
                conflicts.append((port, service))
            else:
                print(f"⚠️ Port {port} ({service}) - Busy (optional)")
    
    return conflicts

def check_file_permissions():
    """Check file permissions for key directories"""
    print_section("File Permissions")
    
    dirs_to_check = [
        "data",
        "logs",
        "tmp",
        "concept_mesh",
        "tori_ui_svelte/node_modules",
        ".venv"
    ]
    
    permission_issues = []
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            if os.access(path, os.W_OK):
                print(f"✅ {dir_path} - Writable")
            else:
                print(f"❌ {dir_path} - NOT writable")
                permission_issues.append(dir_path)
        else:
            print(f"⚠️ {dir_path} - Does not exist (will be created)")
    
    return permission_issues

def check_environment_variables():
    """Check critical environment variables"""
    print_section("Environment Variables")
    
    env_vars = {
        "PYTHONPATH": "Python import path",
        "PYTHONIOENCODING": "UTF-8 encoding",
        "TORI_ENV": "TORI environment mode",
        "NODE_ENV": "Node.js environment"
    }
    
    missing_vars = []
    
    for var, description in env_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✅ {var} = {value}")
        else:
            print(f"⚠️ {var} not set ({description})")
            if var in ["PYTHONIOENCODING"]:
                missing_vars.append(var)
    
    return missing_vars

def check_node_modules():
    """Check if frontend dependencies are installed"""
    print_section("Frontend Dependencies")
    
    package_json = Path("tori_ui_svelte/package.json")
    node_modules = Path("tori_ui_svelte/node_modules")
    
    if not package_json.exists():
        print("❌ package.json not found")
        return ["package.json missing"]
    
    if not node_modules.exists():
        print("❌ node_modules not found - run 'npm install' in tori_ui_svelte/")
        return ["node_modules missing"]
    
    # Check for key dependencies
    key_deps = ["svelte", "vite", "@sveltejs/kit"]
    missing_deps = []
    
    for dep in key_deps:
        dep_path = node_modules / dep
        if dep_path.exists():
            print(f"✅ {dep} - Installed")
        else:
            print(f"❌ {dep} - Missing")
            missing_deps.append(dep)
    
    return missing_deps

def check_config_files():
    """Check configuration files"""
    print_section("Configuration Files")
    
    config_files = {
        "pyproject.toml": "Poetry configuration",
        "tori_ui_svelte/vite.config.js": "Vite configuration",
        "tori_ui_svelte/svelte.config.js": "Svelte configuration",
        ".env": "Environment variables (optional)"
    }
    
    missing_configs = []
    
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path} - Present")
            
            # Special checks for certain files
            if file_path == "tori_ui_svelte/vite.config.js":
                content = Path(file_path).read_text()
                if "8002" in content:
                    print("   ✅ Proxy configured for port 8002")
                else:
                    print("   ⚠️ Proxy might not be configured for port 8002")
        else:
            if "optional" not in description:
                print(f"❌ {file_path} - Missing ({description})")
                missing_configs.append(file_path)
            else:
                print(f"⚠️ {file_path} - Missing ({description})")
    
    return missing_configs

def main():
    print("\n" + "🔧 "*20)
    print("TORI DETAILED DIAGNOSTIC")
    print("🔧 "*20)
    
    all_issues = []
    
    # Run all diagnostics
    syntax_errors = diagnose_syntax_errors()
    if syntax_errors:
        all_issues.append(("Syntax Errors", syntax_errors))
    
    import_errors = check_imports()
    if import_errors:
        all_issues.append(("Import Errors", import_errors))
    
    port_conflicts = check_port_conflicts_detailed()
    if port_conflicts:
        all_issues.append(("Port Conflicts", port_conflicts))
    
    permission_issues = check_file_permissions()
    if permission_issues:
        all_issues.append(("Permission Issues", permission_issues))
    
    missing_env = check_environment_variables()
    if missing_env:
        all_issues.append(("Environment Variables", missing_env))
    
    node_issues = check_node_modules()
    if node_issues:
        all_issues.append(("Frontend Dependencies", node_issues))
    
    config_issues = check_config_files()
    if config_issues:
        all_issues.append(("Configuration Files", config_issues))
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    if not all_issues:
        print("\n✅ No issues found! TORI should be ready to launch.")
    else:
        print(f"\n❌ Found {len(all_issues)} categories of issues:\n")
        
        for category, issues in all_issues:
            print(f"🔸 {category}:")
            for issue in issues:
                if isinstance(issue, tuple):
                    print(f"   - {issue[0]}: {issue[1]}")
                else:
                    print(f"   - {issue}")
            print()
        
        # Provide specific fixes
        print_section("RECOMMENDED FIXES")
        
        for category, issues in all_issues:
            if category == "Syntax Errors":
                print("Fix syntax errors in Python files:")
                for file, error in issues:
                    print(f"  - Edit {file} and fix the syntax error")
                    
            elif category == "Import Errors":
                print("Install missing Python packages:")
                print("  poetry install")
                
            elif category == "Port Conflicts":
                print("Free up required ports:")
                for port, service in issues:
                    print(f"  - Kill process using port {port} ({service})")
                print("  Or use: taskkill /F /IM node.exe")
                print("         taskkill /F /IM python.exe")
                
            elif category == "Frontend Dependencies":
                print("Install frontend dependencies:")
                print("  cd tori_ui_svelte")
                print("  npm install")
                print("  cd ..")
                
            elif category == "Permission Issues":
                print("Fix directory permissions:")
                for dir_path in issues:
                    print(f"  - Grant write permission to {dir_path}")

if __name__ == "__main__":
    main()
