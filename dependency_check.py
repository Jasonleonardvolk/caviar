#!/usr/bin/env python3
"""
TORI Dependency Checker - Comprehensive validation of all system dependencies
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path
import json

def check_python_dependencies():
    """Check Python dependencies from pyproject.toml"""
    print("🐍 Checking Python Dependencies...")
    
    # Core dependencies from pyproject.toml
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "numpy", "scipy", "scikit-learn",
        "pypdf2", "httpx", "sse_starlette", "aiofiles", "websockets",
        "python_dotenv", "psutil", "requests", "pandas", "deepdiff",
        "nltk", "msgpack", "mcp", "python_docx", "whisper", "cv2", "moviepy"
    ]
    
    optional_packages = [
        "yake", "keybert", "sentence_transformers", "spacy",
        "pytesseract", "pdf2image", "torch", "transformers"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == "pypdf2":
                importlib.import_module("PyPDF2")
            elif package == "python_dotenv":
                importlib.import_module("dotenv")
            elif package == "sse_starlette":
                importlib.import_module("sse_starlette")
            elif package == "python_docx":
                importlib.import_module("docx")
            elif package == "whisper":
                importlib.import_module("whisper")
            elif package == "cv2":
                importlib.import_module("cv2")
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            missing_required.append((package, str(e)))
            print(f"  ❌ {package}: {e}")
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} (optional)")
        except ImportError as e:
            missing_optional.append((package, str(e)))
            print(f"  ⚠️  {package} (optional): Missing")
    
    return missing_required, missing_optional

def check_rust_dependencies():
    """Check Rust/Penrose engine availability"""
    print("\n🦀 Checking Rust Dependencies...")
    
    try:
        # Check if Penrose Rust engine is available
        sys.path.insert(0, str(Path(__file__).parent / "concept_mesh"))
        from concept_mesh.similarity import penrose, BACKEND
        print(f"  ✅ Penrose engine: {BACKEND}")
        return True
    except ImportError as e:
        print(f"  ❌ Penrose engine not available: {e}")
        return False

def check_nodejs_dependencies():
    """Check Node.js and frontend dependencies"""
    print("\n📦 Checking Node.js Dependencies...")
    
    # Check if Node.js is available
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Node.js: {result.stdout.strip()}")
        else:
            print("  ❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("  ❌ Node.js not found in PATH")
        return False
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ npm: {result.stdout.strip()}")
        else:
            print("  ❌ npm not found")
            return False
    except FileNotFoundError:
        print("  ❌ npm not found in PATH")
        return False
    
    # Check frontend dependencies
    frontend_dir = Path(__file__).parent / "tori_ui_svelte"
    if frontend_dir.exists():
        node_modules = frontend_dir / "node_modules"
        if node_modules.exists():
            print("  ✅ Frontend dependencies installed")
        else:
            print("  ⚠️  Frontend dependencies not installed (run 'npm install')")
        return True
    else:
        print("  ❌ Frontend directory not found")
        return False

def check_system_dependencies():
    """Check system-level dependencies"""
    print("\n💻 Checking System Dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        print(f"  ✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ❌ Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro} (need 3.10+)")
    
    # Check if Rust is available (for building)
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Rust: {result.stdout.strip()}")
        else:
            print("  ⚠️  Rust not found (needed for building Penrose engine)")
    except FileNotFoundError:
        print("  ⚠️  Rust not found in PATH (needed for building Penrose engine)")
    
    # Check if Cargo is available
    try:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Cargo: {result.stdout.strip()}")
        else:
            print("  ⚠️  Cargo not found")
    except FileNotFoundError:
        print("  ⚠️  Cargo not found in PATH")

def check_optional_tools():
    """Check optional external tools"""
    print("\n🔧 Checking Optional Tools...")
    
    tools = [
        ("ffmpeg", "Video processing for multimedia features"),
        ("tesseract", "OCR functionality"),
        ("git", "Version control")
    ]
    
    for tool, description in tools:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"  ✅ {tool}: {version} ({description})")
            else:
                print(f"  ⚠️  {tool}: Not available ({description})")
        except FileNotFoundError:
            print(f"  ⚠️  {tool}: Not found in PATH ({description})")

def check_data_files():
    """Check for required data files"""
    print("\n📁 Checking Data Files...")
    
    required_paths = [
        "concept_mesh/data.json",
        "logs",
        "data",
        "tmp"
    ]
    
    for path_str in required_paths:
        path = Path(__file__).parent / path_str
        if path.exists():
            print(f"  ✅ {path_str}")
        else:
            print(f"  ⚠️  {path_str}: Missing (will be created automatically)")

def check_environment_variables():
    """Check important environment variables"""
    print("\n🌍 Checking Environment Variables...")
    
    important_vars = [
        ("PYTHONPATH", "Python module path"),
        ("PATH", "System PATH"),
        ("TORI_ENV", "TORI environment mode"),
        ("TMP_ROOT", "Temporary file directory")
    ]
    
    for var, description in important_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: Set ({description})")
        else:
            print(f"  ⚠️  {var}: Not set ({description})")

def main():
    """Run comprehensive dependency check"""
    print("🔍 TORI Comprehensive Dependency Check")
    print("=" * 50)
    
    # Track issues
    critical_issues = []
    warnings = []
    
    # Check all dependencies
    missing_required, missing_optional = check_python_dependencies()
    rust_ok = check_rust_dependencies()
    nodejs_ok = check_nodejs_dependencies()
    
    check_system_dependencies()
    check_optional_tools()
    check_data_files()
    check_environment_variables()
    
    # Summary
    print("\n📋 DEPENDENCY CHECK SUMMARY")
    print("=" * 50)
    
    if missing_required:
        print("\n❌ CRITICAL MISSING DEPENDENCIES:")
        for package, error in missing_required:
            print(f"  - {package}: {error}")
            critical_issues.append(f"Missing required package: {package}")
    
    if missing_optional:
        print("\n⚠️  OPTIONAL MISSING DEPENDENCIES:")
        for package, error in missing_optional:
            print(f"  - {package}")
            warnings.append(f"Missing optional package: {package}")
    
    if not rust_ok:
        critical_issues.append("Penrose Rust engine not available")
    
    if not nodejs_ok:
        critical_issues.append("Node.js/npm not properly configured")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT:")
    if critical_issues:
        print(f"❌ {len(critical_issues)} critical issues found:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\n🔧 Action required: Fix critical issues before proceeding")
        return 1
    else:
        print("✅ All critical dependencies are satisfied!")
        if warnings:
            print(f"⚠️  {len(warnings)} optional components missing (system will work but with reduced functionality)")
        else:
            print("🎉 All dependencies are fully satisfied!")
        print("\n🚀 System is ready for full functionality!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
