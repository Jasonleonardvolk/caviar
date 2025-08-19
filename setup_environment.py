#!/usr/bin/env python3
"""
Complete environment setup for TORI
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        return False

def main():
    print("🔧 TORI Complete Environment Setup")
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Core dependencies
    dependencies = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "spacy>=3.7.0",
        "fastapi[all]>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "deepdiff>=6.3.0",
        "sympy>=1.12",
        "PyPDF2>=3.0.0",
        "websockets>=11.0",
        "redis>=4.6.0",
        "aiohttp>=3.8.0",
        "psutil>=5.9.0",
        "celery[redis]>=5.3.0",
        "requests",
        "numpy",
        "pandas",
        "scikit-learn",
    ]
    
    # Install all dependencies
    for dep in dependencies:
        run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep.split()[0]}")
    
    # Install spaCy model
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Installing spaCy English model")
    
    # Create pip freeze file
    print("\n📦 Creating requirements.lock...")
    with open("requirements.lock", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    
    print("\n✅ Environment setup complete!")
    print("\nTo recreate this exact environment later:")
    print("  pip install -r requirements.lock")

if __name__ == "__main__":
    main()