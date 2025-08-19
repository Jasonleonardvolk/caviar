#!/usr/bin/env python
"""
Enhanced Setup Script for MCP Server Creator v3
Installs all dependencies and configures the system
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def install_dependencies():
    """Install all required and optional dependencies"""
    
    print("Installing MCP Server Creator v3 Dependencies")
    print("=" * 50)
    
    # Core dependencies
    core_deps = [
        "PyPDF2",        # Basic PDF reading
        "httpx",         # Async HTTP
        "aiohttp",       # Parallel downloads
        "requests",      # CrossRef API
    ]
    
    # Enhanced dependencies
    enhanced_deps = [
        "pdfminer.six",  # Advanced PDF extraction
        "pdfplumber",    # Table extraction
        "python-dotenv", # Environment configuration
    ]
    
    # Optional dependencies
    optional_deps = [
        "numpy",         # Statistical analysis
        "scikit-learn",  # Clustering and ML
        "websockets",    # Telemetry
    ]
    
    # Install core
    print("\n1. Installing core dependencies...")
    for dep in core_deps:
        print(f"   Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    # Install enhanced
    print("\n2. Installing enhanced PDF processing...")
    for dep in enhanced_deps:
        print(f"   Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except:
            print(f"   WARNING: {dep} installation failed, some features will be limited")
    
    # Install optional
    print("\n3. Installing optional dependencies...")
    for dep in optional_deps:
        print(f"   Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✓ {dep} installed successfully")
        except:
            print(f"   ⚠ {dep} not installed (optional)")

def setup_environment():
    """Set up environment configuration"""
    print("\n4. Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    env_example = Path(__file__).parent / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("   Creating .env from .env.example...")
        env_file.write_text(env_example.read_text())
        print("   ✓ .env file created")
    else:
        print("   ✓ .env already exists")
    
    # Set development defaults
    if sys.platform == "win32":
        print("\n   Windows environment setup:")
        print("   set DEFAULT_ANALYSIS_INTERVAL=300")
        print("   set KAIZEN_ANALYSIS_INTERVAL=300")
    else:
        print("\n   Unix environment setup:")
        print("   export DEFAULT_ANALYSIS_INTERVAL=300")
        print("   export KAIZEN_ANALYSIS_INTERVAL=300")

def patch_registry():
    """Apply registry patches"""
    print("\n5. Patching agent registry...")
    
    try:
        # Check if registry exists
        registry_path = Path(__file__).parent.parent / "core" / "agent_registry.py"
        if registry_path.exists():
            print("   Found agent_registry.py")
            
            # Check if already patched
            content = registry_path.read_text()
            if "supervised_agent_start" in content:
                print("   ✓ Registry already has supervisor support")
            else:
                print("   ⚠ Registry needs supervisor patch")
                print("   Run: python -c \"from create.registry_watchdog_patch import add_supervisor_to_registry; add_supervisor_to_registry(agent_registry)\"")
        else:
            print("   ⚠ Registry not found at expected location")
    except Exception as e:
        print(f"   Error checking registry: {e}")

def verify_installation():
    """Verify the installation"""
    print("\n6. Verifying installation...")
    
    # Check imports
    imports_ok = True
    
    try:
        import PyPDF2
        print("   ✓ PyPDF2 available")
    except:
        print("   ✗ PyPDF2 not available")
        imports_ok = False
    
    try:
        import aiohttp
        print("   ✓ aiohttp available (parallel downloads)")
    except:
        print("   ⚠ aiohttp not available (limited to sequential)")
    
    try:
        import pdfminer
        print("   ✓ pdfminer available (advanced extraction)")
    except:
        print("   ⚠ pdfminer not available (basic extraction only)")
    
    try:
        import numpy
        import sklearn
        print("   ✓ numpy/sklearn available (full analytics)")
    except:
        print("   ⚠ numpy/sklearn not available (limited analytics)")
    
    # Check cache directory
    cache_dir = Path.home() / ".tori_pdf_cache"
    if cache_dir.exists():
        print(f"   ✓ Cache directory exists: {cache_dir}")
    else:
        cache_dir.mkdir(exist_ok=True)
        print(f"   ✓ Created cache directory: {cache_dir}")
    
    return imports_ok

def create_test_structure():
    """Create test structure for verification"""
    print("\n7. Creating test structure...")
    
    test_dir = Path(__file__).parent / "test"
    test_dir.mkdir(exist_ok=True)
    
    # Create test script
    test_script = test_dir / "test_v3_features.py"
    test_content = '''#!/usr/bin/env python
"""Test v3 features"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def test_enhanced_pipeline():
    """Test enhanced PDF pipeline"""
    try:
        from create.enhanced_pdf_pipeline import EnhancedPDFProcessor
        
        processor = EnhancedPDFProcessor()
        print("✓ Enhanced pipeline available")
        
        # Test cache
        print(f"  Cache dir: {processor.cache_dir}")
        print(f"  Cache entries: {len(processor.cache_index)}")
        
        return True
    except Exception as e:
        print(f"✗ Enhanced pipeline error: {e}")
        return False

async def test_server_creation():
    """Test server creation"""
    try:
        from create.mk_server import create_server
        
        # Create test server
        create_server("test_v3", "Test v3 server", [])
        
        # Check if created
        server_path = Path(__file__).parent.parent.parent / "agents" / "test_v3"
        if server_path.exists():
            print("✓ Server creation works")
            return True
        else:
            print("✗ Server creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Server creation error: {e}")
        return False

async def main():
    """Run all tests"""
    print("Testing MCP Server Creator v3...")
    print("=" * 40)
    
    tests = [
        test_enhanced_pipeline(),
        test_server_creation(),
    ]
    
    results = await asyncio.gather(*tests)
    
    if all(results):
        print("\\n✓ All tests passed!")
    else:
        print("\\n⚠ Some tests failed")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    test_script.write_text(test_content)
    print(f"   ✓ Created test script: {test_script}")

def main():
    """Run complete setup"""
    print("MCP Server Creator v3 Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Apply patches
    patch_registry()
    
    # Verify installation
    if verify_installation():
        print("\n✓ Core installation successful!")
    else:
        print("\n⚠ Some core components missing")
    
    # Create test structure
    create_test_structure()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Set environment variables (see above)")
    print("2. Run tests: python create/test/test_v3_features.py")
    print("3. Create your first v3 server:")
    print("   python create/mk_server.py create myserver 'Enhanced server'")
    print("\nFor PDF processing:")
    print("   python create/mk_server.py create research 'Research server' paper1.pdf paper2.pdf")
    print("\nFor bulk creation:")
    print("   python create/mk_server.py bulk-create ./papers/")

if __name__ == "__main__":
    main()
