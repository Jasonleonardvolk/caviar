#!/usr/bin/env python
"""
Setup script for MCP Server Creator
Ensures all required directories and dependencies are in place
"""

import os
import sys
import json
from pathlib import Path
import subprocess

def ensure_directories():
    """Create required directories if they don't exist"""
    base_dir = Path(__file__).parent.parent
    
    directories = [
        base_dir / "data",
        base_dir / "agents",
        base_dir / "resources",
        base_dir / "_resources",
        base_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory: {directory}")

def check_dependencies():
    """Check and install required Python packages"""
    required = {
        "PyPDF2": "PyPDF2",
        "httpx": "httpx",
        "asyncio": None,  # Built-in
        "logging": None,  # Built-in
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {module} is installed")
        except ImportError:
            if package:
                missing.append(package)
                print(f"✗ {module} is NOT installed")
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✓ All dependencies installed")
    else:
        print("\n✓ All dependencies are already installed")

def create_default_config():
    """Create default configuration file"""
    config_path = Path(__file__).parent.parent / "config" / "servers_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not config_path.exists():
        default_config = {
            "global": {
                "analysis_interval": 300,  # 5 minutes for dev
                "production_analysis_interval": 3600,  # 1 hour for production
                "enable_watchdog": True,
                "watchdog_timeout": 60,
                "max_errors": 5,
                "log_level": "INFO"
            },
            "servers": {
                "kaizen": {
                    "analysis_interval": 3600,
                    "min_data_points": 10,
                    "enable_auto_apply": False
                },
                "daniel": {
                    "model_backend": "mock",
                    "temperature": 0.7,
                    "max_context_length": 4096
                }
            }
        }
        
        config_path.write_text(json.dumps(default_config, indent=2))
        print(f"✓ Created default config at: {config_path}")
    else:
        print(f"✓ Config already exists at: {config_path}")

def verify_agent_registry():
    """Verify the agent registry is set up correctly"""
    registry_path = Path(__file__).parent.parent / "core" / "agent_registry.py"
    
    if registry_path.exists():
        print(f"✓ Agent registry found at: {registry_path}")
        
        # Check if it has watchdog support
        content = registry_path.read_text()
        if "asyncio.wait_for" not in content:
            print("⚠ Warning: Agent registry may need watchdog support added")
            print("  Consider adding timeout handling in the registry's agent execution")
    else:
        print(f"✗ Agent registry not found at: {registry_path}")

def create_test_script():
    """Create a test script to verify server creation"""
    test_path = Path(__file__).parent / "test_server_creation.py"
    
    test_content = '''#!/usr/bin/env python
"""Test script for MCP server creation"""

import sys
import asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_server_creation():
    """Test creating and using a server"""
    from create.mk_server import create_server
    from core.agent_registry import agent_registry
    
    # Create test server
    print("Creating test server...")
    create_server("test_server", "Test server for validation", [])
    
    # Import the created server
    print("Importing test server...")
    from agents.test_server.test_server import TestServerServer
    
    # Test execution
    print("Testing server execution...")
    result = await agent_registry.get("test_server").execute({"test": True})
    print(f"Result: {result}")
    
    # Test continuous loop
    print("Testing continuous loop...")
    server = agent_registry.get("test_server")
    await server.start()
    await asyncio.sleep(2)
    await server.shutdown()
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_server_creation())
'''
    
    test_path.write_text(test_content)
    test_path.chmod(0o755)  # Make executable
    print(f"✓ Created test script at: {test_path}")

def main():
    """Run all setup tasks"""
    print("MCP Server Creator Setup")
    print("=" * 50)
    
    print("\n1. Ensuring directories...")
    ensure_directories()
    
    print("\n2. Checking dependencies...")
    check_dependencies()
    
    print("\n3. Creating default configuration...")
    create_default_config()
    
    print("\n4. Verifying agent registry...")
    verify_agent_registry()
    
    print("\n5. Creating test script...")
    create_test_script()
    
    print("\n" + "=" * 50)
    print("✓ Setup complete!")
    print("\nNext steps:")
    print("1. Run 'python create/test_server_creation.py' to test")
    print("2. Create your first server with:")
    print("   python create/mk_server.py create myserver 'My description'")
    print("\nFor servers with PDFs:")
    print("   python create/mk_server.py create myserver 'My description' paper1.pdf paper2.pdf")

if __name__ == "__main__":
    main()
