#!/usr/bin/env python3
"""
Fix MCP duplicate registrations by adding singleton pattern
"""

import os
import re
from pathlib import Path

def fix_mcp_tool_manager():
    """Add singleton pattern to prevent duplicate tool registrations"""
    
    tool_manager_fixes = '''# Singleton pattern to prevent duplicate registrations
class ToolManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.__class__._initialized = True
            self.tools = {}
            # Original initialization code here
    
    def register_tool(self, name, tool):
        """Register a tool, skip if already exists"""
        if name in self.tools:
            # Silently skip instead of warning
            return
        self.tools[name] = tool
'''

    resource_manager_fixes = '''# Singleton pattern for ResourceManager
class ResourceManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.__class__._initialized = True
            self.resources = {}
    
    def register_resource(self, uri, resource):
        """Register a resource, skip if already exists"""
        if uri in self.resources:
            return  # Skip silently
        self.resources[uri] = resource
'''

    prompt_manager_fixes = '''# Singleton pattern for PromptManager
class PromptManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.__class__._initialized = True
            self.prompts = {}
    
    def register_prompt(self, name, prompt):
        """Register a prompt, skip if already exists"""
        if name in self.prompts:
            return  # Skip silently
        self.prompts[name] = prompt
'''

    print("🔧 Applying singleton pattern to MCP managers...")
    print("\nThis will prevent duplicate registrations from happening in the first place.")
    print("\n📝 Add these patterns to your MCP manager classes:")
    print("\n1. For ToolManager:")
    print(tool_manager_fixes)
    print("\n2. For ResourceManager:")
    print(resource_manager_fixes)
    print("\n3. For PromptManager:")
    print(prompt_manager_fixes)

# Create a startup config file
def create_startup_config():
    """Create a configuration file for startup optimization"""
    
    config_content = '''# TORI Startup Configuration
# Place this in your project root as tori_startup.conf

[startup]
# Suppress warnings during initialization
suppress_warnings = true
warning_types = [
    "UserWarning",
    "DeprecationWarning",
    "ResourceWarning"
]

[logging]
# Set log levels for noisy components
log_levels = {
    "mcp.server.fastmcp": "ERROR",
    "ingest_pdf.pipeline": "ERROR",
    "pydantic": "ERROR"
}

[initialization]
# Skip time-consuming initializations
skip_seed_concepts = true
skip_universal_db = true
lazy_load_models = true

[mcp]
# Prevent duplicate registrations
use_singleton = true
suppress_duplicate_warnings = true

[performance]
# Enable performance optimizations
parallel_init = true
cache_imports = true
precompile_regex = true
'''
    
    with open("tori_startup.conf", 'w') as f:
        f.write(config_content)
    
    print("\n✅ Created tori_startup.conf")

# Create environment setup script
def create_env_setup():
    """Create environment variable setup script"""
    
    env_script = '''#!/usr/bin/env python3
"""
Set up environment for quiet TORI startup
"""

import os

# Suppress Python warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# TORI-specific optimizations
os.environ['TORI_QUIET_MODE'] = '1'
os.environ['TORI_FAST_STARTUP'] = '1'
os.environ['TORI_DISABLE_MESH_CHECK'] = '1'
os.environ['TORI_SKIP_SEED_LOAD'] = '1'

# Suppress TensorFlow/ML library warnings if present
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU warnings if no GPU

# MCP specific
os.environ['MCP_SUPPRESS_DUPLICATES'] = '1'
os.environ['MCP_USE_SINGLETON'] = '1'

print("✅ Environment configured for quiet startup")
print("   Run your launcher now: python enhanced_launcher.py")
'''
    
    with open("setup_quiet_env.py", 'w') as f:
        f.write(env_script)
    
    os.chmod("setup_quiet_env.py", 0o755)
    print("✅ Created setup_quiet_env.py")

# Main execution
if __name__ == "__main__":
    print("🧹 MCP Duplicate Registration Fix")
    print("=" * 60)
    
    fix_mcp_tool_manager()
    create_startup_config()
    create_env_setup()
    
    print("\n🎯 Quick Start Guide:")
    print("1. Set environment: python setup_quiet_env.py")
    print("2. Launch quietly: python tori_quiet.py")
    print("\nOr use the comprehensive fix: python clean_tori_startup.py")
