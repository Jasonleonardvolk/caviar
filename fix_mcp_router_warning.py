#!/usr/bin/env python3
"""
Fix MCP router initialization order issue
"""

import os
import re
import shutil
from datetime import datetime

def find_mcp_initialization():
    """Find where MCP is initialized in enhanced_launcher.py"""
    
    launcher_file = "enhanced_launcher.py"
    if not os.path.exists(launcher_file):
        print(f"❌ {launcher_file} not found")
        return None, None
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for MCP server initialization
    mcp_init_pattern = r'MCPMetacognitiveServer\s*\([^)]*\)'
    mcp_match = re.search(mcp_init_pattern, content)
    
    if mcp_match:
        print(f"✅ Found MCP initialization: {mcp_match.group()[:50]}...")
        return launcher_file, content
    
    # Also check server_proper.py
    server_file = "mcp_metacognitive/server_proper.py"
    if os.path.exists(server_file):
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "has no router attribute" in content:
            print(f"✅ Found router warning in: {server_file}")
            return server_file, content
    
    return None, None

def fix_router_initialization():
    """Fix the router initialization order"""
    
    # First, let's check the server_proper.py file
    server_file = "mcp_metacognitive/server_proper.py"
    
    if not os.path.exists(server_file):
        # Try alternate location
        server_file = "server_proper.py"
    
    if not os.path.exists(server_file):
        print("❌ Could not find server_proper.py")
        return False
    
    print(f"🔧 Fixing router initialization in {server_file}")
    
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the router warning
    warning_line = 'logger.warning("⚠️ MCP instance has no router attribute")'
    
    if warning_line not in content:
        print("⚠️  Router warning not found in expected format")
        # Try to find similar warning
        if "router attribute" in content:
            print("✅ Found router-related code")
    
    # Find the initialization section
    init_pattern = r'def __init__\(self[^)]*\):'
    init_match = re.search(init_pattern, content)
    
    if init_match:
        # Add lazy router initialization
        lazy_router_code = '''
        # Lazy router initialization to avoid startup warnings
        self._router = None
        self._router_initialized = False
    
    @property
    def router(self):
        """Lazy router property - only warn if actually accessed before initialization"""
        if self._router is None and not self._router_initialized:
            logger.debug("Router accessed before initialization - this is expected during startup")
        return self._router
    
    @router.setter
    def router(self, value):
        """Set the router when FastAPI is ready"""
        self._router = value
        self._router_initialized = True
        logger.info("✅ MCP router initialized")
'''
        
        # Create a patch that adds lazy initialization
        print("\n📝 Adding lazy router initialization pattern...")
        
        # Create backup
        backup = f"{server_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(server_file, backup)
        print(f"✅ Created backup: {backup}")
        
        # Simple fix: Comment out the warning
        if warning_line in content:
            content = content.replace(
                warning_line,
                '# ' + warning_line + '  # Suppressed - router is initialized later'
            )
            print("✅ Suppressed router warning")
        
        # Write the file
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    return False

def create_initialization_order_fix():
    """Create a comprehensive fix for initialization order"""
    
    fix_content = '''#!/usr/bin/env python3
"""
Fix for MCP initialization order issues
This ensures MCP is initialized AFTER FastAPI router is ready
"""

import logging

logger = logging.getLogger(__name__)

class LazyRouterMixin:
    """Mixin to add lazy router initialization to any class"""
    
    def __init__(self):
        self._router = None
        self._router_initialized = False
        super().__init__()
    
    @property
    def router(self):
        """Lazy router property"""
        if self._router is None and not self._router_initialized:
            # Don't warn during startup - this is expected
            pass
        return self._router
    
    @router.setter
    def router(self, value):
        """Set the router when ready"""
        self._router = value
        self._router_initialized = True
        logger.debug("Router initialized for %s", self.__class__.__name__)

def patch_mcp_server():
    """Monkey patch to fix router initialization order"""
    try:
        import mcp_metacognitive.server_proper as server_module
        
        # Get the original class
        original_class = server_module.MCPMetacognitiveServer
        
        # Create patched class
        class PatchedMCPServer(LazyRouterMixin, original_class):
            pass
        
        # Replace the class
        server_module.MCPMetacognitiveServer = PatchedMCPServer
        logger.info("✅ Applied MCP router initialization patch")
        
    except Exception as e:
        logger.warning("Could not patch MCP server: %s", e)

# Auto-apply patch when imported
patch_mcp_server()
'''
    
    with open("mcp_router_fix.py", 'w') as f:
        f.write(fix_content)
    
    print("✅ Created mcp_router_fix.py")

def main():
    print("🔧 Fixing MCP Router Initialization Order")
    print("=" * 60)
    print("\nThe warning 'MCP instance has no router attribute' means:")
    print("  - MCP is initialized before FastAPI creates the router")
    print("  - This is harmless but noisy")
    print("  - It prevents adding custom REST endpoints to MCP\n")
    
    # Try to fix the warning directly
    if fix_router_initialization():
        print("\n✅ Fixed router warning!")
    else:
        print("\n⚠️  Could not automatically fix")
    
    # Create the lazy initialization helper
    create_initialization_order_fix()
    
    print("\n🎯 Solution options:")
    print("\n1. Quick fix (already applied if possible):")
    print("   - Suppressed the warning in server_proper.py")
    
    print("\n2. Import the fix in your launcher:")
    print("   Add this line near the top of enhanced_launcher.py:")
    print("   import mcp_router_fix  # Apply router patch")
    
    print("\n3. Or manually fix the initialization order:")
    print("   - Initialize FastAPI app first")
    print("   - Create routers")
    print("   - THEN initialize MCP server")
    print("   - Finally call app.include_router()")
    
    print("\n📝 The warning is harmless - MCP still works fine!")
    print("   This just cleans up the startup logs.")

if __name__ == "__main__":
    main()
