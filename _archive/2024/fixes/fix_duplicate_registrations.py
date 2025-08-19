#!/usr/bin/env python3
"""
Fix duplicate tool/resource registrations by adding existence checks
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

def find_registration_files():
    """Find all files with tool/resource registrations"""
    
    registration_files = []
    
    # Search in mcp_metacognitive directory
    mcp_dir = Path("mcp_metacognitive")
    if mcp_dir.exists():
        for file_path in mcp_dir.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if ("tool_manager.register" in content or 
                        "resource_manager.register" in content or
                        "prompt_manager.register" in content):
                        registration_files.append(str(file_path))
                        print(f"Found registrations in: {file_path}")
            except:
                pass
    
    return registration_files

def add_existence_checks(file_path):
    """Add existence checks before registrations"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    modifications = 0
    
    # Pattern for tool registrations
    tool_patterns = [
        (r'(\s*)tool_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not tool_manager.exists("\2"):\n\1    tool_manager.register("\2", \3)'),
        (r'(\s*)self\.tool_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not self.tool_manager.exists("\2"):\n\1    self.tool_manager.register("\2", \3)'),
    ]
    
    # Pattern for resource registrations
    resource_patterns = [
        (r'(\s*)resource_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not resource_manager.exists("\2"):\n\1    resource_manager.register("\2", \3)'),
        (r'(\s*)self\.resource_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not self.resource_manager.exists("\2"):\n\1    self.resource_manager.register("\2", \3)'),
    ]
    
    # Pattern for prompt registrations
    prompt_patterns = [
        (r'(\s*)prompt_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not prompt_manager.exists("\2"):\n\1    prompt_manager.register("\2", \3)'),
        (r'(\s*)self\.prompt_manager\.register\(\s*["\']([^"\']+)["\'],\s*([^)]+)\)',
         r'\1if not self.prompt_manager.exists("\2"):\n\1    self.prompt_manager.register("\2", \3)'),
    ]
    
    # Apply patterns
    for pattern, replacement in tool_patterns + resource_patterns + prompt_patterns:
        # Check if already has existence check
        if "if not" in content and "exists" in content:
            # Skip if already protected
            continue
        
        matches = list(re.finditer(pattern, content))
        for match in reversed(matches):  # Process in reverse to maintain positions
            # Check if this registration is already protected
            line_start = content.rfind('\n', 0, match.start()) + 1
            prev_line_start = content.rfind('\n', 0, line_start - 1) + 1
            prev_line = content[prev_line_start:line_start].strip()
            
            if "if not" in prev_line and "exists" in prev_line:
                continue  # Already protected
            
            # Apply replacement
            content = content[:match.start()] + match.expand(replacement) + content[match.end():]
            modifications += 1
    
    if modifications > 0:
        # Create backup
        backup = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(file_path, backup)
        
        # Write modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Modified {file_path}: Added {modifications} existence checks")
        return True
    else:
        print(f"No modifications needed for {file_path}")
        return False

def add_exists_methods():
    """Add exists() methods to managers if missing"""
    
    manager_files = [
        ("mcp_metacognitive/tools/tool_manager.py", "ToolManager", "tools"),
        ("mcp_metacognitive/resources/resource_manager.py", "ResourceManager", "resources"),
        ("mcp_metacognitive/prompts/manager.py", "PromptManager", "prompts")
    ]
    
    for file_path, class_name, collection_name in manager_files:
        if not os.path.exists(file_path):
            # Try alternate path
            alt_path = file_path.replace("mcp_metacognitive/", "")
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                print(f"Could not find {file_path}")
                continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has exists method
        if "def exists" in content:
            print(f"{file_path} already has exists() method")
            continue
        
        # Find class definition
        class_pos = content.find(f"class {class_name}")
        if class_pos == -1:
            print(f"Could not find {class_name} in {file_path}")
            continue
        
        # Find where to add method (after __init__ or register method)
        register_pos = content.find("def register", class_pos)
        if register_pos == -1:
            continue
        
        # Find end of register method
        next_method = content.find("\n    def ", register_pos + 1)
        if next_method == -1:
            next_method = len(content)
        
        # Add exists method
        exists_method = f'''
    def exists(self, name: str) -> bool:
        """Check if item exists"""
        return name in self.{collection_name}
'''
        
        # Insert method
        content = content[:next_method] + exists_method + content[next_method:]
        
        # Create backup
        backup = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(file_path, backup)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Added exists() method to {class_name}")

def create_registration_wrapper():
    """Create a wrapper module for safe registrations"""
    
    wrapper_content = '''"""
Safe registration wrapper to prevent duplicates
"""

import logging

logger = logging.getLogger(__name__)

class SafeRegistrationMixin:
    """Mixin to add safe registration to any manager"""
    
    def safe_register(self, name: str, item: any, force: bool = False):
        """Register item only if it doesn't exist (unless force=True)"""
        if hasattr(self, 'exists') and self.exists(name) and not force:
            logger.debug(f"Skipping duplicate registration: {name}")
            return False
        
        # Call original register method
        if hasattr(self, 'register'):
            self.register(name, item)
            return True
        else:
            raise AttributeError(f"{self.__class__.__name__} has no register method")
    
    def register_once(self, name: str, item: any):
        """Register item only if it doesn't exist (never force)"""
        return self.safe_register(name, item, force=False)

def patch_managers():
    """Monkey patch managers to add safe registration"""
    try:
        # Patch tool manager
        from mcp_metacognitive.tools.tool_manager import ToolManager
        if not hasattr(ToolManager, 'safe_register'):
            ToolManager.__bases__ = (SafeRegistrationMixin,) + ToolManager.__bases__
            logger.info("Patched ToolManager with safe registration")
    except:
        pass
    
    try:
        # Patch resource manager
        from mcp_metacognitive.resources.resource_manager import ResourceManager
        if not hasattr(ResourceManager, 'safe_register'):
            ResourceManager.__bases__ = (SafeRegistrationMixin,) + ResourceManager.__bases__
            logger.info("Patched ResourceManager with safe registration")
    except:
        pass
    
    try:
        # Patch prompt manager
        from mcp_metacognitive.prompts.manager import PromptManager
        if not hasattr(PromptManager, 'safe_register'):
            PromptManager.__bases__ = (SafeRegistrationMixin,) + PromptManager.__bases__
            logger.info("Patched PromptManager with safe registration")
    except:
        pass

# Auto-patch on import
patch_managers()
'''
    
    with open("safe_registration.py", 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print("Created safe_registration.py wrapper")

def main():
    print("Fixing Duplicate Tool/Resource Registrations")
    print("=" * 60)
    print("\nProblem: Tools/resources registered multiple times")
    print("Solution: Add existence checks before registration\n")
    
    # First add exists() methods if missing
    print("Step 1: Adding exists() methods to managers...")
    add_exists_methods()
    
    # Find and fix registration files
    print("\nStep 2: Finding registration files...")
    files = find_registration_files()
    
    if not files:
        print("No registration files found")
    else:
        print(f"\nFound {len(files)} files with registrations")
        print("\nStep 3: Adding existence checks...")
        
        fixed = 0
        for file_path in files:
            if add_existence_checks(file_path):
                fixed += 1
        
        print(f"\nModified {fixed} files")
    
    # Create wrapper for future use
    print("\nStep 4: Creating safe registration wrapper...")
    create_registration_wrapper()
    
    print("\nWhat this does:")
    print("1. Adds exists() method to managers if missing")
    print("2. Wraps all register() calls with existence checks")
    print("3. Prevents duplicate registration warnings")
    print("4. Creates safe_registration.py for future use")
    
    print("\nTo use safe registration in new code:")
    print("  import safe_registration  # Auto-patches managers")
    print("  tool_manager.safe_register('tool_name', tool_func)")
    
    print("\nAfter restart:")
    print("- No more 'Tool already exists' warnings")
    print("- Cleaner startup logs")
    print("- No risk of shadowing implementations")

if __name__ == "__main__":
    main()
