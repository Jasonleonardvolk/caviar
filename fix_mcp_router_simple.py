#!/usr/bin/env python3
"""
Add router attribute to MCPMetacognitiveServer to fix startup warning
"""

import os
import re
import shutil
from datetime import datetime

def fix_mcp_router():
    """Add router = APIRouter() to MCPMetacognitiveServer.__init__"""
    
    server_file = "mcp_metacognitive/server_proper.py"
    
    if not os.path.exists(server_file):
        print(f"File not found: {server_file}")
        return False
    
    print(f"Fixing router in {server_file}")
    
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has router
    if "router = APIRouter()" in content:
        print("Already has router = APIRouter()")
        return True
    
    # Find the imports section
    import_section = content.find("from fastapi import")
    if import_section == -1:
        # Add FastAPI import if missing
        import_line = "from fastapi import APIRouter\n"
        # Find other imports
        import_pos = content.find("import ")
        if import_pos != -1:
            # Add after first import
            next_line = content.find("\n", import_pos)
            content = content[:next_line+1] + import_line + content[next_line+1:]
    else:
        # Check if APIRouter is imported
        if "APIRouter" not in content[import_section:import_section+200]:
            # Add APIRouter to imports
            fastapi_line_end = content.find("\n", import_section)
            fastapi_line = content[import_section:fastapi_line_end]
            
            # Add APIRouter to the import
            if " import " in fastapi_line:
                parts = fastapi_line.split(" import ")
                imports = parts[1].split(", ")
                if "APIRouter" not in imports:
                    imports.append("APIRouter")
                    new_line = parts[0] + " import " + ", ".join(imports)
                    content = content.replace(fastapi_line, new_line)
    
    # Find MCPMetacognitiveServer class and its __init__
    class_pattern = r'class MCPMetacognitiveServer[^:]*:'
    class_match = re.search(class_pattern, content)
    
    if not class_match:
        print("Could not find MCPMetacognitiveServer class")
        return False
    
    # Find __init__ method
    init_pattern = r'def __init__\(self[^)]*\):'
    init_match = re.search(init_pattern, content[class_match.end():])
    
    if not init_match:
        print("Could not find __init__ method")
        return False
    
    init_pos = class_match.end() + init_match.end()
    
    # Find where to insert router initialization
    # Look for the first assignment or super().__init__
    next_line_pos = content.find('\n', init_pos)
    indent_match = re.match(r'(\s+)', content[next_line_pos+1:])
    indent = indent_match.group(1) if indent_match else '        '
    
    # Add router initialization
    router_init = f"\n{indent}# Initialize router to avoid startup warning\n{indent}self.router = APIRouter()\n"
    
    # Insert after the first line of __init__
    content = content[:next_line_pos] + router_init + content[next_line_pos:]
    
    # Create backup
    backup = f"{server_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(server_file, backup)
    print(f"Created backup: {backup}")
    
    # Write the fixed file
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully added router = APIRouter() to MCPMetacognitiveServer")
    return True

def main():
    print("Fixing MCP Router Warning")
    print("=" * 60)
    
    if fix_mcp_router():
        print("\nFix applied successfully!")
        print("\nWhat this does:")
        print("  - Adds 'from fastapi import APIRouter' if needed")
        print("  - Adds 'self.router = APIRouter()' in __init__")
        print("  - Eliminates the startup warning")
        print("\nThe MCP server will now have a router attribute from the start!")
    else:
        print("\nCould not apply automatic fix")
        print("\nManual fix:")
        print("1. Open mcp_metacognitive/server_proper.py")
        print("2. Add to imports: from fastapi import APIRouter")
        print("3. In MCPMetacognitiveServer.__init__, add:")
        print("   self.router = APIRouter()")

if __name__ == "__main__":
    main()
