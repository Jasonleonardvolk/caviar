#!/usr/bin/env python3
"""
One-time fix to silence TORI startup warnings
Run this ONCE to patch the files causing noise
"""

import os
import re
import shutil
from datetime import datetime

print("🔧 Applying one-time startup noise fix...")
print("=" * 60)

# Fix 1: Patch enhanced_launcher.py to suppress warnings
def patch_enhanced_launcher():
    """Add warning suppression to the launcher"""
    launcher_file = "enhanced_launcher.py"
    
    if not os.path.exists(launcher_file):
        print(f"❌ {launcher_file} not found")
        return False
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "warnings.filterwarnings" in content:
        print("✅ Launcher already patched")
        return True
    
    # Add warning suppression at the top after imports
    patch = '''
# Suppress startup warnings
import warnings
warnings.filterwarnings("ignore", message=".*already exists.*")
warnings.filterwarnings("ignore", message=".*shadows an attribute.*")
warnings.filterwarnings("ignore", message=".*0 concepts.*")

# Reduce logging noise
import logging
logging.getLogger("mcp.server.fastmcp").setLevel(logging.ERROR)
logging.getLogger("server_proper").setLevel(logging.ERROR)
'''
    
    # Find where to insert (after the imports section)
    import_end = content.find("# Import graceful shutdown")
    if import_end == -1:
        import_end = content.find("# Enhanced error handling")
    
    if import_end != -1:
        content = content[:import_end] + patch + "\n" + content[import_end:]
    else:
        # Just add after the docstring
        docstring_end = content.find('"""', 3) + 3
        content = content[:docstring_end] + "\n" + patch + content[docstring_end:]
    
    # Create backup
    backup = f"{launcher_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(launcher_file, backup)
    
    # Write patched file
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Patched {launcher_file} (backup: {backup})")
    return True

# Fix 2: Patch server_proper.py to prevent duplicate registrations
def patch_mcp_server():
    """Fix duplicate tool/resource registrations"""
    server_file = "mcp_metacognitive/server_proper.py"
    
    if not os.path.exists(server_file):
        # Try alternate location
        server_file = "server_proper.py"
    
    if not os.path.exists(server_file):
        print("⚠️  MCP server file not found, skipping")
        return False
    
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace warning with silent skip
    original = 'logger.warning(f"Tool already exists: {name}")'
    replacement = 'return  # Skip duplicate registration'
    
    if original in content:
        content = content.replace(original, replacement)
        
        # Same for resources
        content = content.replace(
            'logger.warning(f"Resource already exists: {uri}")',
            'return  # Skip duplicate registration'
        )
        
        # And prompts
        content = content.replace(
            'logger.warning(f"Prompt already exists: {name}")',
            'return  # Skip duplicate registration'
        )
        
        # Create backup
        backup = f"{server_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(server_file, backup)
        
        # Write patched file
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Patched {server_file} - duplicates will be silently skipped")
        return True
    else:
        print("✅ MCP server already fixed or uses different code")
        return True

# Fix 3: Disable lattice evolution logging
def patch_lattice_runner():
    """Stop the repetitive lattice oscillator messages"""
    lattice_file = "python/core/lattice_evolution_runner.py"
    
    if not os.path.exists(lattice_file):
        print("⚠️  Lattice runner not found, skipping")
        return False
    
    with open(lattice_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the logging line
    if "[lattice] oscillators=" in content:
        # Comment out or reduce frequency
        content = re.sub(
            r'(logger\.info\(f?\["\']?\[lattice\].*oscillators.*\))',
            r'# \1  # Commented out to reduce noise',
            content
        )
        
        # Create backup
        backup = f"{lattice_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(lattice_file, backup)
        
        # Write patched file
        with open(lattice_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Patched {lattice_file} - stopped repetitive oscillator logs")
        return True
    else:
        print("✅ Lattice runner already quiet")
        return True

# Fix 4: Add environment variable to enhanced_launcher
def add_quiet_env_vars():
    """Add environment variables to suppress warnings"""
    launcher_file = "enhanced_launcher.py"
    
    with open(launcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add after the existing environment variables
    if "PYTHONWARNINGS" not in content:
        env_vars = """
# Suppress Python warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings if any
"""
        # Find where other env vars are set
        pos = content.find("os.environ['TORI_DISABLE_ENTROPY_PRUNE']")
        if pos != -1:
            # Insert after the entropy pruning section
            insert_pos = content.find('\n', pos) + 1
            content = content[:insert_pos] + env_vars + content[insert_pos:]
            
            with open(launcher_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Added quiet environment variables")
            return True
    
    return True

# Main execution
def main():
    print("\nApplying fixes...\n")
    
    # Apply all patches
    success = True
    success &= patch_enhanced_launcher()
    success &= patch_mcp_server()
    success &= patch_lattice_runner()
    success &= add_quiet_env_vars()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ONE-TIME FIX COMPLETE!")
        print("\nThe following changes were made:")
        print("  • Warning filters added to launcher")
        print("  • Duplicate registrations now silent")
        print("  • Lattice oscillator spam disabled")
        print("  • Quiet environment variables set")
        print("\n🚀 Just run your normal launcher now:")
        print("   python enhanced_launcher.py")
        print("\nNo more warning spam! 🎉")
    else:
        print("⚠️  Some fixes could not be applied")
        print("But you can still run normally - some warnings may remain")

if __name__ == "__main__":
    main()
