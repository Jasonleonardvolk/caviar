#!/usr/bin/env python3
"""
Fix Entropy Pruning Import Error
"""

import os
import sys
from pathlib import Path

def fix_entropy_import():
    print("🔧 Fixing Entropy Pruning Import Error...\n")
    
    # Get the paths
    kha_dir = Path("C:/Users/jason/Desktop/tori/kha")
    ingest_pdf_dir = kha_dir / "ingest_pdf"
    pipeline_dir = ingest_pdf_dir / "pipeline"
    pruning_file = pipeline_dir / "pruning.py"
    entropy_file = ingest_pdf_dir / "entropy_prune.py"
    
    # Check if files exist
    if not entropy_file.exists():
        print(f"❌ ERROR: {entropy_file} does not exist!")
        print("\nYou need to create or restore entropy_prune.py")
        return False
    else:
        print(f"✅ Found entropy_prune.py at: {entropy_file}")
    
    if not pruning_file.exists():
        print(f"❌ ERROR: {pruning_file} does not exist!")
        return False
    else:
        print(f"✅ Found pruning.py at: {pruning_file}")
    
    # Read the current pruning.py
    with open(pruning_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a more robust import section
    new_import = '''# Import entropy pruning with multiple fallbacks
try:
    from ..entropy_prune import entropy_prune, entropy_prune_with_categories
    logger.debug("Imported entropy_prune from parent directory")
except ImportError as e1:
    try:
        from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
        logger.debug("Imported entropy_prune using absolute import")
    except ImportError as e2:
        try:
            # Add parent directory to path temporarily
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from entropy_prune import entropy_prune, entropy_prune_with_categories
            logger.debug("Imported entropy_prune after adding to path")
        except ImportError as e3:
            logger.error(f"❌ Failed to import entropy pruning modules")
            logger.error(f"  Relative import error: {e1}")
            logger.error(f"  Absolute import error: {e2}")
            logger.error(f"  Path import error: {e3}")
            logger.error(f"  Current file: {__file__}")
            logger.error(f"  Expected location: {Path(__file__).parent.parent / 'entropy_prune.py'}")
            
            # Provide stub implementations to allow pipeline to continue
            logger.warning("⚠️  Using stub implementations for entropy pruning")
            
            def entropy_prune(concepts, **kwargs):
                """Stub implementation - returns concepts unchanged"""
                return concepts, {"selected": len(concepts), "pruned": 0}
            
            def entropy_prune_with_categories(concepts, **kwargs):
                """Stub implementation - returns concepts unchanged"""
                return concepts, {"selected": len(concepts), "pruned": 0}'''
    
    # Replace the import section
    import_start = content.find("# Import entropy pruning with fallback")
    if import_start == -1:
        import_start = content.find("try:\n    from ..entropy_prune")
    
    if import_start == -1:
        print("❌ Could not find import section in pruning.py")
        return False
    
    # Find the end of the import block
    import_end = content.find("\n\n", import_start)
    if import_end == -1:
        import_end = content.find("def calculate_simple_similarity", import_start)
    
    # Replace the import section
    new_content = content[:import_start] + new_import + content[import_end:]
    
    # Backup the original file
    backup_file = pruning_file.with_suffix('.py.backup_entropy_fix')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_file}")
    
    # Write the fixed content
    with open(pruning_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"✅ Updated pruning.py with robust import handling")
    
    # Create __init__.py files if they don't exist
    init_files = [
        ingest_pdf_dir / "__init__.py",
        pipeline_dir / "__init__.py"
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.write_text("")
            print(f"✅ Created {init_file}")
    
    print("\n✨ Fix complete!")
    print("\n📋 Testing the fix:")
    print("1. Run from the kha directory:")
    print("   cd C:\\Users\\jason\\Desktop\\tori\\kha")
    print("2. Test the import:")
    print("   python -m ingest_pdf.pipeline.quality")
    print("   # or")
    print("   python -c \"from ingest_pdf.pipeline.pruning import apply_entropy_pruning; print('Import successful!')\"")
    
    return True

if __name__ == "__main__":
    success = fix_entropy_import()
    if success:
        print("\n✅ Entropy import issue should be fixed!")
    else:
        print("\n❌ Failed to fix the import issue")
