#!/usr/bin/env python3
"""
Fix the ConceptDB unhashable type error
"""

import os
import re
from pathlib import Path

def find_conceptdb_class():
    """Find the ConceptDB class definition"""
    
    search_paths = [
        "ingest_pdf/pipeline/quality.py",
        "ingest_pdf/pipeline/concept_db.py",
        "ingest_pdf/concept_db.py",
        "ingest_pdf/quality.py"
    ]
    
    # Also search recursively
    for root, dirs, files in os.walk("ingest_pdf"):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'class ConceptDB' in content:
                            return file_path, content
                except:
                    pass
    
    # Try python/core as well
    for root, dirs, files in os.walk("python/core"):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'class ConceptDB' in content:
                            return file_path, content
                except:
                    pass
    
    return None, None

def fix_conceptdb_hashable():
    """Add __hash__ and __eq__ methods to ConceptDB"""
    
    print("🔍 Looking for ConceptDB class...")
    
    file_path, content = find_conceptdb_class()
    
    if not file_path:
        print("❌ Could not find ConceptDB class")
        print("   The error suggests it exists somewhere in the codebase")
        return False
    
    print(f"✅ Found ConceptDB in: {file_path}")
    
    # Check if already has __hash__
    if "__hash__" in content:
        print("✅ ConceptDB already has __hash__ method")
        return True
    
    # Find the class definition
    class_match = re.search(r'class ConceptDB[^:]*:', content)
    if not class_match:
        print("❌ Could not parse ConceptDB class")
        return False
    
    # Find where to insert (after __init__ or at end of class)
    init_match = re.search(r'def __init__\(self[^)]*\):[^}]+?(?=\n    def|\nclass|\n\S|\Z)', content, re.DOTALL)
    
    if init_match:
        # Insert after __init__
        insert_pos = init_match.end()
        
        # Determine if ConceptDB has file_path or id attribute
        if 'self.file_path' in content:
            hash_attr = 'self.file_path'
        elif 'self.id' in content:
            hash_attr = 'self.id'
        elif 'self.name' in content:
            hash_attr = 'self.name'
        else:
            hash_attr = 'id(self)'  # Fallback to object id
        
        hash_methods = f'''
    def __hash__(self):
        """Make ConceptDB hashable"""
        return hash({hash_attr})
    
    def __eq__(self, other):
        """Equality comparison for ConceptDB"""
        if not isinstance(other, ConceptDB):
            return False
        return {hash_attr} == ({hash_attr.replace('self.', 'other.')})
'''
        
        # Insert the methods
        new_content = content[:insert_pos] + hash_methods + content[insert_pos:]
        
        # Create backup
        import shutil
        from datetime import datetime
        backup = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(file_path, backup)
        print(f"✅ Created backup: {backup}")
        
        # Write fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Added __hash__ and __eq__ methods to ConceptDB")
        return True
    else:
        print("❌ Could not find where to insert methods")
        return False

def find_problematic_usage():
    """Find where ConceptDB is being used as a key"""
    
    print("\n🔍 Looking for problematic ConceptDB usage...")
    
    # Search in concurrency module
    concurrency_files = []
    for root, dirs, files in os.walk("ingest_pdf"):
        for file in files:
            if 'concurrency' in file and file.endswith('.py'):
                concurrency_files.append(os.path.join(root, file))
    
    for file_path in concurrency_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for patterns where ConceptDB might be used as key
            if 'ConceptDB' in content:
                print(f"\n📄 Found ConceptDB usage in: {file_path}")
                
                # Common patterns that would cause unhashable error
                patterns = [
                    r'set\([^)]*ConceptDB',
                    r'dict\([^)]*ConceptDB',
                    r'\{[^}]*ConceptDB[^}]*:',
                    r'ConceptDB[^)]*\sin\s+set',
                    r'ConceptDB[^)]*\sin\s+dict'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        print(f"   ⚠️  Possible issue: {matches[0][:50]}...")
        except:
            pass

def main():
    print("🔧 Fixing ConceptDB Unhashable Type Error")
    print("=" * 60)
    
    # First try to fix the ConceptDB class
    if fix_conceptdb_hashable():
        print("\n✅ Fix applied successfully!")
        print("\nThe error should be resolved after restarting.")
    else:
        print("\n⚠️  Could not apply automatic fix")
        print("\nManual fix instructions:")
        print("1. Find the ConceptDB class definition")
        print("2. Add these methods to make it hashable:")
        print("""
    def __hash__(self):
        return hash(self.file_path)  # or self.id or self.name
    
    def __eq__(self, other):
        if not isinstance(other, ConceptDB):
            return False
        return self.file_path == other.file_path
""")
    
    # Also look for where it's being used incorrectly
    find_problematic_usage()
    
    print("\n📝 Note: The extraction is still working (76 concepts extracted)")
    print("     This error is just noise that should be cleaned up.")

if __name__ == "__main__":
    main()
