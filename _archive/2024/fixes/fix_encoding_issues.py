#!/usr/bin/env python3
"""Diagnostic script for TORI concept file encoding issues"""

import chardet
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def check_file_encoding(filepath):
    """Check the encoding of a file"""
    print(f"\n📁 Checking: {filepath.name}")
    
    try:
        # Read raw bytes
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        print(f"  Detected encoding: {encoding} (confidence: {confidence:.2%})")
        
        # Try to decode with detected encoding
        if encoding:
            try:
                text = raw_data.decode(encoding)
                print(f"  ✅ Successfully decoded with {encoding}")
                
                # If it's JSON, try to parse it
                if filepath.suffix == '.json':
                    try:
                        data = json.loads(text)
                        print(f"  ✅ Valid JSON with {len(data) if isinstance(data, (list, dict)) else 1} items")
                    except json.JSONDecodeError as e:
                        print(f"  ❌ JSON parse error: {e}")
                        
            except UnicodeDecodeError as e:
                print(f"  ❌ Decode error with {encoding}: {e}")
        else:
            print("  ❌ Could not detect encoding")
            
        # Check for BOM markers
        if raw_data.startswith(b'\xff\xfe'):
            print("  ⚠️  UTF-16-LE BOM detected")
        elif raw_data.startswith(b'\xfe\xff'):
            print("  ⚠️  UTF-16-BE BOM detected")
        elif raw_data.startswith(b'\xef\xbb\xbf'):
            print("  ⚠️  UTF-8 BOM detected")
            
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")

def fix_encoding(filepath, target_encoding='utf-8'):
    """Convert file to UTF-8"""
    print(f"\n🔧 Attempting to fix encoding for: {filepath.name}")
    
    try:
        # Detect current encoding
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            
        result = chardet.detect(raw_data)
        source_encoding = result['encoding']
        
        if not source_encoding:
            print("  ❌ Could not detect source encoding")
            return False
            
        # Decode and re-encode
        text = raw_data.decode(source_encoding)
        
        # Save backup
        backup_path = filepath.with_suffix(filepath.suffix + '.backup')
        filepath.rename(backup_path)
        print(f"  📋 Created backup: {backup_path.name}")
        
        # Write as UTF-8
        with open(filepath, 'w', encoding=target_encoding) as f:
            f.write(text)
            
        print(f"  ✅ Converted from {source_encoding} to {target_encoding}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error fixing encoding: {e}")
        return False

if __name__ == "__main__":
    # Check concept files
    concept_files = [
        Path(r"{PROJECT_ROOT}\concept_mesh_data.json"),
        Path(r"{PROJECT_ROOT}\concepts.json"),
        Path(r"{PROJECT_ROOT}\file_storage"),  # This seems to be the problem file
    ]
    
    print("🔍 TORI Concept File Encoding Diagnostic")
    print("=" * 50)
    
    for filepath in concept_files:
        if filepath.exists():
            check_file_encoding(filepath)
        else:
            print(f"\n❌ File not found: {filepath}")
    
    # Offer to fix
    print("\n" + "=" * 50)
    fix_input = input("\nWould you like to convert problematic files to UTF-8? (y/n): ")
    
    if fix_input.lower() == 'y':
        for filepath in concept_files:
            if filepath.exists() and filepath.name == 'file_storage':
                fix_encoding(filepath)
