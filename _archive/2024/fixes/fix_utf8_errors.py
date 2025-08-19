#!/usr/bin/env python3
"""
Fix UTF-8 Encoding Errors in TORI Codebase
==========================================

This script scans for and fixes UTF-8 encoding issues that can break the system.
"""

import os
import sys
import chardet
from pathlib import Path
import shutil
import traceback

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'], result['confidence']
    except Exception as e:
        return None, 0

def fix_file_encoding(file_path, target_encoding='utf-8'):
    """Fix encoding issues in a file"""
    try:
        # First, detect current encoding
        detected_encoding, confidence = detect_encoding(file_path)
        
        if not detected_encoding:
            print(f"⚠️  Could not detect encoding for: {file_path}")
            return False
            
        # Skip if already UTF-8 with high confidence
        if detected_encoding.lower() in ['utf-8', 'ascii'] and confidence > 0.9:
            return True
            
        print(f"🔧 Converting {file_path} from {detected_encoding} to UTF-8...")
        
        # Create backup
        backup_path = f"{file_path}.backup_utf8"
        shutil.copy2(file_path, backup_path)
        
        # Read with detected encoding
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Fixed encoding for: {file_path}")
        
        # Remove backup if successful
        os.remove(backup_path)
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        # Restore backup if it exists
        if os.path.exists(f"{file_path}.backup_utf8"):
            shutil.move(f"{file_path}.backup_utf8", file_path)
        return False

def scan_for_encoding_issues(directory):
    """Scan directory for files with encoding issues"""
    
    issues_found = []
    files_fixed = []
    files_scanned = 0
    
    # File extensions to check
    extensions = ['.py', '.js', '.ts', '.json', '.yaml', '.yml', '.txt', '.md', '.rst']
    
    # Directories to skip
    skip_dirs = ['.venv', 'venv_tori_prod', '__pycache__', '.git', 'node_modules', '.svelte-kit', 'build', 'dist']
    
    for root, dirs, files in os.walk(directory):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            # Check file extension
            if not any(file.endswith(ext) for ext in extensions):
                continue
                
            file_path = Path(root) / file
            files_scanned += 1
            
            try:
                # Try to read as UTF-8
                with open(file_path, 'r', encoding='utf-8') as f:
                    _ = f.read()
            except UnicodeDecodeError as e:
                print(f"\n❌ UTF-8 error in: {file_path}")
                print(f"   Error: {e}")
                issues_found.append((file_path, str(e)))
                
                # Try to fix it
                if fix_file_encoding(file_path):
                    files_fixed.append(file_path)
            except Exception as e:
                if 'codec' in str(e).lower() or 'decode' in str(e).lower():
                    print(f"\n⚠️  Encoding issue in: {file_path}")
                    print(f"   Error: {e}")
                    issues_found.append((file_path, str(e)))
    
    return files_scanned, issues_found, files_fixed

def check_critical_files():
    """Check specific critical files for encoding issues"""
    critical_files = [
        "prajna/api/prajna_api.py",
        "api/routes/soliton_production.py",
        "concept_mesh/data.json",
        "enhanced_launcher.py",
        "alan_backend/routes/soliton.py",
        "ingest_pdf/pipeline.py",
        "tori_ui_svelte/src/routes/+page.svelte",
        "mcp_metacognitive/server.py"
    ]
    
    print("\n📋 Checking critical files...")
    for file_path in critical_files:
        if Path(file_path).exists():
            encoding, confidence = detect_encoding(file_path)
            status = "✅" if encoding and encoding.lower() in ['utf-8', 'ascii'] else "❌"
            print(f"{status} {file_path}: {encoding} (confidence: {confidence:.2%})")
        else:
            print(f"⚠️  {file_path}: File not found")

def add_utf8_headers():
    """Add UTF-8 encoding headers to Python files missing them"""
    print("\n🔍 Checking for missing UTF-8 headers in Python files...")
    
    header = "# -*- coding: utf-8 -*-\n"
    files_updated = []
    
    for root, dirs, files in os.walk("."):
        # Skip virtual environments and cache
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv_tori_prod', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        
                    # Check if encoding header is missing
                    if 'coding' not in first_line and '#!/usr/bin/env' not in first_line:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Add header
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(header + content)
                        
                        files_updated.append(file_path)
                        print(f"✅ Added UTF-8 header to: {file_path}")
                        
                except Exception as e:
                    print(f"⚠️  Could not process {file_path}: {e}")
    
    return files_updated

def main():
    """Main function"""
    print("🔍 Scanning for UTF-8 encoding errors in TORI codebase...")
    print("=" * 60)
    
    # Check critical files first
    check_critical_files()
    
    # Scan entire codebase
    print("\n🔍 Scanning entire codebase for encoding issues...")
    files_scanned, issues_found, files_fixed = scan_for_encoding_issues(".")
    
    print(f"\n📊 Scan Summary:")
    print(f"  Files scanned: {files_scanned}")
    print(f"  Issues found: {len(issues_found)}")
    print(f"  Files fixed: {len(files_fixed)}")
    
    if issues_found:
        print(f"\n❌ Files with encoding issues:")
        for filepath, error in issues_found:
            print(f"  - {filepath}")
            if filepath not in files_fixed:
                print(f"    ⚠️  NOT FIXED - manual intervention needed")
    
    # Add UTF-8 headers where missing
    print("\n" + "=" * 60)
    headers_added = add_utf8_headers()
    if headers_added:
        print(f"\n✅ Added UTF-8 headers to {len(headers_added)} files")
    else:
        print("\n✅ All Python files have proper encoding headers")
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("🎯 Recommendations:")
    print("1. Restart the TORI server after fixing encoding issues")
    print("2. Test file uploads and API endpoints")
    print("3. Check logs for any remaining encoding errors")
    
    if not issues_found:
        print("\n✅ No encoding issues found! Your codebase is UTF-8 clean.")
    else:
        unfixed = len(issues_found) - len(files_fixed)
        if unfixed > 0:
            print(f"\n⚠️  {unfixed} files need manual fixing")

if __name__ == "__main__":
    # Install chardet if not available
    try:
        import chardet
    except ImportError:
        print("Installing chardet for encoding detection...")
        os.system(f"{sys.executable} -m pip install chardet")
        import chardet
    
    main()
