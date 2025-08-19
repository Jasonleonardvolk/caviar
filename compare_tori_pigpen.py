#!/usr/bin/env python3
"""
TORI vs Pigpen Comparison Tool
==============================

This tool helps identify differences between your TORI and pigpen directories
to help you understand what changes were made where.

Usage: python compare_tori_pigpen.py
"""

import os
import sys
import hashlib
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime
import filecmp
import difflib
from typing import Dict, List, Tuple, Optional

class DirectoryComparator:
    def __init__(self, tori_path: str, pigpen_path: str):
        self.tori_path = Path(tori_path)
        self.pigpen_path = Path(pigpen_path)
        self.ignore_patterns = {
            '__pycache__', '.pyc', '.git', 'node_modules', 
            '.pytest_cache', 'logs', '*.log', '.env',
            'tori_status.json', 'api_port.json'
        }
        
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored"""
        name = path.name
        
        # Check exact matches
        if name in self.ignore_patterns:
            return True
            
        # Check patterns
        for pattern in self.ignore_patterns:
            if pattern.startswith('*') and name.endswith(pattern[1:]):
                return True
            if pattern.endswith('*') and name.startswith(pattern[:-1]):
                return True
                
        return False
    
    def get_file_hash(self, filepath: Path) -> str:
        """Get MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "ERROR"
    
    def get_file_info(self, filepath: Path) -> Dict:
        """Get file information"""
        try:
            stat = filepath.stat()
            return {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "hash": self.get_file_hash(filepath)
            }
        except:
            return None
    
    def scan_directory(self, root_path: Path) -> Dict[str, Dict]:
        """Scan directory and return file information"""
        files = {}
        
        for path in root_path.rglob('*'):
            if path.is_file() and not self.should_ignore(path):
                relative_path = path.relative_to(root_path)
                files[str(relative_path)] = self.get_file_info(path)
                
        return files
    
    def compare_directories(self) -> Dict:
        """Compare TORI and pigpen directories"""
        print("🔍 Scanning TORI directory...")
        tori_files = self.scan_directory(self.tori_path)
        
        print("🔍 Scanning pigpen directory...")
        pigpen_files = self.scan_directory(self.pigpen_path)
        
        # Find differences
        tori_only = set(tori_files.keys()) - set(pigpen_files.keys())
        pigpen_only = set(pigpen_files.keys()) - set(tori_files.keys())
        common_files = set(tori_files.keys()) & set(pigpen_files.keys())
        
        # Find modified files
        modified_files = []
        for file in common_files:
            if tori_files[file] and pigpen_files[file]:
                if tori_files[file]['hash'] != pigpen_files[file]['hash']:
                    modified_files.append({
                        'file': file,
                        'tori_modified': tori_files[file]['modified'],
                        'pigpen_modified': pigpen_files[file]['modified'],
                        'tori_size': tori_files[file]['size'],
                        'pigpen_size': pigpen_files[file]['size']
                    })
        
        return {
            'tori_only': sorted(list(tori_only)),
            'pigpen_only': sorted(list(pigpen_only)),
            'modified': sorted(modified_files, key=lambda x: x['file']),
            'total_tori': len(tori_files),
            'total_pigpen': len(pigpen_files)
        }
    
    def show_file_diff(self, relative_path: str, context_lines: int = 3):
        """Show diff for a specific file"""
        tori_file = self.tori_path / relative_path
        pigpen_file = self.pigpen_path / relative_path
        
        if not tori_file.exists() or not pigpen_file.exists():
            print(f"❌ File not found in both directories: {relative_path}")
            return
        
        try:
            with open(tori_file, 'r', encoding='utf-8', errors='ignore') as f:
                tori_lines = f.readlines()
            with open(pigpen_file, 'r', encoding='utf-8', errors='ignore') as f:
                pigpen_lines = f.readlines()
            
            diff = difflib.unified_diff(
                tori_lines, pigpen_lines,
                fromfile=f'TORI/{relative_path}',
                tofile=f'pigpen/{relative_path}',
                n=context_lines
            )
            
            diff_text = ''.join(diff)
            if diff_text:
                print(diff_text)
            else:
                print("Files are identical")
                
        except Exception as e:
            print(f"❌ Error reading files: {e}")
    
    def analyze_modifications(self, results: Dict) -> Dict:
        """Analyze modification patterns"""
        analysis = {
            'datasets': [],
            'code_changes': [],
            'config_changes': [],
            'new_features': [],
            'recent_tori_changes': [],
            'recent_pigpen_changes': []
        }
        
        # Check for dataset files in pigpen only
        for file in results['pigpen_only']:
            if any(ext in file for ext in ['.npz', '.json', '.csv', '.pkl', '.h5']):
                analysis['datasets'].append(file)
        
        # Analyze modified files
        now = datetime.now()
        for mod in results['modified']:
            file = mod['file']
            
            # Parse timestamps
            tori_time = datetime.fromisoformat(mod['tori_modified'])
            pigpen_time = datetime.fromisoformat(mod['pigpen_modified'])
            
            # Check if modification is recent (last 48 hours)
            if (now - tori_time).total_seconds() < 48 * 3600:
                analysis['recent_tori_changes'].append(f"{file} (modified {tori_time.strftime('%Y-%m-%d %H:%M')})")
            
            if (now - pigpen_time).total_seconds() < 48 * 3600:
                analysis['recent_pigpen_changes'].append(f"{file} (modified {pigpen_time.strftime('%Y-%m-%d %H:%M')})")
            
            # Categorize by file type
            if file.endswith('.py'):
                analysis['code_changes'].append(file)
            elif file.endswith(('.json', '.yaml', '.yml', '.ini', '.conf')):
                analysis['config_changes'].append(file)
        
        # New features (files only in pigpen)
        for file in results['pigpen_only']:
            if file.endswith('.py'):
                analysis['new_features'].append(file)
        
        return analysis

def main():
    """Main comparison function"""
    tori_path = r"{PROJECT_ROOT}"
    pigpen_path = r"C:\Users\jason\Desktop\pigpen"
    
    print("🔄 TORI vs Pigpen Comparison Tool")
    print("="*60)
    print(f"TORI:   {tori_path}")
    print(f"Pigpen: {pigpen_path}")
    print("="*60)
    
    comparator = DirectoryComparator(tori_path, pigpen_path)
    
    # Run comparison
    results = comparator.compare_directories()
    
    # Analyze results
    analysis = comparator.analyze_modifications(results)
    
    # Save results
    output_file = Path("tori_pigpen_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison': results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total files in TORI: {results['total_tori']}")
    print(f"Total files in pigpen: {results['total_pigpen']}")
    print(f"Files only in TORI: {len(results['tori_only'])}")
    print(f"Files only in pigpen: {len(results['pigpen_only'])}")
    print(f"Modified files: {len(results['modified'])}")
    
    # Show recent changes
    if analysis['recent_pigpen_changes']:
        print(f"\n🆕 Recent changes in pigpen (last 48h):")
        for change in analysis['recent_pigpen_changes'][:10]:
            print(f"   - {change}")
    
    if analysis['recent_tori_changes']:
        print(f"\n🆕 Recent changes in TORI (last 48h):")
        for change in analysis['recent_tori_changes'][:10]:
            print(f"   - {change}")
    
    # Show datasets
    if analysis['datasets']:
        print(f"\n📊 Datasets found in pigpen:")
        for dataset in analysis['datasets'][:10]:
            print(f"   - {dataset}")
    
    # Show new features
    if analysis['new_features']:
        print(f"\n✨ New Python files in pigpen:")
        for feature in analysis['new_features'][:10]:
            print(f"   - {feature}")
    
    # Key modified files
    if analysis['code_changes']:
        print(f"\n🔧 Modified Python files:")
        for code in analysis['code_changes'][:10]:
            print(f"   - {code}")
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Interactive diff viewer
    print("\n" + "="*60)
    print("📝 Interactive Diff Viewer")
    print("Enter a filename to see differences, or 'quit' to exit")
    print("Example: ingest_pdf/pipeline.py")
    print("="*60)
    
    while True:
        filename = input("\nFile to compare (or 'quit'): ").strip()
        if filename.lower() == 'quit':
            break
        if filename:
            comparator.show_file_diff(filename.replace('\\', '/'))

if __name__ == "__main__":
    main()
