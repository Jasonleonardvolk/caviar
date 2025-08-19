#!/usr/bin/env python3
"""
Filesystem Map Generator
========================
Creates a structured map of your project's filesystem, including file sizes,
types, and directory structure. Outputs both JSON and Markdown formats.

Usage
-----
    python tools/gen_map.py
    python tools/gen_map.py --root . --output filesystem_map.md
    python tools/gen_map.py --root . --skip-dir node_modules --max-depth 3
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional, Any

# Directories to skip by default
SKIP_DIRS: Set[str] = {'.git', '.venv', 'node_modules', '__pycache__', 'dist', 'build'}

class FilesystemMapper:
    def __init__(self, skip_dirs: Optional[Set[str]] = None):
        self.skip_dirs = skip_dirs or SKIP_DIRS
        self.file_map = {}
        self.stats = {
            'total_files': 0,
            'total_dirs': 0,
            'total_size': 0,
            'file_types': {},
            'largest_files': [],
            'errors': []
        }
    
    def format_size(self, size: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def scan_directory(self, directory: Path, max_depth: Optional[int] = None) -> Dict[str, Any]:
        """Scan directory and create a map of its structure."""
        base_path = Path(directory).resolve()
        
        def scan_recursive(path: Path, current_depth: int = 0) -> Optional[Dict[str, Any]]:
            if max_depth and current_depth > max_depth:
                return None
            
            # Skip directories in SKIP_DIRS
            if path.is_dir() and path.name in self.skip_dirs:
                return None
            
            try:
                if path.is_file():
                    self.stats['total_files'] += 1
                    size = path.stat().st_size
                    self.stats['total_size'] += size
                    
                    # Track file types
                    ext = path.suffix.lower() or 'no_extension'
                    self.stats['file_types'][ext] = self.stats['file_types'].get(ext, 0) + 1
                    
                    # Track largest files
                    file_info = {
                        'path': str(path.relative_to(base_path)),
                        'size': size
                    }
                    self.stats['largest_files'].append(file_info)
                    # Keep only top 10 largest
                    self.stats['largest_files'].sort(key=lambda x: x['size'], reverse=True)
                    self.stats['largest_files'] = self.stats['largest_files'][:10]
                    
                    return {
                        'type': 'file',
                        'name': path.name,
                        'size': size,
                        'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    }
                    
                elif path.is_dir():
                    self.stats['total_dirs'] += 1
                    children = {}
                    
                    try:
                        for child in path.iterdir():
                            child_data = scan_recursive(child, current_depth + 1)
                            if child_data:
                                children[child.name] = child_data
                    except (PermissionError, OSError) as e:
                        self.stats['errors'].append(f"Cannot read directory {path}: {str(e)}")
                    
                    return {
                        'type': 'directory',
                        'name': path.name,
                        'children': children
                    }
                    
            except PermissionError:
                self.stats['errors'].append(f"Permission denied: {path}")
                return {
                    'type': 'error',
                    'name': path.name,
                    'error': 'Permission denied'
                }
            except Exception as e:
                self.stats['errors'].append(f"Error reading {path}: {str(e)}")
                return {
                    'type': 'error',
                    'name': path.name,
                    'error': str(e)
                }
        
        self.file_map = scan_recursive(base_path)
        return self.file_map
    
    def save_json(self, output_file: Path):
        """Save the filesystem map to a JSON file."""
        output_data = {
            'scan_time': datetime.now().isoformat(),
            'statistics': self.stats,
            'structure': self.file_map
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[gen_map] JSON map saved to: {output_file}")
    
    def save_markdown(self, output_file: Path):
        """Save the filesystem map to a Markdown file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Filesystem Map\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Statistics section
            f.write("## Statistics\n\n")
            f.write(f"- **Total Files**: {self.stats['total_files']:,}\n")
            f.write(f"- **Total Directories**: {self.stats['total_dirs']:,}\n")
            f.write(f"- **Total Size**: {self.format_size(self.stats['total_size'])}\n\n")
            
            # File types section
            f.write("## File Types\n\n")
            sorted_types = sorted(self.stats['file_types'].items(), 
                                key=lambda x: x[1], reverse=True)
            for ext, count in sorted_types[:15]:  # Top 15 file types
                f.write(f"- `{ext}`: {count} files\n")
            
            if len(sorted_types) > 15:
                f.write(f"- ... and {len(sorted_types) - 15} more types\n")
            f.write("\n")
            
            # Largest files section
            if self.stats['largest_files']:
                f.write("## Largest Files\n\n")
                for i, file_info in enumerate(self.stats['largest_files'], 1):
                    f.write(f"{i}. `{file_info['path']}` - {self.format_size(file_info['size'])}\n")
                f.write("\n")
            
            # Directory tree section
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            self._write_tree(f, self.file_map, "", True)
            f.write("```\n")
            
            # Errors section
            if self.stats['errors']:
                f.write("\n## Errors Encountered\n\n")
                for error in self.stats['errors'][:10]:  # Show first 10 errors
                    f.write(f"- {error}\n")
                if len(self.stats['errors']) > 10:
                    f.write(f"- ... and {len(self.stats['errors']) - 10} more errors\n")
        
        print(f"[gen_map] Markdown map saved to: {output_file}")
        print(f"[gen_map] Total files: {self.stats['total_files']:,}")
        print(f"[gen_map] Total directories: {self.stats['total_dirs']:,}")
        print(f"[gen_map] Total size: {self.format_size(self.stats['total_size'])}")
    
    def _write_tree(self, f, node: Dict[str, Any], prefix: str, is_last: bool):
        """Write tree structure to file."""
        if node['type'] == 'error':
            return
        
        connector = "└── " if is_last else "├── "
        f.write(prefix + connector + node['name'])
        
        if node['type'] == 'file':
            f.write(f" ({self.format_size(node['size'])})")
        f.write("\n")
        
        if node['type'] == 'directory' and 'children' in node:
            extension = "    " if is_last else "│   "
            children = list(node['children'].items())
            for i, (name, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                self._write_tree(f, child, prefix + extension, is_last_child)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a filesystem map of your project."
    )
    parser.add_argument(
        "--root", 
        default=".", 
        help="Root directory to scan"
    )
    parser.add_argument(
        "--output", "-o",
        default="filesystem_map.md",
        help="Output file name (use .json for JSON format)"
    )
    parser.add_argument(
        "--max-depth", 
        type=int,
        help="Maximum directory depth to scan"
    )
    parser.add_argument(
        "--skip-dir", 
        action='append', 
        default=[],
        help='Directory name to ignore (can be given multiple times)'
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help='Show more detailed output'
    )
    args = parser.parse_args()
    
    # Update skip directories
    skip_dirs = SKIP_DIRS.copy()
    if args.skip_dir:
        skip_dirs.update(args.skip_dir)
    
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[gen_map] Error: Root path does not exist: {root}")
        return 1
    
    if args.verbose:
        print(f"[gen_map] Scanning: {root}")
        print(f"[gen_map] Skipping dirs: {', '.join(sorted(skip_dirs))}")
        if args.max_depth:
            print(f"[gen_map] Max depth: {args.max_depth}")
    
    # Create mapper and scan
    mapper = FilesystemMapper(skip_dirs)
    print(f"[gen_map] Scanning {root}...")
    mapper.scan_directory(root, args.max_depth)
    
    # Save output
    output_path = Path(args.output)
    if output_path.suffix.lower() == '.json':
        mapper.save_json(output_path)
    else:
        mapper.save_markdown(output_path)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
