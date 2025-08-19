#!/usr/bin/env python3
# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
Add Apache 2.0 license header to Python source files.

This script adds the standard Apache 2.0 license header with patent retaliation
notice to all Python files in the specified directories that don't already have it.
"""

import argparse
import os
import re
import sys
from typing import List, Set


# The standard license header
LICENSE_HEADER = '''# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.
'''

# Regular expression to detect if a file already has a license header
LICENSE_PATTERN = re.compile(r'# Copyright .+ Apache License')


def should_process_file(filepath: str) -> bool:
    """Check if a file should be processed.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        True if the file should be processed, False otherwise
    """
    # Skip files that aren't Python files
    if not filepath.endswith('.py'):
        return False
    
    # Skip empty files
    if os.path.getsize(filepath) == 0:
        return False
    
    # Skip files that already have a license header
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read(1000)  # Read first 1000 chars to check for header
        if LICENSE_PATTERN.search(content):
            return False
    
    return True


def add_header_to_file(filepath: str, dry_run: bool = False) -> bool:
    """Add license header to a file.
    
    Args:
        filepath: Path to the file to add the header to
        dry_run: If True, don't actually modify the file
        
    Returns:
        True if the file was modified, False otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for shebang line
        if content.startswith('#!'):
            # Keep shebang as first line
            shebang_end = content.find('\n') + 1
            new_content = content[:shebang_end] + LICENSE_HEADER + content[shebang_end:]
        else:
            # Add license at the beginning
            new_content = LICENSE_HEADER + content
        
        # Write back to file if not a dry run
        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def process_directory(directory: str, exclude: Set[str], dry_run: bool) -> List[str]:
    """Process all Python files in a directory recursively.
    
    Args:
        directory: Directory to process
        exclude: Set of directories to exclude
        dry_run: If True, don't actually modify any files
        
    Returns:
        List of files that were modified
    """
    modified_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if should_process_file(filepath):
                if add_header_to_file(filepath, dry_run):
                    modified_files.append(filepath)
                    print(f"{'Would add' if dry_run else 'Added'} header to {filepath}")
    
    return modified_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Add license headers to Python files')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    parser.add_argument('--exclude', nargs='*', default=[], help='Directories to exclude')
    parser.add_argument('--dry-run', action='store_true', help='Do not modify files')
    args = parser.parse_args()
    
    exclude_set = set(os.path.abspath(d) for d in args.exclude)
    
    all_modified = []
    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a directory, skipping", file=sys.stderr)
            continue
        
        print(f"Processing {directory}...")
        modified = process_directory(directory, exclude_set, args.dry_run)
        all_modified.extend(modified)
    
    print(f"\nSummary: {len(all_modified)} files {'would be' if args.dry_run else 'were'} modified")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
