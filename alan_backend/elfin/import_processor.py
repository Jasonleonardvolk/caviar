#!/usr/bin/env python3
"""
ELFIN Import Processor

This tool implements the "Acyclic Import Kiwi" feature which enables
`import ... from "..."` in ELFIN files without supporting templates or recursion.

It uses direct text substitution to inline imported helpers and components.
"""

import os
import re
import sys
from pathlib import Path
import filecmp

def resolve_import_path(import_path, base_dir):
    """
    Resolve the import path relative to the base directory.
    
    Args:
        import_path: Path string from the import statement
        base_dir: Base directory of the importing file
        
    Returns:
        Resolved Path object
    """
    # Standardize paths for Windows/Unix compatibility
    import_path = import_path.replace('\\', '/')
    
    # If the path is absolute, use it directly
    if os.path.isabs(import_path):
        return Path(import_path)
    
    # For ELFIN, we need to check a few locations
    
    # 1. Try direct relative path (relative to file location)
    direct_path = base_dir / import_path
    if direct_path.exists():
        return direct_path
    
    # 2. Try relative to standard library location (templates directory)
    # Find the templates directory by going up from the base_dir until we find it
    templates_dir = None
    current_dir = base_dir
    for _ in range(10):  # Limit search depth
        if (current_dir / "templates").exists():
            templates_dir = current_dir / "templates"
            break
        if current_dir.parent == current_dir:  # Reached root
            break
        current_dir = current_dir.parent
    
    if templates_dir:
        std_path = templates_dir / import_path
        if std_path.exists():
            return std_path
    
    # 3. If we only have the section of the path after templates dir,
    # try finding it in the templates dir directly
    templates_path = Path("alan_backend") / "elfin" / "templates" / import_path
    if templates_path.exists():
        return templates_path
    
    # 4. If still not found, default to relative to base_dir
    return base_dir / import_path

def extract_section(file_content, section_name):
    """
    Extract a section from a file, such as a 'helpers' block.
    
    Args:
        file_content: String content of the file
        section_name: Name of the section to extract (e.g., "StdHelpers")
        
    Returns:
        The extracted section as a string, or None if not found
    """
    # Match a helpers block with the given name
    pattern = r'helpers\s+' + re.escape(section_name) + r'\s*\{([^}]*)\}'
    match = re.search(pattern, file_content, re.DOTALL)
    
    if match:
        # Return the content within the braces
        return match.group(1).strip()
    
    return None

def process_imports(file_path, output_path=None):
    """
    Process all imports in an ELFIN file.
    
    Args:
        file_path: Path to the ELFIN file to process
        output_path: Path to write the output file (optional, defaults to file_path + '.processed')
        
    Returns:
        Path to the processed file
    """
    # Setup paths
    file_path = Path(file_path)
    base_dir = file_path.parent
    
    if output_path is None:
        output_path = file_path.with_suffix(file_path.suffix + '.processed')
    
    # Read the input file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track if any changes were made
    changes_made = False
    
    # Find all import statements
    import_pattern = r'import\s+([A-Za-z0-9_]+)\s+from\s+"([^"]+)"'
    
    for match in re.finditer(import_pattern, content):
        section_name = match.group(1)
        import_path_str = match.group(2)
        
        # Get the full import statement
        import_statement = match.group(0)
        
        # Resolve the import path
        import_path = resolve_import_path(import_path_str, base_dir)
        
        # Check if the import file exists
        if not import_path.exists():
            print(f"Error: Import file not found: {import_path}")
            return None
        
        # Read the import file
        with open(import_path, 'r', encoding='utf-8') as f:
            import_content = f.read()
        
        # Extract the section from the import file
        section_content = extract_section(import_content, section_name)
        
        if section_content is None:
            print(f"Error: Section '{section_name}' not found in {import_path}")
            return None
        
        # Create the replacement - a local helpers block with the imported content
        replacement = f"helpers {{\n{section_content}\n}}"
        
        # Replace the import statement with the content
        content = content.replace(import_statement, replacement)
        changes_made = True
        
        print(f"Processed import: {section_name} from {import_path}")
    
    # If no changes were made, just link the input to the output
    if not changes_made:
        print(f"No imports found in {file_path}")
        # If output_path is different from input, copy the file
        if str(output_path) != str(file_path):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        return output_path
    
    # Write the processed content to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Processed file written to: {output_path}")
    
    return output_path

def verify_equivalence(original_file, processed_file):
    """
    Verify that the processed file is functionally equivalent to what would be
    expected after import resolution.
    
    For now, this is a simple line-based comparison that ignores whitespace
    differences. A more sophisticated verification would involve parsing both
    files and comparing their ASTs.
    
    Args:
        original_file: Path to the original file
        processed_file: Path to the processed file
        
    Returns:
        True if the files are equivalent, False otherwise
    """
    # Simple file comparison using filecmp
    if filecmp.cmp(original_file, processed_file, shallow=False):
        return True
    
    # If files differ, do a more detailed comparison that's tolerant of whitespace changes
    with open(original_file, 'r', encoding='utf-8') as f1:
        orig_lines = [line.strip() for line in f1.readlines() if line.strip()]
    
    with open(processed_file, 'r', encoding='utf-8') as f2:
        proc_lines = [line.strip() for line in f2.readlines() if line.strip()]
    
    # Compare line counts
    if len(orig_lines) != len(proc_lines):
        print(f"Files differ in line count: {len(orig_lines)} vs {len(proc_lines)}")
        return False
    
    # Compare each line
    for i, (orig, proc) in enumerate(zip(orig_lines, proc_lines)):
        if orig != proc:
            print(f"Files differ at line {i+1}:")
            print(f"  Original: {orig}")
            print(f"  Processed: {proc}")
            return False
    
    return True

def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python import_processor.py <elfin_file> [output_file]")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Process imports
    processed_file = process_imports(input_file, output_file)
    
    if processed_file is None:
        print("Import processing failed.")
        return 1
    
    # Create a reference file to compare against
    # This would be the expected result after manual substitution
    # In a real implementation, we'd verify against a pre-created reference
    # But for this example, we'll just check that the output is different from the input
    if os.path.getsize(input_file) == os.path.getsize(processed_file):
        print("Warning: Output file has the same size as input. Imports may not have been processed.")
    
    print("Import processing completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
