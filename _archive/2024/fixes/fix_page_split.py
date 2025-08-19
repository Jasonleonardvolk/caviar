#!/usr/bin/env python3
"""
Fix the split +page.svelte file by merging part2 back into the main file
"""

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_split_page(project_root: str):
    """Merge +page_part2.svelte back into +page.svelte"""
    
    svelte_dir = Path(project_root) / 'tori_ui_svelte' / 'src' / 'routes'
    main_page = svelte_dir / '+page.svelte'
    part2_page = svelte_dir / '+page_part2.svelte'
    
    if not part2_page.exists():
        print(f"❌ Part 2 file not found: {part2_page}")
        return False
    
    if not main_page.exists():
        print(f"❌ Main page file not found: {main_page}")
        return False
    
    # Read both files
    with open(main_page, 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    with open(part2_page, 'r', encoding='utf-8') as f:
        part2_content = f.read()
    
    # Find where to insert part2 content
    # The part2 content should go before the closing </script> tag
    script_end_pos = main_content.rfind('</script>')
    
    if script_end_pos == -1:
        print("❌ Could not find </script> tag in main page")
        return False
    
    # Extract the code portion from part2 (before the <script> opening)
    # The part2 file starts with code that belongs inside the script section
    code_end_pos = part2_content.find('</script>')
    if code_end_pos != -1:
        # Extract just the code portion
        part2_code = part2_content[:code_end_pos].strip()
        # Also get the template portion after </script>
        template_start = part2_content.find('</script>') + len('</script>')
        part2_template = part2_content[template_start:].strip()
    else:
        # Assume entire content is code
        part2_code = part2_content.strip()
        part2_template = ""
    
    # Insert the code before the closing script tag
    merged_content = (
        main_content[:script_end_pos] + 
        "\n\n  // === MERGED FROM +page_part2.svelte ===\n" +
        part2_code + "\n" +
        main_content[script_end_pos:]
    )
    
    # If there's template content, we need to merge it too
    if part2_template:
        # Find where the template starts in the merged content
        # This is complex - for now just append at the end
        merged_content += "\n\n<!-- MERGED TEMPLATE FROM +page_part2.svelte -->\n" + part2_template
    
    # Backup the original
    backup_path = main_page.with_suffix('.svelte.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(main_content)
    print(f"✅ Created backup: {backup_path}")
    
    # Write the merged content
    with open(main_page, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    print(f"✅ Merged content written to: {main_page}")
    
    # Delete the part2 file
    part2_page.unlink()
    print(f"✅ Deleted: {part2_page}")
    
    return True

if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else r"{PROJECT_ROOT}"
    
    print("🔧 Fixing split +page.svelte files...")
    if fix_split_page(project_root):
        print("✅ Successfully merged split page files!")
    else:
        print("❌ Failed to merge split page files")
