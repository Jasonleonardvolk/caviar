#!/usr/bin/env python3
"""
Main fix script for Soliton API 500 errors
This script applies all fixes identified in the audit
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=None):
    """Print colored log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if color:
        print(f"{color}[{timestamp}] {message}{Colors.END}")
    else:
        print(f"[{timestamp}] {message}")

def create_backup(file_path):
    """Create backup of a file before modifying"""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        log(f"Backed up: {os.path.basename(file_path)}", Colors.GREEN)
        return backup_path
    return None

def fix_pydantic_imports(root_path):
    """Fix Pydantic v2 BaseSettings imports"""
    log("Fixing Pydantic imports...", Colors.YELLOW)
    fixed_count = 0
    
    for py_file in Path(root_path).rglob("*.py"):
        # Skip unnecessary directories
        if any(skip in str(py_file) for skip in ['node_modules', '__pycache__', '.git', 'venv', '.pytest_cache']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file needs fixing
            if 'from pydantic import' in content and 'BaseSettings' in content:
                original = content
                
                # Pattern to find BaseSettings imports
                pattern = r'^from pydantic import (.*)BaseSettings(.*)$'
                
                lines = content.split('\n')
                new_lines = []
                
                for line in lines:
                    if re.match(pattern, line, re.MULTILINE):
                        # Extract other imports
                        match = re.search(r'from pydantic import (.+)', line)
                        if match:
                            imports = [imp.strip() for imp in match.group(1).split(',')]
                            other_imports = [imp for imp in imports if imp != 'BaseSettings']
                            
                            new_lines.append('try:')
                            new_lines.append('    from pydantic_settings import BaseSettings')
                            new_lines.append('except ModuleNotFoundError:')
                            new_lines.append('    from pydantic import BaseSettings')
                            
                            if other_imports:
                                new_lines.append(f"from pydantic import {', '.join(other_imports)}")
                    else:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
                
                if content != original:
                    create_backup(py_file)
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_count += 1
                    log(f"  Fixed: {py_file.relative_to(root_path)}", Colors.GREEN)
                    
        except Exception as e:
            log(f"  Error processing {py_file}: {e}", Colors.RED)
    
    log(f"Fixed {fixed_count} files with Pydantic imports", Colors.GREEN)
    return fixed_count

def fix_asyncio_run(pipeline_path):
    """Fix asyncio.run() inside event loop"""
    log("Fixing asyncio.run() in pipeline...", Colors.YELLOW)
    
    if not os.path.exists(pipeline_path):
        log(f"  Pipeline not found at {pipeline_path}", Colors.RED)
        return False
    
    create_backup(pipeline_path)
    
    try:
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern for asyncio.run
        pattern = r'asyncio\.run\s*\(\s*process_pdf\s*\(([^)]+)\)\s*\)'
        replacement = '''try:
        loop = asyncio.get_running_loop()
        all_concepts = await process_pdf(\\1)
    except RuntimeError:
        all_concepts = asyncio.run(process_pdf(\\1))'''
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            
            with open(pipeline_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            log("  Fixed asyncio.run() pattern", Colors.GREEN)
            return True
        else:
            log("  asyncio.run() pattern not found or already fixed", Colors.BLUE)
            return False
            
    except Exception as e:
        log(f"  Error fixing asyncio: {e}", Colors.RED)
        return False

def consolidate_pipelines(project_root):
    """Consolidate duplicate pipeline files"""
    log("Consolidating duplicate pipelines...", Colors.YELLOW)
    
    canonical_pipeline = os.path.join(project_root, "ingest_pdf", "pipeline", "pipeline.py")
    duplicates = [
        os.path.join(project_root, "piiiiipeline.py"),
        os.path.join(project_root, "python", "core", "pipeline.py"),
    ]
    
    consolidated = 0
    for dup in duplicates:
        if os.path.exists(dup) and dup != canonical_pipeline:
            old_name = f"{dup}.OLD_DUPLICATE"
            try:
                os.rename(dup, old_name)
                log(f"  Renamed: {os.path.basename(dup)} -> {os.path.basename(old_name)}", Colors.GREEN)
                consolidated += 1
            except Exception as e:
                log(f"  Could not rename {dup}: {e}", Colors.RED)
    
    log(f"Consolidated {consolidated} duplicate files", Colors.GREEN)
    return consolidated

def set_environment_variables():
    """Set required environment variables"""
    log("Setting environment variables...", Colors.YELLOW)
    
    os.environ['TORI_DISABLE_MESH_CHECK'] = '1'
    
    # Try to set permanently on Windows
    if sys.platform == 'win32':
        try:
            subprocess.run([
                'setx', 'TORI_DISABLE_MESH_CHECK', '1'
            ], capture_output=True, text=True)
            log("  Set TORI_DISABLE_MESH_CHECK=1 (permanent)", Colors.GREEN)
        except:
            log("  Set TORI_DISABLE_MESH_CHECK=1 (session only)", Colors.YELLOW)
    else:
        log("  Set TORI_DISABLE_MESH_CHECK=1 (session only)", Colors.YELLOW)

def install_dependencies():
    """Install required Python packages"""
    log("Installing dependencies...", Colors.YELLOW)
    
    packages = ['pydantic_settings', 'httpx', 'pytest-asyncio']
    python_exe = sys.executable
    
    for package in packages:
        try:
            subprocess.run([
                python_exe, '-m', 'pip', 'install', package, '--quiet'
            ], check=True)
            log(f"  Installed: {package}", Colors.GREEN)
        except subprocess.CalledProcessError:
            log(f"  Failed to install: {package}", Colors.RED)

def main():
    """Main fix execution"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("="*60)
    print("       SOLITON API 500 ERROR FIX SCRIPT")
    print("="*60)
    print(f"{Colors.END}")
    
    # Get project root
    project_root = r"{PROJECT_ROOT}"
    os.chdir(project_root)
    
    log(f"Working directory: {project_root}", Colors.BLUE)
    
    # Apply fixes
    fixes_applied = []
    
    # 1. Set environment variables
    set_environment_variables()
    fixes_applied.append("Environment variables")
    
    # 2. Fix Pydantic imports
    pydantic_fixed = fix_pydantic_imports(project_root)
    if pydantic_fixed > 0:
        fixes_applied.append(f"Pydantic imports ({pydantic_fixed} files)")
    
    # 3. Fix asyncio.run
    pipeline_path = os.path.join(project_root, "ingest_pdf", "pipeline", "pipeline.py")
    if fix_asyncio_run(pipeline_path):
        fixes_applied.append("asyncio.run() in pipeline")
    
    # 4. Consolidate pipelines
    consolidated = consolidate_pipelines(project_root)
    if consolidated > 0:
        fixes_applied.append(f"Pipeline consolidation ({consolidated} files)")
    
    # 5. Install dependencies
    install_dependencies()
    fixes_applied.append("Python dependencies")
    
    # Summary
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}\n")
    
    log("Fixes applied:", Colors.GREEN)
    for fix in fixes_applied:
        log(f"  ✓ {fix}", Colors.GREEN)
    
    print(f"\n{Colors.YELLOW}Next steps:{Colors.END}")
    print("1. Apply the fixed soliton API route:")
    print(f"   {Colors.CYAN}python apply_soliton_api_fix.py{Colors.END}")
    print("\n2. Start the backend with debug logging:")
    print(f"   {Colors.CYAN}python start_backend_debug.py{Colors.END}")
    print("\n3. Test the API:")
    print(f"   {Colors.CYAN}python test_soliton_api.py{Colors.END}")
    
    print(f"\n{Colors.GREEN}Fix script completed!{Colors.END}\n")

if __name__ == "__main__":
    main()
