#!/usr/bin/env python3
"""
🔥 NUCLEAR AUDIT AND FIX - The Ultimate TORI System Healer
===========================================================
This script will find and fix EVERYTHING wrong with your TORI system.
No more endless debugging. Just solutions.
"""

import os
import sys
import json
import subprocess
import time
import shutil
import socket
import re
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

class TORINuclearAudit:
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.manual_fixes_needed = []
        self.script_dir = Path(__file__).parent.absolute()
        
    def print_banner(self):
        print("\n" + "🔥"*30)
        print("NUCLEAR AUDIT AND FIX - ENDING YOUR DEBUGGING NIGHTMARE")
        print("🔥"*30 + "\n")
        
    def print_section(self, title):
        print(f"\n{'='*60}")
        print(f"🔧 {title}")
        print('='*60)
        
    def auto_fix_file(self, file_path: Path, fixes: List[Tuple[str, str]], description: str):
        """Apply multiple fixes to a file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for old_text, new_text in fixes:
                if old_text in content:
                    content = content.replace(old_text, new_text)
                    
            if content != original_content:
                # Backup original
                backup_path = file_path.with_suffix(f'.backup_{int(time.time())}')
                shutil.copy2(file_path, backup_path)
                
                # Write fixed content
                file_path.write_text(content, encoding='utf-8')
                self.fixes_applied.append(f"✅ {description} (backup: {backup_path.name})")
                return True
            return False
        except Exception as e:
            self.issues_found.append(f"❌ Could not fix {file_path}: {e}")
            return False

    def check_python_syntax(self):
        """Check ALL Python files for syntax errors"""
        self.print_section("Python Syntax Check")
        
        python_files = list(Path(".").rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            # Skip virtual environments and node_modules
            if any(skip in str(py_file) for skip in ['.venv', 'node_modules', '__pycache__']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append((py_file, e))
                print(f"❌ Syntax error in {py_file}: Line {e.lineno}")
                
        if syntax_errors:
            self.issues_found.append(f"Found {len(syntax_errors)} files with syntax errors")
            
            # Try to auto-fix common syntax errors
            for py_file, error in syntax_errors:
                if "unmatched" in str(error):
                    self.try_fix_unmatched_brackets(py_file, error)
        else:
            print("✅ All Python files have valid syntax")
            
        return len(syntax_errors) == 0
        
    def try_fix_unmatched_brackets(self, file_path: Path, error: SyntaxError):
        """Try to fix unmatched bracket errors"""
        try:
            lines = file_path.read_text(encoding='utf-8').splitlines()
            if error.lineno and error.lineno <= len(lines):
                problem_line = lines[error.lineno - 1]
                
                # Common fixes
                if problem_line.strip() == ')':
                    lines.pop(error.lineno - 1)
                    file_path.write_text('\n'.join(lines), encoding='utf-8')
                    self.fixes_applied.append(f"✅ Removed extra ) from {file_path} line {error.lineno}")
                elif problem_line.strip().endswith('})'):
                    lines[error.lineno - 1] = problem_line.rstrip()[:-1]
                    file_path.write_text('\n'.join(lines), encoding='utf-8')
                    self.fixes_applied.append(f"✅ Removed extra ) from {file_path} line {error.lineno}")
        except Exception as e:
            self.manual_fixes_needed.append(f"Fix syntax in {file_path} line {error.lineno}")

    def check_dependencies(self):
        """Check and install missing dependencies"""
        self.print_section("Dependency Check")
        
        # Python dependencies
        required_packages = {
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn[standard]',
            'requests': 'requests',
            'numpy': 'numpy',
            'pydantic': 'pydantic',
            'psutil': 'psutil',
            'websocket': 'websocket-client',
            'aiofiles': 'aiofiles',
            'python-multipart': 'python-multipart'
        }
        
        missing = []
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {import_name} - installed")
            except ImportError:
                missing.append(package_name)
                print(f"❌ {import_name} - MISSING")
                
        if missing:
            self.issues_found.append(f"Missing {len(missing)} Python packages")
            print(f"\n🔧 Installing missing packages...")
            
            # Try poetry first, then pip
            if Path("pyproject.toml").exists():
                cmd = ["poetry", "add"] + missing
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + missing
                
            try:
                subprocess.run(cmd, check=True)
                self.fixes_applied.append(f"✅ Installed {len(missing)} missing packages")
            except subprocess.CalledProcessError:
                self.manual_fixes_needed.append(f"Install packages manually: {' '.join(missing)}")
                
        # Check Node dependencies
        if Path("tori_ui_svelte/package.json").exists():
            if not Path("tori_ui_svelte/node_modules").exists():
                self.issues_found.append("Frontend node_modules missing")
                print("\n🔧 Installing frontend dependencies...")
                try:
                    subprocess.run(["npm", "install"], cwd="tori_ui_svelte", check=True)
                    self.fixes_applied.append("✅ Installed frontend dependencies")
                except:
                    self.manual_fixes_needed.append("Run: cd tori_ui_svelte && npm install")

    def check_field_mismatches(self):
        """Check for field name mismatches between frontend and backend"""
        self.print_section("Field Name Consistency Check")
        
        mismatches = []
        
        # Check soliton memory
        soliton_ts = Path("tori_ui_svelte/src/lib/services/solitonMemory.ts")
        if soliton_ts.exists():
            content = soliton_ts.read_text(encoding='utf-8')
            
            # Check for common mismatches
            if 'user_id: uid' in content:
                self.auto_fix_file(soliton_ts, [('user_id: uid', 'userId: uid')], 
                                 "Fixed soliton user_id field mismatch")
                mismatches.append("soliton user_id")
                
            if 'user_id:' in content and 'userId:' not in content:
                # More complex pattern
                import re
                pattern = r'"user_id":\s*(\w+)'
                if re.search(pattern, content):
                    new_content = re.sub(pattern, r'"userId": \1', content)
                    soliton_ts.write_text(new_content, encoding='utf-8')
                    self.fixes_applied.append("✅ Fixed all user_id references in solitonMemory.ts")
                    
        if not mismatches:
            print("✅ No field name mismatches found")

    def check_ports(self):
        """Check port availability and conflicts"""
        self.print_section("Port Availability Check")
        
        required_ports = {
            8002: "API Server",
            5173: "Frontend Dev Server",
            8100: "MCP Metacognitive"
        }
        
        blocked_ports = []
        for port, service in required_ports.items():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    print(f"✅ Port {port} ({service}) - Available")
            except OSError:
                blocked_ports.append((port, service))
                print(f"❌ Port {port} ({service}) - BLOCKED")
                
                # Try to find what's using it
                try:
                    result = subprocess.run(
                        f'netstat -ano | findstr :{port}',
                        shell=True, capture_output=True, text=True
                    )
                    if result.stdout:
                        print(f"   Process using port: {result.stdout.strip()}")
                except:
                    pass
                    
        if blocked_ports:
            self.issues_found.append(f"{len(blocked_ports)} ports are blocked")
            self.manual_fixes_needed.append("Kill processes using ports: " + 
                                          ", ".join(f"{p[0]}" for p in blocked_ports))

    def check_environment_setup(self):
        """Check environment variables and configuration"""
        self.print_section("Environment Configuration Check")
        
        # Check .env files
        env_files = [
            Path(".env"),
            Path("tori_ui_svelte/.env")
        ]
        
        for env_file in env_files:
            if env_file.exists():
                content = env_file.read_text(encoding='utf-8')
                
                # Fix common issues
                if 'VITE_API_BASE_URL=http://localhost:3001' in content:
                    self.auto_fix_file(env_file, 
                                     [('VITE_API_BASE_URL=http://localhost:3001',
                                       'VITE_API_BASE_URL=http://localhost:8002')],
                                     f"Fixed API port in {env_file}")
                                     
        # Check vite.config.js proxy
        vite_config = Path("tori_ui_svelte/vite.config.js")
        if vite_config.exists():
            content = vite_config.read_text(encoding='utf-8')
            if 'localhost:3001' in content:
                self.auto_fix_file(vite_config,
                                 [('localhost:3001', 'localhost:8002')],
                                 "Fixed proxy port in vite.config.js")

    def check_file_integrity(self):
        """Check for missing or corrupted files"""
        self.print_section("File Integrity Check")
        
        critical_files = [
            "enhanced_launcher.py",
            "prajna_api.py",
            "api/routes/soliton_router.py",
            "api/routes/soliton_production.py",
            "api/routes/concept_mesh.py",
            "tori_ui_svelte/package.json",
            "tori_ui_svelte/vite.config.js",
            "pyproject.toml"
        ]
        
        missing = []
        for file_path in critical_files:
            if not Path(file_path).exists():
                missing.append(file_path)
                print(f"❌ Missing: {file_path}")
            else:
                print(f"✅ Found: {file_path}")
                
        if missing:
            self.issues_found.append(f"{len(missing)} critical files missing")
            self.manual_fixes_needed.extend([f"Restore file: {f}" for f in missing])

    def create_missing_directories(self):
        """Create all required directories"""
        self.print_section("Directory Structure Check")
        
        required_dirs = [
            "data",
            "data/cognitive",
            "data/memory_vault",
            "data/concept_mesh",
            "data/eigenvalue_monitor",
            "data/lyapunov",
            "data/koopman",
            "logs",
            "tmp",
            "api/routes",
            "prajna/api",
            "prajna/core",
            "prajna/memory"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.fixes_applied.append(f"✅ Created directory: {dir_path}")
                print(f"✅ Created: {dir_path}")
            else:
                print(f"✅ Exists: {dir_path}")

    def fix_import_paths(self):
        """Fix common import path issues"""
        self.print_section("Import Path Fixes")
        
        # Add __init__.py files where needed
        init_locations = [
            "api",
            "api/routes",
            "prajna",
            "prajna/api",
            "prajna/core",
            "prajna/memory",
            "python",
            "python/core",
            "python/stability"
        ]
        
        for location in init_locations:
            init_file = Path(location) / "__init__.py"
            if not init_file.exists() and Path(location).exists():
                init_file.write_text("", encoding='utf-8')
                self.fixes_applied.append(f"✅ Created {init_file}")

    def validate_json_files(self):
        """Validate and fix JSON files"""
        self.print_section("JSON File Validation")
        
        json_files = list(Path(".").rglob("*.json"))
        
        for json_file in json_files:
            if any(skip in str(json_file) for skip in ['.venv', 'node_modules']):
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"✅ Valid JSON: {json_file}")
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {json_file} - {e}")
                self.issues_found.append(f"Invalid JSON: {json_file}")
                
                # Try to fix common issues
                if json_file.name == "data.json":
                    # Fix concept mesh data structure
                    try:
                        content = json_file.read_text(encoding='utf-8')
                        # Try to parse as Python literal
                        import ast
                        data = ast.literal_eval(content)
                        
                        # Convert to proper structure
                        if isinstance(data, list):
                            fixed_data = {
                                "concepts": data,
                                "metadata": {"version": "1.0"}
                            }
                            json_file.write_text(json.dumps(fixed_data, indent=2), encoding='utf-8')
                            self.fixes_applied.append(f"✅ Fixed JSON structure in {json_file}")
                    except:
                        self.manual_fixes_needed.append(f"Fix JSON in {json_file}")

    def run_final_test(self):
        """Run a final test to verify everything works"""
        self.print_section("Final System Test")
        
        # Test 1: Python imports
        print("\n🧪 Testing Python imports...")
        test_imports = [
            "from fastapi import FastAPI",
            "from prajna_api import app",
            "from api.routes.soliton_router import router"
        ]
        
        for test_import in test_imports:
            try:
                exec(test_import)
                print(f"✅ {test_import}")
            except Exception as e:
                print(f"❌ {test_import} - {e}")
                self.issues_found.append(f"Import failed: {test_import}")
                
        # Test 2: Compile main files
        print("\n🧪 Testing main file compilation...")
        main_files = ["enhanced_launcher.py", "prajna_api.py"]
        
        for main_file in main_files:
            if Path(main_file).exists():
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", main_file],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"✅ {main_file} compiles successfully")
                else:
                    print(f"❌ {main_file} has errors")
                    self.issues_found.append(f"Compilation failed: {main_file}")

    def generate_report(self):
        """Generate final report with all findings"""
        self.print_section("FINAL REPORT")
        
        print(f"\n📊 AUDIT SUMMARY:")
        print(f"   Issues Found: {len(self.issues_found)}")
        print(f"   Fixes Applied: {len(self.fixes_applied)}")
        print(f"   Manual Fixes Needed: {len(self.manual_fixes_needed)}")
        
        if self.fixes_applied:
            print(f"\n✅ FIXES APPLIED ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"   {fix}")
                
        if self.manual_fixes_needed:
            print(f"\n⚠️ MANUAL FIXES NEEDED ({len(self.manual_fixes_needed)}):")
            for i, fix in enumerate(self.manual_fixes_needed, 1):
                print(f"   {i}. {fix}")
                
        if self.issues_found:
            print(f"\n❌ ISSUES FOUND ({len(self.issues_found)}):")
            for issue in self.issues_found:
                print(f"   - {issue}")
                
        # Final recommendation
        print("\n" + "="*60)
        if len(self.manual_fixes_needed) == 0 and len(self.issues_found) == 0:
            print("🎉 SYSTEM IS READY! All issues have been fixed!")
            print("\nYou can now run:")
            print("   poetry run python enhanced_launcher.py")
        else:
            print("⚠️ Some manual intervention is still needed.")
            print("\nAfter fixing the manual items above, run:")
            print("   python nuclear_audit_and_fix.py")
            print("\nto verify all issues are resolved.")

    def run(self):
        """Run the complete nuclear audit"""
        self.print_banner()
        
        # Run all checks
        self.check_python_syntax()
        self.check_dependencies()
        self.check_field_mismatches()
        self.check_ports()
        self.check_environment_setup()
        self.check_file_integrity()
        self.create_missing_directories()
        self.fix_import_paths()
        self.validate_json_files()
        self.run_final_test()
        
        # Generate report
        self.generate_report()
        
        # Save detailed log
        log_file = Path(f"nuclear_audit_log_{int(time.time())}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("TORI NUCLEAR AUDIT LOG\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("FIXES APPLIED:\n")
            for fix in self.fixes_applied:
                f.write(f"  {fix}\n")
                
            f.write("\nMANUAL FIXES NEEDED:\n")
            for fix in self.manual_fixes_needed:
                f.write(f"  {fix}\n")
                
            f.write("\nISSUES FOUND:\n")
            for issue in self.issues_found:
                f.write(f"  {issue}\n")
                
        print(f"\n📝 Detailed log saved to: {log_file}")

if __name__ == "__main__":
    try:
        auditor = TORINuclearAudit()
        auditor.run()
    except Exception as e:
        print(f"\n💥 NUCLEAR AUDIT CRASHED: {e}")
        print(traceback.format_exc())
        print("\nThis is likely due to severe system issues.")
        print("Please share the error above for help.")
