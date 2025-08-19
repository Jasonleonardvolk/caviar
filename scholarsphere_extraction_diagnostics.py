#!/usr/bin/env python3
"""
🔬 SURGICAL DIAGNOSTICS - ScholarSphere Extraction Pipeline Analysis
Precisely diagnoses where the extraction pipeline is hanging
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

class ScholarSphereExtractionDiagnostics:
    """Diagnose ScholarSphere extraction pipeline hanging issues"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.main_simple = self.script_dir / "main.py"
        self.main_multi = self.script_dir / "ingest_pdf" / "main.py"
        self.launcher = self.script_dir / "start_unified_tori.py"
        self.pipeline = self.script_dir / "ingest_pdf" / "pipeline.py"
        
    def diagnose_extraction_chain(self):
        """Complete ScholarSphere extraction diagnostic"""
        print("🔬 SURGICAL DIAGNOSTICS - SCHOLARSPHERE EXTRACTION PIPELINE")
        print("=" * 65)
        
        # Step 1: File Structure Analysis
        self.analyze_file_structure()
        
        # Step 2: Import Chain Analysis  
        self.test_import_chain()
        
        # Step 3: Direct Pipeline Test
        self.test_pipeline_directly()
        
        # Step 4: Main Server Tests
        self.test_main_servers()
        
        # Step 5: Launcher Analysis
        self.analyze_launcher_issue()
        
        print("\n🎯 EXTRACTION DIAGNOSTIC COMPLETE")
        
    def analyze_file_structure(self):
        """Analyze the critical file structure"""
        print("\n🔍 STEP 1: FILE STRUCTURE ANALYSIS")
        print("-" * 35)
        
        files_to_check = [
            ("Simple main.py", self.main_simple),
            ("Multi-tenant main", self.main_multi), 
            ("Launcher", self.launcher),
            ("Pipeline", self.pipeline),
            ("ingest_pdf/__init__.py", self.script_dir / "ingest_pdf" / "__init__.py")
        ]
        
        for name, path in files_to_check:
            exists = path.exists()
            print(f"{name}: {'✅ EXISTS' if exists else '❌ MISSING'}")
            if exists and path.suffix == '.py':
                size = path.stat().st_size
                print(f"  Size: {size:,} bytes")
        
        # Check what the launcher is trying to load
        print(f"\n📋 Launcher uvicorn target analysis:")
        if self.launcher.exists():
            with open(self.launcher, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'uvicorn.run(' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'uvicorn.run(' in line:
                            print(f"  Line {i+1}: {line.strip()}")
                            # Look for the app target
                            if '"' in line:
                                app_target = line.split('"')[1]
                                print(f"  🎯 Target: {app_target}")
                                
                                # Check if target exists
                                target_parts = app_target.split(':')
                                if len(target_parts) == 2:
                                    module_path = target_parts[0].replace('.', '/')
                                    module_file = self.script_dir / f"{module_path}.py"
                                    print(f"  📁 Looking for: {module_file}")
                                    print(f"  📁 Exists: {'✅' if module_file.exists() else '❌'}")
        
    def test_import_chain(self):
        """Test the import chain that the launcher uses"""
        print("\n🔍 STEP 2: IMPORT CHAIN ANALYSIS")
        print("-" * 35)
        
        # Test imports that launcher needs
        imports_to_test = [
            ("ingest_pdf", "Basic package import"),
            ("ingest_pdf.main", "Main module import"),
            ("ingest_pdf.pipeline", "Pipeline import"),
            ("ingest_pdf.multi_tenant_manager", "Multi-tenant manager"),
            ("ingest_pdf.knowledge_manager", "Knowledge manager"),
            ("ingest_pdf.user_manager", "User manager")
        ]
        
        for module_name, description in imports_to_test:
            try:
                __import__(module_name)
                print(f"✅ {description}: {module_name}")
            except ImportError as e:
                print(f"❌ {description}: {module_name} - {e}")
            except Exception as e:
                print(f"⚠️ {description}: {module_name} - {e}")
        
        # Test if FastAPI app can be accessed
        try:
            from ingest_pdf.main import app
            print(f"✅ FastAPI app import: ingest_pdf.main:app")
        except ImportError as e:
            print(f"❌ FastAPI app import failed: {e}")
        except Exception as e:
            print(f"⚠️ FastAPI app import error: {e}")
    
    def test_pipeline_directly(self):
        """Test pipeline.py directly"""
        print("\n🔍 STEP 3: DIRECT PIPELINE TEST")
        print("-" * 35)
        
        if not self.pipeline.exists():
            print("❌ Pipeline file not found")
            return
        
        try:
            # Test pipeline import
            sys.path.insert(0, str(self.script_dir / "ingest_pdf"))
            from pipeline import ingest_pdf_clean
            print("✅ Pipeline import successful")
            
            # Test with a dummy file (if we have one)
            print("📄 Looking for test PDF files...")
            test_files = list(self.script_dir.glob("*.pdf"))
            if test_files:
                test_file = test_files[0]
                print(f"📄 Found test file: {test_file.name}")
                print("🧪 Testing pipeline extraction (this may hang)...")
                
                # Run pipeline with timeout
                import signal
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("Pipeline extraction timed out")
                
                # Set timeout (only on Unix systems)
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 second timeout
                
                try:
                    start_time = time.time()
                    result = ingest_pdf_clean(str(test_file))
                    end_time = time.time()
                    
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # Cancel timeout
                    
                    print(f"✅ Pipeline test successful! ({end_time - start_time:.1f}s)")
                    print(f"📊 Result: {result.get('concept_count', 0)} concepts")
                    
                except TimeoutException:
                    print("❌ Pipeline test TIMED OUT after 30 seconds")
                    print("🎯 THIS IS LIKELY THE HANGING ISSUE!")
                except Exception as e:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                    print(f"❌ Pipeline test failed: {e}")
            else:
                print("📄 No test PDF files found")
                
        except ImportError as e:
            print(f"❌ Pipeline import failed: {e}")
        except Exception as e:
            print(f"❌ Pipeline test error: {e}")
    
    def test_main_servers(self):
        """Test both main servers"""
        print("\n🔍 STEP 4: MAIN SERVER TESTS")
        print("-" * 35)
        
        # Test simple main
        if self.main_simple.exists():
            print("🚀 Testing simple main.py server...")
            try:
                cmd = [sys.executable, str(self.main_simple)]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(self.script_dir)
                )
                
                print(f"📊 Process started with PID: {process.pid}")
                
                # Monitor for 10 seconds
                start_time = time.time()
                while time.time() - start_time < 10:
                    if process.poll() is not None:
                        print(f"📊 Process exited with code: {process.poll()}")
                        break
                    
                    line = process.stdout.readline()
                    if line:
                        print(f"OUTPUT: {line.strip()}")
                        if "Uvicorn running" in line:
                            print("✅ Simple main server started!")
                            break
                    time.sleep(0.1)
                
                # Cleanup
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                
            except Exception as e:
                print(f"❌ Simple main test failed: {e}")
        
        # Test multi-tenant main
        if self.main_multi.exists():
            print("\n🚀 Testing multi-tenant main server...")
            print("⚠️ This test is skipped as it may hang")
    
    def analyze_launcher_issue(self):
        """Analyze the specific launcher issue"""
        print("\n🔍 STEP 5: LAUNCHER ISSUE ANALYSIS")
        print("-" * 35)
        
        print("🎯 HYPOTHESIS ANALYSIS:")
        print("1. Launcher tries to load: 'ingest_pdf.main:app'")
        print("2. This loads the MULTI-TENANT main.py")
        print("3. Multi-tenant main.py may hang during:")
        print("   - Import of heavy dependencies")
        print("   - Multi-tenant manager initialization")
        print("   - Database/concept loading")
        print("   - Pipeline initialization")
        
        print("\n💡 SOLUTIONS TO TEST:")
        print("1. Use simple main.py instead of multi-tenant")
        print("2. Fix hanging in multi-tenant initialization")
        print("3. Add timeouts to critical operations")
        print("4. Use subprocess isolation")
        
        # Check if there are concept file_storages that might be huge
        data_dir = self.script_dir / "ingest_pdf" / "data"
        if data_dir.exists():
            print(f"\n📊 Data directory analysis:")
            for file in data_dir.glob("*.json"):
                size = file.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  {file.name}: {size_mb:.1f} MB")
                if size_mb > 10:
                    print(f"    ⚠️ Large file may cause loading delays!")

def main():
    """Run ScholarSphere extraction diagnostics"""
    diagnostics = ScholarSphereExtractionDiagnostics()
    diagnostics.diagnose_extraction_chain()

if __name__ == "__main__":
    main()
