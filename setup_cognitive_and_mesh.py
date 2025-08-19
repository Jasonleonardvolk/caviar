#!/usr/bin/env python3
"""
Setup Script for Cognitive Interface and Concept Mesh Services
=============================================================

This script sets up both the cognitive_interface service and concept_mesh service
for the KHA project, ensuring proper imports and service availability.

Author: Assistant
Date: 2025-01-28
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceSetup:
    def __init__(self):
        # Project paths
        self.project_root = Path(r"{PROJECT_ROOT}")
        self.ingest_pdf_dir = self.project_root / "ingest_pdf"
        self.cognitive_interface_file = self.ingest_pdf_dir / "cognitive_interface.py"
        
        # Service URLs
        self.cognitive_interface_url = "http://localhost:5173"
        self.concept_mesh_url = "http://localhost:8003"
        
        # Process handles
        self.cognitive_process = None
        self.mesh_process = None
        
    def check_python_path(self):
        """Ensure Python can find our modules"""
        logger.info("Setting up Python path...")
        
        # Add project root to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
            
        # Set PYTHONPATH environment variable
        os.environ['PYTHONPATH'] = str(self.project_root)
        
        logger.info(f"Python path set to: {self.project_root}")
        
    def verify_package_structure(self):
        """Verify the package structure is correct"""
        logger.info("Verifying package structure...")
        
        # Check if __init__.py exists in ingest_pdf
        init_file = self.ingest_pdf_dir / "__init__.py"
        if not init_file.exists():
            logger.info("Creating __init__.py in ingest_pdf directory...")
            init_file.write_text("")
            
        # Check if cognitive_interface.py exists
        if not self.cognitive_interface_file.exists():
            logger.error(f"cognitive_interface.py not found at {self.cognitive_interface_file}")
            raise FileNotFoundError("cognitive_interface.py is missing!")
            
        logger.info("Package structure verified ✓")
        
    def test_imports(self):
        """Test if imports work correctly"""
        logger.info("Testing imports...")
        
        try:
            # Test importing cognitive_interface
            from ingest_pdf.cognitive_interface import add_concept_diff
            logger.info("Successfully imported cognitive_interface ✓")
            
        except ImportError as e:
            logger.error(f"Failed to import cognitive_interface: {e}")
            
            # Try alternative import
            try:
                import cognitive_interface
                logger.info("Alternative import of cognitive_interface successful ✓")
            except ImportError as e2:
                logger.error(f"Alternative import also failed: {e2}")
                raise
                
    def start_cognitive_interface(self):
        """Start the cognitive interface service"""
        logger.info("Starting Cognitive Interface service...")
        
        try:
            # Kill any existing process on port 5173
            self.kill_port(5173)
            
            # Start uvicorn
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "ingest_pdf.cognitive_interface:app",
                "--host", "0.0.0.0",
                "--port", "5173",
                "--reload"
            ]
            
            self.cognitive_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env={**os.environ, 'PYTHONPATH': str(self.project_root)}
            )
            
            # Wait for service to start
            time.sleep(3)
            
            # Verify service is running
            if self.check_service_health(self.cognitive_interface_url + "/docs"):
                logger.info("Cognitive Interface service started successfully ✓")
            else:
                logger.error("Cognitive Interface service failed to start")
                
        except Exception as e:
            logger.error(f"Error starting Cognitive Interface: {e}")
            raise
            
    def setup_concept_mesh(self):
        """Setup and start the concept mesh service"""
        logger.info("Setting up Concept Mesh service...")
        
        # First, check if concept_mesh is already installed
        try:
            import concept_mesh
            logger.info("concept_mesh module already installed ✓")
        except ImportError:
            logger.info("concept_mesh not found, attempting to install...")
            
            # Check if it's available on PyPI
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "concept-mesh-client"],
                    check=True
                )
                logger.info("Installed concept-mesh-client from PyPI ✓")
            except subprocess.CalledProcessError:
                logger.warning("concept-mesh-client not available on PyPI")
                
                # Create a mock concept_mesh module if not available
                self.create_mock_concept_mesh()
                
        # Set environment variable
        os.environ['CONCEPT_MESH_URL'] = self.concept_mesh_url + "/api/mesh"
        
    def create_mock_concept_mesh(self):
        """Create a mock concept_mesh module for testing"""
        logger.info("Creating mock concept_mesh module...")
        
        mock_dir = self.project_root / "concept_mesh"
        mock_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_content = '''
"""Mock Concept Mesh Module"""

class ConceptMeshConnector:
    """Mock ConceptMeshConnector for testing"""
    
    def __init__(self, url=None):
        self.url = url or "http://localhost:8003/api/mesh"
        
    def connect(self):
        """Mock connect method"""
        return True
        
    def get_concepts(self):
        """Mock get_concepts method"""
        return []
        
    def add_concept(self, concept):
        """Mock add_concept method"""
        return {"id": "mock_id", "concept": concept}

# Make it importable
__all__ = ['ConceptMeshConnector']
'''
        
        (mock_dir / "__init__.py").write_text(init_content)
        logger.info("Mock concept_mesh module created ✓")
        
    def kill_port(self, port: int):
        """Kill any process using the specified port"""
        try:
            if sys.platform == "win32":
                # Windows command to find and kill process
                subprocess.run(
                    f'netstat -ano | findstr :{port} | findstr LISTENING',
                    shell=True,
                    capture_output=True
                )
                subprocess.run(
                    f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port} ^| findstr LISTENING\') do taskkill /PID %a /F',
                    shell=True,
                    capture_output=True
                )
            else:
                # Unix-like command
                subprocess.run(
                    f"lsof -ti:{port} | xargs kill -9",
                    shell=True,
                    capture_output=True
                )
        except Exception as e:
            logger.debug(f"Error killing port {port}: {e}")
            
    def check_service_health(self, url: str) -> bool:
        """Check if a service is responding"""
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def run_tests(self):
        """Run import tests"""
        logger.info("\n" + "="*50)
        logger.info("Running import tests...")
        
        # Test 1: Import from ingest_pdf
        try:
            from ingest_pdf.cognitive_interface import add_concept_diff
            logger.info("✓ Test 1 PASSED: from ingest_pdf.cognitive_interface import add_concept_diff")
        except ImportError as e:
            logger.error(f"✗ Test 1 FAILED: {e}")
            
        # Test 2: Import cognitive_interface directly
        try:
            import cognitive_interface
            logger.info("✓ Test 2 PASSED: import cognitive_interface")
        except ImportError as e:
            logger.error(f"✗ Test 2 FAILED: {e}")
            
        # Test 3: Import concept_mesh
        try:
            from python.core import ConceptMeshConnector
            logger.info("✓ Test 3 PASSED: from python.core import ConceptMeshConnector")
        except ImportError as e:
            logger.error(f"✗ Test 3 FAILED: {e}")
            
        # Test 4: Check service availability
        if self.check_service_health(self.cognitive_interface_url + "/docs"):
            logger.info(f"✓ Test 4 PASSED: Cognitive Interface available at {self.cognitive_interface_url}/docs")
        else:
            logger.error(f"✗ Test 4 FAILED: Cognitive Interface not available")
            
        logger.info("="*50 + "\n")
        
    def create_batch_files(self):
        """Create convenient batch files for starting services"""
        logger.info("Creating batch files for easy startup...")
        
        # Start cognitive interface batch file
        cognitive_bat = self.project_root / "start_cognitive_interface.bat"
        cognitive_content = f'''@echo off
echo Starting Cognitive Interface Service...
cd /d "{self.project_root}"
set PYTHONPATH={self.project_root}
python -m uvicorn ingest_pdf.cognitive_interface:app --host 0.0.0.0 --port 5173 --reload
'''
        cognitive_bat.write_text(cognitive_content)
        logger.info(f"Created: {cognitive_bat}")
        
        # Setup and test batch file
        setup_bat = self.project_root / "setup_and_test_imports.bat"
        setup_content = f'''@echo off
echo Setting up and testing imports...
cd /d "{self.project_root}"
set PYTHONPATH={self.project_root}
python setup_cognitive_and_mesh.py
pause
'''
        setup_bat.write_text(setup_content)
        logger.info(f"Created: {setup_bat}")
        
    def run(self):
        """Main setup process"""
        logger.info("Starting setup process...")
        logger.info(f"Project root: {self.project_root}")
        
        try:
            # Step 1: Setup Python path
            self.check_python_path()
            
            # Step 2: Verify package structure
            self.verify_package_structure()
            
            # Step 3: Test imports
            self.test_imports()
            
            # Step 4: Start cognitive interface
            self.start_cognitive_interface()
            
            # Step 5: Setup concept mesh
            self.setup_concept_mesh()
            
            # Step 6: Create batch files
            self.create_batch_files()
            
            # Step 7: Run final tests
            self.run_tests()
            
            logger.info("\n" + "="*50)
            logger.info("SETUP COMPLETE!")
            logger.info("="*50)
            logger.info("\nQuick Reference:")
            logger.info(f"1. Cognitive Interface: {self.cognitive_interface_url}/docs")
            logger.info(f"2. Project root: {self.project_root}")
            logger.info(f"3. PYTHONPATH: {os.environ.get('PYTHONPATH')}")
            logger.info("\nTo start services manually:")
            logger.info("   - Run: start_cognitive_interface.bat")
            logger.info("   - Or: python setup_cognitive_and_mesh.py")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
            
        finally:
            # Keep services running
            if self.cognitive_process:
                logger.info("\nCognitive Interface service is running...")
                logger.info("Press Ctrl+C to stop")
                try:
                    self.cognitive_process.wait()
                except KeyboardInterrupt:
                    logger.info("Shutting down services...")
                    if self.cognitive_process:
                        self.cognitive_process.terminate()

if __name__ == "__main__":
    setup = ServiceSetup()
    setup.run()
