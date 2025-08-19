#!/usr/bin/env python3
"""
TORI System Validation Script
Validates that all components are properly implemented and integrated
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TORIValidator:
    """Validates TORI system implementation"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.results = {
            "api_endpoints": {},
            "bps_components": {},
            "memory_system": {},
            "webgpu_shaders": {},
            "documentation": {},
            "overall": "PENDING"
        }
    
    def validate_api_endpoints(self) -> bool:
        """Validate Prajna API endpoints are implemented"""
        logger.info("Validating API endpoints...")
        
        required_files = [
            "api/routes/memory_routes.py",
            "api/routes/chat_routes.py",
            "api/routes/pdf_routes.py",
            "api/routes/__init__.py"
        ]
        
        required_endpoints = [
            ("memory_routes.py", "/api/memory/state/{user_id}"),
            ("chat_routes.py", "/api/chat/history/{user_id}"),
            ("chat_routes.py", "/api/chat/export-all/{user_id}"),
            ("pdf_routes.py", "/api/pdf/stats/{user_id}")
        ]
        
        all_valid = True
        
        # Check files exist
        for file_path in required_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"  ✅ {file_path} exists")
                self.results["api_endpoints"][file_path] = "EXISTS"
            else:
                logger.error(f"  ❌ {file_path} missing")
                self.results["api_endpoints"][file_path] = "MISSING"
                all_valid = False
        
        # Check endpoints defined
        for file_name, endpoint in required_endpoints:
            file_path = self.base_path / "api" / "routes" / file_name
            if file_path.exists():
                content = file_path.read_text()
                if endpoint.replace("{user_id}", "") in content:
                    logger.info(f"  ✅ {endpoint} defined")
                    self.results["api_endpoints"][endpoint] = "DEFINED"
                else:
                    logger.warning(f"  ⚠️  {endpoint} not found in {file_name}")
                    self.results["api_endpoints"][endpoint] = "NOT_FOUND"
        
        return all_valid
    
    def validate_bps_components(self) -> bool:
        """Validate BPS system components"""
        logger.info("Validating BPS components...")
        
        required_components = {
            "python/core/bps_config.py": ["ENABLE_PHASE_SPONGE", "PHASE_SPONGE_DAMPING_FACTOR"],
            "python/core/bps_oscillator.py": ["phase_sponge", "_update_sponge_damping"],
            "python/core/albert_physics_bps.py": ["BPS_INTEGRATED", "compute_topological_charge"],
            "python/core/ghost_collective.py": ["ENABLE_GHOST_COLLECTIVE", "GhostCollective"]
        }
        
        all_valid = True
        
        for file_path, required_items in required_components.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                content = full_path.read_text()
                logger.info(f"  ✅ {file_path} exists")
                
                for item in required_items:
                    if item in content:
                        logger.info(f"    ✅ {item} found")
                        self.results["bps_components"][f"{file_path}:{item}"] = "FOUND"
                    else:
                        logger.error(f"    ❌ {item} missing")
                        self.results["bps_components"][f"{file_path}:{item}"] = "MISSING"
                        all_valid = False
            else:
                logger.error(f"  ❌ {file_path} missing")
                self.results["bps_components"][file_path] = "MISSING"
                all_valid = False
        
        return all_valid
    
    def validate_memory_system(self) -> bool:
        """Validate memory vault enhancements"""
        logger.info("Validating memory system...")
        
        required_files = [
            "python/core/improved_memory_vault/thumbnail_utils.py",
            "python/core/memory_vault_enhancements.py"
        ]
        
        all_valid = True
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"  ✅ {file_path} exists")
                self.results["memory_system"][file_path] = "EXISTS"
                
                # Check for key classes
                content = full_path.read_text()
                if "thumbnail_utils" in file_path:
                    if "ThumbnailGenerator" in content:
                        logger.info(f"    ✅ ThumbnailGenerator class found")
                    else:
                        logger.error(f"    ❌ ThumbnailGenerator class missing")
                        all_valid = False
                elif "memory_vault_enhancements" in file_path:
                    if "CompressionMixin" in content and "MultimodalMixin" in content:
                        logger.info(f"    ✅ Enhancement mixins found")
                    else:
                        logger.error(f"    ❌ Enhancement mixins missing")
                        all_valid = False
            else:
                logger.error(f"  ❌ {file_path} missing")
                self.results["memory_system"][file_path] = "MISSING"
                all_valid = False
        
        return all_valid
    
    def validate_webgpu_shaders(self) -> bool:
        """Validate WebGPU implementation"""
        logger.info("Validating WebGPU shaders...")
        
        required_files = [
            "frontend/lib/webgpu/shaders/topologicalOverlay.wgsl",
            "frontend/lib/webgpu/fallbackSystem.ts"
        ]
        
        all_valid = True
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"  ✅ {file_path} exists")
                self.results["webgpu_shaders"][file_path] = "EXISTS"
                
                content = full_path.read_text()
                if ".wgsl" in file_path:
                    # Check for shader entry points
                    if "@vertex" in content and "@fragment" in content:
                        logger.info(f"    ✅ Vertex and fragment shaders defined")
                    else:
                        logger.error(f"    ❌ Shader entry points missing")
                        all_valid = False
                elif "fallbackSystem" in file_path:
                    # Check for backend implementations
                    if "WebGPUBackend" in content and "WebGL2Backend" in content:
                        logger.info(f"    ✅ Fallback backends implemented")
                    else:
                        logger.error(f"    ❌ Fallback backends missing")
                        all_valid = False
            else:
                logger.error(f"  ❌ {file_path} missing")
                self.results["webgpu_shaders"][file_path] = "MISSING"
                all_valid = False
        
        return all_valid
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness"""
        logger.info("Validating documentation...")
        
        required_docs = [
            "README.md",
            "IMPLEMENTATION_PROGRESS.md",
            "IMPLEMENTATION_COMPLETION.md"
        ]
        
        all_valid = True
        
        for doc_path in required_docs:
            full_path = self.base_path / doc_path
            if full_path.exists():
                content = full_path.read_text()
                lines = len(content.split('\n'))
                logger.info(f"  ✅ {doc_path} exists ({lines} lines)")
                self.results["documentation"][doc_path] = f"EXISTS ({lines} lines)"
                
                # Check for key sections
                if doc_path == "README.md":
                    required_sections = ["Quick Start", "API Endpoints", "BPS Configuration"]
                    for section in required_sections:
                        if section in content:
                            logger.info(f"    ✅ {section} section found")
                        else:
                            logger.warning(f"    ⚠️  {section} section missing")
            else:
                logger.error(f"  ❌ {doc_path} missing")
                self.results["documentation"][doc_path] = "MISSING"
                all_valid = False
        
        return all_valid
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("=" * 60)
        logger.info("TORI SYSTEM VALIDATION")
        logger.info("=" * 60)
        
        validations = [
            ("API Endpoints", self.validate_api_endpoints()),
            ("BPS Components", self.validate_bps_components()),
            ("Memory System", self.validate_memory_system()),
            ("WebGPU Shaders", self.validate_webgpu_shaders()),
            ("Documentation", self.validate_documentation())
        ]
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        all_passed = True
        for name, passed in validations:
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{name}: {status}")
            if not passed:
                all_passed = False
        
        self.results["overall"] = "PASSED" if all_passed else "FAILED"
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 ALL VALIDATIONS PASSED! 🎉")
            logger.info("The TORI system is ready for deployment!")
        else:
            logger.error("⚠️  SOME VALIDATIONS FAILED")
            logger.error("Please review the errors above and fix any issues.")
        logger.info("=" * 60)
        
        return self.results
    
    def save_results(self, output_file: str = "validation_results.json"):
        """Save validation results to JSON file"""
        output_path = self.base_path / output_file
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

def main():
    """Main validation entry point"""
    # Determine base path
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "."
    
    # Run validation
    validator = TORIValidator(base_path)
    results = validator.run_validation()
    
    # Save results
    validator.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall"] == "PASSED" else 1)

if __name__ == "__main__":
    main()
