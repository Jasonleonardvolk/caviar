#!/usr/bin/env python3
"""
TORI/KHA Refactor Verification Script
=====================================

Verifies that all refactor checklist items have been completed.
"""

import os
import sys
import json
import importlib
from pathlib import Path
from datetime import datetime

# Add paths for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "python" / "core"))
sys.path.insert(0, str(script_dir / "python"))
sys.path.insert(0, str(script_dir))

class RefactorVerifier:
    """Verifies refactor checklist completion"""
    
    def __init__(self):
        self.results = {}
        self.total_checks = 0
        self.passed_checks = 0
        
    def check(self, name: str, condition: bool, details: str = ""):
        """Record a verification check"""
        self.total_checks += 1
        if condition:
            self.passed_checks += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        self.results[name] = {
            "status": status,
            "passed": condition,
            "details": details
        }
        
        print(f"{status} | {name}")
        if details:
            print(f"     └─ {details}")
    
    def verify_memory_vault(self):
        """Verify UnifiedMemoryVault refactor items"""
        print("\n📦 Checking UnifiedMemoryVault...")
        
        # Check if NDJSON logging exists
        try:
            from python.core.memory_vault import UnifiedMemoryVault
            vault = UnifiedMemoryVault({'storage_path': 'data/test_verify'})
            
            # Check for NDJSON writer
            has_ndjson = hasattr(vault, '_append_to_live_log') and hasattr(vault, 'live_log_path')
            self.check("Memory Vault: Live NDJSON writer", has_ndjson,
                      f"Found: _append_to_live_log={hasattr(vault, '_append_to_live_log')}, live_log_path={hasattr(vault, 'live_log_path')}")
            
            # Check for get_memory_stats
            has_stats = hasattr(vault, 'get_memory_stats')
            self.check("Memory Vault: get_memory_stats() method", has_stats)
            
            # Check SHA deduplication
            has_sha = hasattr(vault, '_calculate_entry_hash') and hasattr(vault, 'seen_hashes')
            self.check("Memory Vault: SHA deduplication", has_sha,
                      f"Found: _calculate_entry_hash={hasattr(vault, '_calculate_entry_hash')}, seen_hashes={hasattr(vault, 'seen_hashes')}")
            
            # Check unified ID generation
            has_unified_id = "unified_id_generator" in str(vault._generate_id.__code__.co_names)
            self.check("Memory Vault: Unified ID integration", has_unified_id,
                      "Checks if _generate_id imports unified_id_generator")
            
            # Cleanup
            vault.shutdown()
            
        except Exception as e:
            self.check("Memory Vault: Import and initialization", False, str(e))
    
    def verify_concept_mesh(self):
        """Verify ConceptMesh refactor items"""
        print("\n🕸️ Checking ConceptMesh...")
        
        try:
            from python.core.concept_mesh import ConceptMesh
            
            # Check if Penrose integration exists
            mesh = ConceptMesh({'storage_path': 'data/test_mesh', 'similarity_engine': 'penrose'})
            has_penrose_init = hasattr(mesh, '_init_penrose')
            self.check("ConceptMesh: Penrose initialization method", has_penrose_init)
            
            # Check if similarity_engine config is used
            uses_similarity_engine = hasattr(mesh, 'similarity_engine')
            self.check("ConceptMesh: similarity_engine attribute", uses_similarity_engine,
                      f"Value: {getattr(mesh, 'similarity_engine', 'N/A')}")
            
            # Check for PenroseConceptMesh class
            from python.core.concept_mesh import PenroseConceptMesh
            has_penrose_class = True
            self.check("ConceptMesh: PenroseConceptMesh class exists", has_penrose_class)
            
            mesh.shutdown()
            
        except Exception as e:
            self.check("ConceptMesh: Import and Penrose integration", False, str(e))
    
    def verify_enhanced_launcher(self):
        """Verify enhanced_launcher.py refactor items"""
        print("\n🚀 Checking Enhanced Launcher...")
        
        try:
            # Read launcher file to check for changes
            launcher_path = script_dir / "enhanced_launcher.py"
            with open(launcher_path, 'r') as f:
                launcher_content = f.read()
            
            # Check for FractalSolitonMemory
            has_fractal_soliton = "fractal_soliton" in launcher_content and "FractalSolitonMemory" in launcher_content
            self.check("Enhanced Launcher: FractalSolitonMemory integration", has_fractal_soliton)
            
            # Check for Penrose config in ConceptMesh
            has_penrose_config = "'similarity_engine': 'penrose'" in launcher_content
            self.check("Enhanced Launcher: Penrose configuration for ConceptMesh", has_penrose_config)
            
            # Check that memory_sculptor is NOT imported
            no_memory_sculptor = "memory_sculptor" not in launcher_content.lower()
            self.check("Enhanced Launcher: memory_sculptor removed", no_memory_sculptor)
            
        except Exception as e:
            self.check("Enhanced Launcher: File checks", False, str(e))
    
    def verify_prajna_mouth(self):
        """Verify prajna_mouth.py refactor items"""
        print("\n🗣️ Checking Prajna Mouth...")
        
        try:
            # Read prajna mouth file
            prajna_path = script_dir / "prajna" / "core" / "prajna_mouth.py"
            with open(prajna_path, 'r') as f:
                prajna_content = f.read()
            
            # Check for enable_mesh_to_text parameter
            has_mesh_param = "enable_mesh_to_text" in prajna_content
            self.check("Prajna Mouth: enable_mesh_to_text parameter", has_mesh_param)
            
            # Check for template fallback
            has_template_fallback = "_template_fallback_generate" in prajna_content
            self.check("Prajna Mouth: Template fallback method", has_template_fallback)
            
        except Exception as e:
            self.check("Prajna Mouth: File checks", False, str(e))
    
    def verify_ingest_service(self):
        """Verify ingest service refactor items"""
        print("\n📥 Checking Ingest Service...")
        
        try:
            # Read ingest service file
            ingest_path = script_dir / "ingest-bus" / "src" / "services" / "ingest_service.py"
            with open(ingest_path, 'r') as f:
                ingest_content = f.read()
            
            # Check for SSE progress reporter
            has_sse = "SSEProgressReporter" in ingest_content and "send_progress" in ingest_content
            self.check("Ingest Service: SSE progress reporting", has_sse)
            
            # Check for unified ID usage
            has_unified_id = "unified_id_generator" in ingest_content
            self.check("Ingest Service: Unified ID integration", has_unified_id)
            
        except Exception as e:
            self.check("Ingest Service: File checks", False, str(e))
    
    def verify_sse_endpoint(self):
        """Verify SSE endpoint in Prajna API"""
        print("\n🌐 Checking SSE Endpoint...")
        
        try:
            # Read prajna API file
            api_path = script_dir / "prajna" / "api" / "prajna_api.py"
            with open(api_path, 'r') as f:
                api_content = f.read()
            
            # Check for SSE endpoint
            has_sse_endpoint = "/api/upload/progress/" in api_content and "EventSourceResponse" in api_content
            self.check("Prajna API: SSE upload progress endpoint", has_sse_endpoint)
            
            # Check for unified ID in soliton
            has_soliton_unified = "unified_id_generator" in api_content and "generate_soliton_id" in api_content
            self.check("Prajna API: Soliton unified ID integration", has_soliton_unified)
            
        except Exception as e:
            self.check("Prajna API: SSE endpoint checks", False, str(e))
    
    def verify_new_components(self):
        """Verify new components created during refactor"""
        print("\n🆕 Checking New Components...")
        
        # Check penrose_adapter.py
        penrose_path = script_dir / "python" / "core" / "penrose_adapter.py"
        self.check("New Component: penrose_adapter.py exists", penrose_path.exists())
        
        # Check fractal_soliton_memory.py
        fractal_path = script_dir / "python" / "core" / "fractal_soliton_memory.py"
        self.check("New Component: fractal_soliton_memory.py exists", fractal_path.exists())
        
        # Check unified_id_generator.py
        unified_id_path = script_dir / "python" / "core" / "unified_id_generator.py"
        self.check("New Component: unified_id_generator.py exists", unified_id_path.exists())
    
    def generate_report(self):
        """Generate final verification report"""
        print("\n" + "="*60)
        print("📊 REFACTOR VERIFICATION REPORT")
        print("="*60)
        print(f"Total Checks: {self.total_checks}")
        print(f"Passed: {self.passed_checks}")
        print(f"Failed: {self.total_checks - self.passed_checks}")
        print(f"Success Rate: {(self.passed_checks/self.total_checks)*100:.1f}%")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": self.total_checks,
            "passed": self.passed_checks,
            "failed": self.total_checks - self.passed_checks,
            "success_rate": (self.passed_checks/self.total_checks)*100,
            "results": self.results
        }
        
        report_path = script_dir / "refactor_verification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📝 Report saved to: {report_path}")
        
        if self.passed_checks == self.total_checks:
            print("\n🎉 ALL REFACTOR TASKS COMPLETED SUCCESSFULLY! 🎉")
        else:
            print("\n⚠️ Some refactor tasks need attention.")
            print("\nFailed checks:")
            for name, result in self.results.items():
                if not result["passed"]:
                    print(f"  - {name}")
                    if result["details"]:
                        print(f"    {result['details']}")

def main():
    """Run verification"""
    print("🔍 TORI/KHA Refactor Verification")
    print("="*60)
    
    verifier = RefactorVerifier()
    
    # Run all verification checks
    verifier.verify_memory_vault()
    verifier.verify_concept_mesh()
    verifier.verify_enhanced_launcher()
    verifier.verify_prajna_mouth()
    verifier.verify_ingest_service()
    verifier.verify_sse_endpoint()
    verifier.verify_new_components()
    
    # Generate report
    verifier.generate_report()

if __name__ == "__main__":
    main()
