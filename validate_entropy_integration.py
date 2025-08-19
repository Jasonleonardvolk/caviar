#!/usr/bin/env python3
"""
TORI Entropy Pruning - Comprehensive Validation Suite
====================================================

This script performs comprehensive validation of the entropy-based
semantic diversity pruning integration for production deployment.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ValidationSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results = []
        
    def log_pass(self, test_name: str, message: str = ""):
        self.passed += 1
        result = f"✅ {test_name}"
        if message:
            result += f": {message}"
        print(result)
        self.results.append({"test": test_name, "status": "PASS", "message": message})
        
    def log_fail(self, test_name: str, message: str = ""):
        self.failed += 1
        result = f"❌ {test_name}"
        if message:
            result += f": {message}"
        print(result)
        self.results.append({"test": test_name, "status": "FAIL", "message": message})
        
    def log_warn(self, test_name: str, message: str = ""):
        self.warnings += 1
        result = f"⚠️  {test_name}"
        if message:
            result += f": {message}"
        print(result)
        self.results.append({"test": test_name, "status": "WARN", "message": message})

    def test_imports(self):
        """Test all required imports and dependencies"""
        print("\n🔬 Testing Module Imports")
        print("=" * 40)
        
        # Test entropy pruning imports
        try:
            from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories, compute_entropy
            self.log_pass("Entropy pruning imports", "All functions available")
        except ImportError as e:
            self.log_fail("Entropy pruning imports", str(e))
            
        # Test pipeline integration
        try:
            from ingest_pdf.pipeline import ENABLE_ENTROPY_PRUNING, ENTROPY_CONFIG, ingest_pdf_clean
            if ENABLE_ENTROPY_PRUNING:
                self.log_pass("Pipeline integration", "Entropy pruning enabled")
            else:
                self.log_warn("Pipeline integration", "Entropy pruning disabled")
        except ImportError as e:
            self.log_fail("Pipeline integration", str(e))
            
        # Test dependencies
        required_deps = [
            ("numpy", "NumPy"),
            ("sklearn", "Scikit-learn"),
            ("sentence_transformers", "SentenceTransformers")
        ]
        
        for module, name in required_deps:
            try:
                __import__(module)
                self.log_pass(f"Dependency: {name}", "Available")
            except ImportError:
                self.log_fail(f"Dependency: {name}", "Missing")

    def test_configuration(self):
        """Test entropy configuration values"""
        print("\n⚙️  Testing Configuration")
        print("=" * 40)
        
        try:
            from ingest_pdf.pipeline import ENTROPY_CONFIG
            
            # Validate configuration ranges
            config_tests = [
                ("default_top_k", 10, 100, "Standard user concept limit"),
                ("admin_top_k", 50, 500, "Admin user concept limit"), 
                ("entropy_threshold", 0.001, 0.1, "Entropy gain threshold"),
                ("similarity_threshold", 0.7, 0.95, "Similarity threshold"),
                ("concepts_per_category", 3, 25, "Category concept limit")
            ]
            
            for key, min_val, max_val, desc in config_tests:
                if key in ENTROPY_CONFIG:
                    value = ENTROPY_CONFIG[key]
                    if min_val <= value <= max_val:
                        self.log_pass(f"Config: {key}", f"{value} (valid range)")
                    else:
                        self.log_warn(f"Config: {key}", f"{value} outside recommended range {min_val}-{max_val}")
                else:
                    self.log_fail(f"Config: {key}", "Missing from configuration")
                    
            # Check boolean configs
            bool_configs = ["enable_categories"]
            for key in bool_configs:
                if key in ENTROPY_CONFIG:
                    value = ENTROPY_CONFIG[key]
                    self.log_pass(f"Config: {key}", f"{value}")
                else:
                    self.log_warn(f"Config: {key}", "Missing from configuration")
                    
        except ImportError:
            self.log_fail("Configuration test", "Cannot import ENTROPY_CONFIG")

    def test_basic_functionality(self):
        """Test basic entropy pruning functionality"""
        print("\n🧪 Testing Basic Functionality") 
        print("=" * 40)
        
        try:
            from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
            
            # Test basic entropy pruning
            test_concepts = [
                {"name": "machine learning", "score": 0.95, "embedding": None},
                {"name": "deep learning", "score": 0.93, "embedding": None},
                {"name": "neural networks", "score": 0.91, "embedding": None},
                {"name": "quantum computing", "score": 0.88, "embedding": None},
                {"name": "blockchain", "score": 0.85, "embedding": None},
            ]
            
            selected, stats = entropy_prune(test_concepts, top_k=3, verbose=False)
            
            # Validate results
            if len(selected) <= 3:
                self.log_pass("Basic pruning: top_k limit", f"Returned {len(selected)}/3 concepts")
            else:
                self.log_fail("Basic pruning: top_k limit", f"Returned {len(selected)} > 3 concepts")
                
            if stats["total"] == 5:
                self.log_pass("Basic pruning: stats tracking", "Correct total count")
            else:
                self.log_fail("Basic pruning: stats tracking", f"Expected 5, got {stats['total']}")
                
            if "final_entropy" in stats and "avg_similarity" in stats:
                self.log_pass("Basic pruning: metrics", "Entropy and similarity metrics available")
            else:
                self.log_fail("Basic pruning: metrics", "Missing entropy or similarity metrics")
                
            # Test category-aware pruning
            categorized_concepts = [
                {"name": "machine learning", "score": 0.95, "metadata": {"category": "AI"}},
                {"name": "quantum mechanics", "score": 0.92, "metadata": {"category": "Physics"}},
                {"name": "cryptography", "score": 0.87, "metadata": {"category": "Security"}},
            ]
            
            cat_selected, cat_stats = entropy_prune_with_categories(
                categorized_concepts, 
                categories=["AI", "Physics", "Security"],
                concepts_per_category=1
            )
            
            if "by_category" in cat_stats:
                self.log_pass("Category pruning: stats", "Category breakdown available")
            else:
                self.log_fail("Category pruning: stats", "Missing category breakdown")
                
        except Exception as e:
            self.log_fail("Basic functionality test", f"Exception: {str(e)}")

    def test_performance_characteristics(self):
        """Test performance characteristics with larger datasets"""
        print("\n⚡ Testing Performance Characteristics")
        print("=" * 40)
        
        try:
            from ingest_pdf.entropy_prune import entropy_prune
            import time
            
            # Create larger test dataset
            large_concepts = []
            for i in range(100):
                large_concepts.append({
                    "name": f"concept_{i}",
                    "score": 0.9 - (i * 0.001),  # Decreasing scores
                    "embedding": None
                })
            
            # Time the pruning
            start_time = time.time()
            selected, stats = entropy_prune(large_concepts, top_k=20, verbose=False)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if processing_time < 10.0:  # Should complete in under 10 seconds
                self.log_pass("Performance: speed", f"Completed in {processing_time:.2f}s")
            else:
                self.log_warn("Performance: speed", f"Took {processing_time:.2f}s (may be slow)")
                
            if len(selected) == 20:
                self.log_pass("Performance: accuracy", "Returned exact top_k count")
            else:
                self.log_warn("Performance: accuracy", f"Returned {len(selected)} instead of 20")
                
            # Test memory efficiency (concepts should have embeddings added)
            has_embeddings = all("embedding" in c and c["embedding"] is not None for c in selected)
            if has_embeddings:
                self.log_pass("Performance: memory", "Embeddings properly cached")
            else:
                self.log_warn("Performance: memory", "Embeddings may not be cached")
                
        except Exception as e:
            self.log_fail("Performance test", f"Exception: {str(e)}")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🎯 Testing Edge Cases")
        print("=" * 40)
        
        try:
            from ingest_pdf.entropy_prune import entropy_prune
            
            # Test empty input
            selected, stats = entropy_prune([], top_k=10)
            if len(selected) == 0 and stats["total"] == 0:
                self.log_pass("Edge case: empty input", "Handled correctly")
            else:
                self.log_fail("Edge case: empty input", "Not handled correctly")
                
            # Test single concept
            single_concept = [{"name": "single", "score": 0.9, "embedding": None}]
            selected, stats = entropy_prune(single_concept, top_k=10)
            if len(selected) == 1:
                self.log_pass("Edge case: single concept", "Handled correctly")
            else:
                self.log_fail("Edge case: single concept", f"Returned {len(selected)} concepts")
                
            # Test top_k larger than input
            small_concepts = [
                {"name": "concept1", "score": 0.9, "embedding": None},
                {"name": "concept2", "score": 0.8, "embedding": None}
            ]
            selected, stats = entropy_prune(small_concepts, top_k=10)
            if len(selected) == 2:
                self.log_pass("Edge case: top_k > input", "Handled correctly")
            else:
                self.log_fail("Edge case: top_k > input", f"Returned {len(selected)} concepts")
                
        except Exception as e:
            self.log_fail("Edge cases test", f"Exception: {str(e)}")

    def test_pipeline_integration(self):
        """Test integration with main pipeline"""
        print("\n🔗 Testing Pipeline Integration")
        print("=" * 40)
        
        try:
            from ingest_pdf.pipeline import ingest_pdf_clean
            import inspect
            
            # Check function signature
            sig = inspect.signature(ingest_pdf_clean)
            if "admin_mode" in sig.parameters:
                self.log_pass("Pipeline: admin_mode param", "Available in function signature")
            else:
                self.log_fail("Pipeline: admin_mode param", "Missing from function signature")
                
            # Check if function accepts admin_mode
            default_admin = sig.parameters.get("admin_mode", None)
            if default_admin and default_admin.default is False:
                self.log_pass("Pipeline: admin_mode default", "Defaults to False (standard mode)")
            else:
                self.log_warn("Pipeline: admin_mode default", "May not default to False")
                
        except Exception as e:
            self.log_fail("Pipeline integration test", f"Exception: {str(e)}")

    def test_file_integrity(self):
        """Test file integrity and structure"""
        print("\n📁 Testing File Integrity")
        print("=" * 40)
        
        required_files = [
            ("ingest_pdf/entropy_prune.py", "Core entropy pruning module"),
            ("test_entropy_pruning.py", "Test suite"),
            ("ENTROPY_PRUNING_INTEGRATION.md", "Documentation")
        ]
        
        for filepath, description in required_files:
            if os.path.exists(filepath):
                self.log_pass(f"File exists: {filepath}", description)
                
                # Check file is readable
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > 100:  # Non-empty file
                            self.log_pass(f"File readable: {filepath}", f"{len(content)} characters")
                        else:
                            self.log_warn(f"File readable: {filepath}", "File appears to be empty")
                except Exception as e:
                    self.log_fail(f"File readable: {filepath}", f"Cannot read: {str(e)}")
            else:
                self.log_fail(f"File exists: {filepath}", "Missing required file")

    def run_test_suite(self):
        """Run the complete test suite and display results"""
        print("🧪 Running entropy pruning test suite...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "test_entropy_pruning.py"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log_pass("Test suite execution", "All tests passed")
                
                # Check for specific success indicators
                if "All tests complete!" in result.stdout:
                    self.log_pass("Test suite completion", "Completed successfully")
                else:
                    self.log_warn("Test suite completion", "May have completed with warnings")
            else:
                self.log_fail("Test suite execution", f"Exit code: {result.returncode}")
                if result.stderr:
                    print(f"   Error output: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            self.log_fail("Test suite execution", "Timed out after 60 seconds")
        except FileNotFoundError:
            self.log_fail("Test suite execution", "test_entropy_pruning.py not found")
        except Exception as e:
            self.log_fail("Test suite execution", f"Exception: {str(e)}")

    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "="*60)
        print("🚀 ENTROPY PRUNING VALIDATION REPORT")
        print("="*60)
        
        print(f"✅ Passed: {self.passed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"❌ Failed: {self.failed}")
        
        total_tests = self.passed + self.warnings + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        print("\n🎯 DEPLOYMENT RECOMMENDATION:")
        if self.failed == 0:
            if self.warnings == 0:
                print("✅ READY FOR PRODUCTION DEPLOYMENT")
                print("   All tests passed without warnings.")
            else:
                print("⚠️  READY WITH MINOR WARNINGS")
                print("   Review warnings but safe to deploy.")
            return True
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print("   Fix failed tests before deploying.")
            return False

def main():
    """Main validation function"""
    print("🚀 TORI ENTROPY PRUNING - COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print("Validating entropy-based semantic diversity pruning integration...")
    print()
    
    validator = ValidationSuite()
    
    # Run all validation tests
    validator.test_imports()
    validator.test_configuration()
    validator.test_basic_functionality()
    validator.test_performance_characteristics()
    validator.test_edge_cases()
    validator.test_pipeline_integration()
    validator.test_file_integrity()
    validator.run_test_suite()
    
    # Generate final report
    deployment_ready = validator.generate_report()
    
    if deployment_ready:
        print("\n📋 NEXT STEPS:")
        print("1. Start unified TORI: python start_unified_tori.py")
        print("2. Test Prajna admin: http://localhost:8001/docs")
        print("3. Test ScholarSphere: http://localhost:5173")
        print("4. Monitor entropy pruning performance")
        
        print("\n🎯 EXPECTED BEHAVIOR:")
        print("• Standard mode: Up to 50 diverse concepts")
        print("• Admin mode: Up to 200 diverse concepts")
        print("• Similarity pruning: <85% concept similarity")
        print("• Category balance: Even domain distribution")
    
    return 0 if deployment_ready else 1

if __name__ == "__main__":
    sys.exit(main())
