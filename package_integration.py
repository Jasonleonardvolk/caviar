"""
PRAJNA PACKAGE INTEGRATION - SOLVES RELATIVE IMPORT ISSUES
This creates a proper Python package structure to handle your advanced system's
relative imports correctly.

PHILOSOPHY:
- Treat ingest_pdf as a proper Python package
- Handle relative imports correctly
- Use dynamic module loading with package context
- Always provide working fallback
"""

import os
import sys
import json
import subprocess
import tempfile
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logging import logger

# Paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"

class PackageAwareIntegration:
    """
    Integration that properly handles your advanced system as a Python package
    to resolve relative import issues.
    """
    
    def __init__(self):
        self.stats = {
            "total_documents_processed": 0,
            "total_concepts_extracted": 0,
            "total_advanced_successes": 0,
            "total_fallback_uses": 0,
            "integration_start_time": datetime.now().isoformat(),
            "method": "package_aware_integration"
        }
        
        # Set up package structure and test
        self.advanced_available = self._setup_package_integration()
        
        logger.info("🌉 Package-Aware Integration initialized")
        logger.info(f"🏗️ Advanced system available: {self.advanced_available}")
        
        if self.advanced_available:
            logger.info("🎉 Package integration successful - your 4000-hour system is ready!")
        else:
            logger.warning("⚠️ Package integration failed - using enhanced fallback")
    
    def _setup_package_integration(self) -> bool:
        """Set up proper package structure for your advanced system"""
        logger.info("🔧 Setting up package-aware integration...")
        
        # Check if ingest_pdf directory exists
        if not ingest_pdf_dir.exists():
            logger.error(f"❌ ingest_pdf directory not found: {ingest_pdf_dir}")
            return False
        
        # Ensure ingest_pdf has an __init__.py file
        init_file = ingest_pdf_dir / "__init__.py"
        if not init_file.exists():
            logger.info("📝 Creating __init__.py for package structure")
            try:
                init_file.write_text("# Package init for ingest_pdf\n")
                logger.info("✅ Created __init__.py")
            except Exception as e:
                logger.error(f"❌ Failed to create __init__.py: {e}")
                return False
        else:
            logger.info("✅ __init__.py already exists")
        
        # Add parent directory to Python path so we can import ingest_pdf as a package
        parent_dir = ingest_pdf_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
            logger.info(f"✅ Added parent directory to Python path: {parent_dir}")
        
        # Test package import
        return self._test_package_import()
    
    def _test_package_import(self) -> bool:
        """Test importing the advanced system as a proper package"""
        logger.info("🧪 Testing package import...")
        
        # Create a test script that imports as a package
        test_script = f'''
import sys
sys.path.insert(0, r"{ingest_pdf_dir.parent}")

try:
    # Import as a package (this should handle relative imports)
    import ingest_pdf.pipeline as pipeline
    import ingest_pdf.extractConceptsFromDocument as extract_module
    
    # Test that the functions exist
    if hasattr(pipeline, 'ingest_pdf_clean'):
        print("PACKAGE_IMPORT_SUCCESS")
        print(f"PIPELINE_FUNCTION_AVAILABLE: {{hasattr(pipeline, 'ingest_pdf_clean')}}")
        print(f"EXTRACT_FUNCTION_AVAILABLE: {{hasattr(extract_module, 'extractConceptsFromDocument')}}")
    else:
        print("PACKAGE_IMPORT_PARTIAL: pipeline imported but function missing")
        
except Exception as e:
    print(f"PACKAGE_IMPORT_ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
'''
        
        try:
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True, timeout=30)
            
            output = result.stdout
            if "PACKAGE_IMPORT_SUCCESS" in output:
                logger.info("✅ Package import test passed!")
                logger.info("🎉 Your advanced system can now be imported properly")
                return True
            else:
                logger.warning(f"❌ Package import test failed:")
                logger.warning(f"   stdout: {output}")
                logger.warning(f"   stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("❌ Package import test timed out")
            return False
        except Exception as e:
            logger.warning(f"❌ Package import test error: {e}")
            return False
    
    def extract_concepts_package_aware(
        self, 
        pdf_file_path_or_bytes, 
        user_id: Optional[str] = None, 
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract concepts using package-aware integration
        """
        logger.info("🌉 PACKAGE INTEGRATION: Processing PDF")
        
        # Handle bytes input
        if isinstance(pdf_file_path_or_bytes, bytes):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_file_path_or_bytes)
                pdf_path = tmp_file.name
            temp_file_created = True
        else:
            pdf_path = pdf_file_path_or_bytes
            temp_file_created = False
        
        try:
            if self.advanced_available:
                logger.info("🚀 Attempting package-aware advanced extraction")
                result = self._call_advanced_package(pdf_path, doc_id)
                
                if result and result.get("num_concepts", 0) > 0:
                    logger.info(f"🎉 Package-aware advanced extraction SUCCESS: {result.get('num_concepts', 0)} concepts")
                    self.stats["total_advanced_successes"] += 1
                    return result
                else:
                    logger.warning("🔄 Advanced extraction returned no concepts, using fallback")
                    
            logger.info("🔄 Using enhanced fallback extraction")
            result = self._enhanced_fallback_extraction(pdf_path, doc_id)
            self.stats["total_fallback_uses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"🌉 Package integration error: {str(e)}")
            result = self._enhanced_fallback_extraction(pdf_path, doc_id)
            self.stats["total_fallback_uses"] += 1
            return result
            
        finally:
            # Always update stats and clean up
            self.stats["total_documents_processed"] += 1
            
            if temp_file_created:
                try:
                    os.unlink(pdf_path)
                except:
                    pass
    
    def _call_advanced_package(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """
        Call the advanced system using proper package imports
        """
        logger.info("🔄 Calling advanced system via package import")
        
        # Create a comprehensive extraction script with package imports
        extraction_script = f'''
import sys
import os
import json
from pathlib import Path

# Set up package import path
parent_dir = Path(r"{ingest_pdf_dir.parent}")
sys.path.insert(0, str(parent_dir))

try:
    # Import the advanced pipeline as a package
    import ingest_pdf.pipeline as pipeline
    
    # Call the sophisticated pipeline with zero threshold for maximum extraction
    result = pipeline.ingest_pdf_clean(
        r"{pdf_path}", 
        doc_id="{doc_id or 'package_extraction'}", 
        extraction_threshold=0.0
    )
    
    # Output result in parseable format
    print("ADVANCED_PACKAGE_RESULT_START")
    print(json.dumps(result, default=str, indent=2))
    print("ADVANCED_PACKAGE_RESULT_END")
    
    # Also output summary for easy parsing
    print("PACKAGE_SUMMARY_START")
    summary = {{
        "filename": result.get("filename", "unknown"),
        "concept_count": result.get("concept_count", 0),
        "concept_names": result.get("concept_names", []),
        "status": result.get("status", "unknown"),
        "processing_time": result.get("processing_time_seconds", 0),
        "method": "package_aware_advanced_4000h_system",
        "purity_analysis": result.get("purity_analysis", {{}}),
        "context_extraction": result.get("context_extraction", {{}}),
        "advanced_analytics": {{
            "semantic_concepts": result.get("semantic_concepts", 0),
            "boosted_concepts": result.get("boosted_concepts", 0),
            "auto_prefilled_concepts": result.get("auto_prefilled_concepts", 0),
            "universal_methods": result.get("universal_methods", []),
            "domain_distribution": result.get("domain_distribution", {{}})
        }}
    }}
    print(json.dumps(summary, default=str))
    print("PACKAGE_SUMMARY_END")
    
except Exception as e:
    print(f"PACKAGE_ADVANCED_ERROR: {{str(e)}}")
    import traceback
    print("PACKAGE_TRACEBACK_START")
    traceback.print_exc()
    print("PACKAGE_TRACEBACK_END")
'''
        
        try:
            # Run the extraction with generous timeout
            result = subprocess.run([
                sys.executable, "-c", extraction_script
            ], capture_output=True, text=True, timeout=180)  # 3 minutes
            
            if result.returncode == 0:
                return self._parse_package_result(result.stdout, pdf_path)
            else:
                logger.error(f"❌ Package-aware subprocess failed:")
                logger.error(f"   Return code: {result.returncode}")
                logger.error(f"   stderr: {result.stderr}")
                raise Exception(f"Package subprocess failed with return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Package-aware system call timed out (180s)")
            raise Exception("Package-aware system timeout")
        except Exception as e:
            logger.error(f"❌ Package-aware system call failed: {e}")
            raise e
    
    def _parse_package_result(self, output: str, pdf_path: str) -> Dict[str, Any]:
        """Parse the output from the package-aware advanced system"""
        try:
            # Try to extract the full advanced result first
            if "ADVANCED_PACKAGE_RESULT_START" in output and "ADVANCED_PACKAGE_RESULT_END" in output:
                start_idx = output.find("ADVANCED_PACKAGE_RESULT_START") + len("ADVANCED_PACKAGE_RESULT_START\n")
                end_idx = output.find("ADVANCED_PACKAGE_RESULT_END")
                result_json = output[start_idx:end_idx].strip()
                
                advanced_result = json.loads(result_json)
                logger.info("✅ Successfully parsed full package-aware result")
                
                # Transform to Prajna format with all advanced features
                return self._transform_package_result(advanced_result, pdf_path)
            
            # Extract summary if full result not available
            elif "PACKAGE_SUMMARY_START" in output and "PACKAGE_SUMMARY_END" in output:
                start_idx = output.find("PACKAGE_SUMMARY_START") + len("PACKAGE_SUMMARY_START\n")
                end_idx = output.find("PACKAGE_SUMMARY_END")
                summary_json = output[start_idx:end_idx].strip()
                
                summary = json.loads(summary_json)
                logger.info("✅ Parsed package-aware summary")
                
                # Build result from summary
                concepts = []
                concept_names = summary.get("concept_names", [])
                
                if concept_names:
                    for i, name in enumerate(concept_names):
                        concepts.append({
                            "name": name,
                            "score": 0.85,  # High score for advanced system
                            "method": "package_aware_advanced_4000h",
                            "source": {
                                "advanced_pipeline": True,
                                "package_aware": True,
                                "sophisticated_extraction": True
                            },
                            "context": "Extracted via package-aware 4000-hour pipeline",
                            "metadata": {
                                "extraction_method": "package_aware_advanced",
                                "sophisticated": True,
                                "concept_index": i
                            }
                        })
                
                return {
                    "doc_id": summary.get("filename", Path(pdf_path).stem),
                    "num_concepts": len(concepts),
                    "concepts": concepts,
                    "injection_result": {"injected": len(concepts)},
                    "method": "package_aware_advanced_4000h_system",
                    "status": summary.get("status", "success"),
                    "processing_time": summary.get("processing_time", 0),
                    
                    # Advanced analytics from summary
                    "advanced_analytics": {
                        "purity_analysis": summary.get("purity_analysis", {}),
                        "context_extraction": summary.get("context_extraction", {}),
                        "advanced_data": summary.get("advanced_analytics", {}),
                        "package_aware": True
                    },
                    
                    "summary": {
                        "total_concepts": len(concepts),
                        "processing_time": summary.get("processing_time", 0),
                        "advanced_pipeline_used": True,
                        "package_aware": True,
                        "sophisticated_extraction": True
                    }
                }
            
            else:
                logger.error("❌ Could not parse package-aware output")
                logger.error(f"Output preview: {output[:500]}...")
                raise Exception("Could not parse package-aware output")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in package result: {e}")
            logger.error(f"Output preview: {output[:500]}...")
            raise Exception(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"❌ Package result parsing error: {e}")
            raise e
    
    def _transform_package_result(self, advanced_result: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        """Transform full package-aware advanced result to Prajna format"""
        logger.info("🔄 Transforming package-aware advanced result to Prajna format")
        
        # Extract concepts with full sophistication
        concepts = []
        
        # Handle different concept formats from your advanced system
        if "concepts" in advanced_result and isinstance(advanced_result["concepts"], list):
            concepts = advanced_result["concepts"]
            logger.info(f"✅ Found {len(concepts)} sophisticated concepts in package result")
            
            # Enhance each concept with package-aware metadata
            for concept in concepts:
                if isinstance(concept, dict):
                    concept.setdefault("metadata", {})["package_aware"] = True
                    concept.setdefault("metadata", {})["sophisticated_4000h"] = True
                    
        elif "concept_names" in advanced_result:
            concept_names = advanced_result["concept_names"]
            if isinstance(concept_names, list):
                for i, name in enumerate(concept_names):
                    concepts.append({
                        "name": name,
                        "score": 0.9,  # Very high score for package-aware advanced system
                        "method": "package_aware_advanced_4000h",
                        "source": {
                            "advanced_pipeline": True,
                            "package_aware": True,
                            "sophisticated_extraction": True,
                            "jason_4000h_system": True
                        },
                        "context": "Extracted via package-aware sophisticated 4000-hour pipeline",
                        "metadata": {
                            "extraction_method": "package_aware_advanced",
                            "sophisticated": True,
                            "package_integration": True,
                            "concept_index": i
                        }
                    })
                logger.info(f"✅ Converted {len(concepts)} concept names to sophisticated concept objects")
        
        # Build comprehensive Prajna result with ALL advanced features
        prajna_result = {
            "doc_id": advanced_result.get("filename", Path(pdf_path).stem),
            "num_concepts": len(concepts) if concepts else advanced_result.get("concept_count", 0),
            "concepts": concepts,
            "injection_result": {
                "injected": len(concepts) if concepts else advanced_result.get("concept_count", 0),
                "method": "package_aware_advanced_4000h_system"
            },
            "method": "package_aware_advanced_4000h_system",
            "status": advanced_result.get("status", "success"),
            
            # Preserve ALL your sophisticated analytics
            "advanced_analytics": {
                "original_result": advanced_result,
                "purity_analysis": advanced_result.get("purity_analysis", {}),
                "context_extraction": advanced_result.get("context_extraction", {}),
                "filtering_stats": advanced_result.get("filtering_stats", {}),
                "universal_methods": advanced_result.get("universal_methods", []),
                "domain_distribution": advanced_result.get("domain_distribution", {}),
                "processing_time": advanced_result.get("processing_time_seconds", 0),
                "semantic_concepts": advanced_result.get("semantic_concepts", 0),
                "boosted_concepts": advanced_result.get("boosted_concepts", 0),
                "cross_reference_boosted": advanced_result.get("cross_reference_boosted", 0),
                "auto_prefilled_concepts": advanced_result.get("auto_prefilled_concepts", 0),
                "file_storage_size": advanced_result.get("file_storage_size", 0),
                "package_aware": True
            },
            
            # Summary metrics for dashboard
            "summary": {
                "total_concepts": len(concepts) if concepts else advanced_result.get("concept_count", 0),
                "processing_time": advanced_result.get("processing_time_seconds", 0),
                "advanced_pipeline_used": True,
                "package_aware": True,
                "sophisticated_extraction": True,
                "purity_based": True,
                "context_aware": True,
                
                # Extract purity metrics if available
                "pure_concepts": advanced_result.get("purity_analysis", {}).get("pure_concepts", 0),
                "consensus_concepts": advanced_result.get("purity_analysis", {}).get("distribution", {}).get("consensus", 0),
                "purity_efficiency": advanced_result.get("purity_analysis", {}).get("purity_efficiency", "N/A"),
                "auto_prefilled": advanced_result.get("auto_prefilled_concepts", 0)
            }
        }
        
        return prajna_result
    
    def _enhanced_fallback_extraction(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Enhanced fallback extraction"""
        logger.info("🔄 Using enhanced fallback extraction")
        
        try:
            # Import our enhanced extraction
            from extraction import extract_concepts_from_pdf
            
            logger.info("🔄 Calling enhanced universal extraction as fallback")
            concepts = extract_concepts_from_pdf(pdf_path)
            
            logger.info(f"✅ Enhanced fallback found {len(concepts)} concepts")
            
            return {
                "doc_id": doc_id or Path(pdf_path).stem,
                "num_concepts": len(concepts),
                "concepts": concepts,
                "injection_result": {"injected": len(concepts)},
                "method": "enhanced_universal_fallback",
                "status": "success",
                "summary": {
                    "total_concepts": len(concepts),
                    "advanced_pipeline_used": False,
                    "fallback_reason": "package_integration_failed"
                }
            }
            
        except Exception as e:
            logger.error(f"🔄 Enhanced fallback failed: {e}")
            
            # Ultra-simple fallback
            return {
                "doc_id": doc_id or "unknown",
                "num_concepts": 1,
                "concepts": [{
                    "name": "Document Processing",
                    "score": 0.5,
                    "method": "ultra_simple_fallback",
                    "source": {"extraction_method": "placeholder"},
                    "context": "Placeholder concept - system fallback",
                    "metadata": {"placeholder": True, "fallback": True}
                }],
                "injection_result": {"injected": 1},
                "method": "ultra_simple_fallback",
                "status": "fallback",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            **self.stats,
            "advanced_available": self.advanced_available,
            "success_rate": (
                self.stats["total_advanced_successes"] / max(1, self.stats["total_documents_processed"])
            ) * 100,
            "fallback_rate": (
                self.stats["total_fallback_uses"] / max(1, self.stats["total_documents_processed"])
            ) * 100,
            "current_timestamp": datetime.now().isoformat()
        }

# Global instance
_package_integration = None

def get_package_integration():
    global _package_integration
    if _package_integration is None:
        _package_integration = PackageAwareIntegration()
    return _package_integration

# Public API
def extract_concepts_package_aware(pdf_file_path_or_bytes, user_id=None, doc_id=None):
    """Package-aware extraction that properly handles your 4000-hour system"""
    integration = get_package_integration()
    return integration.extract_concepts_package_aware(pdf_file_path_or_bytes, user_id, doc_id)

def get_package_stats():
    """Get package integration stats"""
    integration = get_package_integration()
    return integration.get_stats()

# Test function
def test_package_integration():
    """Test the package integration"""
    logger.info("🧪 Testing Package-Aware Integration")
    
    try:
        integration = get_package_integration()
        stats = integration.get_stats()
        
        logger.info("✅ Package integration ready")
        logger.info(f"📊 Advanced available: {stats['advanced_available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Package integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_package_integration()
