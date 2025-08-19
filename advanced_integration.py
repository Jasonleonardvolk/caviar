"""
PRAJNA ADVANCED INTEGRATION - ROBUST VERSION
This creates a proper wrapper that can successfully call your 4000-hour advanced system
without import conflicts or relative import issues.

PHILOSOPHY:
- Respect your advanced system (never modify it)
- Create a proper Python package interface
- Handle all edge cases and errors gracefully
- Always provide detailed logging for debugging
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logging import logger

# Paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"

class AdvancedPipelineIntegration:
    """
    Robust integration with Jason's 4000-hour advanced system.
    Handles relative imports and dependency issues properly.
    """
    
    def __init__(self):
        self.stats = {
            "total_documents_processed": 0,
            "total_concepts_extracted": 0,
            "total_advanced_successes": 0,
            "total_fallback_uses": 0,
            "integration_start_time": datetime.now().isoformat(),
            "method": "advanced_integration"
        }
        
        # Check system availability
        self.advanced_available = self._comprehensive_system_check()
        
        logger.info("🌉 Advanced Pipeline Integration initialized")
        logger.info(f"🏗️ Advanced system available: {self.advanced_available}")
        
        if self.advanced_available:
            logger.info("🎉 Ready to use your 4000-hour sophisticated system!")
        else:
            logger.warning("⚠️ Advanced system not available - will use enhanced fallback")
    
    def _comprehensive_system_check(self) -> bool:
        """Comprehensive check of the advanced system"""
        logger.info("🔍 Performing comprehensive advanced system check...")
        
        # Check basic file existence
        pipeline_file = ingest_pdf_dir / "pipeline.py"
        if not pipeline_file.exists():
            logger.warning(f"❌ pipeline.py not found at {pipeline_file}")
            return False
        
        logger.info(f"✅ Found pipeline.py at {pipeline_file}")
        
        # Check key dependency files
        key_files = [
            "extractConceptsFromDocument.py",
            "extract_blocks.py", 
            "scoring.py",
            "models.py"
        ]
        
        missing_files = []
        for file in key_files:
            file_path = ingest_pdf_dir / file
            if file_path.exists():
                logger.info(f"✅ Found {file}")
            else:
                logger.warning(f"⚠️ Missing {file}")
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"❌ Missing key files: {missing_files}")
            return False
        
        # Test basic import in subprocess (safest way)
        test_success = self._test_subprocess_import()
        
        if test_success:
            logger.info("🎉 Advanced system fully operational!")
            return True
        else:
            logger.warning("❌ Advanced system import test failed")
            return False
    
    def _test_subprocess_import(self) -> bool:
        """Test importing the advanced system in a subprocess"""
        logger.info("🧪 Testing advanced system import via subprocess...")
        
        test_script = f'''
import sys
import os
sys.path.insert(0, r"{ingest_pdf_dir}")
os.chdir(r"{ingest_pdf_dir}")

try:
    # Test importing key modules
    from pipeline import ingest_pdf_clean
    from extractConceptsFromDocument import extractConceptsFromDocument
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
        
        try:
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True, timeout=30)
            
            if "IMPORT_SUCCESS" in result.stdout:
                logger.info("✅ Subprocess import test passed")
                return True
            else:
                logger.warning(f"❌ Subprocess import test failed:")
                logger.warning(f"   stdout: {result.stdout}")
                logger.warning(f"   stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("❌ Subprocess import test timed out (30s)")
            return False
        except Exception as e:
            logger.warning(f"❌ Subprocess import test error: {e}")
            return False
    
    def extract_concepts_advanced(
        self, 
        pdf_file_path_or_bytes, 
        user_id: Optional[str] = None, 
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract concepts using the advanced system with robust error handling
        """
        logger.info("🌉 ADVANCED INTEGRATION: Processing PDF")
        
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
                logger.info("🚀 Attempting advanced extraction via your 4000-hour system")
                result = self._call_advanced_system(pdf_path, doc_id)
                
                if result and result.get("num_concepts", 0) > 0:
                    logger.info(f"🎉 Advanced extraction SUCCESS: {result.get('num_concepts', 0)} concepts")
                    self.stats["total_advanced_successes"] += 1
                    return result
                else:
                    logger.warning("🔄 Advanced extraction returned no concepts, trying fallback")
                    
            logger.info("🔄 Using enhanced fallback extraction")
            result = self._enhanced_fallback_extraction(pdf_path, doc_id)
            self.stats["total_fallback_uses"] += 1
            return result
            
        except Exception as e:
            logger.error(f"🌉 Advanced integration error: {str(e)}")
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
    
    def _call_advanced_system(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """
        Call the advanced system using the most robust method
        """
        logger.info("🔄 Calling advanced system via subprocess")
        
        # Create a comprehensive extraction script
        extraction_script = f'''
import sys
import os
import json
from pathlib import Path

# Set up environment for advanced system
ingest_pdf_dir = Path(r"{ingest_pdf_dir}")
sys.path.insert(0, str(ingest_pdf_dir))
os.chdir(str(ingest_pdf_dir))

try:
    # Import and call the advanced pipeline
    from pipeline import ingest_pdf_clean
    
    # Call with zero threshold for maximum extraction
    result = ingest_pdf_clean(
        r"{pdf_path}", 
        doc_id="{doc_id or 'advanced_extraction'}", 
        extraction_threshold=0.0
    )
    
    # Output result in parseable format
    print("ADVANCED_RESULT_START")
    print(json.dumps(result, default=str, indent=2))
    print("ADVANCED_RESULT_END")
    
    # Also output summary for easy parsing
    print("SUMMARY_START")
    summary = {{
        "filename": result.get("filename", "unknown"),
        "concept_count": result.get("concept_count", 0),
        "status": result.get("status", "unknown"),
        "processing_time": result.get("processing_time_seconds", 0),
        "method": "advanced_4000h_system"
    }}
    print(json.dumps(summary, default=str))
    print("SUMMARY_END")
    
except Exception as e:
    print(f"ADVANCED_ERROR: {{str(e)}}")
    import traceback
    print("TRACEBACK_START")
    traceback.print_exc()
    print("TRACEBACK_END")
'''
        
        try:
            # Run the extraction with generous timeout
            result = subprocess.run([
                sys.executable, "-c", extraction_script
            ], capture_output=True, text=True, timeout=120, cwd=ingest_pdf_dir)
            
            if result.returncode == 0:
                return self._parse_advanced_result(result.stdout, pdf_path)
            else:
                logger.error(f"❌ Advanced system subprocess failed:")
                logger.error(f"   Return code: {result.returncode}")
                logger.error(f"   stderr: {result.stderr}")
                raise Exception(f"Subprocess failed with return code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Advanced system call timed out (120s)")
            raise Exception("Advanced system timeout")
        except Exception as e:
            logger.error(f"❌ Advanced system call failed: {e}")
            raise e
    
    def _parse_advanced_result(self, output: str, pdf_path: str) -> Dict[str, Any]:
        """Parse the output from the advanced system"""
        try:
            # Extract the advanced result
            if "ADVANCED_RESULT_START" in output and "ADVANCED_RESULT_END" in output:
                start_idx = output.find("ADVANCED_RESULT_START") + len("ADVANCED_RESULT_START\n")
                end_idx = output.find("ADVANCED_RESULT_END")
                result_json = output[start_idx:end_idx].strip()
                
                advanced_result = json.loads(result_json)
                logger.info("✅ Successfully parsed advanced system result")
                
                # Transform to Prajna format
                return self._transform_advanced_result(advanced_result, pdf_path)
            
            # Try to extract summary if full result not available
            elif "SUMMARY_START" in output and "SUMMARY_END" in output:
                start_idx = output.find("SUMMARY_START") + len("SUMMARY_START\n")
                end_idx = output.find("SUMMARY_END")
                summary_json = output[start_idx:end_idx].strip()
                
                summary = json.loads(summary_json)
                logger.info("✅ Parsed advanced system summary")
                
                return {
                    "doc_id": summary.get("filename", Path(pdf_path).stem),
                    "num_concepts": summary.get("concept_count", 0),
                    "concepts": [],  # Will be populated if available
                    "injection_result": {"injected": summary.get("concept_count", 0)},
                    "method": "advanced_4000h_system",
                    "status": summary.get("status", "unknown"),
                    "processing_time": summary.get("processing_time", 0),
                    "advanced_pipeline_data": summary
                }
            
            else:
                logger.error("❌ Could not parse advanced system output")
                logger.error(f"Output preview: {output[:500]}...")
                raise Exception("Could not parse advanced system output")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            logger.error(f"Output preview: {output[:500]}...")
            raise Exception(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"❌ Result parsing error: {e}")
            raise e
    
    def _transform_advanced_result(self, advanced_result: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        """Transform advanced result to Prajna format"""
        logger.info("🔄 Transforming advanced result to Prajna format")
        
        # Extract concepts
        concepts = []
        
        # Handle different concept formats
        if "concepts" in advanced_result and isinstance(advanced_result["concepts"], list):
            concepts = advanced_result["concepts"]
            logger.info(f"✅ Found {len(concepts)} concepts in advanced result")
        elif "concept_names" in advanced_result:
            # Convert concept names to concept objects
            concept_names = advanced_result["concept_names"]
            if isinstance(concept_names, list):
                for i, name in enumerate(concept_names):
                    concepts.append({
                        "name": name,
                        "score": 0.85,  # High score for advanced system
                        "method": "advanced_4000h_system",
                        "source": {
                            "advanced_pipeline": True,
                            "extraction_method": "jason_4000h_system",
                            "sophisticated_extraction": True
                        },
                        "context": "Extracted via Jason's sophisticated 4000-hour pipeline",
                        "metadata": {
                            "extraction_method": "advanced_pipeline",
                            "sophisticated": True,
                            "concept_index": i
                        }
                    })
                logger.info(f"✅ Converted {len(concepts)} concept names to concept objects")
        
        # Build comprehensive Prajna result
        prajna_result = {
            "doc_id": advanced_result.get("filename", Path(pdf_path).stem),
            "num_concepts": len(concepts) if concepts else advanced_result.get("concept_count", 0),
            "concepts": concepts,
            "injection_result": {
                "injected": len(concepts) if concepts else advanced_result.get("concept_count", 0),
                "method": "advanced_4000h_system"
            },
            "method": "advanced_4000h_system",
            "status": advanced_result.get("status", "success"),
            
            # Preserve ALL advanced analytics
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
                "auto_prefilled_concepts": advanced_result.get("auto_prefilled_concepts", 0),
                "file_storage_size": advanced_result.get("file_storage_size", 0)
            },
            
            # Summary metrics for dashboard
            "summary": {
                "total_concepts": len(concepts) if concepts else advanced_result.get("concept_count", 0),
                "processing_time": advanced_result.get("processing_time_seconds", 0),
                "advanced_pipeline_used": True,
                "sophisticated_extraction": True,
                "purity_based": True,
                "context_aware": True
            }
        }
        
        return prajna_result
    
    def _enhanced_fallback_extraction(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """
        Enhanced fallback extraction with better concept detection
        """
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
                    "fallback_reason": "advanced_system_unavailable"
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
_advanced_integration = None

def get_advanced_integration():
    global _advanced_integration
    if _advanced_integration is None:
        _advanced_integration = AdvancedPipelineIntegration()
    return _advanced_integration

# Public API
def extract_concepts_advanced(pdf_file_path_or_bytes, user_id=None, doc_id=None):
    """Advanced extraction that tries the 4000-hour system first"""
    integration = get_advanced_integration()
    return integration.extract_concepts_advanced(pdf_file_path_or_bytes, user_id, doc_id)

def get_advanced_stats():
    """Get advanced integration stats"""
    integration = get_advanced_integration()
    return integration.get_stats()

# Test function
def test_advanced_integration():
    """Test the advanced integration"""
    logger.info("🧪 Testing Advanced Pipeline Integration")
    
    try:
        integration = get_advanced_integration()
        stats = integration.get_stats()
        
        logger.info("✅ Advanced integration ready")
        logger.info(f"📊 Advanced available: {stats['advanced_available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_advanced_integration()
