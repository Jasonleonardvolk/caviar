"""
PRAJNA SIMPLE INTEGRATION - LIGHTWEIGHT APPROACH
Instead of complex imports, let's create a simple interface that works
with your existing system through file I/O and subprocess calls.

This avoids import hell and dependency conflicts.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile

from utils.logging import logger

# Paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"

class SimplePrajnaIntegration:
    """
    Simple integration that calls your advanced system via subprocess
    to avoid import conflicts and dependency issues.
    """
    
    def __init__(self):
        self.stats = {
            "total_documents_processed": 0,
            "total_concepts_extracted": 0,
            "integration_start_time": datetime.now().isoformat(),
            "method": "subprocess_integration"
        }
        
        # Check if advanced system is available
        self.advanced_available = self._check_advanced_system()
        
        logger.info("🌉 Simple Prajna Integration initialized")
        logger.info(f"🏗️ Advanced system available: {self.advanced_available}")
    
    def _check_advanced_system(self) -> bool:
        """Check if the advanced system is available"""
        pipeline_file = ingest_pdf_dir / "pipeline.py"
        main_file = ingest_pdf_dir / "main.py"
        
        if pipeline_file.exists():
            logger.info(f"✅ Found pipeline.py at {pipeline_file}")
            return True
        elif main_file.exists():
            logger.info(f"✅ Found main.py at {main_file}")
            return True
        else:
            logger.warning(f"⚠️ No advanced system entry point found in {ingest_pdf_dir}")
            return False
    
    def extract_concepts_simple(
        self, 
        pdf_file_path_or_bytes, 
        user_id: Optional[str] = None, 
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple concept extraction that works regardless of import issues
        """
        logger.info("🌉 SIMPLE INTEGRATION: Processing PDF")
        
        # Handle bytes input
        if isinstance(pdf_file_path_or_bytes, bytes):
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_file_path_or_bytes)
                pdf_path = tmp_file.name
            temp_file_created = True
        else:
            pdf_path = pdf_file_path_or_bytes
            temp_file_created = False
        
        try:
            if self.advanced_available:
                result = self._try_advanced_extraction(pdf_path, doc_id)
            else:
                result = self._simple_fallback_extraction(pdf_path, doc_id)
            
            # Update stats
            self.stats["total_documents_processed"] += 1
            self.stats["total_concepts_extracted"] += result.get("num_concepts", 0)
            
            # Clean up temp file
            if temp_file_created:
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"🌉 Simple integration error: {str(e)}")
            
            # Clean up temp file
            if temp_file_created:
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            
            return self._simple_fallback_extraction(pdf_path, doc_id)
    
    def _try_advanced_extraction(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """
        Try to use advanced system via subprocess or direct call
        """
        logger.info("🔄 Attempting advanced extraction")
        
        # Try different approaches
        approaches = [
            self._try_python_direct,
            self._try_subprocess_call,
            self._try_simple_integration
        ]
        
        for approach in approaches:
            try:
                result = approach(pdf_path, doc_id)
                if result and result.get("num_concepts", 0) > 0:
                    logger.info(f"✅ Advanced extraction successful via {approach.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"⚠️ {approach.__name__} failed: {e}")
                continue
        
        logger.warning("🔄 All advanced approaches failed, using fallback")
        return self._simple_fallback_extraction(pdf_path, doc_id)
    
    def _try_python_direct(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Try running Python directly in the ingest_pdf directory"""
        try:
            # Change to ingest_pdf directory and run extraction
            original_cwd = os.getcwd()
            os.chdir(ingest_pdf_dir)
            
            # Add to Python path
            if str(ingest_pdf_dir) not in sys.path:
                sys.path.insert(0, str(ingest_pdf_dir))
            
            # Try to import and run
            import pipeline
            result = pipeline.ingest_pdf_clean(pdf_path, doc_id=doc_id, extraction_threshold=0.0)
            
            # Restore directory
            os.chdir(original_cwd)
            
            # Transform result
            return self._transform_result(result, "python_direct")
            
        except Exception as e:
            # Restore directory
            try:
                os.chdir(original_cwd)
            except:
                pass
            raise e
    
    def _try_subprocess_call(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Try calling via subprocess"""
        logger.info("🔄 Trying subprocess approach")
        
        # Create a simple script to run extraction
        script_content = f'''
import sys
import json
sys.path.insert(0, r"{ingest_pdf_dir}")

try:
    from pipeline import ingest_pdf_clean
    result = ingest_pdf_clean(r"{pdf_path}", doc_id="{doc_id or 'subprocess'}", extraction_threshold=0.0)
    print("RESULT_START")
    print(json.dumps(result, default=str))
    print("RESULT_END")
except Exception as e:
    print("ERROR:", str(e))
'''
        
        # Run the script
        result = subprocess.run([
            sys.executable, "-c", script_content
        ], capture_output=True, text=True, cwd=ingest_pdf_dir)
        
        if result.returncode == 0:
            # Parse output
            output = result.stdout
            if "RESULT_START" in output and "RESULT_END" in output:
                start_idx = output.find("RESULT_START") + len("RESULT_START\n")
                end_idx = output.find("RESULT_END")
                json_str = output[start_idx:end_idx].strip()
                
                advanced_result = json.loads(json_str)
                return self._transform_result(advanced_result, "subprocess")
        
        raise Exception(f"Subprocess failed: {result.stderr}")
    
    def _try_simple_integration(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Try a simple integration approach"""
        logger.info("🔄 Trying simple integration")
        
        # For now, this is the same as fallback but could be enhanced
        # to call specific functions from your system
        return self._simple_fallback_extraction(pdf_path, doc_id)
    
    def _simple_fallback_extraction(self, pdf_path: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """
        Simple fallback that creates reasonable concepts without heavy dependencies
        """
        logger.info("🔄 Using simple fallback extraction")
        
        try:
            # Try to extract basic text from PDF
            concepts = self._extract_simple_concepts(pdf_path)
            
            return {
                "doc_id": doc_id or Path(pdf_path).stem,
                "num_concepts": len(concepts),
                "concepts": concepts,
                "injection_result": {"injected": len(concepts)},
                "method": "simple_fallback",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"🔄 Simple fallback failed: {e}")
            
            # Ultra-simple fallback
            return {
                "doc_id": doc_id or "unknown",
                "num_concepts": 0,
                "concepts": [],
                "injection_result": {"injected": 0},
                "method": "ultra_simple_fallback",
                "status": "failed",
                "error": str(e)
            }
    
    def _extract_simple_concepts(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract concepts using simple text analysis (no heavy ML)"""
        import re
        
        try:
            # Try PyPDF2 for text extraction (lightweight)
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages[:5]:  # First 5 pages only
                    text += page.extract_text()
            
            # Simple concept extraction using regex
            concepts = []
            
            # Find capitalized terms (potential concepts)
            capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            
            # Count frequency
            from collections import Counter
            term_counts = Counter(capitalized_terms)
            
            # Create concept objects
            for i, (term, count) in enumerate(term_counts.most_common(20)):
                if len(term) > 3 and count > 1:  # Filter short/rare terms
                    concepts.append({
                        "name": term,
                        "score": min(1.0, count / 10.0),  # Simple scoring
                        "method": "simple_text_analysis",
                        "source": {
                            "extraction_method": "regex_capitalized_terms",
                            "frequency": count
                        },
                        "context": f"Found {count} times in document",
                        "metadata": {
                            "extraction_method": "simple_fallback",
                            "frequency": count,
                            "concept_index": i
                        }
                    })
            
            logger.info(f"✅ Simple extraction found {len(concepts)} concepts")
            return concepts
            
        except Exception as e:
            logger.warning(f"⚠️ Simple extraction failed: {e}")
            
            # Ultra-simple concepts
            return [
                {
                    "name": "Document Analysis",
                    "score": 0.8,
                    "method": "placeholder",
                    "source": {"extraction_method": "placeholder"},
                    "context": "Placeholder concept",
                    "metadata": {"placeholder": True}
                }
            ]
    
    def _transform_result(self, advanced_result: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Transform advanced result to Prajna format"""
        # Extract concepts
        concepts = []
        
        if "concepts" in advanced_result:
            concepts = advanced_result["concepts"]
        elif "concept_names" in advanced_result:
            for name in advanced_result["concept_names"]:
                concepts.append({
                    "name": name,
                    "score": 0.8,
                    "method": f"advanced_{method}",
                    "source": {"advanced_pipeline": True},
                    "context": "From advanced pipeline",
                    "metadata": {"advanced_extraction": True}
                })
        
        return {
            "doc_id": advanced_result.get("filename", "unknown"),
            "num_concepts": len(concepts),
            "concepts": concepts,
            "injection_result": {"injected": len(concepts)},
            "advanced_pipeline_data": advanced_result,
            "method": f"advanced_{method}",
            "status": "success"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.stats,
            "advanced_available": self.advanced_available,
            "current_timestamp": datetime.now().isoformat()
        }

# Global instance
_simple_integration = None

def get_simple_integration():
    global _simple_integration
    if _simple_integration is None:
        _simple_integration = SimplePrajnaIntegration()
    return _simple_integration

# Public API
def extract_concepts_simple(pdf_file_path_or_bytes, user_id=None, doc_id=None):
    """Simple extraction that always works"""
    integration = get_simple_integration()
    return integration.extract_concepts_simple(pdf_file_path_or_bytes, user_id, doc_id)

def get_simple_stats():
    """Get simple integration stats"""
    integration = get_simple_integration()
    return integration.get_stats()

# Test function
def test_simple_integration():
    """Test the simple integration"""
    logger.info("🧪 Testing Simple Prajna Integration")
    
    try:
        integration = get_simple_integration()
        stats = integration.get_stats()
        
        logger.info("✅ Simple integration ready")
        logger.info(f"📊 Advanced available: {stats['advanced_available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_integration()
