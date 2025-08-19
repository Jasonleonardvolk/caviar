#!/usr/bin/env python3
"""
Fix 3: Improve ScholarSphere Upload Error Handling
This script adds better error handling and debugging to the upload process
"""

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_upload_debug_wrapper():
    """Create a debug wrapper for the ScholarSphere upload to identify issues"""
    
    # Path for the debug wrapper
    wrapper_path = Path(r"{PROJECT_ROOT}\api\upload_debug_wrapper.py")
    
    wrapper_content = '''"""
Upload Debug Wrapper - Enhanced error tracking for ScholarSphere uploads
"""

import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('upload_debug')

class UploadDebugger:
    """Debug wrapper for upload operations"""
    
    def __init__(self):
        self.debug_log = []
        self.error_count = 0
        
    def log_step(self, step: str, data: Any = None):
        """Log a debug step"""
        entry = {
            'timestamp': time.time(),
            'step': step,
            'data': data
        }
        self.debug_log.append(entry)
        logger.info(f"UPLOAD STEP: {step} - {data}")
        
    def log_error(self, error: Exception, context: str):
        """Log an error with full context"""
        self.error_count += 1
        error_entry = {
            'timestamp': time.time(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        self.debug_log.append(error_entry)
        logger.error(f"UPLOAD ERROR in {context}: {error}")
        logger.debug(traceback.format_exc())
        
    def validate_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF file before processing"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Check file exists
            if not file_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append('File does not exist')
                return validation_result
                
            # Check file size
            file_size = file_path.stat().st_size
            validation_result['file_info']['size_bytes'] = file_size
            validation_result['file_info']['size_mb'] = file_size / (1024 * 1024)
            
            if file_size == 0:
                validation_result['valid'] = False
                validation_result['errors'].append('File is empty')
            elif file_size > 50 * 1024 * 1024:  # 50MB
                validation_result['warnings'].append('File is larger than 50MB')
                
            # Check file extension
            if not file_path.suffix.lower() == '.pdf':
                validation_result['valid'] = False
                validation_result['errors'].append(f'Invalid file type: {file_path.suffix}')
                
            # Try to read PDF header
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(5)
                    if header != b'%PDF-':
                        validation_result['valid'] = False
                        validation_result['errors'].append('Invalid PDF header')
            except Exception as e:
                validation_result['warnings'].append(f'Could not validate PDF header: {e}')
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Validation error: {e}')
            
        return validation_result
        
    def get_debug_report(self) -> Dict[str, Any]:
        """Get complete debug report"""
        return {
            'total_steps': len(self.debug_log),
            'error_count': self.error_count,
            'log': self.debug_log,
            'summary': self._generate_summary()
        }
        
    def _generate_summary(self) -> str:
        """Generate a human-readable summary"""
        if self.error_count == 0:
            return "Upload completed successfully"
        else:
            error_types = {}
            for entry in self.debug_log:
                if 'error_type' in entry:
                    error_type = entry['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
            summary = f"Upload failed with {self.error_count} errors: "
            summary += ", ".join([f"{count} {etype}" for etype, count in error_types.items()])
            return summary

# Global debugger instance
_debugger = None

def get_debugger() -> UploadDebugger:
    """Get or create the global debugger instance"""
    global _debugger
    if _debugger is None:
        _debugger = UploadDebugger()
    return _debugger

def reset_debugger():
    """Reset the debugger for a new upload"""
    global _debugger
    _debugger = UploadDebugger()
    return _debugger

# Enhanced upload processing with debugging
async def debug_pdf_upload(file_path: str, filename: str) -> Dict[str, Any]:
    """Process PDF upload with comprehensive debugging"""
    
    debugger = reset_debugger()
    debugger.log_step('upload_start', {'filename': filename, 'file_path': file_path})
    
    try:
        # Validate PDF
        debugger.log_step('validating_pdf')
        validation = debugger.validate_pdf(Path(file_path))
        debugger.log_step('validation_complete', validation)
        
        if not validation['valid']:
            raise ValueError(f"PDF validation failed: {validation['errors']}")
            
        # Try to extract concepts
        debugger.log_step('concept_extraction_start')
        
        # Import the actual processing function
        try:
            from ingest_pdf.pipeline import ingest_pdf_clean
            
            debugger.log_step('pipeline_imported')
            
            # Process with maximum verbosity
            result = await ingest_pdf_clean(
                file_path,
                extraction_threshold=0.0,
                admin_mode=True,
                verbose=True
            )
            
            debugger.log_step('concept_extraction_complete', {
                'concept_count': result.get('concept_count', 0),
                'status': result.get('status', 'unknown')
            })
            
            return result
            
        except ImportError as e:
            debugger.log_error(e, 'import_pipeline')
            
            # Try fallback
            debugger.log_step('using_fallback_processing')
            
            # Simple fallback processing
            return {
                'concept_count': 0,
                'concept_names': [],
                'concepts': [],
                'status': 'fallback_processing',
                'debug_report': debugger.get_debug_report()
            }
            
        except Exception as e:
            debugger.log_error(e, 'concept_extraction')
            raise
            
    except Exception as e:
        debugger.log_error(e, 'upload_processing')
        
        # Return error response with debug info
        return {
            'success': False,
            'error': str(e),
            'debug_report': debugger.get_debug_report(),
            'concept_count': 0,
            'concepts': []
        }
        
    finally:
        # Save debug log
        debug_log_path = Path('logs/upload_debug') / f"{int(time.time())}_{filename}.json"
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(debug_log_path, 'w') as f:
            json.dump(debugger.get_debug_report(), f, indent=2)
            
        debugger.log_step('debug_log_saved', str(debug_log_path))

# Test function
def test_pdf_validation():
    """Test PDF validation on a sample file"""
    debugger = UploadDebugger()
    
    # Test with a dummy path
    test_path = Path("test.pdf")
    result = debugger.validate_pdf(test_path)
    
    print("Validation result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_pdf_validation()
'''
    
    # Create the wrapper file
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print(f"✅ Created upload debug wrapper at {wrapper_path}")
    
    # Also create a patch for the prajna_api to use the debug wrapper
    patch_content = '''# Add this to prajna_api.py after the imports:

# Import upload debugger
try:
    from api.upload_debug_wrapper import debug_pdf_upload, get_debugger
    UPLOAD_DEBUG_AVAILABLE = True
except ImportError:
    UPLOAD_DEBUG_AVAILABLE = False
    debug_pdf_upload = None

# Then modify the upload endpoint to use debug mode when needed:
# In the upload_pdf_bulletproof function, add:

if request.headers.get('X-Debug-Upload') == 'true' and UPLOAD_DEBUG_AVAILABLE:
    # Use debug wrapper
    extraction_result = await debug_pdf_upload(str(temp_file_path), safe_filename)
    
    # Add debug report to response if available
    if 'debug_report' in extraction_result:
        response_data['debug_report'] = extraction_result['debug_report']
'''
    
    patch_path = Path(r"{PROJECT_ROOT}\fixes_2025_01_19\upload_debug_patch.py")
    with open(patch_path, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"✅ Created patch instructions at {patch_path}")
    
    return True

if __name__ == "__main__":
    success = create_upload_debug_wrapper()
    if success:
        print("\n✨ Upload debug wrapper created!")
        print("🔍 Features:")
        print("   - Comprehensive PDF validation")
        print("   - Step-by-step debugging logs")
        print("   - Error tracking and reporting")
        print("   - Automatic debug log saving")
        print("   - Fallback processing on errors")
        print("\n📝 To use: Add 'X-Debug-Upload: true' header to upload requests")
