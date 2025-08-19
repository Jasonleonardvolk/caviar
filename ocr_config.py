"""
OCR Configuration Helper
Add this to your PDF processing code to configure Tesseract
"""

import os
import platform
import pytesseract
from pathlib import Path

def configure_tesseract():
    """Auto-configure Tesseract path for Windows"""
    if platform.system() == 'Windows':
        # Common Windows installation paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\%USERNAME%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        ]
        
        # Expand environment variables
        possible_paths = [os.path.expandvars(p) for p in possible_paths]
        
        # Find Tesseract
        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✅ Tesseract configured: {path}")
                return True
        
        # Check if in PATH
        try:
            import subprocess
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                tesseract_path = result.stdout.strip().split('\n')[0]
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                print(f"✅ Tesseract found in PATH: {tesseract_path}")
                return True
        except:
            pass
        
        print("❌ Tesseract not found. Please install from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    return True  # Assume configured on non-Windows

# Usage in your code:
# from ocr_config import configure_tesseract
# configure_tesseract()
