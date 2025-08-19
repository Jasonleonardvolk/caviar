"""
OCR Warning Silencer Patch
Add this to any file that imports pytesseract
"""

import importlib.util
import logging

logger = logging.getLogger(__name__)

# Check if OCR libraries are available
OCR_ENABLED = importlib.util.find_spec("pytesseract") is not None
if not OCR_ENABLED:
    logger.info("🛈 OCR disabled – skipping scanned pages")

# Function to wrap OCR operations
def try_ocr(func):
    """Decorator to skip OCR operations if not available"""
    def wrapper(*args, **kwargs):
        if not OCR_ENABLED:
            return None
        return func(*args, **kwargs)
    return wrapper

# Usage example:
# @try_ocr
# def extract_text_from_image(image):
#     return pytesseract.image_to_string(image)
