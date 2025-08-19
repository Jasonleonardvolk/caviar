# core/universal_file_extractor.py - Extract content from any file type
import os
import logging
from pathlib import Path
from typing import Optional
import chardet
import json

logger = logging.getLogger(__name__)

class UniversalFileExtractor:
    """Extract text content from various file types"""
    
    def __init__(self):
        self.supported_extensions = {
            # Text files
            '.txt', '.md', '.py', '.js', '.json', '.xml', '.html', '.css', 
            '.yml', '.yaml', '.ini', '.cfg', '.conf', '.log', '.csv', '.tsv',
            '.sh', '.bat', '.ps1', '.java', '.c', '.cpp', '.h', '.hpp',
            '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
            
            # Document files
            '.pdf', '.doc', '.docx', '.odt', '.rtf',
            
            # Data files
            '.jsonl', '.ndjson', '.geojson',
            
            # Code/config
            '.toml', '.env', '.gitignore', '.dockerignore',
            
            # Web
            '.jsx', '.tsx', '.vue', '.svelte',
            
            # Audio transcription placeholders
            '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac',
            
            # Video transcription placeholders  
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
            
            # Image OCR placeholders
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
        }
        
    async def extract_content(self, file_path: str) -> str:
        """Extract text content from any supported file"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = file_path.suffix.lower()
        
        try:
            # PDF files
            if ext == '.pdf':
                return await self._extract_pdf(file_path)
            
            # Word documents
            elif ext in ['.doc', '.docx']:
                return await self._extract_word(file_path)
            
            # Audio files (placeholder for transcription)
            elif ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']:
                return await self._extract_audio_placeholder(file_path)
            
            # Video files (placeholder for transcription)
            elif ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
                return await self._extract_video_placeholder(file_path)
            
            # Image files (placeholder for OCR)
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                return await self._extract_image_placeholder(file_path)
            
            # All text-based files
            else:
                return await self._extract_text(file_path)
                
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            # Fallback to basic text extraction
            return await self._extract_text_fallback(file_path)
    
    async def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            logger.warning("PyMuPDF not available, trying pdfplumber")
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            except ImportError:
                return f"[PDF file: {file_path.name} - Install PyMuPDF or pdfplumber to extract content]"
    
    async def _extract_word(self, file_path: Path) -> str:
        """Extract text from Word documents"""
        try:
            import docx
            doc = docx.Document(str(file_path))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            return f"[Word document: {file_path.name} - Install python-docx to extract content]"
    
    async def _extract_audio_placeholder(self, file_path: Path) -> str:
        """Placeholder for audio transcription"""
        return f"""[Audio file: {file_path.name}]
        
This is an audio file that would require transcription to extract content.
To enable audio transcription, integrate a service like:
- OpenAI Whisper
- Google Speech-to-Text
- Azure Speech Services
- AWS Transcribe

File info:
- Name: {file_path.name}
- Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB
- Type: {file_path.suffix}
"""
    
    async def _extract_video_placeholder(self, file_path: Path) -> str:
        """Placeholder for video transcription"""
        return f"""[Video file: {file_path.name}]
        
This is a video file that would require transcription to extract content.
To enable video transcription:
1. Extract audio track
2. Transcribe using speech-to-text service
3. Optionally perform OCR on frames for on-screen text

File info:
- Name: {file_path.name}
- Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB
- Type: {file_path.suffix}
"""
    
    async def _extract_image_placeholder(self, file_path: Path) -> str:
        """Placeholder for image OCR"""
        return f"""[Image file: {file_path.name}]
        
This is an image file that would require OCR to extract text content.
To enable image text extraction, integrate:
- Tesseract OCR
- Google Vision API
- Azure Computer Vision
- AWS Textract

File info:
- Name: {file_path.name}
- Size: {file_path.stat().st_size / 1024:.2f} KB
- Type: {file_path.suffix}
"""
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from text-based files with encoding detection"""
        # Try to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    async def _extract_text_fallback(self, file_path: Path) -> str:
        """Fallback text extraction with aggressive error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                return f"[Extracted with encoding errors replaced]\n{content}"
        except Exception as e:
            return f"[Failed to extract from {file_path.name}: {e}]"

# Global extractor instance
_file_extractor = None

async def get_file_extractor() -> UniversalFileExtractor:
    """Get or create global file extractor"""
    global _file_extractor
    if _file_extractor is None:
        _file_extractor = UniversalFileExtractor()
    return _file_extractor

async def extract_content_from_file(file_path: str) -> str:
    """Main function to extract content from any file"""
    extractor = await get_file_extractor()
    return await extractor.extract_content(file_path)
