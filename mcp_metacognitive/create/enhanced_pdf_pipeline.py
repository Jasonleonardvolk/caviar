#!/usr/bin/env python
"""
Enhanced PDF Pipeline for MCP Server Creator
Includes caching, metadata extraction, parallel downloads, and security
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile
import httpx

# Optional dependencies
try:
    from PyPDF2 import PdfReader
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False

try:
    from pdfplumber import PDF as PDFPlumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text, extract_pages
    from pdfminer.layout import LTTextBox, LTTable
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global cache configuration
CACHE_DIR = Path.home() / ".tori_pdf_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Security settings
TEMP_DIR = Path(tempfile.gettempdir()) / "tori_pdf_temp"
TEMP_DIR.mkdir(exist_ok=True)

# CrossRef API
CROSSREF_API = "https://api.crossref.org/works/"

class EnhancedPDFProcessor:
    """Enhanced PDF processing with caching, metadata, and security"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        self.session = None
        
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load the global cache index"""
        if self.cache_index_file.exists():
            try:
                return json.loads(self.cache_index_file.read_text())
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save the cache index"""
        self.cache_index_file.write_text(json.dumps(self.cache_index, indent=2))
    
    async def process_pdfs(self, pdf_sources: List[Dict[str, str]], 
                          dest_dir: Path, 
                          parallel: int = 5) -> Dict[str, Any]:
        """
        Process multiple PDFs with parallel downloads and caching
        
        Args:
            pdf_sources: List of dicts with 'url' or 'path' and optional 'tier'
            dest_dir: Destination directory for the agent
            parallel: Number of parallel downloads
            
        Returns:
            Combined metadata for all PDFs
        """
        dest_resources = dest_dir / "resources"
        dest_resources.mkdir(parents=True, exist_ok=True)
        
        # Initialize spec
        spec_file = dest_dir / "spec.json"
        if spec_file.exists():
            spec = json.loads(spec_file.read_text())
        else:
            spec = {
                "pdfs": [],
                "total_chars": 0,
                "last_updated": None,
                "seed_max_chars": 50000,
                "tiers": [],
                "failed": []
            }
        
        # Process PDFs with parallel downloads
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Create tasks for parallel processing
            tasks = []
            for source in pdf_sources:
                task = self._process_single_pdf(source, dest_resources)
                tasks.append(task)
            
            # Run with semaphore to limit parallelism
            semaphore = asyncio.Semaphore(parallel)
            async def bounded_process(source):
                async with semaphore:
                    return await self._process_single_pdf(source, dest_resources)
            
            bounded_tasks = [bounded_process(source) for source in pdf_sources]
            results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Process results
        combined_text = ""
        successful_pdfs = []
        
        for source, result in zip(pdf_sources, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {source}: {result}")
                spec["failed"].append({
                    "source": source,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif result:
                successful_pdfs.append(result)
                combined_text += f"\n\n--- From {result['file']} ---\n\n{result.get('text', '')}"
                
                # Update tiers if specified
                if source.get('tier') and source['tier'] not in spec['tiers']:
                    spec['tiers'].append(source['tier'])
        
        # Update spec
        spec["pdfs"].extend(successful_pdfs)
        spec["total_chars"] = sum(pdf.get("chars", 0) for pdf in spec["pdfs"])
        spec["last_updated"] = datetime.utcnow().isoformat()
        
        # Save spec
        spec_file.write_text(json.dumps(spec, indent=2))
        
        # Save combined seed text
        seed_file = dest_dir / "seed.txt"
        max_chars = spec.get("seed_max_chars", 50000)
        seed_file.write_text(combined_text[:max_chars])
        
        # Save sections if available
        sections_file = dest_dir / "sections.json"
        all_sections = {}
        for pdf in successful_pdfs:
            if "sections" in pdf:
                all_sections[pdf["file"]] = pdf["sections"]
        if all_sections:
            sections_file.write_text(json.dumps(all_sections, indent=2))
        
        # Emit telemetry
        await self._emit_telemetry("pdf_batch_complete", {
            "total": len(pdf_sources),
            "successful": len(successful_pdfs),
            "failed": len(spec["failed"]),
            "cached": sum(1 for p in successful_pdfs if p.get("from_cache"))
        })
        
        return spec
    
    async def _process_single_pdf(self, source: Dict[str, str], 
                                 dest_dir: Path) -> Optional[Dict[str, Any]]:
        """Process a single PDF with caching and metadata extraction"""
        try:
            # Download or copy to temp location
            if "url" in source:
                temp_file = await self._download_pdf(source["url"])
            elif "path" in source:
                temp_file = Path(source["path"])
            else:
                raise ValueError("Source must have 'url' or 'path'")
            
            # Calculate MD5
            md5_hash = hashlib.md5(temp_file.read_bytes()).hexdigest()
            
            # Check cache
            if md5_hash in self.cache_index:
                logger.info(f"Cache hit for {md5_hash}")
                cached_data = self.cache_index[md5_hash]
                
                # Symlink or copy from cache
                cache_file = self.cache_dir / f"{md5_hash}.pdf"
                dest_file = dest_dir / Path(cached_data["file"]).name
                
                if hasattr(dest_file, 'symlink_to'):
                    try:
                        dest_file.symlink_to(cache_file)
                    except:
                        shutil.copy2(cache_file, dest_file)
                else:
                    shutil.copy2(cache_file, dest_file)
                
                cached_data["from_cache"] = True
                return cached_data
            
            # Security check
            if not await self._security_check(temp_file):
                raise ValueError("PDF failed security check")
            
            # Extract text and metadata
            pdf_data = await self._extract_pdf_data(temp_file)
            
            # Get CrossRef metadata if DOI found
            if pdf_data.get("doi"):
                crossref_data = await self._fetch_crossref_metadata(pdf_data["doi"])
                pdf_data.update(crossref_data)
            
            # Save to cache
            cache_file = self.cache_dir / f"{md5_hash}.pdf"
            shutil.copy2(temp_file, cache_file)
            
            # Update cache index
            self.cache_index[md5_hash] = pdf_data
            self._save_cache_index()
            
            # Copy to destination
            dest_file = dest_dir / Path(pdf_data["file"]).name
            shutil.copy2(temp_file, dest_file)
            
            # Clean up temp file if downloaded
            if "url" in source and temp_file.parent == TEMP_DIR:
                temp_file.unlink()
            
            return pdf_data
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    async def _download_pdf(self, url: str, max_retries: int = 3) -> Path:
        """Download PDF with retries and progress tracking"""
        filename = url.split("/")[-1] or "download.pdf"
        temp_file = TEMP_DIR / f"{hashlib.md5(url.encode()).hexdigest()}_{filename}"
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    
                    # Stream download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(temp_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (1024 * 1024) == 0:  # Log every MB
                                    logger.debug(f"Download progress: {progress:.1f}%")
                    
                    logger.info(f"Downloaded {filename} ({downloaded} bytes)")
                    return temp_file
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Download failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def _security_check(self, pdf_file: Path) -> bool:
        """Check PDF for security issues"""
        try:
            # Basic file type check
            header = pdf_file.read_bytes()[:5]
            if header != b'%PDF-':
                logger.warning("File is not a valid PDF")
                return False
            
            # Check for JavaScript or Launch actions
            content = pdf_file.read_text(errors='ignore')
            dangerous_patterns = [
                r'/JavaScript',
                r'/JS\s',
                r'/Launch',
                r'/EmbeddedFile',
                r'/OpenAction'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content):
                    logger.warning(f"PDF contains potentially dangerous pattern: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return False
    
    async def _extract_pdf_data(self, pdf_file: Path) -> Dict[str, Any]:
        """Extract text, metadata, and sections from PDF"""
        data = {
            "file": pdf_file.name,
            "md5": hashlib.md5(pdf_file.read_bytes()).hexdigest(),
            "added": datetime.utcnow().isoformat(),
            "source_path": str(pdf_file)
        }
        
        # Try different extraction methods
        text = ""
        sections = {}
        metadata = {}
        
        # Method 1: PDFMiner (best for complex layouts)
        if PDFMINER_AVAILABLE:
            try:
                text = extract_text(str(pdf_file))
                
                # Extract sections
                sections = self._extract_sections_pdfminer(pdf_file)
                
            except Exception as e:
                logger.debug(f"PDFMiner extraction failed: {e}")
        
        # Method 2: PyPDF2 (fallback)
        if not text and PDF_READER_AVAILABLE:
            try:
                reader = PdfReader(pdf_file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                
                # Extract metadata
                if reader.metadata:
                    metadata = {
                        "title": getattr(reader.metadata, 'title', None),
                        "author": getattr(reader.metadata, 'author', None),
                        "subject": getattr(reader.metadata, 'subject', None),
                        "creator": getattr(reader.metadata, 'creator', None),
                    }
                    
            except Exception as e:
                logger.debug(f"PyPDF2 extraction failed: {e}")
        
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.I)
        if doi_match:
            data["doi"] = doi_match.group(0)
        
        # Extract key topics (simple TF-IDF)
        key_topics = self._extract_key_topics(text)
        
        # Update data
        data.update({
            "chars": len(text),
            "text": text,
            "sections": sections,
            "metadata": metadata,
            "key_topics": key_topics
        })
        
        return data
    
    def _extract_sections_pdfminer(self, pdf_file: Path) -> Dict[str, str]:
        """Extract document sections using PDFMiner"""
        sections = {
            "abstract": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        
        if not PDFMINER_AVAILABLE:
            return sections
        
        try:
            current_section = None
            section_text = []
            
            for page_layout in extract_pages(str(pdf_file)):
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        text = element.get_text().strip()
                        
                        # Check for section headers
                        lower_text = text.lower()
                        for section_name in sections.keys():
                            if section_name in lower_text and len(text) < 50:
                                if current_section and section_text:
                                    sections[current_section] = '\n'.join(section_text)
                                current_section = section_name
                                section_text = []
                                break
                        else:
                            if current_section:
                                section_text.append(text)
            
            # Save last section
            if current_section and section_text:
                sections[current_section] = '\n'.join(section_text)
                
        except Exception as e:
            logger.debug(f"Section extraction failed: {e}")
        
        return {k: v for k, v in sections.items() if v}  # Only return non-empty sections
    
    def _extract_key_topics(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key topics using simple TF-IDF"""
        # Simple word frequency (could be enhanced with actual TF-IDF)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Skip common words
        stopwords = {'this', 'that', 'these', 'those', 'then', 'than', 'from', 
                    'with', 'have', 'been', 'were', 'their', 'there', 'which',
                    'would', 'could', 'should', 'about', 'after', 'before'}
        
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, _ in top_words]
    
    async def _fetch_crossref_metadata(self, doi: str) -> Dict[str, Any]:
        """Fetch metadata from CrossRef API using async httpx"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{CROSSREF_API}{doi}",
                    timeout=8.0,
                    headers={"User-Agent": "TORI-MCP/1.0"}
                )
            
            if response.status_code == 200:
                data = response.json()["message"]
                return {
                    "title": data.get("title", [""])[0],
                    "authors": [f"{a.get('given', '')} {a.get('family', '')}" 
                               for a in data.get("author", [])],
                    "year": data.get("issued", {}).get("date-parts", [[None]])[0][0],
                    "journal": data.get("container-title", [""])[0],
                    "publisher": data.get("publisher", ""),
                    "citations": data.get("is-referenced-by-count", 0)
                }
                
        except Exception as e:
            logger.debug(f"CrossRef lookup failed for {doi}: {e}")
        
        return {}
    
    async def _emit_telemetry(self, event: str, data: Dict[str, Any]):
        """Emit telemetry event (placeholder for WebSocket integration)"""
        logger.info(f"Telemetry: {event} - {data}")
        # TODO: Integrate with actual WebSocket emitter
        # websocket.emit("pdf_ingest_done", data)

# Convenience functions for backward compatibility
async def enhanced_copy_and_extract_multiple(pdf_paths: List[Path], 
                                           dest_dir: Path, 
                                           append: bool = False) -> Dict[str, Any]:
    """Enhanced version with caching and parallel processing"""
    processor = EnhancedPDFProcessor()
    
    # Convert paths to sources
    sources = [{"path": str(p)} for p in pdf_paths]
    
    return await processor.process_pdfs(sources, dest_dir)

def bulk_create_servers(pdf_directory: Path, base_description: str = "Auto-generated server"):
    """Create one server per PDF in a directory"""
    from mk_server import create_server
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    pdf_files.extend(list(pdf_directory.glob("*.PDF")))
    
    for pdf_file in pdf_files:
        # Generate server name from filename
        server_name = re.sub(r'[^a-z0-9_]', '_', pdf_file.stem.lower())
        server_name = re.sub(r'_+', '_', server_name).strip('_')
        
        # Create server with PDF
        desc = f"{base_description} for {pdf_file.stem}"
        create_server(server_name, desc, [str(pdf_file)])
        
        print(f"Created server '{server_name}' with {pdf_file.name}")

# Hot-reload support
async def trigger_registry_reload():
    """Trigger hot-reload of agent registry"""
    try:
        from ..core.agent_registry import agent_registry
        
        # Touch the registry file to trigger file watcher
        registry_file = Path(__file__).parent.parent / "core" / "agent_registry.py"
        registry_file.touch()
        
        # Or directly reload if available
        if hasattr(agent_registry, 'reload_all_agents'):
            await agent_registry.reload_all_agents()
            
    except Exception as e:
        logger.error(f"Failed to trigger reload: {e}")

if __name__ == "__main__":
    # Example usage
    async def main():
        processor = EnhancedPDFProcessor()
        
        sources = [
            {"url": "https://example.com/paper1.pdf", "tier": 1},
            {"path": "/local/paper2.pdf", "tier": 2}
        ]
        
        dest = Path("./test_agent")
        result = await processor.process_pdfs(sources, dest)
        
        print(f"Processed {len(result['pdfs'])} PDFs")
        print(f"Total characters: {result['total_chars']}")
    
    asyncio.run(main())
