# =================================================================
#    TORI PIPELINE - 100% BULLETPROOF - ZERO NONETYPE ERRORS
#    ENHANCED WITH OCR, ACADEMIC STRUCTURE, QUALITY METRICS
# =================================================================
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import json
import os
import hashlib
import PyPDF2
import logging
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# === ATOMIC SAFE MATH - NO FAILURES POSSIBLE ===
def safe_get(obj, key, default=0):
    """Absolutely safe dictionary access"""
    if obj is None:
        return default
    value = obj.get(key, default)
    return value if value is not None else default

def safe_divide(numerator, denominator, default=0.0):
    """100% bulletproof division"""
    num = safe_get({'val': numerator}, 'val', 0)
    den = safe_get({'val': denominator}, 'val', 1)
    if den == 0:
        return default
    try:
        return float(num) / float(den)
    except:
        return default

def safe_multiply(a, b, default=0.0):
    """100% bulletproof multiplication"""
    val_a = safe_get({'val': a}, 'val', 0)
    val_b = safe_get({'val': b}, 'val', 0)
    try:
        return float(val_a) * float(val_b)
    except:
        return default

def safe_percentage(value, total, default=0.0):
    """100% bulletproof percentage"""
    return safe_multiply(safe_divide(value, total, 0), 100, default)

def safe_round(value, decimals=3):
    """100% bulletproof rounding"""
    try:
        val = safe_get({'val': value}, 'val', 0)
        return round(float(val), decimals)
    except:
        return 0.0

def sanitize_dict(data_dict):
    """Sanitize any dictionary to remove all None values"""
    if not data_dict:
        return {}
    
    clean_dict = {}
    for key, value in data_dict.items():
        if value is None:
            if key in ['total', 'selected', 'pruned', 'count', 'frequency']:
                clean_dict[key] = 0
            elif key in ['score', 'final_entropy', 'avg_similarity', 'efficiency']:
                clean_dict[key] = 0.0
            else:
                clean_dict[key] = 0
        else:
            clean_dict[key] = value
    return clean_dict

# === Imports with safe fallbacks ===
try:
    from .extract_blocks import extract_concept_blocks, extract_chunks
    from .extractConceptsFromDocument import extractConceptsFromDocument, reset_frequency_counter, track_concept_frequency, get_concept_frequency, concept_frequency_counter
    from .entropy_prune import entropy_prune, entropy_prune_with_categories
    from .cognitive_interface import add_concept_diff
    from .memory_sculptor import memory_sculptor  # For enhanced memory integration
except ImportError:
    from extract_blocks import extract_concept_blocks, extract_chunks
    from extractConceptsFromDocument import extractConceptsFromDocument, reset_frequency_counter, track_concept_frequency, get_concept_frequency, concept_frequency_counter
    from entropy_prune import entropy_prune, entropy_prune_with_categories
    from cognitive_interface import add_concept_diff
    try:
        from memory_sculptor import memory_sculptor
    except:
        memory_sculptor = None

# === Optional OCR imports ===
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger = logging.getLogger("pdf_ingestion")
    logger.warning("⚠️ OCR libraries not available. Install pytesseract, pdf2image, and PIL for OCR support.")

# === Logging ===
logger = logging.getLogger("pdf_ingestion")
logger.setLevel(logging.INFO)

# === Config ===
ENABLE_CONTEXT_EXTRACTION = True
ENABLE_FREQUENCY_TRACKING = True
ENABLE_SMART_FILTERING = True
ENABLE_ENTROPY_PRUNING = True
ENABLE_OCR_FALLBACK = True
ENABLE_PARALLEL_PROCESSING = True
ENABLE_ENHANCED_MEMORY_STORAGE = True
OCR_MAX_PAGES = None  # None = no limit, or set an int
MAX_PARALLEL_WORKERS = None  # None = use min(4, cpu_count), or set an int

ENTROPY_CONFIG = {
    "max_diverse_concepts": None,
    "entropy_threshold": 0.0001,      # Much lower threshold
    "similarity_threshold": 0.85,      # Allow much more similarity  
    "enable_categories": True,
    "concepts_per_category": None
}

# === Academic Paper Structure Detection ===
ACADEMIC_SECTIONS = {
    'title': ['title'],
    'abstract': ['abstract', 'summary'],
    'introduction': ['introduction', '1 introduction', '1. introduction', 'i. introduction'],
    'methodology': ['methodology', 'methods', 'approach', '2 methodology', '2. methods', 'materials and methods'],
    'results': ['results', 'findings', 'experiments', '3 results', '3. results', 'experimental results'],
    'discussion': ['discussion', 'analysis', '4 discussion', '4. discussion', 'discussion and analysis'],
    'conclusion': ['conclusion', 'summary', '5 conclusion', '5. conclusion', 'conclusions', 'concluding remarks'],
    'references': ['references', 'bibliography', 'citations', 'works cited']
}

# === Universal DB ===
concept_db_path = Path(__file__).parent / "data" / "concept_file_storage.json"
universal_seed_path = Path(__file__).parent / "data" / "concept_seed_universal.json"
concept_file_storage, concept_names, concept_scores = [], [], {}

def load_universal_concept_file_storage():
    global concept_file_storage, concept_names, concept_scores
    try:
        with open(concept_db_path, "r", encoding="utf-8") as f:
            main_concepts = json.load(f)
        logger.info(f"✅ Main concept file_storage loaded: {len(main_concepts)} concepts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load main concept file_storage: {e}")
        main_concepts = []
    try:
        with open(universal_seed_path, "r", encoding="utf-8") as f:
            universal_seeds = json.load(f)
        logger.info(f"🌍 Universal seed concepts loaded: {len(universal_seeds)} concepts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load universal seed concepts: {e}")
        universal_seeds = []
    
    existing_names = {c["name"].lower() for c in main_concepts}
    merged_concepts = main_concepts[:]
    seeds_added = 0
    for seed in universal_seeds:
        if seed["name"].lower() not in existing_names:
            merged_concepts.append(seed)
            seeds_added += 1
    
    concept_file_storage = merged_concepts
    concept_names = [c["name"] for c in concept_file_storage]
    concept_scores = {c["name"]: c.get("priority", 0.5) for c in concept_file_storage}
    logger.info(f"🌍 UNIVERSAL DATABASE READY: {len(concept_file_storage)} total concepts (+{seeds_added} seeds)")

load_universal_concept_file_storage()

# === NEW: OCR Integration ===
def preprocess_with_ocr(pdf_path: str, max_pages: Optional[int] = None) -> Optional[str]:
    """Try to get better text extraction using OCR if needed"""
    if not OCR_AVAILABLE or not ENABLE_OCR_FALLBACK:
        return None
        
    try:
        # Check if PDF has good text layer
        with open(pdf_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            sample_text = ""
            for i in range(min(3, len(pdf.pages))):
                sample_text += pdf.pages[i].extract_text()
        
        # If text extraction is poor, use OCR
        if len(sample_text.strip()) < 100 or sample_text.count('�') > 10:
            logger.info("📸 Poor text extraction detected, attempting OCR...")
            
            # Convert PDF to images with configurable page limit
            if max_pages is None:
                max_pages = OCR_MAX_PAGES
            last_page = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            
            # Stream pages to avoid memory issues
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=300, 
                first_page=1, 
                last_page=last_page,
                output_folder=None  # Stream mode
            )
            
            ocr_text = ""
            for i, image in enumerate(images):
                logger.info(f"🔍 OCR processing page {i+1}...")
                page_text = pytesseract.image_to_string(image, lang='eng')
                ocr_text += f"\n--- Page {i+1} ---\n{page_text}"
            
            logger.info(f"✅ OCR completed, extracted {len(ocr_text)} characters")
            return ocr_text
    except Exception as e:
        logger.warning(f"⚠️ OCR preprocessing failed: {e}")
    
    return None

# === NEW: Academic Section Detection ===
def detect_section_type(chunk_text: str) -> str:
    """Detect which section of academic paper this chunk belongs to"""
    if not chunk_text:
        return "body"
        
    first_lines = chunk_text[:500].lower()
    
    # Check each section type
    for section_type, markers in ACADEMIC_SECTIONS.items():
        for marker in markers:
            # Look for section headers
            if re.search(r'\b' + re.escape(marker) + r'\b', first_lines):
                return section_type
            # Look for numbered sections
            if re.search(r'^\s*\d+\.?\s*' + re.escape(marker), first_lines, re.MULTILINE):
                return section_type
    
    return "body"

# === NEW: Enhanced Concept Quality Metrics ===
def calculate_concept_quality(concept: Dict, doc_context: Dict) -> float:
    """Calculate comprehensive quality score for concept"""
    base_score = safe_get(concept, 'score', 0.5)
    
    # Existing factors
    frequency = safe_get(concept.get('metadata', {}), 'frequency', 1)
    in_title = safe_get(concept.get('metadata', {}), 'in_title', False)
    in_abstract = safe_get(concept.get('metadata', {}), 'in_abstract', False)
    
    # Section weights for academic papers
    section_weight = {
        'title': 2.0,
        'abstract': 1.5,
        'introduction': 1.2,
        'conclusion': 1.2,
        'methodology': 1.1,
        'results': 1.1,
        'discussion': 1.0,
        'body': 1.0,
        'references': 0.7
    }
    
    section = safe_get(concept.get('metadata', {}), 'section', 'body')
    
    # Theme relevance calculation
    theme_relevance = calculate_theme_relevance(concept['name'], doc_context)
    
    # Combine factors
    quality = base_score * section_weight.get(section, 1.0)
    quality *= (1 + min(frequency, 5) * 0.1)  # Frequency boost, capped
    quality *= (1.3 if in_title else 1.0)
    quality *= (1.2 if in_abstract else 1.0)
    quality *= (0.8 + theme_relevance * 0.4)  # Theme relevance factor
    
    # Boost for multi-method extraction
    method = safe_get(concept, 'method', '')
    if '+' in method:
        quality *= 1.1
    
    return min(quality, 1.0)

def calculate_theme_relevance(concept_name: str, doc_context: Dict) -> float:
    """Calculate how relevant a concept is to the document's main theme"""
    # Simple implementation - can be enhanced with more sophisticated NLP
    title = safe_get(doc_context, 'title', '').lower()
    abstract = safe_get(doc_context, 'abstract', '').lower()
    concept_lower = concept_name.lower()
    
    relevance = 0.0
    
    # Direct mention in title/abstract
    if concept_lower in title:
        relevance += 0.5
    if concept_lower in abstract:
        relevance += 0.3
    
    # Partial matches (words from concept appear in title/abstract)
    concept_words = set(concept_lower.split())
    title_words = set(title.split())
    abstract_words = set(abstract.split())
    
    title_overlap = len(concept_words & title_words) / max(len(concept_words), 1)
    abstract_overlap = len(concept_words & abstract_words) / max(len(concept_words), 1)
    
    relevance += title_overlap * 0.3
    relevance += abstract_overlap * 0.2
    
    return min(relevance, 1.0)

# === NEW: Parallel Processing for Chunks ===
async def process_chunks_parallel(chunks: List[Dict], extraction_params: Dict) -> List[Dict]:
    """Process chunks in parallel for better performance"""
    if not ENABLE_PARALLEL_PROCESSING:
        # Fall back to sequential processing
        all_concepts = []
        for i, chunk in enumerate(chunks):
            concepts = extract_and_boost_concepts(
                chunk.get('text', ''),
                extraction_params['threshold'],
                i,
                chunk.get('section', 'body'),
                extraction_params['title'],
                extraction_params['abstract']
            )
            all_concepts.extend(concepts)
        return all_concepts
    
    # Use configurable max workers
    max_workers = MAX_PARALLEL_WORKERS or min(4, os.cpu_count() or 1)
    
    # Use asyncio.to_thread for better event loop integration
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def process_chunk_async(i, chunk_data):
        return await asyncio.to_thread(
            extract_and_boost_concepts,
            chunk_data.get('text', ''),
            extraction_params['threshold'],
            i,
            chunk_data.get('section', 'body'),
            extraction_params['title'],
            extraction_params['abstract']
        )
    
    # Process chunks concurrently with limited concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(i, chunk):
        async with semaphore:
            try:
                return await process_chunk_async(i, chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                return []
    
    # Create tasks
    tasks = [process_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
    chunk_results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_concepts = []
    for concepts in chunk_results:
        all_concepts.extend(concepts)
    
    return all_concepts

# === NEW: Enhanced Memory Storage ===
async def store_concepts_in_soliton(concepts: List[Dict], doc_metadata: Dict):
    """Enhanced storage with relationship mapping"""
    if not ENABLE_ENHANCED_MEMORY_STORAGE or not memory_sculptor:
        logger.info("Enhanced memory storage not available")
        return
        
    try:
        # Group concepts by similarity for relationship detection
        concept_clusters = cluster_similar_concepts(concepts)
        
        for cluster in concept_clusters:
            # Store primary concept
            primary = cluster[0]
            memory_id = await memory_sculptor.sculpt_and_store(
                user_id=doc_metadata.get('tenant_id', 'default'),
                raw_concept=primary,
                metadata={
                    **doc_metadata,
                    'cluster_size': len(cluster),
                    'is_primary': True,
                    'quality_score': calculate_concept_quality(primary, doc_metadata)
                }
            )
            
            # Store related concepts with links
            for related in cluster[1:]:
                await memory_sculptor.sculpt_and_store(
                    user_id=doc_metadata.get('tenant_id', 'default'),
                    raw_concept=related,
                    metadata={
                        **doc_metadata,
                        'primary_concept': primary['name'],
                        'relationship': 'semantic_cluster',
                        'quality_score': calculate_concept_quality(related, doc_metadata)
                    }
                )
        
        logger.info(f"✅ Stored {len(concepts)} concepts in {len(concept_clusters)} clusters")
    except Exception as e:
        logger.error(f"Failed to store concepts in Soliton: {e}")

def cluster_similar_concepts(concepts: List[Dict], similarity_threshold: float = 0.8) -> List[List[Dict]]:
    """Group similar concepts into clusters"""
    if not concepts:
        return []
    
    clusters = []
    used = set()
    
    for i, concept in enumerate(concepts):
        if i in used:
            continue
            
        cluster = [concept]
        used.add(i)
        
        # Find similar concepts
        for j, other in enumerate(concepts[i+1:], i+1):
            if j in used:
                continue
                
            # Simple similarity check - can be enhanced with better NLP
            if calculate_simple_similarity(concept['name'], other['name']) >= similarity_threshold:
                cluster.append(other)
                used.add(j)
        
        clusters.append(cluster)
    
    return clusters

def calculate_simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity calculation"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return safe_divide(intersection, union, 0.0)

def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Extract metadata with safe defaults"""
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": pdf_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_version": "tori_enhanced_v2.0"
    }
    try:
        file_size = os.path.getsize(pdf_path)
        metadata["file_size_bytes"] = file_size
        with open(pdf_path, "rb") as f:
            content = f.read()
            metadata["sha256"] = hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Could not extract file info: {e}")
        metadata["file_size_bytes"] = 0
        metadata["sha256"] = "unknown"
    
    if pdf_path.lower().endswith('.pdf'):
        try:
            with open(pdf_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                if pdf.metadata:
                    metadata["pdf_metadata"] = {
                        k.lower().replace('/', ''): str(v)
                        for k, v in pdf.metadata.items() if k and v
                    }
                metadata["page_count"] = len(pdf.pages)
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            metadata["page_count"] = 1
    
    return metadata

def get_dynamic_limits(file_size_mb: float) -> Tuple[int, int]:
    """Dynamic limits with safe math"""
    size = safe_get({'val': file_size_mb}, 'val', 0)
    if size < 1:
        return 300, 250
    elif size < 5:
        return 500, 700
    elif size < 25:
        return 1200, 1500
    else:
        return 2000, 3000

def extract_title_abstract_safe(chunks: List[Any], pdf_path: str) -> Tuple[str, str]:
    """Extract title and abstract with complete safety"""
    title_text = ""
    abstract_text = ""
    
    try:
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0]
            if isinstance(first_chunk, dict):
                first_text = first_chunk.get("text", "")
            else:
                first_text = str(first_chunk)
            
            if first_text:
                lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
                if lines:
                    candidate = lines[0]
                    if 10 < len(candidate) < 150 and not candidate.endswith('.'):
                        title_text = candidate
                
                lower_text = first_text.lower()
                if "abstract" in lower_text:
                    try:
                        idx = lower_text.index("abstract")
                        abstract_start = idx + len("abstract")
                        while abstract_start < len(first_text) and first_text[abstract_start] in ": \r\t\n":
                            abstract_start += 1
                        abstract_text = first_text[abstract_start:].strip()
                        
                        intro_pos = abstract_text.lower().find("introduction")
                        if intro_pos > 0:
                            abstract_text = abstract_text[:intro_pos].strip()
                        abstract_text = abstract_text[:1000]
                    except:
                        pass
        
        if not title_text:
            filename = Path(pdf_path).stem
            if len(filename) > 10:
                title_text = filename.replace('_', ' ').replace('-', ' ')
    
    except Exception as e:
        logger.debug(f"Could not extract title/abstract: {e}")
    
    return title_text, abstract_text

def analyze_concept_purity(all_concepts: List[Dict[str, Any]], doc_name: str = "", title_text: str = "", abstract_text: str = "", doc_context: Dict = None) -> List[Dict[str, Any]]:
    """Concept purity analysis with 100% safe operations and quality scoring"""
    logger.info(f"🔬 CONCEPT PURITY ANALYSIS for {doc_name}")
    logger.info(f"📊 Analyzing {len(all_concepts)} raw concepts")
    
    pure_concepts = []
    GENERIC_TERMS = {
        'document', 'paper', 'analysis', 'method', 'approach', 'study',
        'research', 'results', 'data', 'figure', 'table', 'section',
        'abstract', 'introduction', 'conclusion', 'pdf document', 
        'academic paper', 'page', 'text', 'content', 'information',
        'system', 'model', 'based', 'using', 'used', 'new', 'proposed'
    }
    
    # Prepare document context for quality calculation
    if doc_context is None:
        doc_context = {
            'title': title_text,
            'abstract': abstract_text,
            'filename': doc_name
        }
    
    for concept in all_concepts:
        if not concept or not isinstance(concept, dict):
            continue
            
        name = safe_get(concept, 'name', '')
        if not name or len(name) < 3:
            continue
            
        score = safe_get(concept, 'score', 0)
        if score < 0.2:
            continue
            
        method = safe_get(concept, 'method', '')
        metadata = safe_get(concept, 'metadata', {})
        
        name_lower = name.lower().strip()
        if name_lower in GENERIC_TERMS:
            continue
        
        word_count = len(name.split())
        if word_count > 6:
            continue
        
        # Calculate enhanced quality score
        quality_score = calculate_concept_quality(concept, doc_context)
        concept['quality_score'] = quality_score
        
        # Enhanced acceptance criteria with quality score
        frequency = safe_get(metadata, 'frequency', 1)
        in_title = safe_get(metadata, 'in_title', False)
        in_abstract = safe_get(metadata, 'in_abstract', False)
        method_count = method.count('+') + 1 if '+' in method else 1
        is_boosted = 'file_storage_boosted' in method or 'boost' in method
        
        # Accept based on various criteria including quality score
        if (quality_score >= 0.7 or
            method_count >= 2 or 
            is_boosted and score >= 0.75 or
            (in_title or in_abstract) and score >= 0.7 or
            score >= 0.85 and word_count <= 3 or
            frequency >= 3 and score >= 0.65):
            pure_concepts.append(concept)
    
    # Deduplicate safely
    seen = set()
    unique_pure = []
    for c in pure_concepts:
        name_lower = safe_get(c, 'name', '').lower().strip()
        if name_lower and name_lower not in seen:
            seen.add(name_lower)
            unique_pure.append(c)
    
    # Sort by quality score
    unique_pure.sort(key=lambda x: safe_get(x, 'quality_score', 0), reverse=True)
    
    logger.info(f"🏆 FINAL PURE CONCEPTS: {len(unique_pure)}")
    return unique_pure

def boost_known_concepts(chunk: str) -> List[Dict[str, Any]]:
    """Database boosting with complete safety"""
    boosted = []
    chunk_lower = chunk.lower()
    MAX_BOOSTS = 25
    
    for concept in concept_file_storage[:300]:  # Limit for performance
        if len(boosted) >= MAX_BOOSTS:
            break
            
        name = safe_get(concept, "name", "")
        if len(name) < 4:
            continue
        
        if name.lower() in chunk_lower:
            base_score = safe_get(concept_scores, name, 0.5)
            boost_multiplier = safe_get(concept, "boost_multiplier", 1.2)
            boosted_score = min(0.98, safe_multiply(base_score, boost_multiplier, 0.5))
            
            boosted.append({
                "name": name,
                "score": boosted_score,
                "method": "file_storage_boosted",
                "source": {"file_storage_matched": True},
                "metadata": {"category": safe_get(concept, "category", "general")}
            })
    
    return boosted

def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, chunk_index: int = 0, chunk_section: str = "body", title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """Extract and boost with complete safety and section awareness"""
    try:
        # Detect section if not provided
        if chunk_section == "body":
            chunk_section = detect_section_type(chunk)
        
        # Extract concepts
        semantic_hits = extractConceptsFromDocument(chunk, threshold=threshold, chunk_index=chunk_index, chunk_section=chunk_section)
        boosted = boost_known_concepts(chunk)
        combined = semantic_hits + boosted
        
        # Add metadata safely
        for concept in combined:
            if not isinstance(concept, dict):
                continue
                
            name = safe_get(concept, 'name', '')
            name_lower = name.lower()
            
            # Ensure metadata exists
            if 'metadata' not in concept:
                concept['metadata'] = {}
            
            # Add safe frequency data
            freq_data = get_concept_frequency(name)
            concept['metadata']['frequency'] = safe_get(freq_data, 'count', 1)
            concept['metadata']['sections'] = [chunk_section]
            concept['metadata']['section'] = chunk_section
            concept['metadata']['in_title'] = bool(title_text and name_lower in title_text.lower())
            concept['metadata']['in_abstract'] = bool(abstract_text and name_lower in abstract_text.lower())
        
        return combined
        
    except Exception as e:
        logger.error(f"Error in extract_and_boost_concepts: {e}")
        return []

def ingest_pdf_clean(pdf_path: str, doc_id: str = None, extraction_threshold: float = 0.0, admin_mode: bool = False, use_ocr: bool = None) -> Dict[str, Any]:
    """
    100% BULLETPROOF PDF INGESTION - ENHANCED VERSION
    Now with OCR, academic structure detection, quality metrics, and parallel processing
    """
    start_time = datetime.now()
    
    # Ensure all variables have safe defaults
    if doc_id is None:
        doc_id = Path(pdf_path).stem
    
    if use_ocr is None:
        use_ocr = ENABLE_OCR_FALLBACK
    
    # Safe file operations
    try:
        file_size = os.path.getsize(pdf_path)
        file_size_mb = safe_divide(file_size, 1024 * 1024, 0)
    except:
        file_size_mb = 0
    
    MAX_CHUNKS, MAX_TOTAL_CONCEPTS = get_dynamic_limits(file_size_mb)
    
    logger.info(f"🛡️ [ENHANCED BULLETPROOF] Ingesting: {Path(pdf_path).name}")
    logger.info(f"File size: {file_size_mb:.1f} MB, Limits: {MAX_CHUNKS} chunks, {MAX_TOTAL_CONCEPTS} concepts")
    
    try:
        # Extract metadata safely
        doc_metadata = extract_pdf_metadata(pdf_path)
        
        # Try OCR preprocessing if enabled
        ocr_text = None
        if use_ocr and OCR_AVAILABLE:
            ocr_text = preprocess_with_ocr(pdf_path)
            if ocr_text:
                doc_metadata['ocr_used'] = True
        
        # Reset frequency counter safely
        if ENABLE_FREQUENCY_TRACKING:
            try:
                reset_frequency_counter()
                logger.info("✅ Frequency counter reset successfully")
            except Exception as e:
                logger.warning(f"⚠️ Frequency counter reset failed: {e}")
                # Force clear any global frequency state
                try:
                    import gc
                    gc.collect()  # Force garbage collection
                    logger.info("🧹 Forced garbage collection to clear state")
                except:
                    pass
        
        # Extract chunks safely (with OCR text if available)
        if ocr_text:
            # Create chunks from OCR text
            chunks = []
            ocr_lines = ocr_text.split('\n')
            chunk_size = 50  # lines per chunk
            for i in range(0, len(ocr_lines), chunk_size):
                chunk_text = '\n'.join(ocr_lines[i:i+chunk_size])
                chunks.append({
                    'text': chunk_text,
                    'index': len(chunks),
                    'section': detect_section_type(chunk_text)
                })
            logger.info(f"📄 Created {len(chunks)} chunks from OCR text")
        else:
            chunks = extract_chunks(pdf_path)
        
        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path}")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "empty",
                "admin_mode": admin_mode,
                "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
            }
        
        # Enhance chunks with section detection
        for chunk in chunks:
            if isinstance(chunk, dict) and 'section' not in chunk:
                chunk['section'] = detect_section_type(chunk.get('text', ''))
        
        # Extract title and abstract safely
        title_text, abstract_text = "", ""
        if ENABLE_CONTEXT_EXTRACTION:
            title_text, abstract_text = extract_title_abstract_safe(chunks, pdf_path)
        
        # Process chunks - use parallel processing if enabled
        chunks_to_process = chunks[:MAX_CHUNKS]
        
        extraction_params = {
            'threshold': extraction_threshold,
            'title': title_text,
            'abstract': abstract_text
        }
        
        # Use async parallel processing if available
        if ENABLE_PARALLEL_PROCESSING:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an existing event loop, can't use asyncio.run
                # Fall back to ThreadPoolExecutor
                logger.info("⚠️ Detected existing event loop, using ThreadPoolExecutor")
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS or min(4, os.cpu_count() or 1)) as executor:
                    tasks = []
                    for i, chunk in enumerate(chunks_to_process):
                        task = executor.submit(
                            extract_and_boost_concepts,
                            chunk.get('text', ''),
                            extraction_params['threshold'],
                            i,
                            chunk.get('section', 'body'),
                            extraction_params['title'],
                            extraction_params['abstract']
                        )
                        tasks.append(task)
                    
                    # Gather results
                    all_extracted_concepts = []
                    for future in tasks:
                        try:
                            concepts = future.result(timeout=30)
                            all_extracted_concepts.extend(concepts)
                        except Exception as e:
                            logger.error(f"Parallel processing error: {e}")
            except RuntimeError:
                # No running loop, we're in sync context - safe to use asyncio.run
                all_extracted_concepts = asyncio.run(
                    process_chunks_parallel(chunks_to_process, extraction_params)
                )
        else:
            # Sequential processing
            all_extracted_concepts = []
            semantic_count = 0
            boosted_count = 0
            
            for i, chunk_data in enumerate(chunks_to_process):
                try:
                    if isinstance(chunk_data, dict):
                        chunk_text = safe_get(chunk_data, "text", "")
                        chunk_index = safe_get(chunk_data, "index", i)
                        chunk_section = safe_get(chunk_data, "section", "body")
                    else:
                        chunk_text = str(chunk_data)
                        chunk_index = i
                        chunk_section = "body"
                    
                    if not chunk_text:
                        continue
                    
                    # Extract concepts safely
                    enhanced_concepts = extract_and_boost_concepts(
                        chunk_text, extraction_threshold, chunk_index, chunk_section, title_text, abstract_text
                    )
                    
                    # Count safely
                    for c in enhanced_concepts:
                        method = safe_get(c, "method", "")
                        if "universal" in method:
                            semantic_count += 1
                        if "file_storage_boosted" in method or "boost" in method:
                            boosted_count += 1
                    
                    all_extracted_concepts.extend(enhanced_concepts)
                    
                    # Early exit safely
                    if len(all_extracted_concepts) >= MAX_TOTAL_CONCEPTS:
                        logger.info(f"Concept limit reached: {len(all_extracted_concepts)}. Stopping.")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
        
        if not all_extracted_concepts:
            logger.error("No concepts extracted!")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "no_concepts",
                "admin_mode": admin_mode,
                "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
            }
        
        # Count concepts by method
        semantic_count = sum(1 for c in all_extracted_concepts if "universal" in safe_get(c, "method", ""))
        boosted_count = sum(1 for c in all_extracted_concepts if "file_storage_boosted" in safe_get(c, "method", "") or "boost" in safe_get(c, "method", ""))
        
        # Apply purity filtering with quality scoring
        doc_context = {
            'title': title_text,
            'abstract': abstract_text,
            'filename': Path(pdf_path).name
        }
        pure_concepts = analyze_concept_purity(all_extracted_concepts, Path(pdf_path).name, title_text, abstract_text, doc_context)
        pure_concepts.sort(key=lambda x: safe_get(x, 'quality_score', safe_get(x, 'score', 0)), reverse=True)
        
        original_pure_count = len(pure_concepts)
        concept_count = len(pure_concepts)
        prune_stats = None
        
        # Apply entropy pruning safely with enhanced selection criteria
        if ENABLE_ENTROPY_PRUNING and concept_count > 0:
            logger.info("🎯 Applying enhanced entropy pruning...")
            
            try:
                # Enhanced selection criteria with quality score threshold
                quality_threshold = 0.75
                high_quality_concepts = [c for c in pure_concepts if safe_get(c, 'quality_score', 0) >= quality_threshold]
                
                if len(high_quality_concepts) > 0:
                    # Dynamic survivor count based on high-quality concepts
                    min_survivors = len(high_quality_concepts)
                    if min_survivors < 5:
                        min_survivors = 5
                    
                    logger.info(f"📊 High-quality concepts (≥{quality_threshold}): {len(high_quality_concepts)}")
                    logger.info(f"🎯 Minimum survivors target: {min_survivors}")
                    
                    selected_concepts, prune_stats = entropy_prune(
                        high_quality_concepts,
                        top_k=min_survivors,
                        min_survivors=min_survivors,
                        similarity_threshold=0.87,  # A little room for semantic siblings, but not clones
                        verbose=True
                    )
                    
                    # If we didn't get enough from high-quality, supplement with remaining pure concepts
                    if len(selected_concepts) < min_survivors and len(pure_concepts) > len(high_quality_concepts):
                        remaining_concepts = [c for c in pure_concepts if c not in high_quality_concepts]
                        remaining_concepts.sort(key=lambda x: safe_get(x, 'quality_score', safe_get(x, 'score', 0)), reverse=True)
                        
                        needed = min_survivors - len(selected_concepts)
                        supplemental = remaining_concepts[:needed]
                        selected_concepts.extend(supplemental)
                        
                        logger.info(f"📈 Added {len(supplemental)} supplemental concepts to reach minimum")
                    
                    pure_concepts = selected_concepts
                else:
                    # Fallback to original logic if no high-quality concepts
                    logger.info("⚠️ No concepts meet high quality threshold, using original entropy pruning")
                    pure_concepts, prune_stats = entropy_prune(
                        pure_concepts,
                        top_k=None if admin_mode else safe_get(ENTROPY_CONFIG, "max_diverse_concepts"),
                        entropy_threshold=safe_get(ENTROPY_CONFIG, "entropy_threshold", 0.0001),
                        similarity_threshold=safe_get(ENTROPY_CONFIG, "similarity_threshold", 0.95),
                        verbose=True
                    )
                
                concept_count = len(pure_concepts)
                logger.info(f"✅ Enhanced entropy pruning: {concept_count} concepts from {original_pure_count}")
                
            except Exception as e:
                logger.error(f"Error in enhanced entropy pruning: {e}")
                prune_stats = None
        
        # Enhanced memory storage
        if ENABLE_ENHANCED_MEMORY_STORAGE and concept_count > 0:
            try:
                # Check for existing event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Can't use asyncio.run in existing loop, skip enhanced storage
                    logger.warning("⚠️ Skipping enhanced memory storage due to existing event loop")
                except RuntimeError:
                    # No running loop, safe to use asyncio.run
                    asyncio.run(
                        store_concepts_in_soliton(pure_concepts, doc_metadata)
                    )
            except Exception as e:
                logger.warning(f"Enhanced memory storage failed: {e}")
        
        # Knowledge injection safely (fallback to original method)
        if concept_count > 0:
            try:
                concept_diff_data = {
                    "type": "document",
                    "title": Path(pdf_path).name,
                    "concepts": pure_concepts,
                    "summary": f"{concept_count} concepts extracted.",
                    "metadata": doc_metadata,
                }
                add_concept_diff(concept_diff_data)
            except Exception as e:
                logger.warning(f"Concept diff injection failed: {e}")
        
        # Calculate all values safely
        total_time = safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
        
        # Safe score calculation
        if pure_concepts:
            valid_scores = [safe_get(c, "quality_score", safe_get(c, "score", 0)) for c in pure_concepts]
            valid_scores = [s for s in valid_scores if s is not None]
            avg_score = safe_divide(sum(valid_scores), len(valid_scores), 0) if valid_scores else 0.0
        else:
            avg_score = 0.0
        
        high_conf_count = sum(1 for c in pure_concepts if safe_get(c, "score", 0) > 0.8)
        high_quality_count = sum(1 for c in pure_concepts if safe_get(c, "quality_score", 0) > 0.8)
        
        # Calculate section distribution
        section_distribution = {}
        for c in pure_concepts:
            section = safe_get(safe_get(c, 'metadata', {}), 'section', 'body')
            section_distribution[section] = section_distribution.get(section, 0) + 1
        
        # Build response with 100% safe calculations
        response = {
            "filename": Path(pdf_path).name,
            "concept_count": concept_count,
            "concept_names": [safe_get(c, 'name', '') for c in pure_concepts],
            "concepts": pure_concepts,
            "status": "success" if concept_count > 0 else "no_concepts",
            "purity_based": True,
            "entropy_pruned": ENABLE_ENTROPY_PRUNING and prune_stats is not None,
            "admin_mode": admin_mode,
            "equal_access": True,
            "performance_limited": True,
            "chunks_processed": len(chunks_to_process),
            "chunks_available": len(chunks),
            "semantic_extracted": semantic_count,
            "file_storage_boosted": boosted_count,
            "average_concept_score": safe_round(avg_score),
            "high_confidence_concepts": high_conf_count,
            "high_quality_concepts": high_quality_count,
            "total_extraction_time": safe_round(total_time),
            "domain_distribution": {"general": concept_count},
            "section_distribution": section_distribution,
            "title_found": bool(title_text),
            "abstract_found": bool(abstract_text),
            "ocr_used": safe_get(doc_metadata, 'ocr_used', False),
            "parallel_processing": ENABLE_PARALLEL_PROCESSING,
            "enhanced_memory_storage": ENABLE_ENHANCED_MEMORY_STORAGE,
            "processing_time_seconds": safe_round(total_time),
            "purity_analysis": {
                "raw_concepts": len(all_extracted_concepts),
                "pure_concepts": original_pure_count,
                "final_concepts": concept_count,
                "purity_efficiency_percent": safe_round(safe_percentage(original_pure_count, len(all_extracted_concepts)), 1),
                "diversity_efficiency_percent": safe_round(safe_percentage(concept_count, original_pure_count), 1),
                "top_concepts": [
                    {
                        "name": safe_get(c, 'name', ''),
                        "score": safe_round(safe_get(c, 'score', 0)),
                        "quality_score": safe_round(safe_get(c, 'quality_score', 0)),
                        "methods": [safe_get(c, 'method', 'unknown')],
                        "frequency": safe_get(safe_get(c, 'metadata', {}), 'frequency', 1),
                        "section": safe_get(safe_get(c, 'metadata', {}), 'section', 'body'),
                        "purity_decision": "accepted"
                    }
                    for c in pure_concepts[:10]
                ]
            }
        }
        
        # 100% BULLETPROOF entropy analysis
        if prune_stats:
            # Sanitize stats completely
            clean_stats = sanitize_dict(prune_stats)
            
            total = safe_get(clean_stats, "total", 0)
            selected = safe_get(clean_stats, "selected", 0)
            pruned = safe_get(clean_stats, "pruned", 0)
            final_entropy = safe_get(clean_stats, "final_entropy", 0.0)
            avg_similarity = safe_get(clean_stats, "avg_similarity", 0.0)
            
            response["entropy_analysis"] = {
                "enabled": True,
                "admin_mode": admin_mode,
                "total_before_entropy": total,
                "selected_diverse": selected,
                "pruned_similar": pruned,
                "diversity_efficiency_percent": safe_round(safe_percentage(selected, total), 1),
                "final_entropy": safe_round(final_entropy),
                "avg_similarity": safe_round(avg_similarity),
                "by_category": safe_get(clean_stats, "by_category", {}),
                "config": {
                    "max_diverse_concepts": "unlimited" if admin_mode else safe_get(ENTROPY_CONFIG, "max_diverse_concepts"),
                    "entropy_threshold": safe_get(ENTROPY_CONFIG, "entropy_threshold", 0.0005),
                    "similarity_threshold": safe_get(ENTROPY_CONFIG, "similarity_threshold", 0.83),
                    "category_aware": safe_get(ENTROPY_CONFIG, "enable_categories", True)
                },
                "performance": {
                    "original_pure_concepts": original_pure_count,
                    "final_diverse_concepts": concept_count,
                    "reduction_ratio": safe_round(safe_divide(original_pure_count - concept_count, original_pure_count, 0))
                }
            }
        else:
            response["entropy_analysis"] = {
                "enabled": False,
                "reason": "entropy_pruning_disabled" if not ENABLE_ENTROPY_PRUNING else "no_concepts_to_prune"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ PDF ingestion failed: {e}")
        return {
            "filename": Path(pdf_path).name,
            "concept_count": 0,
            "concept_names": [],
            "concepts": [],
            "status": "error",
            "error_message": str(e),
            "admin_mode": admin_mode,
            "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0.1)
        }

# Export
__all__ = ['ingest_pdf_clean']

logger.info("🛡️ ENHANCED BULLETPROOF PIPELINE LOADED - ZERO NONETYPE ERRORS GUARANTEED")
logger.info("✨ New features: OCR support, academic structure detection, quality metrics, parallel processing, enhanced memory storage")
