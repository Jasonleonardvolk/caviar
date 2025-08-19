"""source_validator.py - Validate PDF source quality based on curated standards.

This module implements ALAN's source curation system, ensuring only high-quality,
formally structured content is ingested into the knowledge base. It supports the
"No Pretraining" commitment by filtering out undesirable content and tracking
document provenance.
"""

import re
from typing import Dict, Any, Tuple, List, Optional
import logging
import PyPDF2
from pathlib import Path

# Configure logger
logger = logging.getLogger("source_validator")

# Curated list of trusted academic sources and publishers
TRUSTED_DOMAINS = [
    'arxiv.org', 'ieee.org', 'acm.org', 'nature.com', 'science.org', 
    'springer.com', 'wiley.com', 'elsevier.com', 'mit.edu', 'stanford.edu',
    'berkeley.edu', 'harvard.edu', 'princeton.edu', 'caltech.edu',
    'ox.ac.uk', 'cam.ac.uk', 'ethz.ch', 'epfl.ch', 'toronto.edu',
    'nips.cc', 'icml.cc', 'cvpr.thecvf.com', 'iclr.cc'
]

# Preferred subject areas aligned with ALAN's knowledge domains
PRIORITY_SUBJECTS = [
    # Mathematics
    'mathematics', 'algebra', 'geometry', 'topology', 'analysis', 'calculus',
    # Physics
    'physics', 'quantum', 'mechanics', 'dynamics', 'relativity',
    # Computer Science
    'computer science', 'algorithm', 'data structure', 'computation', 'complexity',
    'machine learning', 'artificial intelligence', 'neural network',
    # Biology & Neuroscience
    'neuroscience', 'cognitive', 'biology', 'neuron', 'brain',
    # Engineering/Systems
    'control theory', 'system', 'engineering', 'cybernetics', 'robotics',
    # Philosophy of Science/Logic
    'logic', 'reasoning', 'philosophy of science', 'epistemology',
    # Formal specifications
    'specification', 'standard', 'protocol', 'formal method'
]

# Keywords that suggest informal or undesirable content
NEGATIVE_INDICATORS = [
    'blog', 'forum', 'comment', 'opinion', 'personal', 'news', 
    'advertisement', 'marketing', 'social media', 'tweet', 'facebook',
    'instagram', 'reddit', 'chat', 'gpt', 'chatgpt', 'llm',
    'clickbait', 'viral', 'trending', 'gossip'
]

class SourceValidationResult:
    """Container for source validation results with detailed quality metrics."""
    
    def __init__(self):
        self.is_valid: bool = False
        self.quality_score: float = 0.0
        self.domain_score: float = 0.0
        self.structure_score: float = 0.0
        self.content_score: float = 0.0
        self.reasons: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.source_type: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "domain_score": self.domain_score,
            "structure_score": self.structure_score,
            "content_score": self.content_score,
            "reasons": self.reasons,
            "source_type": self.source_type
        }

def extract_text_sample(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from the first few pages of a PDF for analysis."""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        pages = min(len(reader.pages), max_pages)
        text = ""
        for i in range(pages):
            page_text = reader.pages[i].extract_text() or ""
            text += page_text + "\n\n"
        return text
    except Exception as e:
        logger.warning(f"Error extracting text sample from {pdf_path}: {e}")
        return ""

def analyze_structure(text: str) -> Tuple[float, List[str]]:
    """Analyze document structure for formal academic patterns."""
    reasons = []
    score = 0.0
    
    # Check for abstract
    if re.search(r'abstract|summary', text.lower()):
        score += 0.2
        reasons.append("Contains abstract")
    
    # Check for references/bibliography
    if re.search(r'references|bibliography|citations', text.lower()):
        score += 0.2
        reasons.append("Contains references section")
    
    # Check for sections/headings
    if re.search(r'\n\s*\d+[\.\)]\s+[A-Z]|\n\s*[A-Z][a-z]+\s+\d+[\.\)]|\n\s*[IVXLCDM]+\.\s+[A-Z]', text):
        score += 0.2
        reasons.append("Contains structured sections")
    
    # Check for equations (simple heuristic)
    if re.search(r'[a-z]\s*=\s*[a-z0-9]|[a-z]\s*\([a-z]\)|[a-z]\s*\^\s*[0-9]', text):
        score += 0.2
        reasons.append("Contains equations")
    
    # Check for figures/tables references
    if re.search(r'(figure|fig\.|table)\s+\d+', text.lower()):
        score += 0.2
        reasons.append("Contains figure/table references")
    
    return min(1.0, score), reasons

def check_domain_quality(metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Check if document comes from trusted academic sources."""
    reasons = []
    score = 0.0
    
    # Check for trusted publishers in metadata
    publisher = metadata.get('publisher', '').lower()
    producer = metadata.get('producer', '').lower()
    author = metadata.get('author', '').lower()
    source_info = publisher + ' ' + producer + ' ' + author
    
    # Check for trusted domains
    for domain in TRUSTED_DOMAINS:
        if domain.lower() in source_info:
            score += 0.5
            reasons.append(f"Published by trusted source: {domain}")
            break
    
    # Look for academic identifiers
    if re.search(r'doi:|arxiv:|isbn:|issn:', source_info):
        score += 0.3
        reasons.append("Contains academic identifier")
    
    # Check for university or research institution affiliation
    if re.search(r'university|institute|laboratory|dept\.|department', source_info):
        score += 0.2
        reasons.append("Academic institution affiliation")
    
    return min(1.0, score), reasons

def analyze_content_quality(text: str) -> Tuple[float, str, float, List[str]]:
    """Analyze content for subject relevance and quality indicators."""
    text_lower = text.lower()
    reasons = []
    score = 0.0
    
    # Check for priority subjects
    subject_matches = []
    for subject in PRIORITY_SUBJECTS:
        if subject.lower() in text_lower:
            subject_matches.append(subject)
    
    subject_score = min(1.0, len(subject_matches) * 0.1)
    score += subject_score
    
    if subject_matches:
        reasons.append(f"Relevant subjects: {', '.join(subject_matches[:5])}")
    
    # Detect negative indicators
    negative_matches = []
    for indicator in NEGATIVE_INDICATORS:
        if indicator.lower() in text_lower:
            negative_matches.append(indicator)
    
    if negative_matches:
        penalty = min(0.8, len(negative_matches) * 0.2)
        score -= penalty
        reasons.append(f"Contains undesirable content types: {', '.join(negative_matches)}")
    
    # Determine document type
    doc_type = "unknown"
    if "arxiv" in text_lower or re.search(r'submitted to|proceedings of|journal of', text_lower):
        doc_type = "academic_paper"
    elif re.search(r'manual|guide|documentation|specification|standard|protocol', text_lower):
        doc_type = "technical_manual"
    elif re.search(r'chapter|textbook|course|lecture', text_lower):
        doc_type = "textbook"
    
    # Add score for formal document types
    if doc_type in ["academic_paper", "technical_manual", "textbook"]:
        score += 0.3
        reasons.append(f"Document type: {doc_type}")
    
    return min(1.0, max(0.0, score)), doc_type, subject_score, reasons

def validate_source(pdf_path: str, min_quality_score: float = None) -> SourceValidationResult:
    """
    Validate PDF source quality based on multiple criteria.
    
    Args:
        pdf_path: Path to the PDF file
        min_quality_score: Minimum overall quality score (0-1) required for validation
                           If None, uses SOURCE_VALIDATOR_MIN_SCORE from environment
                           
    Returns:
        SourceValidationResult object with validation details
    """
    # Get minimum score from environment if not provided
    if min_quality_score is None:
        import os
        min_quality_score = float(os.environ.get('SOURCE_VALIDATOR_MIN_SCORE', '0.6'))
    result = SourceValidationResult()
    
    try:
        # Extract metadata
        reader = PyPDF2.PdfReader(pdf_path)
        
        # Basic metadata
        metadata = {}
        if reader.metadata:
            for key, value in reader.metadata.items():
                if key and value:
                    clean_key = key.lower().replace('/', '')
                    metadata[clean_key] = str(value)
        
        # Add filename and path
        metadata["filename"] = Path(pdf_path).name
        metadata["path"] = pdf_path
        
        # Extract text sample for analysis
        text_sample = extract_text_sample(pdf_path)
        
        # Run analysis
        domain_score, domain_reasons = check_domain_quality(metadata)
        structure_score, structure_reasons = analyze_structure(text_sample)
        content_score, doc_type, subject_score, content_reasons = analyze_content_quality(text_sample)
        
        # Get weights from environment variables
        import os
        domain_weight = float(os.environ.get('SOURCE_VALIDATOR_DOMAIN_WEIGHT', '0.4'))
        structure_weight = float(os.environ.get('SOURCE_VALIDATOR_STRUCTURE_WEIGHT', '0.3'))
        content_weight = float(os.environ.get('SOURCE_VALIDATOR_CONTENT_WEIGHT', '0.3'))
        
        # Calculate overall quality score (weighted)
        quality_score = (
            domain_score * domain_weight +
            structure_score * structure_weight +
            content_score * content_weight
        )
        
        # Populate result
        result.is_valid = quality_score >= min_quality_score
        result.quality_score = quality_score
        result.domain_score = domain_score
        result.structure_score = structure_score
        result.content_score = content_score
        result.reasons = domain_reasons + structure_reasons + content_reasons
        result.metadata = metadata
        result.source_type = doc_type
        
        # Log validation result
        if result.is_valid:
            logger.info(f"Source validated: {pdf_path} (Score: {quality_score:.2f}, Type: {doc_type})")
        else:
            logger.warning(f"Source rejected: {pdf_path} (Score: {quality_score:.2f}, Type: {doc_type})")
            logger.debug(f"Rejection reasons: {', '.join(result.reasons)}")
            
    except Exception as e:
        logger.error(f"Error validating source {pdf_path}: {e}")
        result.is_valid = False
        result.reasons.append(f"Validation error: {str(e)}")
    
    return result

def batch_validate_sources(file_paths: List[str], min_quality_score: float = 0.6) -> Dict[str, SourceValidationResult]:
    """
    Validate multiple PDF sources.
    
    Args:
        file_paths: List of paths to PDF files
        min_quality_score: Minimum quality score threshold
    
    Returns:
        Dictionary mapping file paths to validation results
    """
    results = {}
    for path in file_paths:
        results[path] = validate_source(path, min_quality_score)
    
    # Log summary statistics
    valid_count = sum(1 for r in results.values() if r.is_valid)
    logger.info(f"Batch validation complete: {valid_count}/{len(results)} sources passed validation")
    
    return results
