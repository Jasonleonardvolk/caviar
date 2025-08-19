# core/concept_extractor_full.py - Extract more concepts from full text
import logging
from typing import List, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class Concept:
    text: str
    score: float = 1.0
    metadata: dict = None

class FullTextConceptExtractor:
    """Extract concepts from full text by chunking into sentences"""
    
    def __init__(self, max_concepts_per_chunk=100, min_sentence_length=20):
        self.max_concepts = max_concepts_per_chunk
        self.min_length = min_sentence_length
        logger.info(f"Full Text Concept Extractor initialized (max {self.max_concepts} concepts)")
    
    async def extract_concepts(self, content: str) -> List[Concept]:
        """Extract concepts by splitting into meaningful sentences"""
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', content)
        
        concepts = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences
            if len(sentence) < self.min_length:
                continue
                
            # Skip sentences that are mostly numbers or special chars
            if len(re.findall(r'[a-zA-Z]', sentence)) < len(sentence) * 0.5:
                continue
            
            # Add as concept
            concepts.append(Concept(
                text=sentence,
                score=0.8,
                metadata={"source": "sentence_split"}
            ))
            
            # Limit number of concepts
            if len(concepts) >= self.max_concepts:
                break
        
        logger.info(f"Extracted {len(concepts)} concepts from {len(content)} chars")
        return concepts
