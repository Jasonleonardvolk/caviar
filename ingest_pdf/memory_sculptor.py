"""
ingest_pdf/memory_sculptor.py

Sculpts incoming extracted knowledge into the Soliton Memory lattice,
ensuring proper formatting, enrichment, and integration with the wave-based
memory system instead of legacy storage mechanisms.
"""

import os
import re
import time
import logging
import asyncio
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set

# Import Soliton client
try:
    from core.soliton_client import SolitonClient
    SOLITON_IMPORT_SOURCE = "core"
except ImportError:
    try:
        from mcp_metacognitive.core.soliton_memory import SolitonMemoryClient as SolitonClient
        SOLITON_IMPORT_SOURCE = "mcp_metacognitive"
    except ImportError:
        try:
            # Last resort - import from relative path
            from ..core.soliton_client import SolitonClient
            SOLITON_IMPORT_SOURCE = "relative"
        except ImportError:
            logging.error("❌ CRITICAL: Could not import SolitonClient from any known location")
            raise RuntimeError("SolitonClient not available - cannot initialize memory sculptor")

# Import NLP utilities if available
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLP_AVAILABLE = True
    ENABLE_SENTIMENT = False  # Disabled by default for scientific text
    
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except:
            # Fallback to punkt if punkt_tab doesn't exist
            try:
                nltk.download('punkt')
            except:
                logger.warning("Could not download NLTK punkt data")
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    
except ImportError:
    logging.warning("⚠️ NLTK not available - will use basic text processing")
    NLP_AVAILABLE = False

# Setup logger
logger = logging.getLogger("memory_sculptor")

class MemorySculptor:
    """
    Sculpts extracted concepts into well-formed memories in the Soliton system,
    handling enrichment, segmentation, and proper integration with the 
    wave-based memory architecture.
    """
    
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the memory sculptor
        
        Args:
            api_url: Optional URL to the Soliton API. If not provided,
                    will use the default from environment variables.
        """
        self.api_url = api_url or os.environ.get("SOLITON_API_URL", "http://localhost:8002/api/soliton")
        self.client = SolitonClient(api_url=self.api_url)
        
        # Initialize NLP components if available
        if NLP_AVAILABLE and ENABLE_SENTIMENT:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info(f"🌊 MemorySculptor initialized (imported from {SOLITON_IMPORT_SOURCE})")
        logger.info(f"🧠 NLP enrichment: {'ENABLED' if NLP_AVAILABLE else 'DISABLED'}")
    
    async def sculpt_and_store(self, 
                              user_id: str, 
                              raw_concept: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None,
                              related_concepts: Optional[List[str]] = None) -> List[str]:
        """
        Sculpt a raw extracted concept into one or more refined memories,
        then store them in the Soliton Memory system.
        
        Args:
            user_id: ID of the user who owns the memory
            raw_concept: Raw concept data from extraction
            metadata: Optional additional metadata
            
        Returns:
            List of memory IDs that were created
        """
        # Validate user_id
        if not user_id or user_id == "default":
            # Only log once to avoid spam
            if not hasattr(self, '_warned_default_user'):
                logger.warning("⚠️ Skipping memory store - no valid user_id provided (default user). This warning will only show once.")
                self._warned_default_user = True
            return []
            
        # Ensure user is initialized in Soliton
        await self.client.initialize_user(user_id)
        
        # Extract basic concept properties
        concept_id = raw_concept.get('id', f"concept_{int(time.time())}")
        content = raw_concept.get('text', '')
        
        # Validate content
        if not content or not content.strip():
            logger.warning(f"⚠️ Skipping concept {concept_id} with empty content")
            return []
            
        concept_name = raw_concept.get('name', '')
        concept_metadata = raw_concept.get('metadata', {})
        score = raw_concept.get('score', 0.7)
        quality_score = raw_concept.get('quality_score', score)
        source_page = raw_concept.get('source_page', 0)
        
        # Merge metadata
        combined_metadata = {
            "source_page": source_page,
            "concept_score": score,
            "quality_score": quality_score,
            "concept_name": concept_name,
            "sculpted_at": datetime.now().isoformat(),
            "sculptor_version": "2.0.0",
            **concept_metadata
        }
        
        if metadata:
            combined_metadata.update(metadata)
        
        # Detect relationships if we have related concepts
        if related_concepts and concept_name:
            relationships = self.detect_concept_relationships(content, related_concepts)
            if concept_name in relationships:
                combined_metadata['related_concepts'] = relationships[concept_name]
            # Also check if this concept appears in others' relationships
            appears_in = []
            for other_concept, related_list in relationships.items():
                if concept_name in related_list:
                    appears_in.append(other_concept)
            if appears_in:
                combined_metadata['referenced_by'] = appears_in
        elif related_concepts and not concept_name:
            logger.warning(f"⚠️ Cannot resolve relationships for unnamed concept: {concept_id}")
        
        # Set default tags
        tags = ["ingested", "pdf", "sculpted"]
        
        # Add confidence tags
        if score > 0.8:
            tags.append("high_confidence")
        elif score < 0.4:
            tags.append("low_confidence")
        
        # Check if content should be segmented
        segments = self._segment_if_needed(content)
        memory_ids = []
        
        if len(segments) == 1:
            # Single segment - just enrich and store
            enriched_content, enriched_metadata, enriched_tags = self._enrich_content(
                segments[0], combined_metadata, tags
            )
            
            # Final validation before storing
            if not enriched_content.strip():
                logger.warning(f"⚠️ Enriched content is empty for concept {concept_id}, skipping storage")
                return []
            
            # Calculate memory strength based on concept score, quality score, and enrichment
            # Use quality score if available, otherwise fall back to concept score
            base_score = quality_score if quality_score > 0 else score
            strength = min(1.0, max(0.3, base_score * 0.8 + 0.1))  # Base on quality/concept score
            
            # Adjust strength based on sentiment if available
            if 'sentiment_score' in enriched_metadata:
                # Higher strength for more emotional content
                sentiment_abs = abs(enriched_metadata['sentiment_score'])
                strength = min(1.0, strength + (sentiment_abs * 0.1))
            
            # Generate unique memory ID
            memory_id = f"sculpted_{concept_id}_{int(time.time())}"
            
            # Log parameters for debugging
            logger.debug(f"Storing memory with params: user_id={user_id}, memory_id={memory_id}, content_length={len(enriched_content)}, strength={strength}")
            
            # Store in Soliton Memory
            success = await self.client.store_memory(
                user_id=user_id,
                memory_id=memory_id,
                content=enriched_content,
                strength=strength,
                tags=enriched_tags,
                metadata=enriched_metadata
            )
            
            if success:
                logger.info(f"✅ Stored sculpted memory {memory_id} for user {user_id}")
                memory_ids.append(memory_id)
            else:
                logger.warning(f"⚠️ Failed to store sculpted memory for user {user_id}")
        
        else:
            # Multiple segments - process each one
            for i, segment in enumerate(segments):
                # Validate segment content
                if not segment.strip():
                    logger.warning(f"⚠️ Skipping empty segment {i} for concept {concept_id}")
                    continue
                    
                # Segment-specific metadata
                segment_metadata = {
                    **combined_metadata,
                    "segment_index": i,
                    "segment_count": len(segments),
                    "parent_concept_id": concept_id
                }
                
                # Segment-specific tags
                segment_tags = tags + ["segmented"]
                
                # Enrich the segment
                enriched_content, enriched_metadata, enriched_tags = self._enrich_content(
                    segment, segment_metadata, segment_tags
                )
                
                # Final validation before storing
                if not enriched_content.strip():
                    logger.warning(f"⚠️ Enriched segment {i} is empty for concept {concept_id}, skipping")
                    continue
                
                # Calculate memory strength for this segment
                segment_strength = min(1.0, max(0.3, score * 0.7 + 0.1))  # Slightly lower base than main concept
                
                # Adjust strength based on sentiment if available
                if 'sentiment_score' in enriched_metadata:
                    sentiment_abs = abs(enriched_metadata['sentiment_score'])
                    segment_strength = min(1.0, segment_strength + (sentiment_abs * 0.1))
                
                # Generate unique memory ID for segment
                segment_memory_id = f"sculpted_{concept_id}_seg{i}_{int(time.time())}"
                
                # Log parameters for debugging
                logger.debug(f"Storing segment with params: user_id={user_id}, memory_id={segment_memory_id}, content_length={len(enriched_content)}, strength={segment_strength}")
                
                # Store segment in Soliton Memory
                success = await self.client.store_memory(
                    user_id=user_id,
                    memory_id=segment_memory_id,
                    content=enriched_content,
                    strength=segment_strength,
                    tags=enriched_tags,
                    metadata=enriched_metadata
                )
                
                if success:
                    logger.info(f"✅ Stored sculpted segment {segment_memory_id} for user {user_id}")
                    memory_ids.append(segment_memory_id)
                else:
                    logger.warning(f"⚠️ Failed to store sculpted segment for user {user_id}")
        
        logger.info(f"🌊 Sculpted and stored {len(memory_ids)}/{len(segments)} memories for concept {concept_id}")
        return memory_ids
    
    def _segment_if_needed(self, content: str) -> List[str]:
        """
        Segment content into smaller pieces if it's too long or contains
        natural breaking points.
        
        Args:
            content: The text content to potentially segment
            
        Returns:
            List of segments (may be just the original content)
        """
        # If content is short, don't segment
        if len(content) < 500:
            return [content]
        
        # Try to use NLTK for better segmentation if available
        if NLP_AVAILABLE:
            try:
                # Split into sentences
                sentences = sent_tokenize(content)
                
                # If only a few sentences, don't segment
                if len(sentences) <= 3:
                    return [content]
                
                # Group sentences into segments of ~300-500 chars each
                segments = []
                current_segment = []
                current_length = 0
                
                for sentence in sentences:
                    # If adding this sentence would make segment too long, start a new one
                    if current_length + len(sentence) > 500 and current_length > 300:
                        segments.append(' '.join(current_segment))
                        current_segment = [sentence]
                        current_length = len(sentence)
                    else:
                        current_segment.append(sentence)
                        current_length += len(sentence)
                
                # Add the last segment if it's not empty
                if current_segment:
                    segments.append(' '.join(current_segment))
                
                return segments
                
            except Exception as e:
                logger.warning(f"⚠️ Error in NLTK segmentation: {str(e)} - falling back to basic")
        
        # Basic segmentation - split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # If only one paragraph or very short, don't segment
        if len(paragraphs) <= 1 or len(content) < 800:
            return [content]
        
        return paragraphs
    
    def _enrich_content(self, 
                        content: str, 
                        metadata: Dict[str, Any], 
                        tags: List[str]) -> Tuple[str, Dict[str, Any], List[str]]:
        """
        Enrich content with additional metadata and tags based on analysis
        
        Args:
            content: The text content to enrich
            metadata: Existing metadata
            tags: Existing tags
            
        Returns:
            Tuple of (enriched_content, enriched_metadata, enriched_tags)
        """
        enriched_metadata = dict(metadata)  # Copy to avoid modifying original
        enriched_tags = list(tags)  # Copy to avoid modifying original
        
        # Analyze content using NLP if available
        if NLP_AVAILABLE:
            try:
                # Sentiment analysis (disabled by default for scientific text)
                if ENABLE_SENTIMENT and hasattr(self, 'sentiment_analyzer'):
                    sentiment = self.sentiment_analyzer.polarity_scores(content)
                    enriched_metadata['sentiment'] = sentiment
                    enriched_metadata['sentiment_score'] = sentiment['compound']
                    
                    # Tag based on sentiment
                    if sentiment['compound'] > 0.3:
                        enriched_tags.append('positive')
                    elif sentiment['compound'] < -0.3:
                        enriched_tags.append('negative')
                
                # Extract key entities - use advanced extraction
                entities = self._extract_advanced_entities(content)
                if entities:
                    enriched_metadata['entities'] = entities
                    enriched_metadata['entity_count'] = len(entities)
                    
                    # Add entity types as tags
                    entity_types = set(e['type'] for e in entities)
                    for entity_type in entity_types:
                        enriched_tags.append(f"entity_{entity_type}")
                    
                    # Add special tags for academic content
                    if 'citation' in entity_types:
                        enriched_tags.append('academic')
                    if 'math' in entity_types:
                        enriched_tags.append('technical')
                
                # Calculate readability (simplified)
                word_count = len(content.split())
                sentence_count = max(1, len(sent_tokenize(content)))
                avg_words_per_sentence = word_count / sentence_count
                
                enriched_metadata['readability'] = {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_words_per_sentence': avg_words_per_sentence
                }
                
                # Tag based on complexity
                if avg_words_per_sentence > 25:
                    enriched_tags.append('complex')
                elif avg_words_per_sentence < 10:
                    enriched_tags.append('simple')
                
            except Exception as e:
                logger.warning(f"⚠️ Error in NLP enrichment: {str(e)}")
        
        # Extract potential topic labels (simplified)
        potential_topics = self._extract_topic_labels(content)
        if potential_topics:
            enriched_metadata['potential_topics'] = potential_topics
            
            # Add top topics as tags
            for topic in potential_topics[:3]:  # Just the top 3
                # Clean up topic for tag (lowercase, no spaces, etc.)
                topic_tag = re.sub(r'[^a-z0-9]', '_', topic.lower())
                topic_tag = f"topic_{topic_tag}"
                enriched_tags.append(topic_tag)
        
        # Detect languages (simplified)
        lang = self._detect_language(content)
        if lang != 'en':
            enriched_metadata['language'] = lang
            enriched_tags.append(f"lang_{lang}")
        
        # Entity Phase Binding Logic
        # Check if this content has a Wikidata ID for phase locking
        if 'wikidata_id' in metadata:
            wikidata_id = metadata['wikidata_id']
            
            # Extract numeric ID from Wikidata format (e.g., Q12345 -> 12345)
            numeric_match = re.match(r'Q(\d+)', str(wikidata_id))
            if numeric_match:
                numeric_id = int(numeric_match.group(1))
                
                # Golden ratio φ (phi)
                phi = 1.618033988749895
                
                # Apply golden-ratio-based phase mapping
                # θ_entity = (2π * numeric_id / φ) mod 2π
                entity_phase = (2 * math.pi * numeric_id / phi) % (2 * math.pi)
                
                # Store phase information in metadata
                enriched_metadata['entity_phase'] = entity_phase
                enriched_metadata['phase_locked'] = True
                enriched_metadata['kb_id'] = wikidata_id
                
                # Add knowledge base tag
                enriched_tags.append(f'kb_{wikidata_id}')
                enriched_tags.append('phase_locked')
                enriched_tags.append('entity_linked')
                
                logger.info(f"🔗 Phase locked entity {wikidata_id} with phase θ={entity_phase:.4f}")
            else:
                logger.warning(f"⚠️ Invalid Wikidata ID format: {wikidata_id}")
        
        # Deduplicate tags
        enriched_tags = list(set(enriched_tags))
        
        return content, enriched_metadata, enriched_tags
    
    def _extract_simple_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract simple entities from text using pattern matching
        
        Args:
            text: The text to analyze
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Very simple pattern matching - in a real system, use a proper NER
        
        # Look for dates (simplified)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY or DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(0),
                    'type': 'date',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Look for numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        for match in re.finditer(number_pattern, text):
            # Skip if it's part of a date we already found
            if any(e['start'] <= match.start() and e['end'] >= match.end() for e in entities):
                continue
                
            entities.append({
                'text': match.group(0),
                'type': 'number',
                'start': match.start(),
                'end': match.end()
            })
        
        # Look for potential organization names (simplified)
        org_pattern = r'\b[A-Z][a-z]+ (?:Inc|LLC|Corp|Corporation|Company|Co|Ltd)\b'
        for match in re.finditer(org_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'organization',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def _extract_advanced_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract advanced entities including emails, citations, acronyms, and math
        
        Args:
            text: The text to analyze
            
        Returns:
            List of entity dictionaries
        """
        # Start with simple entities
        entities = self._extract_simple_entities(text)
        
        # Add: Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'email',
                'start': match.start(),
                'end': match.end()
            })
        
        # Add: Citation patterns (common in academic texts)
        citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}\)',  # (Author et al., 2020)
            r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s+\d{4}\)',  # (Author and Author, 2020)
            r'\[[A-Z][a-z]+\s+\d{4}\]',  # [Author 2020]
            r'\[\d+\]',  # [1], [2], etc.
        ]
        
        for pattern in citation_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(0),
                    'type': 'citation',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Add: Technical terms (acronyms)
        acronym_pattern = r'\b[A-Z]{2,}[s]?\b'
        for match in re.finditer(acronym_pattern, text):
            # Filter out common words that might match
            if match.group(0) not in ['I', 'A', 'US', 'UK', 'EU']:
                entities.append({
                    'text': match.group(0),
                    'type': 'acronym',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Add: Mathematical expressions
        math_patterns = [
            r'(?:[0-9]+\.?[0-9]*\s*[+\-*/=]\s*[0-9]+\.?[0-9]*)',  # Basic arithmetic
            r'(?:\d+\.?\d*\s*[<>≤≥]\s*\d+\.?\d*)',  # Comparisons
            r'(?:[a-zA-Z]\s*=\s*[0-9]+\.?[0-9]*)',  # Variable assignments
            r'(?:∑|∏|∫|√|∞|π|θ|α|β|γ|δ|ε)',  # Mathematical symbols
        ]
        
        for pattern in math_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(0),
                    'type': 'math',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Add: URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                'text': match.group(0),
                'type': 'url',
                'start': match.start(),
                'end': match.end()
            })
        
        # Sort by start position and remove duplicates
        entities.sort(key=lambda x: x['start'])
        
        # Remove overlapping entities (keep the first one)
        filtered_entities = []
        for entity in entities:
            if not filtered_entities or entity['start'] >= filtered_entities[-1]['end']:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _extract_topic_labels(self, text: str) -> List[str]:
        """
        Extract potential topic labels from text
        
        Args:
            text: The text to analyze
            
        Returns:
            List of potential topic labels
        """
        # Very simplified topic extraction - in a real system, use proper topic modeling
        
        # Extract noun phrases (simplified)
        noun_phrase_patterns = [
            r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:Analysis|Theory|Method|Approach|System|Framework)\b',
            r'\b(?:Data|Information|Knowledge|Learning) (?:[A-Z][a-z]+ )?(?:Management|Processing|Analysis)\b',
            r'\b[A-Z][a-z]+ (?:Architecture|Engineering|Science|Technology)\b'
        ]
        
        topics = []
        
        for pattern in noun_phrase_patterns:
            for match in re.finditer(pattern, text):
                topics.append(match.group(0))
        
        # If we didn't find any with patterns, use frequent capitalized terms
        if not topics:
            # Find all capitalized words (likely proper nouns)
            cap_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
            
            # Count frequencies
            word_counts = {}
            for word in cap_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get top 5 most frequent
            topics = [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return topics
    
    def detect_concept_relationships(self, content: str, all_concepts: List[str]) -> Dict[str, List[str]]:
        """
        Detect relationships between concepts in the same segment
        
        Args:
            content: The text content to analyze
            all_concepts: List of all concept names to look for
            
        Returns:
            Dictionary mapping concepts to their related concepts
        """
        relationships = {}
        
        # Get sentences
        if NLP_AVAILABLE:
            try:
                sentences = sent_tokenize(content)
            except:
                sentences = content.split('. ')
        else:
            sentences = content.split('. ')
        
        # Analyze each sentence for co-occurring concepts
        for sentence in sentences:
            sentence_lower = sentence.lower()
            concepts_in_sentence = []
            
            # Find all concepts that appear in this sentence (with word boundaries)
            for concept in all_concepts:
                # Use word boundary matching to avoid substring false positives
                if self._contains_exact(sentence_lower, concept.lower()):
                    concepts_in_sentence.append(concept)
            
            # If multiple concepts appear together, they're likely related
            if len(concepts_in_sentence) > 1:
                for concept in concepts_in_sentence:
                    if concept not in relationships:
                        relationships[concept] = []
                    
                    # Add all other concepts as related
                    for other_concept in concepts_in_sentence:
                        if other_concept != concept and other_concept not in relationships[concept]:
                            relationships[concept].append(other_concept)
        
        # Also detect relationships based on proximity (within N words)
        words = content.split()
        word_positions = {}
        
        # Build position index for each concept
        for concept in all_concepts:
            concept_lower = concept.lower()
            positions = []
            
            # Find all positions where this concept appears (exact match)
            for i, word in enumerate(words):
                # Clean word of punctuation for better matching
                clean_word = re.sub(r'[^\w\s-]', '', word).lower()
                if concept_lower == clean_word or concept_lower in self._get_word_variants(clean_word):
                    positions.append(i)
            
            if positions:
                word_positions[concept] = positions
        
        # Check proximity relationships (within 10 words)
        PROXIMITY_THRESHOLD = 10
        
        for concept1, positions1 in word_positions.items():
            for concept2, positions2 in word_positions.items():
                if concept1 == concept2:
                    continue
                
                # Check if any positions are within threshold
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= PROXIMITY_THRESHOLD:
                            if concept1 not in relationships:
                                relationships[concept1] = []
                            if concept2 not in relationships[concept1]:
                                relationships[concept1].append(concept2)
                            break
        
        return relationships
    
    def _contains_exact(self, haystack: str, needle: str) -> bool:
        """Check if needle appears as exact word in haystack"""
        # Use word boundaries to avoid substring matches
        pattern = r'\b' + re.escape(needle) + r'\b'
        return bool(re.search(pattern, haystack))
    
    def _get_word_variants(self, word: str) -> Set[str]:
        """Get common variants of a word (plural, etc.)"""
        variants = {word}
        # Simple pluralization rules
        if word.endswith('y'):
            variants.add(word[:-1] + 'ies')
        elif word.endswith('s') or word.endswith('x') or word.endswith('ch'):
            variants.add(word + 'es')
        else:
            variants.add(word + 's')
        
        # Remove plural
        if word.endswith('ies'):
            variants.add(word[:-3] + 'y')
        elif word.endswith('es'):
            variants.add(word[:-2])
        elif word.endswith('s'):
            variants.add(word[:-1])
        
        return variants
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text (simplified)
        
        Args:
            text: The text to analyze
            
        Returns:
            Language code (default 'en')
        """
        # Very simplified language detection
        # In a real system, use a proper language detection library
        
        # Check for common non-English characters
        if re.search(r'[áéíóúüñ¿¡]', text):
            return 'es'  # Spanish
        elif re.search(r'[àâçéèêëîïôùûüÿœ]', text):
            return 'fr'  # French
        elif re.search(r'[äöüß]', text):
            return 'de'  # German
        
        # Default to English
        return 'en'
    
    async def sculpt_and_store_batch(self,
                                   user_id: str,
                                   concepts: List[Dict[str, Any]],
                                   doc_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple concepts in batch with relationship detection
        
        Args:
            user_id: ID of the user who owns the memories
            concepts: List of concept dictionaries
            doc_metadata: Optional document-level metadata
            
        Returns:
            Dictionary with results including memory IDs and relationships
        """
        # Validate user_id at batch level
        if not user_id or user_id == "default":
            # Only log once to avoid spam
            if not hasattr(self, '_warned_default_user'):
                logger.warning("⚠️ Skipping memory store - no valid user_id provided (default user). This warning will only show once.")
                self._warned_default_user = True
            return {
                'total_concepts': len(concepts),
                'memories_created': [],
                'relationships_detected': {},
                'processing_time': 0,
                'errors': [{
                    'error': f"Invalid user_id: '{user_id}'",
                    'type': 'validation_error'
                }]
            }
            
        results = {
            'total_concepts': len(concepts),
            'memories_created': [],
            'relationships_detected': {},
            'processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Extract all concept names for relationship detection
        concept_names = [c.get('name', '') for c in concepts if c.get('name')]
        
        # Process each concept
        for i, concept in enumerate(concepts):
            try:
                # Add document metadata to each concept
                concept_metadata = doc_metadata.copy() if doc_metadata else {}
                concept_metadata['batch_index'] = i
                concept_metadata['batch_size'] = len(concepts)
                
                # Store with relationship information
                memory_ids = await self.sculpt_and_store(
                    user_id=user_id,
                    raw_concept=concept,
                    metadata=concept_metadata,
                    related_concepts=concept_names
                )
                
                results['memories_created'].extend(memory_ids)
                
                # Track relationships
                if concept.get('name') and 'related_concepts' in concept.get('metadata', {}):
                    results['relationships_detected'][concept['name']] = concept['metadata']['related_concepts']
                    
            except Exception as e:
                logger.error(f"Error processing concept {i}: {str(e)}")
                results['errors'].append({
                    'concept_index': i,
                    'concept_name': concept.get('name', 'unknown'),
                    'error': str(e)
                })
        
        results['processing_time'] = time.time() - start_time
        results['success_rate'] = (len(results['memories_created']) / len(concepts)) if concepts else 0
        
        logger.info(f"✅ Batch processing complete: {len(results['memories_created'])} memories created in {results['processing_time']:.2f}s")
        
        return results

# Create singleton instance
memory_sculptor = MemorySculptor()

# Export for API usage
async def sculpt_and_store(user_id: str, 
                         raw_concept: Dict[str, Any],
                         metadata: Optional[Dict[str, Any]] = None,
                         related_concepts: Optional[List[str]] = None) -> List[str]:
    """Sculpt and store a concept in Soliton Memory"""
    return await memory_sculptor.sculpt_and_store(user_id, raw_concept, metadata, related_concepts)

async def sculpt_and_store_batch(user_id: str,
                               concepts: List[Dict[str, Any]],
                               doc_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process multiple concepts in batch with relationship detection"""
    return await memory_sculptor.sculpt_and_store_batch(user_id, concepts, doc_metadata)

# Test function
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python memory_sculptor.py <user_id> <text_or_json_file>")
        sys.exit(1)
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    async def test_sculptor():
        user_id = sys.argv[1]
        text_or_file = sys.argv[2]
        
        # Validate user_id at CLI level
        if not user_id or user_id == "default":
            print("❌ ERROR: Invalid user_id passed. Please provide a real user ID.")
            sys.exit(1)
        
        # Check if input is a file
        if os.path.exists(text_or_file):
            with open(text_or_file, 'r', encoding='utf-8') as f:
                # Try to parse as JSON
                try:
                    raw_concept = json.load(f)
                except json.JSONDecodeError:
                    # Not JSON, treat as raw text
                    raw_concept = {
                        'id': f"test_{int(time.time())}",
                        'text': f.read(),
                        'score': 0.7,
                        'metadata': {'source': 'test_file'}
                    }
        else:
            # Input is raw text
            raw_concept = {
                'id': f"test_{int(time.time())}",
                'text': text_or_file,
                'score': 0.7,
                'metadata': {'source': 'command_line'}
            }
        
        print(f"Sculpting and storing concept for user {user_id}...")
        memory_ids = await memory_sculptor.sculpt_and_store(
            user_id=user_id,
            raw_concept=raw_concept,
            metadata={"test_run": True}
        )
        
        print(f"Created {len(memory_ids)} memories:")
        for memory_id in memory_ids:
            print(f"  - {memory_id}")
    
    asyncio.run(test_sculptor())
