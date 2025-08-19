"""
Concept Extraction Module for Semantic PDF Ingestion
Includes spaCy fallback, YAKE keyword extraction, and semantic SVO parsing
"""

import logging
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import handling with proper error messages
try:
    import spacy
    from spacy.tokens import Doc, Token, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - concept extraction will be limited")

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    logging.warning("YAKE not available - keyword extraction will be limited")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available - PDF extraction will be limited")

logger = logging.getLogger(__name__)

# Log availability at module load
logger.info(f"spaCy available for concept extraction: {SPACY_AVAILABLE}")
logger.info(f"YAKE available for keyword extraction: {YAKE_AVAILABLE}")
logger.info(f"PyPDF2 available for PDF extraction: {PYPDF2_AVAILABLE}")

# -----------------------------------------------------------------------------
# ⚙️ CONFIG
# -----------------------------------------------------------------------------
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM_SIZE = 3
YAKE_DEDUPLICATION_THRESHOLD = 0.9
YAKE_TOP_K = 25
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_FALLBACK_MODEL = "en_core_web_sm"

# -----------------------------------------------------------------------------
# 📦 STRUCTURES
# -----------------------------------------------------------------------------
@dataclass
class Relationship:
    """Represents a semantic relationship between concepts"""
    type: str  # e.g., "subject_of", "object_of", "related_to"
    target: str  # The target concept
    source: Optional[str] = None  # The source concept
    verb: Optional[str] = None  # The connecting verb
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Concept:
    """Enhanced concept with semantic relationships"""
    name: str
    score: float = 1.0
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with guaranteed label field"""
        # Ensure we always have a label
        if not self.name:
            logger.warning("⚠️ Concept has no name/label, using fallback")
            label = self.metadata.get('text', '')[:50].strip() or "unknown_concept"
        else:
            label = self.name
            
        return {
            "name": label,  # Backward compatibility
            "label": label,  # Guaranteed field
            "score": self.score,
            "method": self.method,
            "metadata": {
                **self.metadata,
                "relationships": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.relationships]
            }
        }


# -----------------------------------------------------------------------------
# 🧠 MODEL MANAGEMENT
# -----------------------------------------------------------------------------
# Global NLP model cache
_nlp_model = None
_nlp_model_name = None


def install_spacy_model(model_name: str) -> bool:
    """Attempt to install a spaCy model"""
    try:
        logger.info(f"📦 Installing spaCy model: {model_name}")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"✅ Successfully installed {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error installing {model_name}: {e}")
        return False


def load_spacy_model(model_name: str = SPACY_MODEL_NAME, auto_install: bool = True) -> Optional['spacy.Language']:
    """Load spaCy model with automatic fallback and installation"""
    global _nlp_model, _nlp_model_name
    
    if not SPACY_AVAILABLE:
        logger.error("❌ spaCy is not installed. Run: pip install spacy")
        return None
    
    # Return cached model if already loaded
    if _nlp_model is not None and _nlp_model_name == model_name:
        return _nlp_model
    
    # Try to load the model
    try:
        _nlp_model = spacy.load(model_name)
        _nlp_model_name = model_name
        
        # Ensure dependency parser is enabled
        if "parser" not in _nlp_model.pipe_names:
            logger.warning("⚠️ Dependency parser not in pipeline. Adding it...")
            _nlp_model.add_pipe("parser")
        
        logger.info(f"✅ Loaded spaCy model: {model_name}")
        logger.info(f"📦 Pipeline components: {_nlp_model.pipe_names}")
        return _nlp_model
        
    except OSError as e:
        logger.error(f"Model {model_name} not found: {e}")
        
        if auto_install:
            logger.info(f"🔄 Attempting to install {model_name}...")
            if install_spacy_model(model_name):
                # Try loading again after installation
                try:
                    _nlp_model = spacy.load(model_name)
                    _nlp_model_name = model_name
                    logger.info(f"✅ Successfully loaded {model_name} after installation")
                    return _nlp_model
                except Exception as e2:
                    logger.error(f"❌ Failed to load after installation: {e2}")
        
        # Try fallback model if different
        if model_name != SPACY_FALLBACK_MODEL:
            logger.info(f"🔄 Trying fallback model: {SPACY_FALLBACK_MODEL}")
            return load_spacy_model(SPACY_FALLBACK_MODEL, auto_install=auto_install)
    
    logger.error(f"❌ Could not load any spaCy model")
    return None


# -----------------------------------------------------------------------------
# 📘 KEYWORD EXTRACTION
# -----------------------------------------------------------------------------
def extract_keywords_yake(text: str, top_k: int = YAKE_TOP_K) -> List[Concept]:
    """Extract keywords using YAKE"""
    if not YAKE_AVAILABLE:
        logger.warning("⚠️ YAKE not available for keyword extraction")
        return []
    
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        kw_extractor = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM_SIZE,
            dedupLim=YAKE_DEDUPLICATION_THRESHOLD,
            top=top_k,
            features=None
        )
        
        keywords = kw_extractor.extract_keywords(text)
        
        concepts = []
        for keyword, score in keywords:
            # YAKE scores are inverted (lower is better)
            normalized_score = 1.0 / (1.0 + score)
            concepts.append(Concept(
                name=keyword,
                score=normalized_score,
                method="yake",
                metadata={"yake_score": score}
            ))
        
        return concepts
        
    except Exception as e:
        logger.error(f"❌ YAKE extraction failed: {e}")
        return []


# -----------------------------------------------------------------------------
# 🔗 RELATIONSHIP EXTRACTION
# -----------------------------------------------------------------------------
def extract_svo_relationships(doc: 'Doc') -> List[Dict[str, Any]]:
    """Extract Subject-Verb-Object relationships from spaCy doc"""
    relationships = []
    
    for sent in doc.sents:
        # Find main verb
        root = sent.root
        if root.pos_ != "VERB":
            continue
        
        # Find subjects
        subjects = []
        for token in sent:
            if "subj" in token.dep_ and token.head == root:
                subjects.append(token)
        
        # Find objects
        objects = []
        for token in sent:
            if "obj" in token.dep_ and token.head == root:
                objects.append(token)
        
        # Create relationships
        for subj in subjects:
            for obj in objects:
                rel = {
                    "subject": subj.text,
                    "verb": root.text,
                    "object": obj.text,
                    "sentence": sent.text.strip(),
                    "subject_pos": subj.pos_,
                    "object_pos": obj.pos_
                }
                relationships.append(rel)
    
    return relationships


def extract_entity_relationships(doc: 'Doc') -> List[Tuple[str, str, str]]:
    """Extract relationships between named entities"""
    relationships = []
    entities = list(doc.ents)
    
    for sent in doc.sents:
        sent_ents = [e for e in entities if e.start >= sent.start and e.end <= sent.end]
        
        if len(sent_ents) >= 2:
            # Find verb between entities
            for i, ent1 in enumerate(sent_ents[:-1]):
                for ent2 in sent_ents[i+1:]:
                    # Find connecting verb
                    verb_tokens = []
                    for token in sent:
                        if token.i > ent1.end and token.i < ent2.start and token.pos_ == "VERB":
                            verb_tokens.append(token.text)
                    
                    if verb_tokens:
                        relationships.append((ent1.text, " ".join(verb_tokens), ent2.text))
    
    return relationships


# -----------------------------------------------------------------------------
# 🧠 ENHANCED EXTRACTION
# -----------------------------------------------------------------------------
def extract_concepts_with_nlp(text: str, nlp_model: Optional['spacy.Language'] = None) -> List[Concept]:
    """Extract concepts using spaCy NLP with entity recognition"""
    if nlp_model is None:
        nlp_model = load_spacy_model()
    
    if nlp_model is None:
        logger.error("❌ No spaCy model available")
        return []
    
    try:
        doc = nlp_model(text)
        concepts = []
        
        # Extract named entities
        for ent in doc.ents:
            concept = Concept(
                name=ent.text,
                score=0.8,  # Base score for entities
                method="spacy_ner",
                metadata={
                    "entity_type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
            )
            concepts.append(concept)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            # Skip if already captured as entity
            if any(chunk.text == c.name for c in concepts):
                continue
            
            concept = Concept(
                name=chunk.text,
                score=0.6,  # Lower score for noun phrases
                method="spacy_noun_phrase",
                metadata={
                    "root": chunk.root.text,
                    "root_pos": chunk.root.pos_
                }
            )
            concepts.append(concept)
        
        return concepts
        
    except Exception as e:
        logger.error(f"❌ spaCy extraction failed: {e}")
        return []


# -----------------------------------------------------------------------------
# 🎯 MAIN EXTRACTION FUNCTIONS
# -----------------------------------------------------------------------------
def extract_concepts_from_text(text: str) -> List[Dict[str, Any]]:
    """Basic concept extraction without relationships (backward compatibility)"""
    concepts = []
    
    # Try YAKE keywords
    if YAKE_AVAILABLE:
        yake_concepts = extract_keywords_yake(text)
        concepts.extend(yake_concepts)
    
    # Try spaCy entities
    if SPACY_AVAILABLE:
        nlp = load_spacy_model()
        if nlp:
            spacy_concepts = extract_concepts_with_nlp(text, nlp)
            concepts.extend(spacy_concepts)
    
    # Deduplicate by name
    seen = set()
    unique_concepts = []
    for concept in concepts:
        if concept.name.lower() not in seen:
            seen.add(concept.name.lower())
            unique_concepts.append(concept.to_dict())
    
    return unique_concepts


def extract_semantic_concepts(text: str, use_nlp: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced semantic concept extraction with relationships
    
    Args:
        text: Input text to process
        use_nlp: Whether to use NLP for relationship extraction
        
    Returns:
        List of concept dictionaries with relationships
    """
    if not text or len(text.strip()) < 10:
        logger.warning("⚠️ Empty or too-short text for extraction")
        return []
    
    concepts_dict = {}  # name -> Concept mapping
    
    # Step 1: Extract keywords with YAKE
    if YAKE_AVAILABLE:
        logger.info("🔤 Extracting keywords with YAKE...")
        yake_concepts = extract_keywords_yake(text)
        for concept in yake_concepts:
            concepts_dict[concept.name.lower()] = concept
    
    # Step 2: Extract entities and relationships with spaCy
    if use_nlp and SPACY_AVAILABLE:
        nlp = load_spacy_model()
        if nlp:
            logger.info("🧠 Processing with spaCy NLP...")
            doc = nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                name_lower = ent.text.lower()
                if name_lower not in concepts_dict:
                    concepts_dict[name_lower] = Concept(
                        name=ent.text,
                        score=0.8,
                        method="spacy_entity",
                        metadata={"entity_type": ent.label_}
                    )
                else:
                    # Enhance existing concept
                    concepts_dict[name_lower].method += "+spacy_entity"
                    concepts_dict[name_lower].metadata["entity_type"] = ent.label_
            
            # Extract SVO relationships
            logger.info("🔗 Extracting SVO relationships...")
            svo_rels = extract_svo_relationships(doc)
            
            # Map relationships to concepts
            for rel in svo_rels:
                subj_lower = rel["subject"].lower()
                obj_lower = rel["object"].lower()
                
                # Ensure subject exists as concept
                if subj_lower not in concepts_dict:
                    concepts_dict[subj_lower] = Concept(
                        name=rel["subject"],
                        score=0.7,
                        method="svo_subject"
                    )
                
                # Ensure object exists as concept
                if obj_lower not in concepts_dict:
                    concepts_dict[obj_lower] = Concept(
                        name=rel["object"],
                        score=0.7,
                        method="svo_object"
                    )
                
                # Add relationship to subject
                concepts_dict[subj_lower].relationships.append(
                    Relationship(
                        type="subject_of",
                        target=rel["object"],
                        verb=rel["verb"],
                        source=rel["subject"]
                    )
                )
                
                # Add inverse relationship to object
                concepts_dict[obj_lower].relationships.append(
                    Relationship(
                        type="object_of",
                        target=rel["subject"],
                        verb=rel["verb"],
                        source=rel["object"]
                    )
                )
            
            # Extract entity-to-entity relationships
            logger.info("🔗 Extracting entity relationships...")
            entity_rels = extract_entity_relationships(doc)
            
            for subj, verb, obj in entity_rels:
                subj_lower = subj.lower()
                obj_lower = obj.lower()
                
                if subj_lower in concepts_dict:
                    concepts_dict[subj_lower].relationships.append(
                        Relationship(
                            type="related_to",
                            target=obj,
                            verb=verb,
                            source=subj
                        )
                    )
    
    # Convert to list and sort by score
    concepts = list(concepts_dict.values())
    concepts.sort(key=lambda c: c.score, reverse=True)
    
    # Log statistics
    total_concepts = len(concepts)
    total_relationships = sum(len(c.relationships) for c in concepts)
    concepts_with_rels = sum(1 for c in concepts if c.relationships)
    
    logger.info(f"📊 Extracted {total_concepts} concepts with {total_relationships} relationships")
    logger.info(f"🔗 {concepts_with_rels}/{total_concepts} concepts have relationships")
    
    # Convert to dictionaries for output
    return [c.to_dict() for c in concepts]


# -----------------------------------------------------------------------------
# 📄 PDF EXTRACTION
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file"""
    if not PYPDF2_AVAILABLE:
        logger.error("❌ PyPDF2 not available for PDF extraction")
        return ""
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        return ""


# -----------------------------------------------------------------------------
# 🚀 MODULE INITIALIZATION
# -----------------------------------------------------------------------------
# Pre-load spaCy model on module import for performance
if SPACY_AVAILABLE:
    logger.info("🔄 Pre-loading spaCy model...")
    _nlp_model = load_spacy_model(auto_install=True)
    if _nlp_model:
        logger.info("✅ spaCy model ready for extraction")
    else:
        logger.warning("⚠️ spaCy model could not be loaded - extraction will be limited")
