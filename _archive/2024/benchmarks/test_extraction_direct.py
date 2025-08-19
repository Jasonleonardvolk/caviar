#!/usr/bin/env python3
"""
Direct test of semantic extraction on the Living Soliton Memory Systems PDF
"""

import asyncio
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_extraction():
    """Test semantic extraction with the new concept_extraction module"""
    
    # Import extraction functions
    from ingest_pdf.extraction.concept_extraction import (
        extract_semantic_concepts,
        extract_text_from_pdf,
        load_spacy_model
    )
    
    # Path to PDF
    pdf_path = r"{PROJECT_ROOT}\docs\LivingSolitonMemorySystemsCopy.pdf"
    
    logger.info(f"📄 Testing extraction on: {pdf_path}")
    
    # Extract text from PDF
    logger.info("📖 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        logger.error("❌ Failed to extract text from PDF")
        return
    
    logger.info(f"📊 Extracted {len(text)} characters from PDF")
    logger.info(f"📝 First 500 chars: {text[:500]}...")
    
    # Load spaCy model first
    logger.info("🧠 Loading spaCy model...")
    nlp = load_spacy_model(auto_install=True)
    if nlp:
        logger.info(f"✅ spaCy model loaded with components: {nlp.pipe_names}")
    else:
        logger.error("❌ Failed to load spaCy model")
    
    # Run semantic extraction
    logger.info("🚀 Running semantic concept extraction...")
    concepts = extract_semantic_concepts(text, use_nlp=True)
    
    logger.info(f"\n✅ Extraction complete!")
    logger.info(f"📊 Found {len(concepts)} concepts")
    
    # Analyze results
    concepts_with_rels = [c for c in concepts if c.get('metadata', {}).get('relationships')]
    total_relationships = sum(
        len(c.get('metadata', {}).get('relationships', [])) 
        for c in concepts
    )
    
    logger.info(f"🔗 {len(concepts_with_rels)} concepts have relationships")
    logger.info(f"🔗 Total relationships: {total_relationships}")
    
    # Show top concepts
    logger.info("\n🏆 Top 10 concepts:")
    for i, concept in enumerate(concepts[:10]):
        rels = concept.get('metadata', {}).get('relationships', [])
        logger.info(f"{i+1}. {concept['name']} (score: {concept['score']:.3f}, method: {concept['method']}, relationships: {len(rels)})")
        
        # Show relationships for this concept
        if rels:
            for rel in rels[:2]:  # Show first 2 relationships
                logger.info(f"   → {rel['type']}: {rel['target']}")
                if 'verb' in rel:
                    logger.info(f"     (via '{rel['verb']}')")
    
    # Try to store in memory vault
    try:
        from python.core.memory_vault import UnifiedMemoryVault
        from python.core.memory_types import MemoryType
        
        logger.info("\n💾 Attempting to store in memory vault...")
        
        # Create vault config
        vault_config = {
            'storage_path': 'data/memory_vault',
            'max_working_memory': 100,
            'ghost_memory_ttl': 3600
        }
        
        # Initialize vault
        vault = UnifiedMemoryVault(vault_config)
        
        # Store concepts
        stored = 0
        for concept in concepts[:20]:  # Store top 20
            try:
                await vault.store(
                    content={
                        "name": concept['name'],
                        "score": concept['score'],
                        "extraction": concept,
                        "source": "LivingSolitonMemorySystemsCopy.pdf"
                    },
                    memory_type=MemoryType.SEMANTIC,
                    tags=["pdf", "soliton", "test", "semantic_extraction"]
                )
                stored += 1
            except Exception as e:
                logger.error(f"Failed to store {concept['name']}: {e}")
        
        # Save to disk
        await vault.save_all()
        logger.info(f"✅ Stored {stored} concepts in memory vault")
        
    except Exception as e:
        logger.warning(f"⚠️ Memory vault storage failed: {e}")
    
    # Show extraction diagnostics
    if total_relationships == 0:
        logger.warning("\n⚠️ No relationships were extracted!")
        logger.info("Diagnostics:")
        logger.info("1. Check if spaCy model has parser: " + str('parser' in nlp.pipe_names if nlp else 'N/A'))
        logger.info("2. Sample text for SVO analysis:")
        
        # Try a simple SVO extraction on sample
        if nlp:
            sample = "Solitons maintain their shape. They propagate through media."
            doc = nlp(sample)
            logger.info(f"   Sample: '{sample}'")
            logger.info(f"   Sentences: {len(list(doc.sents))}")
            for sent in doc.sents:
                logger.info(f"   Tokens: {[(t.text, t.pos_, t.dep_) for t in sent]}")


if __name__ == "__main__":
    asyncio.run(test_extraction())
