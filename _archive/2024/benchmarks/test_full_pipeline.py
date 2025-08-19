#!/usr/bin/env python3
"""
Complete Integration Test: PDF → Semantic Extraction → Vault Storage
Tests the entire pipeline with the Living Soliton Memory Systems PDF
"""

import asyncio
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_full_pipeline():
    """Test the complete semantic extraction to vault storage pipeline"""
    
    # Import all components
    from ingest_pdf.extraction import (
        extract_semantic_concepts,
        extract_text_from_pdf,
        load_spacy_model
    )
    from python.core.vault_writer import VaultWriter
    from python.core.memory_vault import UnifiedMemoryVault
    from python.core.memory_types import MemoryType
    
    # Configuration
    pdf_path = r"{PROJECT_ROOT}\docs\LivingSolitonMemorySystemsCopy.pdf"
    vault_path = "data/memory_vault"
    
    logger.info("🚀 Starting full pipeline test...")
    logger.info(f"📄 PDF: {pdf_path}")
    logger.info(f"💾 Vault: {vault_path}")
    
    # Step 1: Extract text from PDF
    logger.info("\n📖 Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        logger.error("❌ Failed to extract text from PDF")
        return
    
    logger.info(f"✅ Extracted {len(text)} characters")
    logger.info(f"📝 Preview: {text[:200]}...")
    
    # Step 2: Load spaCy model
    logger.info("\n🧠 Step 2: Loading NLP model...")
    nlp = load_spacy_model(auto_install=True)
    if nlp:
        logger.info(f"✅ Model loaded with components: {nlp.pipe_names}")
    
    # Step 3: Extract semantic concepts
    logger.info("\n🔍 Step 3: Extracting semantic concepts...")
    concepts = extract_semantic_concepts(text, use_nlp=True)
    
    logger.info(f"✅ Extracted {len(concepts)} raw concepts")
    
    # Analyze extraction results
    concepts_with_rels = sum(1 for c in concepts if c.get('metadata', {}).get('relationships'))
    total_relationships = sum(len(c.get('metadata', {}).get('relationships', [])) for c in concepts)
    
    logger.info(f"🔗 {concepts_with_rels} concepts have relationships")
    logger.info(f"🔗 Total relationships: {total_relationships}")
    
    # Show top concepts
    logger.info("\n🏆 Top 5 concepts:")
    for i, concept in enumerate(concepts[:5]):
        rels = len(concept.get('metadata', {}).get('relationships', []))
        logger.info(f"  {i+1}. {concept['name']} (score: {concept['score']:.3f}, rels: {rels})")
    
    # Step 4: Convert to memory concepts
    logger.info("\n📦 Step 4: Converting to memory concepts...")
    memory_concepts = VaultWriter.convert(
        concepts=concepts,
        doc_id="LivingSolitonMemorySystemsCopy.pdf",
        additional_tags=["soliton", "physics", "memory_systems"],
        metadata={
            "source_path": pdf_path,
            "extraction_method": "semantic_enhanced",
            "pipeline_version": "2.0"
        }
    )
    
    logger.info(f"✅ Converted to {len(memory_concepts)} memory concepts (after deduplication)")
    
    # Show deduplication results
    if len(memory_concepts) < len(concepts):
        logger.info(f"🧬 Deduplication reduced concepts by {len(concepts) - len(memory_concepts)}")
    
    # Step 5: Initialize memory vault
    logger.info("\n💾 Step 5: Initializing memory vault...")
    vault_config = {
        'storage_path': vault_path,
        'max_working_memory': 100,
        'ghost_memory_ttl': 3600
    }
    
    try:
        vault = UnifiedMemoryVault(vault_config)
        logger.info("✅ Memory vault initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize vault: {e}")
        return
    
    # Step 6: Store concepts in vault
    logger.info("\n📥 Step 6: Storing concepts in vault...")
    stored_count = 0
    failed_count = 0
    
    for mc in memory_concepts[:50]:  # Store top 50 to avoid overload
        try:
            memory_id = await vault.store(
                content=mc.to_dict(),
                memory_type=MemoryType.SEMANTIC,
                tags=["soliton_pdf", "test_run", datetime.now().strftime("%Y%m%d")]
            )
            if memory_id:
                stored_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Failed to store '{mc.label}': {e}")
            failed_count += 1
    
    logger.info(f"✅ Stored {stored_count} concepts")
    if failed_count > 0:
        logger.warning(f"⚠️ Failed to store {failed_count} concepts")
    
    # Step 7: Save vault to disk
    logger.info("\n💾 Step 7: Saving vault to disk...")
    try:
        await vault.save_all()
        logger.info("✅ Vault saved to disk")
    except Exception as e:
        logger.error(f"❌ Failed to save vault: {e}")
    
    # Step 8: Generate summary report
    logger.info("\n📊 Step 8: Generating summary report...")
    
    summary = {
        "pipeline_run": {
            "timestamp": datetime.utcnow().isoformat(),
            "pdf_file": pdf_path,
            "pdf_chars": len(text),
            "pipeline_version": "2.0"
        },
        "extraction_results": {
            "raw_concepts": len(concepts),
            "concepts_with_relationships": concepts_with_rels,
            "total_relationships": total_relationships,
            "extraction_methods": list(set(c.get('method', '') for c in concepts))
        },
        "deduplication_results": {
            "input_concepts": len(concepts),
            "output_concepts": len(memory_concepts),
            "reduction_percentage": round((1 - len(memory_concepts)/len(concepts)) * 100, 2)
        },
        "vault_storage": {
            "concepts_stored": stored_count,
            "concepts_failed": failed_count,
            "storage_path": vault_path
        },
        "top_concepts": [
            {
                "label": mc.label,
                "score": mc.score,
                "method": mc.method,
                "relationships": len(mc.relationships),
                "tags": mc.metadata.get('tags', [])[:3]  # First 3 tags
            }
            for mc in memory_concepts[:10]
        ]
    }
    
    # Save summary
    summary_path = Path("extraction_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved to: {summary_path}")
    
    # Display final statistics
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL STATISTICS:")
    logger.info("="*60)
    logger.info(f"📄 Source: {Path(pdf_path).name}")
    logger.info(f"📝 Text extracted: {len(text):,} characters")
    logger.info(f"🔍 Concepts found: {len(concepts)}")
    logger.info(f"🔗 Relationships: {total_relationships}")
    logger.info(f"🧬 After deduplication: {len(memory_concepts)} concepts")
    logger.info(f"💾 Stored in vault: {stored_count} concepts")
    logger.info(f"📁 Vault location: {vault_path}")
    logger.info("="*60)
    
    return summary


async def verify_vault_contents():
    """Verify what's actually stored in the vault"""
    vault_path = Path("data/memory_vault/memories")
    
    if not vault_path.exists():
        logger.warning("⚠️ Vault memories directory doesn't exist")
        return
    
    memory_files = list(vault_path.glob("*.json"))
    logger.info(f"\n📁 Found {len(memory_files)} memory files in vault")
    
    if memory_files:
        # Sample a memory file
        with open(memory_files[0], 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        logger.info("\n📄 Sample memory structure:")
        logger.info(f"  ID: {sample.get('id', 'N/A')}")
        logger.info(f"  Type: {sample.get('type', 'N/A')}")
        
        content = sample.get('content', {})
        logger.info(f"  Label: {content.get('label', 'N/A')}")
        logger.info(f"  Score: {content.get('score', 'N/A')}")
        logger.info(f"  Relationships: {len(content.get('relationships', []))}")


if __name__ == "__main__":
    # Run the full pipeline test
    summary = asyncio.run(test_full_pipeline())
    
    # Verify vault contents
    asyncio.run(verify_vault_contents())
    
    logger.info("\n✅ Full pipeline test complete!")
