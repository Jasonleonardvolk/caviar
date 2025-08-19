#!/usr/bin/env python3
"""
🧬 Clean PDF Pipeline Demo

This script demonstrates your streamlined PDF concept extraction pipeline:
- Dense semantic matching from extractConceptsFromDocument()
- Priority boosting from concept_file_storage.json
- Quality auditing in conceptMesh.ts

Usage:
    python demo_clean_pipeline.py path/to/document.pdf
    python demo_clean_pipeline.py path/to/pdf_directory/ --batch
"""

import sys
import logging
from pathlib import Path

# Add the ingest_pdf directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from ingest_pdf.pipeline import (
    ingest_pdf_clean,
    batch_ingest_pdfs_clean,
    extract_and_boost_concepts,
    concept_file_storage
)

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_single_pdf(pdf_path: str):
    """Demo processing a single PDF"""
    print(f"\n🧬 CLEAN PDF PIPELINE DEMO")
    print(f"📄 Processing: {pdf_path}")
    print(f"🚀 Concept Database: {len(concept_file_storage)} boost concepts loaded")
    print("-" * 60)
    
    result = ingest_pdf_clean(pdf_path)
    
    print(f"\n📊 RESULTS:")
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Total Concepts: {result.get('concept_count', 0)}")
    print(f"   Semantic Concepts: {result.get('semantic_concepts', 0)}")
    print(f"   Boosted Concepts: {result.get('boosted_concepts', 0)}")
    print(f"   Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
    
    if result.get('concept_names'):
        print(f"\n🧠 EXTRACTED CONCEPTS:")
        for i, name in enumerate(result['concept_names'][:10], 1):  # Show first 10
            print(f"   {i:2}. {name}")
        if len(result['concept_names']) > 10:
            print(f"   ... and {len(result['concept_names']) - 10} more")
    
    print(f"\n✅ ConceptMesh Integration: {result.get('concept_mesh_injected', False)}")
    print("-" * 60)

def demo_batch_processing(pdf_directory: str):
    """Demo batch processing multiple PDFs"""
    print(f"\n🧬 CLEAN BATCH PIPELINE DEMO")
    print(f"📁 Processing directory: {pdf_directory}")
    print(f"🚀 Concept Database: {len(concept_file_storage)} boost concepts loaded")
    print("-" * 60)
    
    results = batch_ingest_pdfs_clean(pdf_directory)
    
    print(f"\n📊 BATCH RESULTS:")
    successful = [r for r in results if r.get('status') == 'success']
    total_concepts = sum(r.get('concept_count', 0) for r in successful)
    total_boosted = sum(r.get('boosted_concepts', 0) for r in successful)
    
    print(f"   PDFs Processed: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Total Concepts: {total_concepts}")
    print(f"   Boosted Concepts: {total_boosted}")
    if total_concepts > 0:
        print(f"   Boost Rate: {(total_boosted/total_concepts*100):.1f}%")
    
    print(f"\n📄 PER-FILE BREAKDOWN:")
    for result in results:
        status = result.get('status', 'unknown')
        concepts = result.get('concept_count', 0)
        boosted = result.get('boosted_concepts', 0)
        filename = result.get('filename', 'unknown')
        print(f"   {filename}: {status} - {concepts} concepts ({boosted} boosted)")
    
    print("-" * 60)

def demo_concept_extraction(text_sample: str):
    """Demo the concept extraction process on a text sample"""
    print(f"\n🧬 CONCEPT EXTRACTION DEMO")
    print(f"📝 Sample Text: {text_sample[:100]}...")
    print("-" * 60)
    
    concepts = extract_and_boost_concepts(text_sample)
    
    print(f"\n🔍 EXTRACTION RESULTS:")
    print(f"   Total Concepts Found: {len(concepts)}")
    
    semantic_concepts = [c for c in concepts if c.get('method') == 'semantic_extraction']
    boosted_concepts = [c for c in concepts if c.get('method') == 'file_storage_boosted']
    
    print(f"   Semantic Concepts: {len(semantic_concepts)}")
    print(f"   Database Boosted: {len(boosted_concepts)}")
    
    if concepts:
        print(f"\n🧠 FOUND CONCEPTS:")
        for i, concept in enumerate(concepts, 1):
            method = concept.get('method', 'unknown')
            score = concept.get('score', 0)
            print(f"   {i:2}. {concept['name']} (score: {score:.3f}, method: {method})")
    
    print("-" * 60)

def main():
    """Main demo function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo_clean_pipeline.py path/to/document.pdf")
        print("  python demo_clean_pipeline.py path/to/pdf_directory/ --batch")
        print("  python demo_clean_pipeline.py --demo-text")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "--demo-text":
        # Demo with sample text about AI and machine learning
        sample_text = """
        Artificial intelligence and machine learning are revolutionizing modern technology.
        Deep learning neural networks enable sophisticated pattern recognition capabilities.
        Natural language processing allows computers to understand human communication.
        Computer vision systems can analyze and interpret visual information automatically.
        Quantum computing promises to solve complex computational problems exponentially faster.
        """
        demo_concept_extraction(sample_text)
        return
    
    path = Path(arg)
    
    if not path.exists():
        print(f"❌ Path does not exist: {arg}")
        sys.exit(1)
    
    if "--batch" in sys.argv or path.is_dir():
        # Batch processing
        demo_batch_processing(str(path))
    else:
        # Single file
        if not path.suffix.lower() == '.pdf':
            print(f"❌ File must be a PDF: {arg}")
            sys.exit(1)
        demo_single_pdf(str(path))
    
    print("\n🎯 Demo complete! Your concepts have been injected into the ConceptMesh system.")

if __name__ == "__main__":
    main()
