# process_pdf_enhanced.py - Use the enhanced bulletproof pipeline
import asyncio
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the pipeline module to path if needed
sys.path.insert(0, str(Path(__file__).parent))

async def process_with_enhanced_pipeline(pdf_path: str):
    """Process using the enhanced bulletproof pipeline"""
    
    try:
        # Import the enhanced pipeline
        from pipeline import ingest_pdf_clean
        
        print(f"🚀 Using ENHANCED BULLETPROOF PIPELINE")
        print(f"📄 Processing: {Path(pdf_path).name}")
        print(f"✨ Features: OCR, academic structure, quality scoring, parallel processing")
        
        # Process with enhanced pipeline
        result = ingest_pdf_clean(
            pdf_path=pdf_path,
            extraction_threshold=0.0,  # Accept all concepts initially
            admin_mode=False,  # Use entropy pruning
            use_ocr=False  # Skip OCR for speed (set True if needed)
        )
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"  - Status: {result.get('status', 'unknown')}")
        print(f"  - Concepts extracted: {result.get('concept_count', 0)}")
        print(f"  - Chunks processed: {result.get('chunks_processed', 0)}")
        print(f"  - Average quality: {result.get('average_concept_score', 0):.3f}")
        print(f"  - High quality concepts: {result.get('high_quality_concepts', 0)}")
        
        if 'section_distribution' in result:
            print(f"\n📚 Section Distribution:")
            for section, count in result['section_distribution'].items():
                print(f"  - {section}: {count} concepts")
        
        if 'purity_analysis' in result:
            purity = result['purity_analysis']
            print(f"\n🔬 Purity Analysis:")
            print(f"  - Raw concepts: {purity.get('raw_concepts', 0)}")
            print(f"  - Pure concepts: {purity.get('pure_concepts', 0)}")
            print(f"  - Final concepts: {purity.get('final_concepts', 0)}")
            
        if 'entropy_analysis' in result and result['entropy_analysis'].get('enabled'):
            entropy = result['entropy_analysis']
            print(f"\n🎯 Entropy Analysis:")
            print(f"  - Diversity efficiency: {entropy.get('diversity_efficiency_percent', 0)}%")
            print(f"  - Final entropy: {entropy.get('final_entropy', 0)}")
        
        # Show top concepts
        if 'concepts' in result and result['concepts']:
            print(f"\n🏆 Top 5 Concepts:")
            for i, concept in enumerate(result['concepts'][:5], 1):
                name = concept.get('name', 'Unknown')
                quality = concept.get('quality_score', 0)
                section = concept.get('metadata', {}).get('section', 'unknown')
                print(f"  {i}. {name} (quality: {quality:.3f}, section: {section})")
        
        return result
        
    except ImportError as e:
        print(f"❌ Could not import enhanced pipeline: {e}")
        print("Falling back to standard pipeline...")
        
        from core.canonical_ingestion_production_fixed import ingest_file_production
        result = await ingest_file_production(pdf_path)
        
        if result.success:
            print(f"✅ Standard pipeline: {result.concepts_extracted} concepts")
        else:
            print(f"❌ Standard pipeline failed: {result.archive_id}")
        
        return result

if __name__ == "__main__":
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    asyncio.run(process_with_enhanced_pipeline(pdf_path))
