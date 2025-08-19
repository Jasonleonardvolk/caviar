# process_pdf_with_pipeline.py - Use the enhanced pipeline directly
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the enhanced pipeline
from pipeline import ingest_pdf_clean

def process_with_enhanced_pipeline(pdf_path: str):
    """Process PDF with all the advanced features"""
    
    print(f"🚀 Using ENHANCED PIPELINE with:")
    print(f"   ✨ Quality scoring (section, frequency, theme relevance)")
    print(f"   📚 Academic structure detection") 
    print(f"   🔬 Purity analysis (filters generic terms)")
    print(f"   🎯 Entropy pruning (keeps diverse concepts)")
    print(f"   💯 100% bulletproof (no NoneType errors)")
    print(f"\n📄 Processing: {Path(pdf_path).name}")
    
    # Process with all features enabled
    result = ingest_pdf_clean(
        pdf_path=pdf_path,
        extraction_threshold=0.0,  # Accept all concepts initially
        admin_mode=False,  # Use entropy pruning
        use_ocr=False  # Set True if PDF has poor text extraction
    )
    
    # Display comprehensive results
    print(f"\n📊 RESULTS:")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Concepts extracted: {result.get('concept_count', 0)}")
    print(f"Processing time: {result.get('processing_time_seconds', 0):.1f}s")
    
    if result.get('status') == 'success':
        print(f"\n📈 QUALITY METRICS:")
        print(f"Average concept score: {result.get('average_concept_score', 0):.3f}")
        print(f"High quality concepts: {result.get('high_quality_concepts', 0)}")
        
        # Section distribution
        if 'section_distribution' in result:
            print(f"\n📚 ACADEMIC SECTIONS:")
            for section, count in result['section_distribution'].items():
                print(f"  {section}: {count} concepts")
        
        # Purity analysis
        if 'purity_analysis' in result:
            purity = result['purity_analysis']
            print(f"\n🔬 PURITY ANALYSIS:")
            print(f"  Raw concepts: {purity.get('raw_concepts', 0)}")
            print(f"  After purity filter: {purity.get('pure_concepts', 0)}")
            print(f"  Final (after entropy): {purity.get('final_concepts', 0)}")
            print(f"  Purity efficiency: {purity.get('purity_efficiency_percent', 0):.1f}%")
        
        # Entropy analysis
        if result.get('entropy_analysis', {}).get('enabled'):
            entropy = result['entropy_analysis']
            print(f"\n🎯 ENTROPY PRUNING:")
            print(f"  Diversity preserved: {entropy.get('diversity_efficiency_percent', 0):.1f}%")
            print(f"  Similar concepts pruned: {entropy.get('pruned_similar', 0)}")
            print(f"  Final entropy score: {entropy.get('final_entropy', 0):.3f}")
        
        # Top concepts with quality scores
        if 'purity_analysis' in result and 'top_concepts' in result['purity_analysis']:
            print(f"\n🏆 TOP CONCEPTS (with quality scores):")
            for i, concept in enumerate(result['purity_analysis']['top_concepts'][:10], 1):
                name = concept.get('name', 'Unknown')
                quality = concept.get('quality_score', 0)
                section = concept.get('section', 'unknown')
                freq = concept.get('frequency', 1)
                print(f"  {i}. {name}")
                print(f"     Quality: {quality:.3f} | Section: {section} | Frequency: {freq}")
    
    return result

if __name__ == "__main__":
    # Your PDF
    pdf_path = r"{PROJECT_ROOT}\anewapproach.pdf"
    
    # Check dependencies
    try:
        from pipeline import ingest_pdf_clean
        print("✅ Pipeline module loaded successfully")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure all pipeline dependencies are available")
        sys.exit(1)
    
    # Process with enhanced features
    result = process_with_enhanced_pipeline(pdf_path)
