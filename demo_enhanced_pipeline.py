#!/usr/bin/env python3
"""
🧬 Enhanced PDF Pipeline Demo - WITH THRESHOLD FIXES

This script demonstrates your enhanced PDF concept extraction pipeline with:
- LOWERED threshold (0.35) for better concept recall
- Enhanced diagnostic logging to see what's being filtered
- Score boosting mechanisms for domain-specific terms
- Real-time concept extraction visibility

Usage:
    python demo_enhanced_pipeline.py path/to/document.pdf
    python demo_enhanced_pipeline.py path/to/pdf_directory/ --batch
    python demo_enhanced_pipeline.py --test-thresholds  # Test different thresholds
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

# Configure logging for demo with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_single_pdf(pdf_path: str, threshold: float = 0.35):
    """Demo processing a single PDF with configurable threshold"""
    print(f"\n🧬 ENHANCED PDF PIPELINE DEMO")
    print(f"📄 Processing: {pdf_path}")
    print(f"🚀 Concept Database: {len(concept_file_storage)} boost concepts loaded")
    print(f"🔧 Extraction Threshold: {threshold} (LOWERED for better recall)")
    print("-" * 80)
    
    result = ingest_pdf_clean(pdf_path, extraction_threshold=threshold)
    
    print(f"\n📊 RESULTS:")
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Total Concepts: {result.get('concept_count', 0)}")
    print(f"   🧬 Semantic Concepts: {result.get('semantic_concepts', 0)}")
    print(f"   🚀 Database Boosted: {result.get('boosted_concepts', 0)}")
    print(f"   🎯 Score Boosted: {result.get('score_boosted_concepts', 0)}")
    print(f"   📈 Enhancement Rate: {result.get('boost_effectiveness', 'N/A')}")
    print(f"   ⚖️ Average Score: {result.get('average_score', 0):.3f}")
    print(f"   🏆 High Confidence (>0.8): {result.get('high_confidence_concepts', 0)}")
    print(f"   ⏱️ Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
    print(f"   🔧 Threshold Used: {result.get('extraction_threshold', threshold)}")
    
    if result.get('concept_names'):
        print(f"\n🧠 EXTRACTED CONCEPTS:")
        for i, name in enumerate(result['concept_names'][:20], 1):  # Show first 20
            print(f"   {i:2}. {name}")
        if len(result['concept_names']) > 20:
            print(f"   ... and {len(result['concept_names']) - 20} more")
    
    print(f"\n✅ ConceptMesh Integration: {result.get('concept_mesh_injected', False)}")
    
    if result.get('concept_count', 0) < 5:
        print(f"\n⚠️  LOW YIELD WARNING:")
        print(f"    Only {result.get('concept_count', 0)} concepts extracted!")
        print(f"    This suggests the threshold might still be too high.")
        print(f"    Try running with --test-thresholds to find optimal value.")
    
    print("-" * 80)

def demo_batch_processing(pdf_directory: str, threshold: float = 0.35):
    """Demo batch processing multiple PDFs with threshold control"""
    print(f"\n🧬 ENHANCED BATCH PIPELINE DEMO")
    print(f"📁 Processing directory: {pdf_directory}")
    print(f"🚀 Concept Database: {len(concept_file_storage)} boost concepts loaded")
    print(f"🔧 Extraction Threshold: {threshold} (LOWERED for better recall)")
    print("-" * 80)
    
    results = batch_ingest_pdfs_clean(pdf_directory, extraction_threshold=threshold)
    
    print(f"\n📊 BATCH RESULTS:")
    successful = [r for r in results if r.get('status') == 'success']
    total_concepts = sum(r.get('concept_count', 0) for r in successful)
    total_boosted = sum(r.get('boosted_concepts', 0) for r in successful)
    total_score_boosted = sum(r.get('score_boosted_concepts', 0) for r in successful)
    
    print(f"   PDFs Processed: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Total Concepts: {total_concepts}")
    print(f"   🚀 Database Boosted: {total_boosted}")
    print(f"   🎯 Score Boosted: {total_score_boosted}")
    if total_concepts > 0:
        enhancement_rate = (total_boosted + total_score_boosted) / total_concepts * 100
        print(f"   📈 Enhancement Rate: {enhancement_rate:.1f}%")
    print(f"   🔧 Threshold Used: {threshold}")
    
    print(f"\n📄 PER-FILE BREAKDOWN:")
    for result in results:
        status = result.get('status', 'unknown')
        concepts = result.get('concept_count', 0)
        boosted = result.get('boosted_concepts', 0)
        score_boosted = result.get('score_boosted_concepts', 0)
        filename = result.get('filename', 'unknown')
        avg_score = result.get('average_score', 0)
        
        status_icon = "✅" if status == "success" else "❌"
        yield_warning = " ⚠️" if concepts < 5 else ""
        
        print(f"   {status_icon} {filename}: {concepts} concepts (DB:{boosted}, Score:{score_boosted}, avg:{avg_score:.2f}){yield_warning}")
    
    print("-" * 80)

def demo_threshold_testing(pdf_path: str):
    """Demo testing different thresholds to find optimal value"""
    print(f"\n🧬 THRESHOLD TESTING DEMO")
    print(f"📄 Testing thresholds on: {Path(pdf_path).name}")
    print(f"🔧 This will help you find the optimal extraction threshold")
    print("-" * 80)
    
    thresholds = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    
    print(f"Testing thresholds: {thresholds}")
    print()
    
    results = []
    for threshold in thresholds:
        print(f"🔧 Testing threshold: {threshold}")
        result = ingest_pdf_clean(pdf_path, extraction_threshold=threshold)
        
        concept_count = result.get('concept_count', 0)
        avg_score = result.get('average_score', 0)
        high_conf = result.get('high_confidence_concepts', 0)
        
        results.append({
            'threshold': threshold,
            'concepts': concept_count,
            'avg_score': avg_score,
            'high_conf': high_conf,
            'status': result.get('status', 'unknown')
        })
        
        yield_status = "✅ Good" if concept_count >= 5 else "⚠️ Low" if concept_count > 0 else "❌ None"
        print(f"   Result: {concept_count} concepts, avg score: {avg_score:.3f}, {yield_status}")
        print()
    
    print("📊 THRESHOLD ANALYSIS:")
    print("Threshold | Concepts | Avg Score | High Conf | Status")
    print("-" * 55)
    for r in results:
        status_icon = "✅" if r['concepts'] >= 5 else "⚠️" if r['concepts'] > 0 else "❌"
        print(f"   {r['threshold']:4.2f}   |    {r['concepts']:3d}   |   {r['avg_score']:5.3f}   |    {r['high_conf']:3d}    | {status_icon}")
    
    # Find optimal threshold
    valid_results = [r for r in results if r['concepts'] > 0]
    if valid_results:
        # Optimal = highest concept count with reasonable average score
        optimal = max(valid_results, key=lambda x: x['concepts'] + (x['avg_score'] * 2))
        print(f"\n🎯 RECOMMENDED THRESHOLD: {optimal['threshold']}")
        print(f"   Concepts: {optimal['concepts']}")
        print(f"   Average Score: {optimal['avg_score']:.3f}")
        print(f"   High Confidence: {optimal['high_conf']}")
        
        if optimal['concepts'] >= 10:
            print("   ✅ Excellent concept extraction!")
        elif optimal['concepts'] >= 5:
            print("   👍 Good concept extraction")
        else:
            print("   ⚠️ Consider lowering threshold further or checking document content")
    
    print("-" * 80)

def demo_concept_extraction(text_sample: str, threshold: float = 0.35):
    """Demo the concept extraction process on a text sample with threshold control"""
    print(f"\n🧬 CONCEPT EXTRACTION DEMO")
    print(f"📝 Sample Text: {text_sample[:100]}...")
    print(f"🔧 Extraction Threshold: {threshold}")
    print("-" * 80)
    
    concepts = extract_and_boost_concepts(text_sample, threshold=threshold)
    
    print(f"\n🔍 EXTRACTION RESULTS:")
    print(f"   Total Concepts Found: {len(concepts)}")
    
    semantic_concepts = [c for c in concepts if 'semantic' in c.get('method', '')]
    boosted_concepts = [c for c in concepts if c.get('method') == 'file_storage_boosted']
    score_boosted = [c for c in concepts if c.get('metadata', {}).get('score_boosted', False)]
    
    print(f"   🧬 Semantic Concepts: {len(semantic_concepts)}")
    print(f"   🚀 Database Boosted: {len(boosted_concepts)}")
    print(f"   🎯 Score Boosted: {len(score_boosted)}")
    
    if concepts:
        print(f"\n🧠 FOUND CONCEPTS:")
        for i, concept in enumerate(concepts, 1):
            method = concept.get('method', 'unknown')
            score = concept.get('score', 0)
            boosted = "🎯" if concept.get('metadata', {}).get('score_boosted', False) else ""
            print(f"   {i:2}. {concept['name']} (score: {score:.3f}, method: {method}) {boosted}")
    
    print("-" * 80)

def main():
    """Main demo function with enhanced options"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo_enhanced_pipeline.py path/to/document.pdf")
        print("  python demo_enhanced_pipeline.py path/to/document.pdf --threshold 0.3")
        print("  python demo_enhanced_pipeline.py path/to/pdf_directory/ --batch")
        print("  python demo_enhanced_pipeline.py path/to/document.pdf --test-thresholds")
        print("  python demo_enhanced_pipeline.py --demo-text")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Parse threshold argument
    threshold = 0.35  # Default lowered threshold
    if "--threshold" in sys.argv:
        try:
            threshold_idx = sys.argv.index("--threshold") + 1
            threshold = float(sys.argv[threshold_idx])
            print(f"🔧 Using custom threshold: {threshold}")
        except (IndexError, ValueError):
            print("⚠️ Invalid threshold value, using default: 0.35")
    
    if arg == "--demo-text":
        # Demo with sample text about AI and machine learning
        sample_text = """
        Artificial intelligence and machine learning are revolutionizing modern technology.
        Deep learning neural networks enable sophisticated pattern recognition capabilities.
        Natural language processing allows computers to understand human communication.
        Computer vision systems can analyze and interpret visual information automatically.
        Quantum computing promises to solve complex computational problems exponentially faster.
        Crystallizing spacetime geometries reveal new insights into gravitational wave propagation.
        Non-Abelian anyons demonstrate topological quantum computing advantages.
        """
        demo_concept_extraction(sample_text, threshold)
        return
    
    path = Path(arg)
    
    if not path.exists():
        print(f"❌ Path does not exist: {arg}")
        sys.exit(1)
    
    if "--test-thresholds" in sys.argv:
        if path.is_file() and path.suffix.lower() == '.pdf':
            demo_threshold_testing(str(path))
        else:
            print("❌ Threshold testing requires a single PDF file")
            sys.exit(1)
    elif "--batch" in sys.argv or path.is_dir():
        # Batch processing
        demo_batch_processing(str(path), threshold)
    else:
        # Single file
        if not path.suffix.lower() == '.pdf':
            print(f"❌ File must be a PDF: {arg}")
            sys.exit(1)
        demo_single_pdf(str(path), threshold)
    
    print("\n🎯 Demo complete! Your concepts have been injected into the ConceptMesh system.")
    print("💡 If you're still seeing low concept yields, try:")
    print("   - Using --test-thresholds to find optimal extraction threshold")
    print("   - Adding domain-specific terms to your concept_file_storage.json")
    print("   - Checking the console logs for concepts that are being filtered out")

if __name__ == "__main__":
    main()
