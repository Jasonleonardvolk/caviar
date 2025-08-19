#!/usr/bin/env python3
"""
Test the unified enriched ingestion system
Shows how all modalities now use the same enrichment pipeline
"""

import logging
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_all_ingestion():
    """Test all ingestion routes with sample data"""
    
    # Import all ingestion modules
    from ingest_bus.audio.audio_ingest import ingest_audio_clean
    from ingest_bus.video.video_ingest import ingest_video_clean
    from ingest_pdf.pipeline.docx_ingest import ingest_docx_clean
    
    logger.info("="*70)
    logger.info("UNIFIED ENRICHED INGESTION TEST")
    logger.info("="*70)
    
    # Test paths (you'll need to provide actual files)
    test_files = {
        'audio': Path('test_data/sample.wav'),
        'video': Path('test_data/sample.mp4'),
        'docx': Path('test_data/sample.docx'),
        'pdf': Path('test_data/sample.pdf')
    }
    
    # For demo purposes, let's create dummy test files
    test_data_dir = Path('test_data')
    test_data_dir.mkdir(exist_ok=True)
    
    # Create dummy files for testing
    for file_type, file_path in test_files.items():
        if not file_path.exists():
            logger.info(f"Creating dummy {file_type} file: {file_path}")
            file_path.write_text(f"This is a dummy {file_type} file for testing")
    
    # Test each ingestion route
    results = {}
    
    # Test audio ingestion
    if test_files['audio'].exists():
        logger.info("\n🎧 Testing audio ingestion...")
        try:
            count = ingest_audio_clean(test_files['audio'])
            results['audio'] = f"✅ Success: {count} concepts"
        except Exception as e:
            results['audio'] = f"❌ Failed: {e}"
    
    # Test video ingestion
    if test_files['video'].exists():
        logger.info("\n🎥 Testing video ingestion...")
        try:
            count = ingest_video_clean(test_files['video'])
            results['video'] = f"✅ Success: {count} concepts"
        except Exception as e:
            results['video'] = f"❌ Failed: {e}"
    
    # Test DOCX ingestion
    if test_files['docx'].exists():
        logger.info("\n📄 Testing DOCX ingestion...")
        try:
            count = ingest_docx_clean(test_files['docx'])
            results['docx'] = f"✅ Success: {count} concepts"
        except Exception as e:
            results['docx'] = f"❌ Failed: {e}"
    
    # Test PDF ingestion (if you have the existing pipeline)
    if test_files['pdf'].exists():
        logger.info("\n📑 Testing PDF ingestion...")
        try:
            # Your existing PDF pipeline
            # from ingest_pdf.pipeline import ingest_pdf_clean
            # result = ingest_pdf_clean(test_files['pdf'])
            results['pdf'] = "⏭️ Skipped (use existing pipeline)"
        except Exception as e:
            results['pdf'] = f"❌ Failed: {e}"
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    for modality, result in results.items():
        logger.info(f"{modality.upper()}: {result}")
    
    # Check vault
    vault_dir = Path("data/memory_vault/memories")
    if vault_dir.exists():
        files = list(vault_dir.glob("tori_mem_*.json"))
        logger.info(f"\n💾 Total concepts in vault: {len(files)}")
        
        # Show a sample enriched concept
        if files:
            import json
            with open(files[-1], 'r') as f:
                sample = json.load(f)
            
            logger.info("\n📊 Sample enriched concept:")
            logger.info(json.dumps(sample, indent=2))


if __name__ == "__main__":
    test_all_ingestion()
