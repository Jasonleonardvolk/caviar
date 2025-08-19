#!/usr/bin/env python3
"""
Continue the pipeline test and show results
"""

import asyncio
import logging
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def continue_pipeline():
    """Continue from where the test left off"""
    
    # Import the full pipeline test
    from test_full_pipeline import test_full_pipeline
    
    # Run it
    logger.info("🚀 Running full pipeline with enhanced deduplication...")
    summary = await test_full_pipeline()
    
    return summary


def analyze_results():
    """Analyze the extraction results"""
    summary_file = Path("extraction_summary.json")
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        logger.info("\n" + "="*60)
        logger.info("📊 EXTRACTION ANALYSIS")
        logger.info("="*60)
        
        # Show deduplication effectiveness
        dedup = summary.get('deduplication_results', {})
        logger.info(f"\n🧬 Deduplication Results:")
        logger.info(f"  Input concepts: {dedup.get('input_concepts', 0)}")
        logger.info(f"  Output concepts: {dedup.get('output_concepts', 0)}")
        logger.info(f"  Reduction: {dedup.get('reduction_percentage', 0)}%")
        
        # Show top concepts with relationships
        logger.info(f"\n🏆 Top Concepts with Relationships:")
        for i, concept in enumerate(summary.get('top_concepts', [])[:10]):
            logger.info(f"\n  {i+1}. {concept['label']}")
            logger.info(f"     Score: {concept['score']:.3f}")
            logger.info(f"     Method: {concept['method']}")
            logger.info(f"     Relationships: {concept['relationships']}")
            logger.info(f"     Tags: {', '.join(concept['tags'])}")
        
        # Show extraction methods used
        methods = summary.get('extraction_results', {}).get('extraction_methods', [])
        logger.info(f"\n🛠️ Extraction Methods Used:")
        for method in methods:
            logger.info(f"  - {method}")
        
        return summary
    else:
        logger.warning("⚠️ No extraction summary found. Run the pipeline first.")
        return None


if __name__ == "__main__":
    # Run the pipeline
    summary = asyncio.run(continue_pipeline())
    
    # Analyze results
    if summary:
        analyze_results()
        
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info("📁 Check the following locations:")
        logger.info("  - extraction_summary.json - Full extraction report")
        logger.info("  - data/memory_vault/memories/ - Stored concepts")
        logger.info("  - data/memory_vault/logs/ - Vault logs")
