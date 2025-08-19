#!/usr/bin/env python3
"""
Test script to verify concept storage is working
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_concept_storage():
    """Test that concepts can be stored without errors"""
    
    # Import the memory sculptor
    from ingest_pdf.memory_sculptor import MemorySculptor
    
    sculptor = MemorySculptor()
    
    # Test with proper user ID
    test_concept = {
        "name": "Test Concept",
        "text": "This is a test concept to verify storage is working",
        "score": 0.85,
        "metadata": {"source": "test"}
    }
    
    logger.info("Testing with valid user ID...")
    result = await sculptor.sculpt_and_store(
        user_id="testuser",
        raw_concept=test_concept,
        metadata={"test": True}
    )
    
    if result:
        logger.info("✅ Concept stored successfully!")
    else:
        logger.error("❌ Failed to store concept")
    
    # Test with default user (should warn once)
    logger.info("\nTesting with default user ID...")
    result2 = await sculptor.sculpt_and_store(
        user_id="default",
        raw_concept=test_concept,
        metadata={"test": True}
    )
    
    # Try again to verify it only warns once
    result3 = await sculptor.sculpt_and_store(
        user_id="default",
        raw_concept=test_concept,
        metadata={"test": True}
    )
    
    logger.info("\n✅ Test complete - check logs above")


if __name__ == "__main__":
    asyncio.run(test_concept_storage())
