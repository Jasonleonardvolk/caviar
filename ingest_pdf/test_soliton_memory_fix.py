"""
test_soliton_memory_fix.py

Test script to verify the Soliton Memory Storage Bug fixes
"""

import asyncio
import logging
from datetime import datetime
from memory_sculptor import MemorySculptor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_soliton_fix")

async def test_edge_cases():
    """Test various edge cases that could cause the missing parameters bug"""
    sculptor = MemorySculptor()
    
    test_cases = [
        {
            "name": "Valid user_id and content",
            "user_id": "test_user_123",
            "concept": {
                "id": "concept_1",
                "text": "This is valid test content about quantum computing.",
                "score": 0.8
            },
            "should_succeed": True
        },
        {
            "name": "Empty user_id",
            "user_id": "",
            "concept": {
                "id": "concept_2",
                "text": "This content has an empty user_id.",
                "score": 0.7
            },
            "should_succeed": False
        },
        {
            "name": "Default user_id",
            "user_id": "default",
            "concept": {
                "id": "concept_3",
                "text": "This content has the default user_id.",
                "score": 0.7
            },
            "should_succeed": False
        },
        {
            "name": "Empty content",
            "user_id": "test_user_456",
            "concept": {
                "id": "concept_4",
                "text": "",
                "score": 0.6
            },
            "should_succeed": False
        },
        {
            "name": "Whitespace-only content",
            "user_id": "test_user_789",
            "concept": {
                "id": "concept_5",
                "text": "   \n\t   ",
                "score": 0.5
            },
            "should_succeed": False
        },
        {
            "name": "None user_id",
            "user_id": None,
            "concept": {
                "id": "concept_6",
                "text": "This content has None as user_id.",
                "score": 0.7
            },
            "should_succeed": False
        },
        {
            "name": "Valid multi-segment content",
            "user_id": "test_user_multi",
            "concept": {
                "id": "concept_7",
                "text": "This is a very long content that should be segmented. " * 50,  # Long content
                "score": 0.9
            },
            "should_succeed": True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test_case['name']}")
        logger.info(f"User ID: '{test_case['user_id']}'")
        logger.info(f"Content length: {len(test_case['concept']['text'])}")
        
        try:
            memory_ids = await sculptor.sculpt_and_store(
                user_id=test_case["user_id"],
                raw_concept=test_case["concept"],
                metadata={"test_case": test_case["name"]}
            )
            
            success = len(memory_ids) > 0
            expected = test_case["should_succeed"]
            
            if success == expected:
                logger.info(f"✅ PASSED: Got expected result (success={success})")
            else:
                logger.error(f"❌ FAILED: Expected success={expected}, got success={success}")
            
            results.append({
                "test": test_case["name"],
                "passed": success == expected,
                "memory_ids": memory_ids,
                "expected_success": expected,
                "actual_success": success
            })
            
        except Exception as e:
            logger.error(f"❌ EXCEPTION: {str(e)}")
            results.append({
                "test": test_case["name"],
                "passed": False,
                "error": str(e),
                "expected_success": test_case["should_succeed"]
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success rate: {(passed/total)*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        logger.info(f"{status} - {result['test']}")
        if not result["passed"]:
            if "error" in result:
                logger.info(f"    Error: {result['error']}")
            else:
                logger.info(f"    Expected: {result['expected_success']}, Got: {result['actual_success']}")

async def test_batch_processing():
    """Test batch processing with relationship detection"""
    sculptor = MemorySculptor()
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing Batch Processing with Relationships")
    logger.info(f"{'='*60}")
    
    # Create concepts with relationships
    concepts = [
        {
            "id": "batch_concept_1",
            "name": "Quantum Computing",
            "text": "Quantum computing uses quantum mechanics principles for computation.",
            "score": 0.9
        },
        {
            "id": "batch_concept_2",
            "name": "Quantum Mechanics",
            "text": "Quantum mechanics is a fundamental theory in physics.",
            "score": 0.85
        },
        {
            "id": "batch_concept_3",
            "name": "Superposition",
            "text": "Superposition is a principle of quantum mechanics used in quantum computing.",
            "score": 0.8
        }
    ]
    
    # Test with valid user_id
    result = await sculptor.sculpt_and_store_batch(
        user_id="test_batch_user",
        concepts=concepts,
        doc_metadata={"source": "test_document", "timestamp": datetime.now().isoformat()}
    )
    
    logger.info(f"Batch processing results:")
    logger.info(f"  Total concepts: {result['total_concepts']}")
    logger.info(f"  Memories created: {len(result['memories_created'])}")
    logger.info(f"  Success rate: {result['success_rate']*100:.1f}%")
    logger.info(f"  Processing time: {result['processing_time']:.2f}s")
    
    if result['relationships_detected']:
        logger.info(f"  Relationships detected:")
        for concept, related in result['relationships_detected'].items():
            logger.info(f"    {concept} -> {', '.join(related)}")
    
    # Test with invalid user_id
    logger.info("\nTesting batch processing with invalid user_id...")
    result_invalid = await sculptor.sculpt_and_store_batch(
        user_id="default",
        concepts=concepts,
        doc_metadata={"source": "test_document"}
    )
    
    if result_invalid['errors']:
        logger.info("✅ Correctly rejected batch with invalid user_id")
        logger.info(f"  Error: {result_invalid['errors'][0]}")
    else:
        logger.error("❌ Failed to reject batch with invalid user_id")

async def main():
    """Run all tests"""
    logger.info("Starting Soliton Memory Storage Bug Fix Tests")
    
    # Run edge case tests
    await test_edge_cases()
    
    # Run batch processing tests
    await test_batch_processing()
    
    logger.info(f"\n{'='*60}")
    logger.info("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
