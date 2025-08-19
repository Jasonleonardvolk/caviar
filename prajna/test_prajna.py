#!/usr/bin/env python3
"""
Prajna Test Suite
================

Quick test to verify all Prajna components are working correctly.
Run this after installation to ensure everything is set up properly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add prajna to path
sys.path.insert(0, str(Path(__file__).parent))

from prajna.config.prajna_config import load_config
from prajna.core.prajna_mouth import PrajnaLanguageModel
from prajna.memory.context_builder import build_context, ContextResult, MemorySnippet
from prajna.audit.alien_overlay import audit_prajna_answer, ghost_feedback_analysis

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prajna.test")

async def test_configuration():
    """Test configuration system"""
    logger.info("🧪 Testing configuration system...")
    
    try:
        config = load_config(
            use_env=False,
            model_type="demo",
            debug_mode=True
        )
        
        assert config.model_type == "demo"
        assert config.debug_mode == True
        
        logger.info("✅ Configuration system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def test_language_model():
    """Test Prajna language model"""
    logger.info("🧪 Testing Prajna language model...")
    
    try:
        # Initialize demo model
        prajna = PrajnaLanguageModel(model_type="demo")
        await prajna.load_model()
        
        # Create mock context
        context = ContextResult(
            text="This is test context about quantum dynamics.",
            sources=["test_source.txt"],
            snippets=[MemorySnippet(
                text="Quantum dynamics test content",
                source="test_source.txt",
                relevance_score=0.9,
                memory_type="test"
            )],
            total_relevance=0.9,
            retrieval_time=0.1,
            concept_coverage={"quantum", "dynamics"}
        )
        
        # Generate response
        output = await prajna.generate_answer(
            "What is quantum dynamics?", 
            context
        )
        
        assert output.answer
        assert len(output.answer) > 0
        
        await prajna.cleanup()
        
        logger.info("✅ Language model working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Language model test failed: {e}")
        return False

async def test_context_builder():
    """Test context building system"""
    logger.info("🧪 Testing context builder...")
    
    try:
        # Test with no memory systems (should return fallback)
        context = await build_context(
            user_query="What is TORI?",
            focus_concept="TORI",
            soliton_memory=None,  # No memory systems for test
            concept_mesh=None
        )
        
        assert context.text
        assert context.sources
        
        logger.info("✅ Context builder working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Context builder test failed: {e}")
        return False

async def test_audit_system():
    """Test alien overlay audit system"""
    logger.info("🧪 Testing audit system...")
    
    try:
        # Create test context
        context = ContextResult(
            text="Test context about phase dynamics and coherence.",
            sources=["test_doc.pdf"],
            snippets=[],
            total_relevance=1.0,
            retrieval_time=0.1,
            concept_coverage={"phase", "dynamics"}
        )
        
        # Test answer with potential alien content
        test_answer = "Phase dynamics involves coherent oscillations. However, according to general knowledge, this is commonly understood."
        
        # Run audit
        audit_result = await audit_prajna_answer(test_answer, context)
        
        assert "trust_score" in audit_result
        assert "alien_detections" in audit_result
        assert "phase_analysis" in audit_result
        
        # Run ghost analysis
        ghost_result = await ghost_feedback_analysis(
            test_answer, 
            context, 
            original_query="What is phase dynamics?"
        )
        
        assert "ghost_questions" in ghost_result
        assert "completeness_score" in ghost_result
        
        logger.info("✅ Audit system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Audit system test failed: {e}")
        return False

async def test_end_to_end():
    """Test complete end-to-end pipeline"""
    logger.info("🧪 Testing end-to-end pipeline...")
    
    try:
        # Initialize components
        config = load_config(model_type="demo", debug_mode=True)
        prajna = PrajnaLanguageModel(model_type="demo")
        await prajna.load_model()
        
        # Build context
        context = await build_context(
            user_query="What is Prajna?",
            focus_concept="Prajna"
        )
        
        # Generate answer
        output = await prajna.generate_answer("What is Prajna?", context)
        
        # Audit answer
        audit_result = await audit_prajna_answer(output.answer, context)
        ghost_result = await ghost_feedback_analysis(
            output.answer, 
            context,
            original_query="What is Prajna?"
        )
        
        # Verify pipeline
        assert output.answer
        assert audit_result["trust_score"] >= 0.0
        assert ghost_result["completeness_score"] >= 0.0
        
        await prajna.cleanup()
        
        logger.info("✅ End-to-end pipeline working")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        return False

async def run_all_tests():
    """Run all Prajna tests"""
    logger.info("🚀 Starting Prajna test suite...")
    
    tests = [
        ("Configuration", test_configuration()),
        ("Language Model", test_language_model()),
        ("Context Builder", test_context_builder()),
        ("Audit System", test_audit_system()),
        ("End-to-End", test_end_to_end())
    ]
    
    results = []
    for test_name, test_coro in tests:
        logger.info(f"🔄 Running {test_name} test...")
        result = await test_coro
        results.append((test_name, result))
    
    # Report results
    logger.info("\n📊 Test Results:")
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status} - {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    total = len(results)
    logger.info(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Prajna is ready to serve.")
        return True
    else:
        logger.error(f"💥 {failed} tests failed. Please check the logs above.")
        return False

def main():
    """Main test entry point"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
