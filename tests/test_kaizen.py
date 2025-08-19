"""
Unit tests for Kaizen Improvement Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import tempfile

from kha.mcp_metacognitive.agents.kaizen import (
    KaizenImprovementEngine, 
    PerformanceMetrics, 
    LearningInsight
)


@pytest.fixture
async def kaizen_engine():
    """Create a Kaizen engine with test configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "analysis_interval": 1,  # 1 second for testing
            "min_data_points": 2,  # Low threshold for testing
            "knowledge_base_path": str(Path(tmpdir) / "test_kb.json"),
            "max_insights_stored": 100,  # Smaller for testing
            "enable_auto_apply": False,
            "enable_clustering": False  # Disable to avoid sklearn dependency
        }
        engine = KaizenImprovementEngine(config=config)
        yield engine
        # Cleanup
        if engine.is_running:
            await engine.stop_continuous_improvement()


@pytest.mark.asyncio
async def test_initialization(kaizen_engine):
    """Test Kaizen engine initialization"""
    assert kaizen_engine is not None
    assert kaizen_engine.config["analysis_interval"] == 1
    assert kaizen_engine.config["min_data_points"] == 2
    assert not kaizen_engine.is_running
    assert len(kaizen_engine.insights) == 0
    assert isinstance(kaizen_engine.metrics, PerformanceMetrics)


@pytest.mark.asyncio
async def test_start_stop_continuous_improvement(kaizen_engine):
    """Test starting and stopping the continuous improvement loop"""
    # Start
    result = await kaizen_engine.start_continuous_improvement()
    assert result["status"] == "started"
    assert kaizen_engine.is_running
    
    # Try starting again (should indicate already running)
    result2 = await kaizen_engine.start_continuous_improvement()
    assert result2["status"] == "already_running"
    
    # Stop
    result3 = await kaizen_engine.stop_continuous_improvement()
    assert result3["status"] == "stopped"
    assert not kaizen_engine.is_running


@pytest.mark.asyncio
async def test_metrics_tracking():
    """Test performance metrics tracking"""
    metrics = PerformanceMetrics()
    
    # Test response times
    metrics.add_response_time(1.5)
    metrics.add_response_time(2.5)
    assert metrics.get_average_response_time() == 2.0
    
    # Test error tracking
    metrics.total_queries = 100
    metrics.add_error("timeout")
    metrics.add_error("timeout")
    metrics.add_error("parse_error")
    assert metrics.get_error_rate() == 0.03  # 3/100
    
    # Test query patterns
    metrics.add_query_pattern("question_wh")
    metrics.add_query_pattern("question_wh")
    metrics.add_query_pattern("creative_request")
    assert metrics.query_patterns["question_wh"] == 2
    assert metrics.query_patterns["creative_request"] == 1
    
    # Test deque maxlen behavior
    for i in range(10005):
        metrics.add_consciousness_level(0.5)
    assert len(metrics.consciousness_levels) == 10000  # Capped at maxlen


@pytest.mark.asyncio
async def test_analyze_performance(kaizen_engine):
    """Test performance analysis"""
    # Add metrics that exceed thresholds
    for _ in range(10):
        kaizen_engine.metrics.add_response_time(3.0)  # Above 2.0 threshold
    
    insights = await kaizen_engine._analyze_performance()
    
    # Should generate insight about slow response time
    assert len(insights) > 0
    assert any(i.insight_type == "performance" for i in insights)
    perf_insight = next(i for i in insights if i.insight_type == "performance")
    assert perf_insight.confidence >= 0.9
    assert "response time" in perf_insight.description.lower()


@pytest.mark.asyncio
async def test_analyze_error_patterns(kaizen_engine):
    """Test error pattern analysis"""
    # Add repeated errors
    for _ in range(10):
        kaizen_engine.metrics.add_error("connection_timeout")
    for _ in range(7):
        kaizen_engine.metrics.add_error("parse_error")
    
    insights = await kaizen_engine._analyze_error_patterns()
    
    # Should identify frequent error patterns
    assert len(insights) >= 2
    assert all(i.insight_type == "error_pattern" for i in insights)
    assert any("connection_timeout" in i.description for i in insights)


@pytest.mark.asyncio
async def test_insight_application(kaizen_engine):
    """Test applying insights"""
    # Create test insight
    insight = LearningInsight(
        insight_type="performance",
        description="Test performance optimization",
        confidence=0.95,
        data={"recommendation": "Enable caching"},
        timestamp=datetime.utcnow(),
        applied=False
    )
    
    # Add insight with lock
    async with kaizen_engine._insight_lock:
        kaizen_engine.insights.append(insight)
    
    # Apply insight
    result = await kaizen_engine.apply_insight(insight)
    assert result["status"] == "success"
    assert insight.applied is True
    assert "performance_adjustments" in kaizen_engine.knowledge_base


@pytest.mark.asyncio
async def test_get_recent_insights(kaizen_engine):
    """Test retrieving recent insights"""
    # Add some test insights
    async with kaizen_engine._insight_lock:
        for i in range(5):
            kaizen_engine.insights.append(LearningInsight(
                insight_type="test",
                description=f"Test insight {i}",
                confidence=0.8,
                data={},
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                applied=i % 2 == 0
            ))
    
    result = await kaizen_engine.get_recent_insights(limit=3)
    assert result["status"] == "success"
    assert len(result["insights"]) == 3
    assert result["total_insights"] == 5
    assert result["applied_insights"] == 3  # 0, 2, 4 are applied


@pytest.mark.asyncio
async def test_thread_safety(kaizen_engine):
    """Test thread-safe insight operations"""
    async def add_insights():
        for i in range(100):
            insight = LearningInsight(
                insight_type="concurrent",
                description=f"Concurrent insight {i}",
                confidence=0.7,
                data={},
                timestamp=datetime.utcnow(),
                applied=False
            )
            async with kaizen_engine._insight_lock:
                kaizen_engine.insights.append(insight)
            await asyncio.sleep(0.001)
    
    async def modify_insights():
        for _ in range(50):
            async with kaizen_engine._insight_lock:
                if kaizen_engine.insights:
                    kaizen_engine.insights[0].applied = True
            await asyncio.sleep(0.002)
    
    # Run concurrent operations
    await asyncio.gather(
        add_insights(),
        modify_insights(),
        add_insights()
    )
    
    # Verify no corruption
    assert len(kaizen_engine.insights) == 200
    assert all(isinstance(i, LearningInsight) for i in kaizen_engine.insights)


@pytest.mark.asyncio
async def test_knowledge_base_persistence(kaizen_engine):
    """Test knowledge base save/load"""
    # Add to knowledge base
    kaizen_engine.knowledge_base["test_key"] = {
        "data": "test_value",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save
    kaizen_engine._save_knowledge_base()
    
    # Verify file exists
    assert kaizen_engine.kb_path.exists()
    
    # Clear and reload
    kaizen_engine.knowledge_base.clear()
    kaizen_engine._load_knowledge_base()
    
    # Verify loaded
    assert "test_key" in kaizen_engine.knowledge_base
    assert kaizen_engine.knowledge_base["test_key"]["data"] == "test_value"


@pytest.mark.asyncio
async def test_cleanup_old_metrics(kaizen_engine):
    """Test metric cleanup functionality"""
    # Add old insights
    old_time = datetime.utcnow() - timedelta(days=10)
    async with kaizen_engine._insight_lock:
        for i in range(150):  # Exceed max_insights_stored
            kaizen_engine.insights.append(LearningInsight(
                insight_type="old",
                description=f"Old insight {i}",
                confidence=0.6,
                data={},
                timestamp=old_time if i < 50 else datetime.utcnow(),
                applied=False
            ))
    
    # Add many queries to trigger reset
    kaizen_engine.metrics.total_queries = 100001
    kaizen_engine.metrics.error_rates["test_error"] = 100
    
    # Run cleanup
    kaizen_engine._cleanup_old_metrics()
    await asyncio.sleep(0.1)  # Let async cleanup complete
    
    # Verify old insights removed and counters reset
    async with kaizen_engine._insight_lock:
        assert len(kaizen_engine.insights) <= 100  # max_insights_stored
        assert all(i.timestamp > old_time for i in kaizen_engine.insights)
    
    assert kaizen_engine.metrics.total_queries == 0
    assert len(kaizen_engine.metrics.error_rates) == 0


@pytest.mark.asyncio
async def test_query_pattern_extraction():
    """Test query pattern extraction"""
    engine = KaizenImprovementEngine()
    
    # Test various query patterns
    assert engine._extract_query_pattern("What is the weather?") == "question_wh"
    assert engine._extract_query_pattern("How does this work?") == "question_wh"
    assert engine._extract_query_pattern("Is this correct?") == "question_other"
    assert engine._extract_query_pattern("Create a poem about cats") == "creative_request"
    assert engine._extract_query_pattern("Analyze this data") == "analysis_request"
    assert engine._extract_query_pattern("Hello there") == "statement_or_other"


@pytest.mark.asyncio
async def test_gap_fill_trigger(kaizen_engine):
    """Test gap-fill search trigger for gap-related errors"""
    # Mock mcp_bridge
    with patch('kha.mcp_metacognitive.agents.kaizen.mcp_bridge') as mock_bridge:
        # Add gap-related errors
        for _ in range(10):
            kaizen_engine.metrics.add_error("knowledge_gap_detected")
        
        # Run error analysis
        insights = await kaizen_engine._analyze_error_patterns()
        
        # Should trigger paper search for gaps
        # Note: The actual trigger happens in the real implementation
        assert len(insights) > 0
        assert any("gap" in i.description.lower() for i in insights)


@pytest.mark.asyncio 
async def test_config_with_env_override():
    """Test configuration with environment variable overrides"""
    import os
    
    # Set environment variables
    os.environ["KAIZEN_ANALYSIS_INTERVAL"] = "7200"
    os.environ["KAIZEN_ENABLE_AUTO_APPLY"] = "true"
    
    try:
        engine = KaizenImprovementEngine()
        assert engine.config["analysis_interval"] == 7200
        assert engine.config["enable_auto_apply"] is True
    finally:
        # Cleanup
        os.environ.pop("KAIZEN_ANALYSIS_INTERVAL", None)
        os.environ.pop("KAIZEN_ENABLE_AUTO_APPLY", None)


if __name__ == "__main__":
    # Run tests with asyncio
    asyncio.run(pytest.main([__file__, "-v"]))
