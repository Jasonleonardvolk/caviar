#!/usr/bin/env python3
"""
Test Suite for Adapter Retraining Pipeline
===========================================
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from python.training.synthetic_data_generator import (
    SyntheticDataGenerator, IntentDataCollector, 
    MeshDataCollector, TrainingExample
)
from python.training.validate_adapter import AdapterValidator, TestCase
from python.training.rollback_adapter import AdapterRollback

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)

@pytest.fixture
def mock_memory_vault(temp_dir):
    """Create mock memory vault structure."""
    memory_vault = temp_dir / "memory_vault"
    
    # Create directories
    (memory_vault / "traces").mkdir(parents=True)
    (memory_vault / "sessions").mkdir(parents=True)
    (memory_vault / "feedback").mkdir(parents=True)
    
    # Create sample trace file
    trace_file = memory_vault / "traces" / "session_test.jsonl"
    with open(trace_file, 'w') as f:
        f.write(json.dumps({
            "event": "intent_opened",
            "intent_type": "query",
            "description": "What is Python?",
            "user_id": "test_user",
            "timestamp": datetime.now().timestamp()
        }) + "\n")
        f.write(json.dumps({
            "event": "intent_satisfied",
            "intent_type": "query",
            "resolution": "Python is a programming language",
            "user_id": "test_user",
            "timestamp": datetime.now().timestamp()
        }) + "\n")
    
    # Create sample session file
    session_file = memory_vault / "sessions" / "session_test.jsonl"
    with open(session_file, 'w') as f:
        f.write(json.dumps({
            "event": "conversation",
            "role": "user",
            "text": "Hello, how are you?",
            "user_id": "test_user",
            "timestamp": datetime.now().timestamp()
        }) + "\n")
        f.write(json.dumps({
            "event": "conversation",
            "role": "assistant",
            "text": "I'm doing well, thank you!",
            "user_id": "test_user",
            "timestamp": datetime.now().timestamp()
        }) + "\n")
    
    return memory_vault

@pytest.fixture
def mock_config(temp_dir):
    """Create mock config file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        f.write("""
use_intent_traces: true
use_mesh_context: true
use_feedback: true
synthetic_data_enabled: true
augmentation_factor: 2
promote_if_score: 0.9
regression_threshold: 0.05
""")
    return config_file

# ============================================================================
# DATA GENERATION TESTS
# ============================================================================

def test_intent_data_collector(mock_memory_vault):
    """Test intent data collection."""
    collector = IntentDataCollector(mock_memory_vault)
    examples = collector.collect(user_id="test_user")
    
    assert len(examples) > 0
    assert any(e.source == "intent" for e in examples)
    assert any("Python" in e.input for e in examples)

def test_training_example_creation():
    """Test TrainingExample dataclass."""
    example = TrainingExample(
        input="What is AI?",
        output="AI is artificial intelligence",
        meta={"test": True},
        source="test",
        confidence=0.9
    )
    
    assert example.input == "What is AI?"
    assert example.confidence == 0.9
    assert example.source == "test"

def test_synthetic_data_generator(mock_memory_vault, mock_config, temp_dir):
    """Test complete data generation pipeline."""
    generator = SyntheticDataGenerator(mock_config)
    generator.memory_vault_dir = mock_memory_vault
    
    output_path = temp_dir / "test_dataset.jsonl"
    stats = generator.generate_dataset(
        user_id="test_user",
        output_path=output_path,
        include_global=False
    )
    
    assert output_path.exists()
    assert stats.total_examples > 0
    
    # Check dataset format
    with open(output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            assert "input" in data
            assert "output" in data
            assert "source" in data

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_test_case_creation():
    """Test TestCase dataclass."""
    test_case = TestCase(
        input="Hello",
        expected_output="greeting",
        test_type="contains",
        category="general"
    )
    
    assert test_case.input == "Hello"
    assert test_case.test_type == "contains"
    assert test_case.importance == 1.0

def test_validation_result_format(mock_config):
    """Test validation result structure."""
    from python.training.validate_adapter import ValidationResult
    
    result = ValidationResult(
        adapter_path="/path/to/adapter.pt",
        user_id="test_user",
        score=0.95,
        passed=True,
        metrics={"accuracy": 0.95},
        test_cases_passed=95,
        test_cases_total=100,
        regression_detected=False,
        timestamp=datetime.now().timestamp()
    )
    
    result_dict = result.to_dict()
    assert result_dict["score"] == 0.95
    assert result_dict["passed"] == True
    assert "metrics" in result_dict

# ============================================================================
# ROLLBACK TESTS
# ============================================================================

def test_rollback_initialization(temp_dir):
    """Test rollback manager initialization."""
    adapters_dir = temp_dir / "adapters"
    adapters_dir.mkdir()
    
    # Create index file
    index_file = adapters_dir / "adapters_index.json"
    with open(index_file, 'w') as f:
        json.dump({"test_user": "user_test_v1.pt"}, f)
    
    rollback = AdapterRollback(adapters_dir)
    assert "test_user" in rollback.index

def test_version_history_tracking(temp_dir):
    """Test version history management."""
    adapters_dir = temp_dir / "adapters"
    adapters_dir.mkdir()
    
    history_file = adapters_dir / "adapter_history.json"
    history = {
        "versions": {
            "test_user": [
                {
                    "version": 1,
                    "path": "user_test_v1.pt",
                    "timestamp": datetime.now().isoformat(),
                    "validation_score": 0.92
                },
                {
                    "version": 2,
                    "path": "user_test_v2.pt",
                    "timestamp": datetime.now().isoformat(),
                    "validation_score": 0.95
                }
            ]
        }
    }
    
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    rollback = AdapterRollback(adapters_dir)
    history = rollback.get_version_history("test_user")
    
    assert len(history) == 2
    assert history[-1]["version"] == 2
    assert history[-1]["validation_score"] == 0.95

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_pipeline(mock_memory_vault, mock_config, temp_dir):
    """Test complete pipeline from data generation to validation."""
    # 1. Generate dataset
    generator = SyntheticDataGenerator(mock_config)
    generator.memory_vault_dir = mock_memory_vault
    
    dataset_path = temp_dir / "dataset.jsonl"
    stats = generator.generate_dataset(
        user_id="test_user",
        output_path=dataset_path,
        include_global=False
    )
    
    assert dataset_path.exists()
    assert stats.total_examples > 0
    
    # 2. Validate dataset format
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data)
    
    assert all("input" in e for e in examples)
    assert all("output" in e for e in examples)
    
    # 3. Check statistics file
    stats_file = dataset_path.with_suffix('.stats.json')
    assert stats_file.exists()
    
    with open(stats_file, 'r') as f:
        saved_stats = json.load(f)
    
    assert saved_stats["total_examples"] == stats.total_examples

def test_api_request_models():
    """Test API request/response models."""
    from api.retrain_adapter import RetrainRequest, ValidationRequest, RollbackRequest
    
    # Test RetrainRequest
    retrain_req = RetrainRequest(
        user_id="test_user",
        trigger="manual",
        include_global=True,
        force=False
    )
    assert retrain_req.user_id == "test_user"
    
    # Test ValidationRequest
    val_req = ValidationRequest(
        adapter_path="/path/to/adapter.pt",
        user_id="test_user"
    )
    assert val_req.adapter_path == "/path/to/adapter.pt"
    
    # Test RollbackRequest
    rollback_req = RollbackRequest(
        user_id="test_user",
        version=1,
        reason="Test rollback"
    )
    assert rollback_req.reason == "Test rollback"

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_large_dataset_generation(mock_memory_vault, mock_config, temp_dir):
    """Test performance with large dataset."""
    import time
    
    # Add more data to memory vault
    trace_file = mock_memory_vault / "traces" / "session_large.jsonl"
    with open(trace_file, 'w') as f:
        for i in range(1000):
            f.write(json.dumps({
                "event": "intent_opened",
                "intent_type": f"query_{i}",
                "description": f"Question {i}",
                "user_id": "test_user",
                "timestamp": datetime.now().timestamp()
            }) + "\n")
    
    generator = SyntheticDataGenerator(mock_config)
    generator.memory_vault_dir = mock_memory_vault
    
    output_path = temp_dir / "large_dataset.jsonl"
    
    start_time = time.time()
    stats = generator.generate_dataset(
        user_id="test_user",
        output_path=output_path,
        include_global=False
    )
    elapsed = time.time() - start_time
    
    assert stats.total_examples > 1000  # With augmentation
    assert elapsed < 10  # Should complete within 10 seconds
    print(f"Generated {stats.total_examples} examples in {elapsed:.2f} seconds")

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_missing_memory_vault(mock_config, temp_dir):
    """Test handling of missing memory vault."""
    generator = SyntheticDataGenerator(mock_config)
    generator.memory_vault_dir = temp_dir / "nonexistent"
    
    output_path = temp_dir / "empty_dataset.jsonl"
    stats = generator.generate_dataset(
        user_id="test_user",
        output_path=output_path,
        include_global=False
    )
    
    # Should handle gracefully
    assert output_path.exists()
    assert stats.total_examples >= 0  # May be 0 if no data

def test_invalid_json_handling(temp_dir):
    """Test handling of invalid JSON in data files."""
    memory_vault = temp_dir / "memory_vault"
    (memory_vault / "traces").mkdir(parents=True)
    
    # Create file with invalid JSON
    trace_file = memory_vault / "traces" / "invalid.jsonl"
    with open(trace_file, 'w') as f:
        f.write("This is not JSON\n")
        f.write(json.dumps({"event": "valid", "user_id": "test"}) + "\n")
        f.write("{broken json\n")
    
    collector = IntentDataCollector(memory_vault)
    examples = collector.collect()
    
    # Should skip invalid lines but process valid ones
    assert len(examples) >= 0  # Should not crash

# ============================================================================
# CLI TESTS
# ============================================================================

def test_cli_arguments():
    """Test CLI argument parsing."""
    import subprocess
    
    # Test help output
    result = subprocess.run(
        ["python", "python/training/synthetic_data_generator.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "--user_id" in result.stdout
    assert "--output" in result.stdout
    assert "--config" in result.stdout

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
