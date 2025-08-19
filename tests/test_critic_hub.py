"""
Test suite for the Critics Hub functionality
"""

import pytest
from kha.meta_genome.critics.critic_hub import evaluate

def test_veto():
    """Test that safety failures result in rejection"""
    accepted, consensus, _ = evaluate({
        "safety_pass": False,
        "tests_passed": 10,
        "tests_total": 10
    })
    assert accepted is False

def test_pass():
    """Test that good metrics result in approval"""
    accepted, consensus, _ = evaluate({
        "safety_pass": True,
        "tests_passed": 10,
        "tests_total": 10,
        "lambda_max": 0.01,
        "energy_overshoot_pct": 0
    })
    assert accepted is True and consensus >= 0.75

def test_partial_test_failure():
    """Test behavior with partial test failures"""
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 5,
        "tests_total": 10,  # 50% pass rate
        "lambda_max": 0.02,
        "energy_overshoot_pct": 10
    })
    # With 50% test pass rate, should likely be rejected
    assert "test_critic" in per_critic
    assert per_critic["test_critic"][0] == 0.5  # Score should reflect pass rate

def test_high_lambda_rejection():
    """Test that high lambda values (instability) cause rejection"""
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 10,
        "tests_total": 10,
        "lambda_max": 0.1,  # Very high instability
        "energy_overshoot_pct": 0
    })
    # High lambda should trigger stability critic rejection
    assert "stability_critic" in per_critic
    assert per_critic["stability_critic"][0] < 0.75  # Should fail threshold

def test_energy_overshoot():
    """Test energy overshoot handling"""
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 10,
        "tests_total": 10,
        "lambda_max": 0.01,
        "energy_overshoot_pct": 80  # 80% overshoot
    })
    # High energy overshoot should reduce energy critic score
    assert "energy_critic" in per_critic
    assert per_critic["energy_critic"][0] == 0.2  # 1.0 - 80/100

def test_critic_scores_structure():
    """Test that per_critic returns expected structure"""
    _, _, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 10,
        "tests_total": 10,
        "lambda_max": 0.02,
        "energy_overshoot_pct": 20
    })
    
    # Check structure: {critic_id: (score, passed)}
    for critic_id, result in per_critic.items():
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, passed = result
        assert 0.0 <= score <= 1.0
        assert isinstance(passed, bool)

def test_missing_metrics():
    """Test graceful handling of missing metrics"""
    # Minimal report with missing optional metrics
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 8,
        "tests_total": 10
        # lambda_max and energy_overshoot_pct missing
    })
    
    # Should still work, using default values
    assert isinstance(accepted, bool)
    assert isinstance(consensus, float)
    assert 0.0 <= consensus <= 1.0

def test_edge_case_zero_tests():
    """Test handling when no tests exist"""
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 0,
        "tests_total": 0,  # Edge case: no tests
        "lambda_max": 0.01,
        "energy_overshoot_pct": 0
    })
    
    # Should handle gracefully, likely with low test_critic score
    assert "test_critic" in per_critic
    # Implementation should handle division by zero

def test_perfect_scores():
    """Test perfect scores across all critics"""
    accepted, consensus, per_critic = evaluate({
        "safety_pass": True,
        "tests_passed": 100,
        "tests_total": 100,
        "lambda_max": 0.0,  # Perfect stability
        "energy_overshoot_pct": 0  # Perfect energy usage
    })
    
    assert accepted is True
    assert consensus >= 0.9  # Should be very high
    
    # All critics should pass
    for critic_id, (score, passed) in per_critic.items():
        assert passed is True
        assert score >= 0.9  # All should have high scores

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
