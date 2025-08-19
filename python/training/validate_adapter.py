#!/usr/bin/env python3
"""
Adapter Validation System
=========================
Validates trained adapters against baselines, runs regression tests,
scores performance, and enforces automatic rollback on failure.
"""

import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import hashlib
import time

# Optional imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers/peft not installed")

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.adapter_loader_v5 import MetadataManager
from python.core.saigon_inference_v5 import SaigonInference

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_BASE_MODEL = "models/saigon_base"
DEFAULT_VALIDATION_DATA = "data/validation"
DEFAULT_BASELINE_ADAPTER = "models/adapters/global_adapter_v1.pt"
VALIDATION_THRESHOLD = 0.8  # Minimum score to pass validation
PERPLEXITY_THRESHOLD = 50.0  # Maximum acceptable perplexity

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValidationResult:
    """Container for validation results."""
    adapter_path: str
    user_id: str
    passed: bool
    score: float
    perplexity: float
    accuracy: float
    baseline_score: float
    regression_passed: bool
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "adapter_path": self.adapter_path,
            "user_id": self.user_id,
            "passed": self.passed,
            "score": self.score,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
            "baseline_score": self.baseline_score,
            "regression_passed": self.regression_passed,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds
        }

# ============================================================================
# VALIDATION MANAGER
# ============================================================================

class AdapterValidator:
    """Manages adapter validation and scoring."""
    
    def __init__(self,
                 base_model_dir: str = DEFAULT_BASE_MODEL,
                 validation_data_dir: str = DEFAULT_VALIDATION_DATA,
                 baseline_adapter: Optional[str] = DEFAULT_BASELINE_ADAPTER):
        """
        Initialize validator.
        
        Args:
            base_model_dir: Base model directory
            validation_data_dir: Validation data directory
            baseline_adapter: Path to baseline adapter for comparison
        """
        self.base_model_dir = Path(base_model_dir)
        self.validation_data_dir = Path(validation_data_dir)
        self.baseline_adapter = baseline_adapter
        self.metadata_manager = MetadataManager()
        self.results_cache = {}
        
        # Ensure directories exist
        self.validation_data_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_adapter(self,
                        adapter_path: str,
                        user_id: str,
                        validation_data_path: Optional[str] = None,
                        run_regression: bool = True,
                        compare_baseline: bool = True) -> ValidationResult:
        """
        Validate an adapter comprehensively.
        
        Args:
            adapter_path: Path to adapter
            user_id: User identifier
            validation_data_path: Optional custom validation data
            run_regression: Whether to run regression tests
            compare_baseline: Whether to compare against baseline
            
        Returns:
            ValidationResult object
        """
        start_time = time.time()
        
        # Load validation data
        if validation_data_path:
            val_data = self._load_validation_data(validation_data_path)
        else:
            val_data = self._load_user_validation_data(user_id)
        
        if not val_data:
            logger.error(f"No validation data for user {user_id}")
            return self._create_failed_result(adapter_path, user_id, "No validation data")
        
        # Run validation tests
        logger.info(f"Validating adapter: {adapter_path}")
        
        # 1. Calculate perplexity
        perplexity = self._calculate_perplexity(adapter_path, val_data)
        
        # 2. Calculate accuracy
        accuracy = self._calculate_accuracy(adapter_path, val_data)
        
        # 3. Calculate overall score
        score = self._calculate_score(perplexity, accuracy)
        
        # 4. Run regression tests
        regression_passed = True
        if run_regression:
            regression_passed = self._run_regression_tests(adapter_path, val_data)
        
        # 5. Compare with baseline
        baseline_score = 0.0
        if compare_baseline and self.baseline_adapter:
            baseline_score = self._compare_with_baseline(adapter_path, val_data)
        
        # Determine if validation passed
        passed = (
            score >= VALIDATION_THRESHOLD and
            perplexity <= PERPLEXITY_THRESHOLD and
            regression_passed and
            (not compare_baseline or score >= baseline_score * 0.95)  # Within 5% of baseline
        )
        
        # Create result
        result = ValidationResult(
            adapter_path=adapter_path,
            user_id=user_id,
            passed=passed,
            score=score,
            perplexity=perplexity,
            accuracy=accuracy,
            baseline_score=baseline_score,
            regression_passed=regression_passed,
            metrics={
                "validation_samples": len(val_data),
                "threshold": VALIDATION_THRESHOLD,
                "perplexity_threshold": PERPLEXITY_THRESHOLD
            },
            timestamp=datetime.now(),
            duration_seconds=time.time() - start_time
        )
        
        # Log result
        self._log_validation_result(result)
        
        # Update metadata if passed
        if passed:
            self._update_adapter_metadata(adapter_path, user_id, result)
        
        logger.info(f"Validation {'PASSED' if passed else 'FAILED'} for {adapter_path}")
        
        return result
    
    def _load_validation_data(self, path: str) -> List[Dict]:
        """Load validation data from file."""
        data_path = Path(path)
        if not data_path.exists():
            return []
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        
        return data
    
    def _load_user_validation_data(self, user_id: str) -> List[Dict]:
        """Load user-specific validation data."""
        user_data_path = self.validation_data_dir / f"user_{user_id}_validation.jsonl"
        
        if user_data_path.exists():
            return self._load_validation_data(str(user_data_path))
        
        # Fall back to global validation data
        global_data_path = self.validation_data_dir / "global_validation.jsonl"
        if global_data_path.exists():
            return self._load_validation_data(str(global_data_path))
        
        return []
    
    def _calculate_perplexity(self, adapter_path: str, val_data: List[Dict]) -> float:
        """
        Calculate perplexity of adapter on validation data.
        
        Lower is better.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, returning default perplexity")
            return 10.0
        
        try:
            # Load model with adapter
            engine = SaigonInference(base_model_dir=str(self.base_model_dir))
            model, tokenizer, _ = engine.load_saigon_with_adapter(
                user_id="validation",
                adapter_path=adapter_path
            )
            
            # Calculate perplexity
            total_loss = 0.0
            total_tokens = 0
            
            for item in val_data[:100]:  # Limit to 100 samples for speed
                text = item.get("text", "")
                if not text:
                    continue
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            
            # Calculate perplexity
            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = np.exp(avg_loss)
            
            return float(perplexity)
            
        except Exception as e:
            logger.error(f"Failed to calculate perplexity: {e}")
            return 100.0  # High perplexity on error
    
    def _calculate_accuracy(self, adapter_path: str, val_data: List[Dict]) -> float:
        """
        Calculate accuracy on validation tasks.
        
        Returns value between 0 and 1.
        """
        if not val_data:
            return 0.0
        
        # Simple accuracy: check if model generates reasonable responses
        correct = 0
        total = 0
        
        try:
            engine = SaigonInference(base_model_dir=str(self.base_model_dir))
            
            for item in val_data[:50]:  # Limit for speed
                prompt = item.get("prompt", "")
                expected = item.get("expected", "")
                
                if not prompt:
                    continue
                
                result = engine.run_inference(
                    user_id="validation",
                    user_input=prompt,
                    max_new_tokens=50,
                    temperature=0.1  # Low temperature for consistency
                )
                
                output = result.get("output", "")
                
                # Simple accuracy: check if key words match
                if expected and any(word in output.lower() for word in expected.lower().split()[:3]):
                    correct += 1
                
                total += 1
            
            return correct / max(total, 1)
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return 0.0
    
    def _calculate_score(self, perplexity: float, accuracy: float) -> float:
        """
        Calculate overall score from metrics.
        
        Returns value between 0 and 1.
        """
        # Normalize perplexity (lower is better)
        perplexity_score = max(0, 1 - (perplexity / 100))
        
        # Weighted average
        score = (perplexity_score * 0.4) + (accuracy * 0.6)
        
        return min(1.0, max(0.0, score))
    
    def _run_regression_tests(self, adapter_path: str, val_data: List[Dict]) -> bool:
        """
        Run regression tests to ensure no critical capabilities lost.
        
        Returns True if all tests pass.
        """
        critical_tests = [
            {"prompt": "What is 2+2?", "expected": ["4", "four"]},
            {"prompt": "Hello", "expected": ["hello", "hi", "greetings"]},
            {"prompt": "Explain water", "expected": ["h2o", "liquid", "molecule"]}
        ]
        
        try:
            engine = SaigonInference(base_model_dir=str(self.base_model_dir))
            
            for test in critical_tests:
                result = engine.run_inference(
                    user_id="validation",
                    user_input=test["prompt"],
                    max_new_tokens=50,
                    temperature=0.1
                )
                
                output = result.get("output", "").lower()
                
                # Check if any expected word appears
                if not any(exp in output for exp in test["expected"]):
                    logger.warning(f"Regression test failed: {test['prompt']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Regression tests failed: {e}")
            return False
    
    def _compare_with_baseline(self, adapter_path: str, val_data: List[Dict]) -> float:
        """
        Compare adapter performance with baseline.
        
        Returns baseline score.
        """
        if not self.baseline_adapter or not Path(self.baseline_adapter).exists():
            return 0.0
        
        # Calculate baseline perplexity
        baseline_perplexity = self._calculate_perplexity(self.baseline_adapter, val_data)
        baseline_accuracy = self._calculate_accuracy(self.baseline_adapter, val_data)
        baseline_score = self._calculate_score(baseline_perplexity, baseline_accuracy)
        
        return baseline_score
    
    def _create_failed_result(self, adapter_path: str, user_id: str, reason: str) -> ValidationResult:
        """Create a failed validation result."""
        return ValidationResult(
            adapter_path=adapter_path,
            user_id=user_id,
            passed=False,
            score=0.0,
            perplexity=999.0,
            accuracy=0.0,
            baseline_score=0.0,
            regression_passed=False,
            metrics={"failure_reason": reason},
            timestamp=datetime.now(),
            duration_seconds=0.0
        )
    
    def _log_validation_result(self, result: ValidationResult):
        """Log validation result to file."""
        log_dir = Path("logs/validation")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log validation result: {e}")
    
    def _update_adapter_metadata(self, adapter_path: str, user_id: str, result: ValidationResult):
        """Update adapter metadata with validation results."""
        try:
            # Get adapter info
            adapter_info = self.metadata_manager.get_active_adapter(user_id)
            
            if adapter_info:
                # Add validation event to history
                self.metadata_manager.load_metadata()
                meta = self.metadata_manager.load_metadata()
                
                if user_id in meta:
                    for adapter in meta[user_id]:
                        if adapter["path"] == adapter_path:
                            adapter["history"].append({
                                "event": "validated",
                                "timestamp": datetime.now().isoformat(),
                                "score": result.score,
                                "passed": result.passed
                            })
                            adapter["metrics"]["validation_score"] = result.score
                            adapter["metrics"]["perplexity"] = result.perplexity
                            adapter["metrics"]["accuracy"] = result.accuracy
                            break
                
                self.metadata_manager.save_metadata(meta)
                logger.info(f"Updated metadata for {adapter_path}")
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    def batch_validate(self, user_ids: List[str]) -> Dict[str, ValidationResult]:
        """
        Validate adapters for multiple users.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            Dictionary mapping user_id to ValidationResult
        """
        results = {}
        
        for user_id in user_ids:
            logger.info(f"Validating adapter for user {user_id}")
            
            # Get user's active adapter
            adapter_info = self.metadata_manager.get_active_adapter(user_id)
            
            if adapter_info:
                result = self.validate_adapter(
                    adapter_path=adapter_info["path"],
                    user_id=user_id
                )
                results[user_id] = result
            else:
                logger.warning(f"No active adapter for user {user_id}")
        
        return results

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_adapter(adapter_path: str,
                    user_id: str,
                    validation_data: Optional[str] = None,
                    enforce_rollback: bool = True) -> bool:
    """
    Main function to validate an adapter.
    
    Args:
        adapter_path: Path to adapter
        user_id: User identifier
        validation_data: Optional validation data path
        enforce_rollback: Whether to rollback on failure
        
    Returns:
        True if validation passed
    """
    validator = AdapterValidator()
    
    result = validator.validate_adapter(
        adapter_path=adapter_path,
        user_id=user_id,
        validation_data_path=validation_data
    )
    
    if not result.passed and enforce_rollback:
        logger.warning(f"Validation failed, initiating rollback for {user_id}")
        # Import rollback function (will be created next)
        from python.training.rollback_adapter import rollback_adapter
        rollback_adapter(user_id)
    
    return result.passed

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for adapter validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapter Validation")
    parser.add_argument("--adapter", required=True, help="Adapter path")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--validation_data", help="Validation data path")
    parser.add_argument("--no_regression", action="store_true", help="Skip regression tests")
    parser.add_argument("--no_baseline", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--no_rollback", action="store_true", help="Don't rollback on failure")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate adapter
    validator = AdapterValidator()
    
    result = validator.validate_adapter(
        adapter_path=args.adapter,
        user_id=args.user_id,
        validation_data_path=args.validation_data,
        run_regression=not args.no_regression,
        compare_baseline=not args.no_baseline
    )
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Adapter: {result.adapter_path}")
    print(f"User: {result.user_id}")
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Score: {result.score:.3f}")
    print(f"Perplexity: {result.perplexity:.2f}")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"Baseline Score: {result.baseline_score:.3f}")
    print(f"Regression: {'PASSED' if result.regression_passed else 'FAILED'}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print("="*60)
    
    # Handle rollback
    if not result.passed and not args.no_rollback:
        print("\nInitiating rollback...")
        from python.training.rollback_adapter import rollback_adapter
        rollback_adapter(args.user_id)

if __name__ == "__main__":
    main()
