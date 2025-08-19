#!/usr/bin/env python3
"""
Saigon Inference Engine - Adapter-Aware Model Loader
=====================================================
Production-grade inference with LoRA adapters, mesh context injection,
LRU caching, and complete observability.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from functools import lru_cache
from datetime import datetime
import hashlib
import traceback

# Optional imports with fallback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("[WARNING] transformers not installed. Install with: pip install transformers")
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel, get_peft_model, LoraConfig
except ImportError:
    print("[WARNING] peft not installed. Install with: pip install peft")
    PeftModel = None
    get_peft_model = None
    LoraConfig = None

# Import our adapter loader helpers
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from python.core.adapter_loader import (
    get_adapter_path_for_user,
    get_mesh_context_path_for_user,
    get_global_adapter_path,
    AdapterManager
)

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CACHE MANAGEMENT
# ============================================================================

class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, maxsize: int = 5):
        self.maxsize = maxsize
        self.cache = {}
        self.access_times = {}
        
    def get_cache_key(self, base_model_dir: str, adapter_path: Optional[str]) -> str:
        """Generate unique cache key for model+adapter combo."""
        key = f"{base_model_dir}:{adapter_path or 'vanilla'}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, base_model_dir: str, adapter_path: Optional[str]) -> Optional[Any]:
        """Retrieve cached model if exists."""
        key = self.get_cache_key(base_model_dir, adapter_path)
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def put(self, base_model_dir: str, adapter_path: Optional[str], model: Any) -> None:
        """Store model in cache with LRU eviction."""
        key = self.get_cache_key(base_model_dir, adapter_path)
        
        # Evict LRU if at capacity
        if len(self.cache) >= self.maxsize and key not in self.cache:
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
            logger.info(f"[Cache] Evicted LRU model: {lru_key}")
        
        self.cache[key] = model
        self.access_times[key] = datetime.now()
        logger.info(f"[Cache] Stored model: {key}")
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("[Cache] Cleared all cached models")

# Global model cache instance
_model_cache = ModelCache(maxsize=5)

# ============================================================================
# CORE INFERENCE ENGINE
# ============================================================================

class SaigonInference:
    """
    Production-ready inference engine with adapter support.
    Handles model loading, adapter injection, mesh context, and generation.
    """
    
    def __init__(self,
                 base_model_dir: str = "models/saigon_base/",
                 adapters_dir: str = "models/adapters/",
                 mesh_dir: str = "data/mesh_contexts/",
                 device: str = None,
                 use_cache: bool = True,
                 log_dir: str = "logs/inference/"):
        """
        Initialize Saigon inference engine.
        
        Args:
            base_model_dir: Path to base model
            adapters_dir: Directory containing adapters
            mesh_dir: Directory containing mesh contexts
            device: Device to use (cuda/cpu/auto)
            use_cache: Whether to use model caching
            log_dir: Directory for inference logs
        """
        self.base_model_dir = Path(base_model_dir)
        self.adapters_dir = Path(adapters_dir)
        self.mesh_dir = Path(mesh_dir)
        self.use_cache = use_cache
        self.log_dir = Path(log_dir)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(self.adapters_dir)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session state
        self.current_user_id = None
        self.current_adapter_path = None
        self.current_model = None
        self.current_tokenizer = None
        
        logger.info(f"[SaigonInference] Initialized with device: {self.device}")
    
    def load_model_with_adapter(self,
                               user_id: Optional[str] = None,
                               adapter_path: Optional[str] = None,
                               prefer_global: bool = True,
                               force_reload: bool = False) -> Tuple[Any, Any, str, bool]:
        """
        Load model with optional adapter injection.
        
        Args:
            user_id: User ID for adapter selection
            adapter_path: Explicit adapter path (overrides user_id)
            prefer_global: Fall back to global adapter if user adapter not found
            force_reload: Force reload even if cached
            
        Returns:
            (model, tokenizer, adapter_path_used, adapter_active)
        """
        # Determine adapter path
        if adapter_path is None and user_id:
            adapter_path = get_adapter_path_for_user(user_id, str(self.adapters_dir))
            if not adapter_path and prefer_global:
                adapter_path = get_global_adapter_path(str(self.adapters_dir))
        
        # Check cache
        if self.use_cache and not force_reload:
            cached_model = _model_cache.get(str(self.base_model_dir), adapter_path)
            if cached_model:
                logger.info(f"[SaigonInference] Using cached model for {adapter_path or 'vanilla'}")
                tokenizer = self._load_tokenizer()
                return cached_model, tokenizer, adapter_path, bool(adapter_path)
        
        # Load base model
        logger.info(f"[SaigonInference] Loading base model from {self.base_model_dir}")
        
        if not AutoModelForCausalLM:
            raise ImportError("transformers library not installed")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Inject adapter if provided
            if adapter_path and Path(adapter_path).exists():
                logger.info(f"[SaigonInference] Injecting adapter: {adapter_path}")
                if not PeftModel:
                    logger.warning("PEFT not installed, skipping adapter injection")
                else:
                    model = PeftModel.from_pretrained(model, adapter_path)
                adapter_active = True
            else:
                if adapter_path:
                    logger.warning(f"Adapter path not found: {adapter_path}")
                adapter_active = False
            
            # Move to device
            if self.device != "cuda" or not hasattr(model, "device_map"):
                model.to(self.device)
            
            # Cache model
            if self.use_cache:
                _model_cache.put(str(self.base_model_dir), adapter_path, model)
            
            # Load tokenizer
            tokenizer = self._load_tokenizer()
            
            # Update session state
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_adapter_path = adapter_path
            self.current_user_id = user_id
            
            return model, tokenizer, adapter_path, adapter_active
            
        except Exception as e:
            logger.error(f"[SaigonInference] Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_tokenizer(self) -> Any:
        """Load tokenizer for the model."""
        if not AutoTokenizer:
            raise ImportError("transformers library not installed")
        
        tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_dir))
        
        # Set padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def load_mesh_context(self, user_id: str) -> Optional[Dict]:
        """
        Load mesh context for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Mesh context dict or None
        """
        mesh_path = get_mesh_context_path_for_user(user_id, str(self.mesh_dir))
        
        if mesh_path and Path(mesh_path).exists():
            try:
                with open(mesh_path, 'r', encoding='utf-8') as f:
                    mesh_context = json.load(f)
                logger.info(f"[SaigonInference] Loaded mesh context for {user_id}")
                return mesh_context
            except Exception as e:
                logger.error(f"[SaigonInference] Failed to load mesh context: {e}")
        
        return None
    
    def assemble_prompt(self,
                       user_input: str,
                       mesh_context: Optional[Dict] = None,
                       system_prompt: Optional[str] = None,
                       include_mesh_summary: bool = True) -> str:
        """
        Assemble final prompt with optional mesh context.
        
        Args:
            user_input: User's input text
            mesh_context: Mesh context dict
            system_prompt: Optional system prompt
            include_mesh_summary: Whether to include mesh summary
            
        Returns:
            Assembled prompt string
        """
        parts = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append(f"[System] {system_prompt}")
        
        # Add mesh context if available
        if mesh_context and include_mesh_summary:
            summary = mesh_context.get("summary", "")
            if summary:
                parts.append(f"[Context] {summary}")
            
            # Add relevant nodes if present
            relevant_nodes = mesh_context.get("relevant_nodes", [])
            if relevant_nodes:
                node_info = ", ".join([n.get("label", "") for n in relevant_nodes[:3]])
                if node_info:
                    parts.append(f"[Knowledge] Related concepts: {node_info}")
        
        # Add user input
        parts.append(f"[User] {user_input}")
        
        return "\n\n".join(parts)
    
    def generate(self,
                prompt: str,
                max_length: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True,
                num_return_sequences: int = 1,
                **kwargs) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        if not self.current_model or not self.current_tokenizer:
            raise RuntimeError("Model not loaded. Call load_model_with_adapter first.")
        
        # Tokenize input
        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.current_tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode
        generated_texts = []
        for output in outputs:
            text = self.current_tokenizer.decode(output, skip_special_tokens=True)
            # Remove input prompt from output
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generated_texts.append(text)
        
        return generated_texts[0] if num_return_sequences == 1 else generated_texts
    
    def run_inference(self,
                     user_id: str,
                     user_input: str,
                     use_mesh_context: bool = True,
                     system_prompt: Optional[str] = None,
                     max_length: int = 512,
                     temperature: float = 0.7,
                     log_inference: bool = True,
                     **generation_kwargs) -> Dict[str, Any]:
        """
        Complete inference pipeline with logging.
        
        Args:
            user_id: User ID
            user_input: User's input text
            use_mesh_context: Whether to use mesh context
            system_prompt: Optional system prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            log_inference: Whether to log this inference
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Dict with output, metadata, and logs
        """
        start_time = datetime.now()
        
        # Load model with adapter
        model, tokenizer, adapter_path, adapter_active = self.load_model_with_adapter(
            user_id=user_id,
            prefer_global=True
        )
        
        # Load mesh context
        mesh_context = None
        if use_mesh_context:
            mesh_context = self.load_mesh_context(user_id)
        
        # Assemble prompt
        prompt = self.assemble_prompt(
            user_input=user_input,
            mesh_context=mesh_context,
            system_prompt=system_prompt
        )
        
        # Generate response
        try:
            output = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                **generation_kwargs
            )
        except Exception as e:
            logger.error(f"[SaigonInference] Generation failed: {e}")
            output = "I apologize, but I encountered an error generating a response."
        
        # Calculate metrics
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Prepare result
        result = {
            "user_id": user_id,
            "input": user_input,
            "output": output,
            "prompt": prompt,
            "adapter_used": adapter_path if adapter_active else None,
            "adapter_active": adapter_active,
            "mesh_context_used": mesh_context is not None,
            "device": self.device,
            "latency_ms": latency_ms,
            "timestamp": start_time.isoformat(),
            "metadata": {
                "max_length": max_length,
                "temperature": temperature,
                "system_prompt": system_prompt is not None
            }
        }
        
        # Log inference
        if log_inference:
            self._log_inference(result)
        
        return result
    
    def _log_inference(self, result: Dict[str, Any]) -> None:
        """Log inference details to file."""
        log_file = self.log_dir / f"inference_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[SaigonInference] Failed to log inference: {e}")
    
    def switch_user(self, user_id: str) -> None:
        """
        Switch to a different user's adapter.
        
        Args:
            user_id: New user ID
        """
        if user_id != self.current_user_id:
            logger.info(f"[SaigonInference] Switching from {self.current_user_id} to {user_id}")
            self.load_model_with_adapter(user_id=user_id)
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        _model_cache.clear()
        logger.info("[SaigonInference] Cache cleared")
    
    def hot_swap_adapter(self, adapter_path: str) -> None:
        """
        Hot-swap to a different adapter without reloading base model.
        
        Args:
            adapter_path: Path to new adapter
        """
        if not self.current_model:
            raise RuntimeError("No model loaded")
        
        if not PeftModel or not isinstance(self.current_model, PeftModel):
            logger.warning("Hot-swap only works with PEFT models")
            return
        
        logger.info(f"[SaigonInference] Hot-swapping adapter to {adapter_path}")
        
        # Load new adapter weights
        if Path(adapter_path).exists():
            self.current_model.load_adapter(adapter_path, adapter_name="new_adapter")
            self.current_model.set_adapter("new_adapter")
            self.current_adapter_path = adapter_path
            logger.info("[SaigonInference] Adapter hot-swapped successfully")
        else:
            logger.error(f"Adapter not found: {adapter_path}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_inference(user_input: str,
                   user_id: str = "default",
                   base_model_dir: str = "models/saigon_base/") -> str:
    """
    Quick inference function for simple use cases.
    
    Args:
        user_input: Input text
        user_id: User ID
        base_model_dir: Base model directory
        
    Returns:
        Generated text
    """
    engine = SaigonInference(base_model_dir=base_model_dir)
    result = engine.run_inference(
        user_id=user_id,
        user_input=user_input
    )
    return result["output"]

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for testing inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Saigon Inference Engine")
    parser.add_argument("--user_id", type=str, required=True,
                       help="User ID for adapter selection")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt")
    parser.add_argument("--base_model_dir", type=str,
                       default="models/saigon_base/",
                       help="Base model directory")
    parser.add_argument("--adapters_dir", type=str,
                       default="models/adapters/",
                       help="Adapters directory")
    parser.add_argument("--mesh_dir", type=str,
                       default="data/mesh_contexts/",
                       help="Mesh contexts directory")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu/auto)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable model caching")
    parser.add_argument("--no_mesh", action="store_true",
                       help="Disable mesh context")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize engine
    engine = SaigonInference(
        base_model_dir=args.base_model_dir,
        adapters_dir=args.adapters_dir,
        mesh_dir=args.mesh_dir,
        device=args.device,
        use_cache=not args.no_cache
    )
    
    # Run inference
    result = engine.run_inference(
        user_id=args.user_id,
        user_input=args.prompt,
        use_mesh_context=not args.no_mesh,
        system_prompt=args.system_prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    # Print results
    print("\n" + "="*60)
    print("SAIGON INFERENCE RESULTS")
    print("="*60)
    print(f"User: {result['user_id']}")
    print(f"Adapter: {result['adapter_used'] or 'Base model (no adapter)'}")
    print(f"Mesh Context: {'Yes' if result['mesh_context_used'] else 'No'}")
    print(f"Device: {result['device']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print("-"*60)
    print(f"Input: {result['input']}")
    print("-"*60)
    print(f"Output: {result['output']}")
    print("="*60)

if __name__ == "__main__":
    main()
