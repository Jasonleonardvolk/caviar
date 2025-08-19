#!/usr/bin/env python3
"""
Main Inference Engine - Phase 5 (Modular, Multi-User Safe)
===========================================================
Handles prompt assembly, model and adapter loading, and context injection
with complete multi-user safety, domain support, and production features.
"""

import json
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime
import time
import hashlib
from functools import lru_cache

# Optional imports with fallback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not installed. Install with: pip install transformers")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] peft not installed. Install with: pip install peft")

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.adapter_loader_v5 import (
    get_adapter_path_for_user,
    get_domain_adapter_path,
    get_global_adapter_path,
    MetadataManager
)
from python.core.concept_mesh_v5 import (
    MeshManager,
    load_mesh_context
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_BASE_MODEL_DIR = "models/saigon_base"
DEFAULT_ADAPTERS_DIR = "models/adapters"
DEFAULT_MESH_DIR = "data/mesh_contexts"
DEFAULT_LOGS_DIR = "logs/inference"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_CACHE_SIZE = 5

# ============================================================================
# MODEL CACHE
# ============================================================================

class ModelCache:
    """Thread-safe LRU model cache for multi-user inference."""
    
    def __init__(self, maxsize: int = DEFAULT_MAX_CACHE_SIZE):
        self.maxsize = maxsize
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cache_key(self, base_model: str, adapter_path: Optional[str], user_id: str) -> str:
        """Generate unique cache key."""
        key_str = f"{base_model}:{adapter_path or 'vanilla'}:{user_id}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, base_model: str, adapter_path: Optional[str], user_id: str) -> Optional[Any]:
        """Retrieve cached model."""
        key = self.get_cache_key(base_model, adapter_path, user_id)
        
        if key in self.cache:
            self.access_times[key] = datetime.now()
            self.hit_count += 1
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, base_model: str, adapter_path: Optional[str], user_id: str, model: Any):
        """Store model in cache with LRU eviction."""
        key = self.get_cache_key(base_model, adapter_path, user_id)
        
        # Evict LRU if at capacity
        if len(self.cache) >= self.maxsize and key not in self.cache:
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
            logger.info(f"Evicted LRU model: {lru_key}")
        
        self.cache[key] = model
        self.access_times[key] = datetime.now()
        logger.debug(f"Cached model: {key}")
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Model cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

# Global cache instance
_model_cache = ModelCache()

# ============================================================================
# SAIGON INFERENCE ENGINE
# ============================================================================

class SaigonInference:
    """
    Production-ready multi-user inference engine with adapter support.
    """
    
    def __init__(self,
                 base_model_dir: str = DEFAULT_BASE_MODEL_DIR,
                 adapters_dir: str = DEFAULT_ADAPTERS_DIR,
                 mesh_dir: str = DEFAULT_MESH_DIR,
                 device: str = DEFAULT_DEVICE,
                 use_cache: bool = True,
                 log_dir: str = DEFAULT_LOGS_DIR):
        """
        Initialize Saigon inference engine.
        
        Args:
            base_model_dir: Path to base model
            adapters_dir: Directory containing adapters
            mesh_dir: Directory containing mesh contexts
            device: Device to use (cuda/cpu)
            use_cache: Whether to use model caching
            log_dir: Directory for inference logs
        """
        self.base_model_dir = Path(base_model_dir)
        self.adapters_dir = Path(adapters_dir)
        self.mesh_dir = Path(mesh_dir)
        self.device = device
        self.use_cache = use_cache
        self.log_dir = Path(log_dir)
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.adapter_manager = MetadataManager()
        self.mesh_manager = MeshManager(str(self.mesh_dir))
        
        # Session state
        self.current_model = None
        self.current_tokenizer = None
        self.current_user_id = None
        self.current_adapter_path = None
        
        logger.info(f"Initialized SaigonInference on device: {self.device}")
    
    def load_saigon_with_adapter(self,
                                user_id: str,
                                domain: Optional[str] = None,
                                adapter_path: Optional[str] = None,
                                force_reload: bool = False) -> Tuple[Any, Any, Optional[str]]:
        """
        Load model with appropriate adapter for user.
        
        Args:
            user_id: User identifier
            domain: Optional domain for adapter selection
            adapter_path: Explicit adapter path (overrides auto-selection)
            force_reload: Force reload even if cached
            
        Returns:
            (model, tokenizer, adapter_path_used)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not installed")
        
        # Determine adapter path
        if adapter_path is None:
            if domain:
                adapter_path = get_domain_adapter_path(user_id, domain, str(self.adapters_dir))
            
            if not adapter_path:
                adapter_path = get_adapter_path_for_user(user_id, str(self.adapters_dir))
            
            if not adapter_path:
                adapter_path = get_global_adapter_path(str(self.adapters_dir))
        
        # Check cache
        if self.use_cache and not force_reload:
            cached_model = _model_cache.get(
                str(self.base_model_dir),
                adapter_path,
                user_id
            )
            if cached_model:
                tokenizer = self._load_tokenizer()
                logger.info(f"Using cached model for user {user_id}")
                return cached_model, tokenizer, adapter_path
        
        # Load base model
        logger.info(f"Loading base model from {self.base_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load adapter if available
        if adapter_path and Path(adapter_path).exists():
            if PEFT_AVAILABLE:
                logger.info(f"Loading adapter: {adapter_path}")
                
                # Verify adapter integrity
                adapter_info = self.adapter_manager.get_active_adapter(user_id)
                if adapter_info and adapter_info.get("sha256"):
                    if not self.adapter_manager.verify_adapter(
                        adapter_path,
                        adapter_info["sha256"]
                    ):
                        logger.warning(f"Adapter integrity check failed for {adapter_path}")
                
                model = PeftModel.from_pretrained(model, adapter_path)
            else:
                logger.warning("PEFT not available, skipping adapter injection")
        
        # Move to device
        if self.device != "cuda" or not hasattr(model, "device_map"):
            model.to(self.device)
        
        # Load tokenizer
        tokenizer = self._load_tokenizer()
        
        # Cache model
        if self.use_cache:
            _model_cache.put(
                str(self.base_model_dir),
                adapter_path,
                user_id,
                model
            )
        
        # Update session state
        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_user_id = user_id
        self.current_adapter_path = adapter_path
        
        return model, tokenizer, adapter_path
    
    def _load_tokenizer(self) -> Any:
        """Load tokenizer for the model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not installed")
        
        tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_dir))
        
        # Set padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def assemble_prompt(self,
                       user_input: str,
                       mesh_context: Optional[Dict[str, Any]] = None,
                       system_prompt: Optional[str] = None,
                       include_mesh: bool = True,
                       include_nodes: bool = True) -> str:
        """
        Assemble final prompt with context.
        
        Args:
            user_input: User's input text
            mesh_context: Mesh context dictionary
            system_prompt: Optional system prompt
            include_mesh: Whether to include mesh summary
            include_nodes: Whether to include relevant nodes
            
        Returns:
            Assembled prompt string
        """
        parts = []
        
        # Add system prompt
        if system_prompt:
            parts.append(f"[System] {system_prompt}")
        
        # Add mesh context
        if mesh_context and include_mesh:
            # Add summary
            summary = mesh_context.get("summary", "")
            if summary:
                parts.append(f"[Context] {summary}")
            
            # Add relevant nodes
            if include_nodes:
                nodes = mesh_context.get("nodes", [])[:3]  # Top 3 nodes
                if nodes:
                    node_labels = ", ".join([n.get("label", "") for n in nodes])
                    parts.append(f"[Knowledge] Related concepts: {node_labels}")
        
        # Add user input
        parts.append(f"[User] {user_input}")
        
        return "\n\n".join(parts)
    
    def generate(self,
                prompt: str,
                model: Any,
                tokenizer: Any,
                max_new_tokens: int = 128,
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True,
                **kwargs) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            model: Model instance
            tokenizer: Tokenizer instance
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated
    
    def run_inference(self,
                     user_id: str,
                     user_input: str,
                     domain: Optional[str] = None,
                     use_mesh_context: bool = True,
                     system_prompt: Optional[str] = None,
                     max_new_tokens: int = 128,
                     temperature: float = 0.7,
                     log_inference: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Complete inference pipeline.
        
        Args:
            user_id: User identifier
            user_input: User's input text
            domain: Optional domain for adapter selection
            use_mesh_context: Whether to use mesh context
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            log_inference: Whether to log this inference
            **kwargs: Additional generation arguments
            
        Returns:
            Dict with output and metadata
        """
        start_time = time.time()
        
        # Load model with adapter
        model, tokenizer, adapter_path = self.load_saigon_with_adapter(
            user_id=user_id,
            domain=domain
        )
        
        # Load mesh context
        mesh_context = None
        if use_mesh_context:
            mesh_context = self.mesh_manager.load_mesh(user_id)
        
        # Assemble prompt
        prompt = self.assemble_prompt(
            user_input=user_input,
            mesh_context=mesh_context,
            system_prompt=system_prompt
        )
        
        # Generate
        try:
            output = self.generate(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            output = "I apologize, but I encountered an error generating a response."
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare result
        result = {
            "user_id": user_id,
            "domain": domain,
            "input": user_input,
            "output": output,
            "prompt": prompt,
            "adapter_used": adapter_path,
            "mesh_context_used": mesh_context is not None,
            "device": self.device,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt is not None
            }
        }
        
        # Log inference
        if log_inference:
            self._log_inference(result)
        
        return result
    
    def _log_inference(self, result: Dict[str, Any]):
        """Log inference details."""
        log_file = self.log_dir / f"inference_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log inference: {e}")
    
    def clear_cache(self):
        """Clear model cache."""
        _model_cache.clear()
        logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        cache_stats = _model_cache.get_statistics()
        
        return {
            "device": self.device,
            "base_model": str(self.base_model_dir),
            "current_user": self.current_user_id,
            "current_adapter": self.current_adapter_path,
            "cache": cache_stats
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_inference(user_input: str,
                   user_id: str = "default",
                   domain: Optional[str] = None) -> str:
    """
    Quick inference for simple use cases.
    
    Args:
        user_input: Input text
        user_id: User identifier
        domain: Optional domain
        
    Returns:
        Generated text
    """
    engine = SaigonInference()
    result = engine.run_inference(
        user_id=user_id,
        user_input=user_input,
        domain=domain
    )
    return result["output"]

def batch_inference(requests: List[Dict[str, Any]],
                   parallel: bool = False) -> List[Dict[str, Any]]:
    """
    Process batch of inference requests.
    
    Args:
        requests: List of request dictionaries
        parallel: Whether to process in parallel (not implemented)
        
    Returns:
        List of results
    """
    engine = SaigonInference()
    results = []
    
    for request in requests:
        result = engine.run_inference(**request)
        results.append(result)
    
    return results

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for Saigon inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Saigon Inference Engine")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--domain", default=None, help="Domain for adapter selection")
    parser.add_argument("--base_model_dir", default=DEFAULT_BASE_MODEL_DIR,
                       help="Base model directory")
    parser.add_argument("--adapters_dir", default=DEFAULT_ADAPTERS_DIR,
                       help="Adapters directory")
    parser.add_argument("--mesh_dir", default=DEFAULT_MESH_DIR,
                       help="Mesh contexts directory")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                       help="Device (cuda/cpu)")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable model caching")
    parser.add_argument("--no_mesh", action="store_true",
                       help="Disable mesh context")
    parser.add_argument("--system_prompt", default=None,
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
        domain=args.domain,
        use_mesh_context=not args.no_mesh,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Print results
    print("\n" + "="*60)
    print("SAIGON INFERENCE RESULTS")
    print("="*60)
    print(f"User: {result['user_id']}")
    print(f"Domain: {result.get('domain', 'general')}")
    print(f"Adapter: {result['adapter_used'] or 'Base model'}")
    print(f"Mesh Context: {'Yes' if result['mesh_context_used'] else 'No'}")
    print(f"Device: {result['device']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print("-"*60)
    print(f"Input: {result['input']}")
    print("-"*60)
    print(f"Output:\n{result['output']}")
    print("="*60)
    
    # Print cache statistics
    stats = engine.get_statistics()
    cache_stats = stats["cache"]
    if cache_stats["total_requests"] > 0:
        print(f"\nCache Statistics:")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Size: {cache_stats['size']}/{cache_stats['maxsize']}")

if __name__ == "__main__":
    main()
