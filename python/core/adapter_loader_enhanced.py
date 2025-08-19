#!/usr/bin/env python3
"""
Enhanced Adapter Loader with Blending Support
Manages single and multi-adapter loading with hierarchical composition
"""

import os
import json
import torch
import logging
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import torch.nn as nn

# Import blending support
try:
    from adapter_blending import (
        AdapterBlender,
        BlendConfig,
        BlendingMode,
        BlendedAdapter,
        AdapterType
    )
    BLENDING_AVAILABLE = True
except ImportError:
    BLENDING_AVAILABLE = False
    print("[WARNING] Adapter blending not available - check adapter_blending.py import")

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AdapterConfig:
    """Configuration for a LoRA adapter."""
    name: str
    path: str
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["lstm", "linear"])
    user_id: Optional[str] = None
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of adapter file."""
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        return "no_file"

@dataclass
class LoadedAdapter:
    """Represents a loaded adapter with metadata."""
    config: AdapterConfig
    state_dict: Dict[str, torch.Tensor]
    device: str = "cpu"
    load_time: float = 0.0
    
# ============================================================================
# LORA LAYER
# ============================================================================

class LoRALayer(nn.Module):
    """LoRA layer for parameter-efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Freeze by default
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return 0
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# ============================================================================
# ADAPTER MANAGER
# ============================================================================

class AdapterManager:
    """Manages loading, caching, and injection of LoRA adapters with blending support."""
    
    def __init__(self,
                 adapters_dir: str = "models/adapters",
                 cache_size: int = 10,
                 auto_reload: bool = True,
                 enable_blending: bool = True):
        """
        Initialize adapter manager.
        
        Args:
            adapters_dir: Directory containing adapters
            cache_size: Maximum adapters to keep in memory
            auto_reload: Auto-reload if adapter file changes
            enable_blending: Enable multi-adapter blending
        """
        self.adapters_dir = Path(adapters_dir)
        self.index_path = self.adapters_dir / "adapters_index.json"
        self.cache_size = cache_size
        self.auto_reload = auto_reload
        self.enable_blending = enable_blending
        
        # Create directory if needed
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Caches
        self.loaded_adapters: Dict[str, LoadedAdapter] = {}
        self.file_hashes: Dict[str, str] = {}
        self.user_mapping: Dict[str, str] = {}
        
        # Initialize blender if available
        self.blender = None
        if enable_blending and BLENDING_AVAILABLE:
            self.blender = AdapterBlender(adapters_dir=str(adapters_dir))
            logger.info("✓ Adapter blending enabled")
        elif enable_blending and not BLENDING_AVAILABLE:
            logger.warning("⚠️ Blending requested but adapter_blending module not available")
        
        # Statistics
        self.stats = {
            "loads": 0,
            "cache_hits": 0,
            "reloads": 0,
            "fallbacks": 0,
            "blends": 0
        }
        
        # Load user mapping
        self._load_user_mapping()
        
        logger.info(f"AdapterManager initialized: {self.adapters_dir}")
        logger.info(f"  Blending: {'enabled' if self.blender else 'disabled'}")
        logger.info(f"  Cache size: {self.cache_size}")
    
    def _load_user_mapping(self):
        """Load user to adapter mapping from index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                
                # Handle both old (flat) and new (nested) formats
                if "users" in data:
                    # New format
                    for user_id, adapter_info in data.get("users", {}).items():
                        if isinstance(adapter_info, dict):
                            self.user_mapping[user_id] = adapter_info.get("personal", adapter_info.get("adapter"))
                        else:
                            self.user_mapping[user_id] = adapter_info
                else:
                    # Old format - treat as user mapping
                    for key, value in data.items():
                        if not key.startswith("_") and not key in ["metadata", "teams", "global"]:
                            self.user_mapping[key] = value
                
                logger.info(f"Loaded mappings for {len(self.user_mapping)} users")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
    
    def load_adapter(self,
                    user_id: Optional[str] = None,
                    adapter_path: Optional[str] = None,
                    device: str = "cpu") -> Optional[LoadedAdapter]:
        """
        Load a single adapter for a user.
        
        Args:
            user_id: User identifier
            adapter_path: Explicit adapter path
            device: Target device
            
        Returns:
            Loaded adapter or None
        """
        import time
        start_time = time.time()
        
        # Determine adapter file
        if adapter_path:
            adapter_file = Path(adapter_path)
        elif user_id and user_id in self.user_mapping:
            adapter_file = self.adapters_dir / self.user_mapping[user_id]
        else:
            logger.warning(f"No adapter found for user {user_id}")
            self.stats["fallbacks"] += 1
            return None
        
        # Check cache
        cache_key = str(adapter_file)
        if cache_key in self.loaded_adapters:
            self.stats["cache_hits"] += 1
            logger.debug(f"Using cached adapter: {cache_key}")
            return self.loaded_adapters[cache_key]
        
        # Load adapter
        if not adapter_file.exists():
            logger.error(f"Adapter file not found: {adapter_file}")
            return None
        
        try:
            # Load state dict
            state_dict = torch.load(adapter_file, map_location=device)
            
            # Extract LoRA weights if nested
            if isinstance(state_dict, dict) and "lora_weights" in state_dict:
                lora_weights = state_dict["lora_weights"]
                metadata = state_dict.get("metadata", {})
            else:
                lora_weights = state_dict
                metadata = {}
            
            # Create config
            config = AdapterConfig(
                name=metadata.get("name", adapter_file.stem),
                path=str(adapter_file),
                rank=metadata.get("rank", 8),
                user_id=user_id
            )
            
            # Create loaded adapter
            adapter = LoadedAdapter(
                config=config,
                state_dict=lora_weights,
                device=device,
                load_time=time.time() - start_time
            )
            
            # Cache it
            self.loaded_adapters[cache_key] = adapter
            self.stats["loads"] += 1
            
            logger.info(f"Loaded adapter: {config.name} ({adapter.load_time:.3f}s)")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_file}: {e}")
            return None
    
    def load_blended_adapters(self,
                             user_id: Optional[str] = None,
                             team_ids: Optional[List[str]] = None,
                             use_global: bool = True,
                             blend_mode: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> Optional[BlendedAdapter]:
        """
        Load and blend multiple adapters.
        
        Args:
            user_id: User identifier
            team_ids: Team identifiers
            use_global: Include global adapter
            blend_mode: Blending mode override
            context: Context for dynamic blending
            
        Returns:
            Blended adapter or None
        """
        if not self.blender:
            logger.warning("Blending not available - falling back to single adapter")
            # Fallback to single adapter
            adapter = self.load_adapter(user_id)
            return None
        
        # Configure blending mode if specified
        if blend_mode:
            try:
                config = BlendConfig(mode=BlendingMode[blend_mode.upper()])
                self.blender.blend_config = config
            except KeyError:
                logger.error(f"Invalid blend mode: {blend_mode}")
        
        # Load and blend
        blended = self.blender.load_blended_adapters(
            user_id=user_id,
            team_ids=team_ids,
            use_global=use_global,
            context=context
        )
        
        if blended:
            self.stats["blends"] += 1
            info = blended.get_info()
            logger.info(f"Blended {info['num_adapters']} adapters: {info['composition']}")
        
        return blended
    
    def inject_adapter_to_model(self,
                               model: nn.Module,
                               adapter: LoadedAdapter) -> nn.Module:
        """
        Inject single adapter into model.
        
        Args:
            model: Base model
            adapter: Adapter to inject
            
        Returns:
            Model with adapter
        """
        # Apply LoRA weights to matching modules
        for name, module in model.named_modules():
            if any(target in name for target in adapter.config.target_modules):
                # Add LoRA layers
                if hasattr(module, 'weight'):
                    # Check for matching LoRA weights
                    lora_A_key = f"{name}.lora_A"
                    lora_B_key = f"{name}.lora_B"
                    
                    if lora_A_key in adapter.state_dict and lora_B_key in adapter.state_dict:
                        # Create and attach LoRA layer
                        in_features = module.weight.shape[1]
                        out_features = module.weight.shape[0]
                        
                        lora = LoRALayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=adapter.config.rank,
                            alpha=adapter.config.alpha
                        )
                        
                        # Load weights
                        lora.lora_A.data = adapter.state_dict[lora_A_key]
                        lora.lora_B.data = adapter.state_dict[lora_B_key]
                        
                        # Attach to module
                        module.lora = lora
        
        logger.info(f"Injected adapter {adapter.config.name} into model")
        return model
    
    def inject_blended_adapter(self,
                              model: nn.Module,
                              blended: BlendedAdapter) -> nn.Module:
        """
        Inject blended adapter into model.
        
        Args:
            model: Base model
            blended: Blended adapter
            
        Returns:
            Model with blended adapter
        """
        if not self.blender:
            logger.error("Blender not available")
            return model
        
        return self.blender.apply_blended_adapter(model, blended)
    
    def update_user_mapping(self, user_id: str, adapter_filename: str):
        """Update user mapping and save to index."""
        self.user_mapping[user_id] = adapter_filename
        
        # Update index file
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"metadata": {"version": "2.0", "supports_blending": True}}
        
        # Ensure new format
        if "users" not in data:
            data["users"] = {}
        
        data["users"][user_id] = {"personal": adapter_filename}
        
        # Save
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Updated mapping: {user_id} -> {adapter_filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = {
            **self.stats,
            "cached_adapters": len(self.loaded_adapters),
            "user_mappings": len(self.user_mapping),
            "blending_available": self.blender is not None
        }
        
        if self.blender:
            stats["blending_stats"] = self.blender.get_statistics()
        
        return stats

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_saigon_with_adapter(
    base_model_path: str,
    user_id: Optional[str] = None,
    team_ids: Optional[List[str]] = None,
    adapter_path: Optional[str] = None,
    blend_mode: Optional[str] = "hierarchical",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    manager: Optional[AdapterManager] = None
) -> Tuple[nn.Module, Optional[Any]]:
    """
    Load Saigon model with adapter(s).
    
    Args:
        base_model_path: Path to base model
        user_id: User identifier
        team_ids: Team identifiers for blending
        adapter_path: Explicit adapter path
        blend_mode: Blending mode
        device: Target device
        manager: AdapterManager instance
        
    Returns:
        Tuple of (model, adapter_or_blend)
    """
    # Load base model
    logger.info(f"Loading base model from {base_model_path}")
    
    if Path(base_model_path).exists():
        model = torch.load(base_model_path, map_location=device)
    else:
        # Create dummy model for testing
        logger.warning(f"Model not found, creating dummy model")
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.LSTM(256, 256, 2, batch_first=True),
            nn.Linear(256, 256)
        )
    
    model = model.to(device)
    
    # Create manager if needed
    if not manager:
        manager = AdapterManager()
    
    # Try blending first if teams specified
    if team_ids and manager.blender:
        blended = manager.load_blended_adapters(
            user_id=user_id,
            team_ids=team_ids,
            blend_mode=blend_mode
        )
        if blended:
            model = manager.inject_blended_adapter(model, blended)
            logger.info(f"Model ready with blended adapters")
            return model, blended
    
    # Fall back to single adapter
    adapter = manager.load_adapter(
        user_id=user_id,
        adapter_path=adapter_path,
        device=device
    )
    
    if adapter:
        model = manager.inject_adapter_to_model(model, adapter)
        logger.info(f"Model ready with adapter: {adapter.config.name}")
        return model, adapter
    
    logger.info("Model ready without adapter (base only)")
    return model, None

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "AdapterConfig",
    "LoadedAdapter",
    "LoRALayer",
    "AdapterManager",
    "load_saigon_with_adapter",
    "BLENDING_AVAILABLE"
]

# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    print("\n" + "="*60)
    print("ADAPTER LOADER TEST")
    print("="*60)
    
    # Check blending availability
    print(f"\nBlending available: {BLENDING_AVAILABLE}")
    
    # Initialize manager
    manager = AdapterManager(enable_blending=True)
    
    # Test single adapter loading
    print("\n--- Testing Single Adapter ---")
    adapter = manager.load_adapter(user_id="jason")
    if adapter:
        print(f"✓ Loaded: {adapter.config.name}")
    else:
        print("✗ No adapter found for jason")
    
    # Test blended adapter loading
    if manager.blender:
        print("\n--- Testing Blended Adapters ---")
        blended = manager.load_blended_adapters(
            user_id="jason",
            team_ids=["ProjectX"],
            use_global=True,
            blend_mode="hierarchical"
        )
        if blended:
            info = blended.get_info()
            print(f"✓ Blended {info['num_adapters']} adapters")
            print(f"  Composition: {info['composition']}")
        else:
            print("✗ No adapters to blend")
    
    # Show statistics
    print("\n--- Statistics ---")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
