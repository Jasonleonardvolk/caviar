#!/usr/bin/env python3
"""
Adapter Blending System for TORI
Supports hierarchical composition of personal, team, and global LoRA adapters
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch.nn as nn
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class AdapterType(Enum):
    """Types of adapters in the hierarchy."""
    PERSONAL = "personal"
    TEAM = "team"
    GROUP = "group"  # Alias for team
    DEPARTMENT = "department"
    ORGANIZATION = "organization"
    GLOBAL = "global"

class BlendingMode(Enum):
    """Adapter blending strategies."""
    NONE = "none"                    # No blending (single adapter)
    SEQUENTIAL = "sequential"        # Apply adapters in sequence
    WEIGHTED = "weighted"            # Weighted average of adapter weights
    HIERARCHICAL = "hierarchical"   # Smart hierarchical composition
    DYNAMIC = "dynamic"             # Context-aware selection

class MergeStrategy(Enum):
    """Weight merging strategies for adapters."""
    AVERAGE = "average"          # Simple average
    WEIGHTED_SUM = "weighted_sum"  # Weighted sum
    CONCATENATE = "concatenate"  # Concatenate (for different layers)
    INTERPOLATE = "interpolate"  # Linear interpolation

# Default blending weights for different adapter types
DEFAULT_BLEND_WEIGHTS = {
    AdapterType.PERSONAL: 0.5,
    AdapterType.TEAM: 0.25,
    AdapterType.DEPARTMENT: 0.15,
    AdapterType.GLOBAL: 0.10
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AdapterSpec:
    """Specification for a single adapter."""
    adapter_type: AdapterType
    name: str
    path: str
    weight: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.adapter_type.value,
            "name": self.name,
            "path": self.path,
            "weight": self.weight,
            "enabled": self.enabled,
            "metadata": self.metadata
        }

@dataclass
class BlendConfig:
    """Configuration for adapter blending."""
    mode: BlendingMode = BlendingMode.HIERARCHICAL
    merge_strategy: MergeStrategy = MergeStrategy.WEIGHTED_SUM
    weights: Dict[AdapterType, float] = field(default_factory=lambda: DEFAULT_BLEND_WEIGHTS.copy())
    
    # Hierarchical settings
    enable_personal: bool = True
    enable_team: bool = True
    enable_department: bool = False
    enable_global: bool = True
    
    # Performance settings
    cache_blended: bool = True
    max_adapters: int = 5
    
    # Dynamic selection
    context_aware: bool = False
    min_relevance_score: float = 0.3

@dataclass
class BlendedAdapter:
    """Result of blending multiple adapters."""
    adapters: List[AdapterSpec]
    blended_weights: Dict[str, torch.Tensor]
    blend_config: BlendConfig
    composition_order: List[str]
    total_weight: float
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the blend."""
        return {
            "num_adapters": len(self.adapters),
            "types": [a.adapter_type.value for a in self.adapters],
            "names": [a.name for a in self.adapters],
            "weights": {a.name: a.weight for a in self.adapters},
            "mode": self.blend_config.mode.value,
            "composition": self.composition_order
        }

# ============================================================================
# ADAPTER BLENDER
# ============================================================================

class AdapterBlender:
    """
    Manages blending of multiple LoRA adapters for hierarchical personalization.
    Supports personal, team, department, and global adapter composition.
    """
    
    def __init__(self,
                 adapters_dir: str = "models/adapters",
                 blend_config: Optional[BlendConfig] = None):
        """
        Initialize adapter blender.
        
        Args:
            adapters_dir: Directory containing adapters
            blend_config: Blending configuration
        """
        self.adapters_dir = Path(adapters_dir)
        self.blend_config = blend_config or BlendConfig()
        
        # Adapter cache
        self.adapter_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.blended_cache: Dict[str, BlendedAdapter] = {}
        
        # Load adapter index
        self.adapter_index = self._load_adapter_index()
        
        # Statistics
        self.blend_stats = {
            "total_blends": 0,
            "cache_hits": 0,
            "blend_times": []
        }
        
        logger.info(f"AdapterBlender initialized with mode: {self.blend_config.mode.value}")
    
    def _load_adapter_index(self) -> Dict[str, Any]:
        """Load adapter index mapping users/groups to adapters."""
        index_file = self.adapters_dir / "adapters_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        else:
            # Create default index
            default_index = {
                "users": {},
                "teams": {},
                "departments": {},
                "global": "global_adapter_v1.pt",
                "metadata": {
                    "version": "2.0",
                    "supports_blending": True
                }
            }
            
            # Save default
            with open(index_file, 'w') as f:
                json.dump(default_index, f, indent=2)
            
            return default_index
    
    def load_blended_adapters(self,
                            user_id: Optional[str] = None,
                            team_ids: Optional[List[str]] = None,
                            department_id: Optional[str] = None,
                            use_global: bool = True,
                            context: Optional[Dict[str, Any]] = None) -> Optional[BlendedAdapter]:
        """
        Load and blend multiple adapters based on hierarchy.
        
        Args:
            user_id: User identifier
            team_ids: Team/group identifiers
            department_id: Department identifier
            use_global: Whether to include global adapter
            context: Optional context for dynamic selection
            
        Returns:
            Blended adapter or None
        """
        # Check cache
        cache_key = self._get_cache_key(user_id, team_ids, department_id, use_global)
        if self.blend_config.cache_blended and cache_key in self.blended_cache:
            self.blend_stats["cache_hits"] += 1
            logger.debug(f"Using cached blend: {cache_key}")
            return self.blended_cache[cache_key]
        
        # Collect adapters to blend
        adapters_to_blend = self._collect_adapters(
            user_id, team_ids, department_id, use_global, context
        )
        
        if not adapters_to_blend:
            logger.warning("No adapters found to blend")
            return None
        
        # Perform blending based on mode
        import time
        start_time = time.time()
        
        if self.blend_config.mode == BlendingMode.NONE:
            # Use first adapter only
            blended = self._no_blending(adapters_to_blend[0])
        elif self.blend_config.mode == BlendingMode.SEQUENTIAL:
            blended = self._sequential_blending(adapters_to_blend)
        elif self.blend_config.mode == BlendingMode.WEIGHTED:
            blended = self._weighted_blending(adapters_to_blend)
        elif self.blend_config.mode == BlendingMode.HIERARCHICAL:
            blended = self._hierarchical_blending(adapters_to_blend)
        elif self.blend_config.mode == BlendingMode.DYNAMIC:
            blended = self._dynamic_blending(adapters_to_blend, context)
        else:
            raise ValueError(f"Unknown blending mode: {self.blend_config.mode}")
        
        blend_time = time.time() - start_time
        self.blend_stats["blend_times"].append(blend_time)
        self.blend_stats["total_blends"] += 1
        
        # Cache result
        if self.blend_config.cache_blended:
            self.blended_cache[cache_key] = blended
        
        logger.info(f"Blended {len(adapters_to_blend)} adapters in {blend_time:.3f}s")
        logger.info(f"Composition: {blended.composition_order}")
        
        return blended
    
    def _collect_adapters(self,
                         user_id: Optional[str],
                         team_ids: Optional[List[str]],
                         department_id: Optional[str],
                         use_global: bool,
                         context: Optional[Dict[str, Any]]) -> List[AdapterSpec]:
        """
        Collect all relevant adapters for blending.
        
        Returns:
            List of adapter specifications
        """
        adapters = []
        
        # Personal adapter
        if user_id and self.blend_config.enable_personal:
            personal_path = self._get_adapter_path(user_id, AdapterType.PERSONAL)
            if personal_path:
                adapters.append(AdapterSpec(
                    adapter_type=AdapterType.PERSONAL,
                    name=f"user_{user_id}",
                    path=personal_path,
                    weight=self.blend_config.weights.get(AdapterType.PERSONAL, 0.5)
                ))
        
        # Team adapters
        if team_ids and self.blend_config.enable_team:
            for team_id in team_ids[:2]:  # Limit to 2 teams
                team_path = self._get_adapter_path(team_id, AdapterType.TEAM)
                if team_path:
                    adapters.append(AdapterSpec(
                        adapter_type=AdapterType.TEAM,
                        name=f"team_{team_id}",
                        path=team_path,
                        weight=self.blend_config.weights.get(AdapterType.TEAM, 0.25) / len(team_ids)
                    ))
        
        # Department adapter
        if department_id and self.blend_config.enable_department:
            dept_path = self._get_adapter_path(department_id, AdapterType.DEPARTMENT)
            if dept_path:
                adapters.append(AdapterSpec(
                    adapter_type=AdapterType.DEPARTMENT,
                    name=f"dept_{department_id}",
                    path=dept_path,
                    weight=self.blend_config.weights.get(AdapterType.DEPARTMENT, 0.15)
                ))
        
        # Global adapter
        if use_global and self.blend_config.enable_global:
            global_path = self._get_global_adapter_path()
            if global_path:
                adapters.append(AdapterSpec(
                    adapter_type=AdapterType.GLOBAL,
                    name="global",
                    path=global_path,
                    weight=self.blend_config.weights.get(AdapterType.GLOBAL, 0.1)
                ))
        
        # Apply context filtering if in dynamic mode
        if self.blend_config.context_aware and context:
            adapters = self._filter_by_context(adapters, context)
        
        # Limit to max adapters
        if len(adapters) > self.blend_config.max_adapters:
            adapters = self._prioritize_adapters(adapters)[:self.blend_config.max_adapters]
        
        return adapters
    
    def _get_adapter_path(self, identifier: str, adapter_type: AdapterType) -> Optional[str]:
        """Get adapter path from index."""
        if adapter_type == AdapterType.PERSONAL:
            user_adapters = self.adapter_index.get("users", {})
            if identifier in user_adapters:
                adapter_info = user_adapters[identifier]
                if isinstance(adapter_info, dict):
                    return adapter_info.get("personal")
                return adapter_info
        
        elif adapter_type in [AdapterType.TEAM, AdapterType.GROUP]:
            team_adapters = self.adapter_index.get("teams", {})
            if identifier in team_adapters:
                return team_adapters[identifier]
        
        elif adapter_type == AdapterType.DEPARTMENT:
            dept_adapters = self.adapter_index.get("departments", {})
            if identifier in dept_adapters:
                return dept_adapters[identifier]
        
        return None
    
    def _get_global_adapter_path(self) -> Optional[str]:
        """Get global adapter path."""
        global_adapter = self.adapter_index.get("global")
        if global_adapter:
            return str(self.adapters_dir / global_adapter)
        return None
    
    def _load_adapter_weights(self, adapter_spec: AdapterSpec) -> Dict[str, torch.Tensor]:
        """Load adapter weights from file."""
        # Check cache
        if adapter_spec.path in self.adapter_cache:
            return self.adapter_cache[adapter_spec.path]
        
        adapter_path = Path(adapter_spec.path)
        if not adapter_path.exists():
            adapter_path = self.adapters_dir / adapter_spec.path
        
        if not adapter_path.exists():
            logger.warning(f"Adapter not found: {adapter_path}")
            return {}
        
        try:
            # Load adapter weights
            adapter_data = torch.load(adapter_path, map_location='cpu')
            
            # Extract LoRA weights
            if isinstance(adapter_data, dict):
                weights = adapter_data.get("lora_weights", adapter_data)
            else:
                weights = {"model": adapter_data}
            
            # Cache for future use
            self.adapter_cache[adapter_spec.path] = weights
            
            return weights
        
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_path}: {e}")
            return {}
    
    def _no_blending(self, adapter: AdapterSpec) -> BlendedAdapter:
        """No blending - use single adapter."""
        weights = self._load_adapter_weights(adapter)
        
        return BlendedAdapter(
            adapters=[adapter],
            blended_weights=weights,
            blend_config=self.blend_config,
            composition_order=[adapter.name],
            total_weight=1.0
        )
    
    def _sequential_blending(self, adapters: List[AdapterSpec]) -> BlendedAdapter:
        """
        Sequential blending - apply adapters one after another.
        Later adapters can override earlier ones.
        """
        blended_weights = {}
        composition_order = []
        
        for adapter in adapters:
            weights = self._load_adapter_weights(adapter)
            
            # Apply sequentially (later overrides earlier)
            for key, weight in weights.items():
                if key in blended_weights:
                    # Blend with existing weight
                    alpha = adapter.weight
                    blended_weights[key] = (1 - alpha) * blended_weights[key] + alpha * weight
                else:
                    blended_weights[key] = weight * adapter.weight
            
            composition_order.append(adapter.name)
        
        return BlendedAdapter(
            adapters=adapters,
            blended_weights=blended_weights,
            blend_config=self.blend_config,
            composition_order=composition_order,
            total_weight=sum(a.weight for a in adapters)
        )
    
    def _weighted_blending(self, adapters: List[AdapterSpec]) -> BlendedAdapter:
        """
        Weighted blending - weighted average of all adapter weights.
        """
        all_weights = {}
        weight_sums = {}
        
        # Collect all weights
        for adapter in adapters:
            weights = self._load_adapter_weights(adapter)
            
            for key, weight in weights.items():
                if key not in all_weights:
                    all_weights[key] = []
                    weight_sums[key] = 0
                
                all_weights[key].append(weight * adapter.weight)
                weight_sums[key] += adapter.weight
        
        # Compute weighted average
        blended_weights = {}
        for key, weight_list in all_weights.items():
            if weight_sums[key] > 0:
                # Stack and average
                stacked = torch.stack(weight_list)
                blended_weights[key] = stacked.sum(dim=0) / weight_sums[key]
        
        composition_order = [f"{a.name}({a.weight:.2f})" for a in adapters]
        
        return BlendedAdapter(
            adapters=adapters,
            blended_weights=blended_weights,
            blend_config=self.blend_config,
            composition_order=composition_order,
            total_weight=sum(a.weight for a in adapters)
        )
    
    def _hierarchical_blending(self, adapters: List[AdapterSpec]) -> BlendedAdapter:
        """
        Hierarchical blending - apply adapters based on hierarchy level.
        Personal > Team > Department > Global
        """
        # Sort by hierarchy
        hierarchy_order = [
            AdapterType.GLOBAL,
            AdapterType.DEPARTMENT,
            AdapterType.TEAM,
            AdapterType.PERSONAL
        ]
        
        sorted_adapters = sorted(
            adapters,
            key=lambda a: hierarchy_order.index(a.adapter_type) if a.adapter_type in hierarchy_order else 999
        )
        
        # Apply hierarchically (base to specific)
        blended_weights = {}
        composition_order = []
        
        for adapter in sorted_adapters:
            weights = self._load_adapter_weights(adapter)
            
            # Hierarchical blending with decay
            decay_factor = 0.8 ** hierarchy_order.index(adapter.adapter_type)
            effective_weight = adapter.weight * decay_factor
            
            for key, weight in weights.items():
                if key in blended_weights:
                    # Blend with decay
                    blended_weights[key] = blended_weights[key] * (1 - effective_weight) + weight * effective_weight
                else:
                    blended_weights[key] = weight * effective_weight
            
            composition_order.append(f"{adapter.name}[{adapter.adapter_type.value}]")
        
        return BlendedAdapter(
            adapters=adapters,
            blended_weights=blended_weights,
            blend_config=self.blend_config,
            composition_order=composition_order,
            total_weight=1.0
        )
    
    def _dynamic_blending(self, 
                         adapters: List[AdapterSpec],
                         context: Optional[Dict[str, Any]]) -> BlendedAdapter:
        """
        Dynamic blending based on context.
        Adjusts weights based on query type, domain, etc.
        """
        if not context:
            # Fallback to hierarchical
            return self._hierarchical_blending(adapters)
        
        # Analyze context
        query_type = context.get("query_type", "general")
        domain = context.get("domain", "unknown")
        
        # Adjust weights based on context
        for adapter in adapters:
            if adapter.adapter_type == AdapterType.PERSONAL:
                # Boost personal for user-specific queries
                if query_type in ["personal", "preference"]:
                    adapter.weight *= 1.5
            
            elif adapter.adapter_type == AdapterType.TEAM:
                # Boost team for collaborative queries
                if query_type in ["team", "project", "collaboration"]:
                    adapter.weight *= 1.3
            
            elif adapter.adapter_type == AdapterType.GLOBAL:
                # Boost global for general knowledge
                if query_type in ["general", "factual"]:
                    adapter.weight *= 1.2
        
        # Normalize weights
        total_weight = sum(a.weight for a in adapters)
        if total_weight > 0:
            for adapter in adapters:
                adapter.weight /= total_weight
        
        # Use weighted blending with adjusted weights
        return self._weighted_blending(adapters)
    
    def _filter_by_context(self,
                          adapters: List[AdapterSpec],
                          context: Dict[str, Any]) -> List[AdapterSpec]:
        """Filter adapters based on context relevance."""
        # Placeholder for context-based filtering
        # Could integrate with context_filter.py for scoring
        return adapters
    
    def _prioritize_adapters(self, adapters: List[AdapterSpec]) -> List[AdapterSpec]:
        """Prioritize adapters when exceeding max limit."""
        # Sort by hierarchy priority
        priority = {
            AdapterType.PERSONAL: 4,
            AdapterType.TEAM: 3,
            AdapterType.DEPARTMENT: 2,
            AdapterType.GLOBAL: 1
        }
        
        return sorted(
            adapters,
            key=lambda a: priority.get(a.adapter_type, 0),
            reverse=True
        )
    
    def _get_cache_key(self,
                      user_id: Optional[str],
                      team_ids: Optional[List[str]],
                      department_id: Optional[str],
                      use_global: bool) -> str:
        """Generate cache key for blend configuration."""
        parts = []
        if user_id:
            parts.append(f"u:{user_id}")
        if team_ids:
            parts.append(f"t:{','.join(sorted(team_ids))}")
        if department_id:
            parts.append(f"d:{department_id}")
        if use_global:
            parts.append("g:1")
        
        return "|".join(parts) if parts else "default"
    
    def apply_blended_adapter(self,
                             model: nn.Module,
                             blended_adapter: BlendedAdapter) -> nn.Module:
        """
        Apply blended adapter to a model.
        
        Args:
            model: Base model
            blended_adapter: Blended adapter to apply
            
        Returns:
            Model with blended adapter applied
        """
        # Apply blended weights to model
        for name, module in model.named_modules():
            # Check if this module has LoRA weights
            lora_A_key = f"{name}.lora_A"
            lora_B_key = f"{name}.lora_B"
            
            if lora_A_key in blended_adapter.blended_weights:
                # Apply LoRA weights
                if hasattr(module, 'lora_A'):
                    module.lora_A.data = blended_adapter.blended_weights[lora_A_key]
                if hasattr(module, 'lora_B'):
                    module.lora_B.data = blended_adapter.blended_weights[lora_B_key]
        
        logger.info(f"Applied blended adapter with {len(blended_adapter.adapters)} components")
        return model
    
    def save_blended_adapter(self,
                           blended_adapter: BlendedAdapter,
                           save_path: str) -> str:
        """
        Save blended adapter to file.
        
        Args:
            blended_adapter: Blended adapter to save
            save_path: Where to save
            
        Returns:
            Path where saved
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            "blended_weights": blended_adapter.blended_weights,
            "blend_info": blended_adapter.get_info(),
            "config": {
                "mode": blended_adapter.blend_config.mode.value,
                "merge_strategy": blended_adapter.blend_config.merge_strategy.value,
                "weights": {k.value: v for k, v in blended_adapter.blend_config.weights.items()}
            }
        }
        
        torch.save(save_data, save_path)
        logger.info(f"Saved blended adapter to {save_path}")
        
        return str(save_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blending statistics."""
        avg_blend_time = (
            sum(self.blend_stats["blend_times"]) / len(self.blend_stats["blend_times"])
            if self.blend_stats["blend_times"] else 0
        )
        
        return {
            "total_blends": self.blend_stats["total_blends"],
            "cache_hits": self.blend_stats["cache_hits"],
            "cache_size": len(self.blended_cache),
            "avg_blend_time": avg_blend_time,
            "loaded_adapters": len(self.adapter_cache),
            "mode": self.blend_config.mode.value
        }
    
    def update_adapter_index(self,
                            identifier: str,
                            adapter_type: AdapterType,
                            adapter_path: str):
        """
        Update adapter index with new adapter.
        
        Args:
            identifier: User/team/dept identifier
            adapter_type: Type of adapter
            adapter_path: Path to adapter file
        """
        if adapter_type == AdapterType.PERSONAL:
            if "users" not in self.adapter_index:
                self.adapter_index["users"] = {}
            self.adapter_index["users"][identifier] = adapter_path
        
        elif adapter_type in [AdapterType.TEAM, AdapterType.GROUP]:
            if "teams" not in self.adapter_index:
                self.adapter_index["teams"] = {}
            self.adapter_index["teams"][identifier] = adapter_path
        
        elif adapter_type == AdapterType.DEPARTMENT:
            if "departments" not in self.adapter_index:
                self.adapter_index["departments"] = {}
            self.adapter_index["departments"][identifier] = adapter_path
        
        # Save updated index
        index_file = self.adapters_dir / "adapters_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.adapter_index, f, indent=2)
        
        logger.info(f"Updated adapter index: {identifier} -> {adapter_path}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_global_blender: Optional[AdapterBlender] = None

def get_global_blender(config: Optional[BlendConfig] = None) -> AdapterBlender:
    """
    Get or create global blender instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        Global AdapterBlender instance
    """
    global _global_blender
    
    if _global_blender is None:
        _global_blender = AdapterBlender(blend_config=config)
    
    return _global_blender

def blend_adapters_for_user(user_id: str,
                           team_ids: Optional[List[str]] = None,
                           use_global: bool = True,
                           mode: Optional[BlendingMode] = None) -> Optional[BlendedAdapter]:
    """
    Quick function to blend adapters for a user.
    
    Args:
        user_id: User identifier
        team_ids: Team identifiers
        use_global: Include global adapter
        mode: Optional mode override
        
    Returns:
        Blended adapter or None
    """
    config = BlendConfig(mode=mode) if mode else None
    blender = get_global_blender(config)
    
    return blender.load_blended_adapters(
        user_id=user_id,
        team_ids=team_ids,
        use_global=use_global
    )

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "AdapterType",
    "BlendingMode",
    "MergeStrategy",
    "AdapterSpec",
    "BlendConfig",
    "BlendedAdapter",
    "AdapterBlender",
    "get_global_blender",
    "blend_adapters_for_user"
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
    
    # Test blending
    blender = AdapterBlender()
    
    # Test different blending scenarios
    test_cases = [
        ("alice", ["ProjectX"], True, "Hierarchical"),
        ("bob", ["ProjectX", "TeamBeta"], False, "Weighted"),
        ("charlie", None, True, "Sequential")
    ]
    
    for user_id, teams, use_global, mode_name in test_cases:
        print(f"\n{'='*60}")
        print(f"User: {user_id}, Teams: {teams}, Global: {use_global}")
        print(f"Mode: {mode_name}")
        print("="*60)
        
        config = BlendConfig(mode=BlendingMode[mode_name.upper()])
        blender.blend_config = config
        
        blended = blender.load_blended_adapters(
            user_id=user_id,
            team_ids=teams,
            use_global=use_global
        )
        
        if blended:
            info = blended.get_info()
            print(f"Blended {info['num_adapters']} adapters:")
            for name, weight in info['weights'].items():
                print(f"  - {name}: {weight:.2f}")
            print(f"Composition: {' -> '.join(info['composition'])}")
        else:
            print("No adapters found to blend")
    
    # Show statistics
    print(f"\n{'='*60}")
    print("Blending Statistics:")
    stats = blender.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
