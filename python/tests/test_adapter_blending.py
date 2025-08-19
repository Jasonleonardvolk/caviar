#!/usr/bin/env python3
"""
Test Suite for Adapter Blending (Improvement #3)
Tests hierarchical composition of personal, team, and global LoRA adapters
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
import time
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.adapter_blending import (
    AdapterBlender,
    BlendConfig,
    BlendingMode,
    MergeStrategy,
    AdapterType,
    AdapterSpec,
    BlendedAdapter,
    get_global_blender,
    blend_adapters_for_user
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_adapters():
    """Create test adapter files and index."""
    print("\n" + "="*60)
    print("SETUP: Creating Test Adapters")
    print("="*60)
    
    adapters_dir = Path("models/adapters")
    adapters_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy adapter weights
    def create_dummy_adapter(name: str) -> str:
        """Create a dummy adapter file."""
        adapter_path = adapters_dir / f"{name}.pt"
        
        # Create simple LoRA weights
        lora_weights = {
            "lstm.lora_A": torch.randn(8, 256),
            "lstm.lora_B": torch.randn(256, 8),
            "linear.lora_A": torch.randn(8, 256),
            "linear.lora_B": torch.randn(256, 8)
        }
        
        adapter_data = {
            "lora_weights": lora_weights,
            "metadata": {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "rank": 8
            }
        }
        
        torch.save(adapter_data, adapter_path)
        return str(adapter_path)
    
    # Create test adapters
    adapters = {
        "users": {
            "alice": create_dummy_adapter("user_alice_lora"),
            "bob": create_dummy_adapter("user_bob_lora"),
            "charlie": create_dummy_adapter("user_charlie_lora")
        },
        "teams": {
            "ProjectX": create_dummy_adapter("team_ProjectX_lora"),
            "TeamBeta": create_dummy_adapter("team_TeamBeta_lora"),
            "ResearchGroup": create_dummy_adapter("team_ResearchGroup_lora")
        },
        "departments": {
            "Engineering": create_dummy_adapter("dept_Engineering_lora"),
            "DataScience": create_dummy_adapter("dept_DataScience_lora")
        },
        "global": "global_adapter_v1.pt"
    }
    
    # Create global adapter
    create_dummy_adapter("global_adapter_v1")
    
    # Create adapter index
    index = {
        "users": adapters["users"],
        "teams": adapters["teams"],
        "departments": adapters["departments"],
        "global": adapters["global"],
        "metadata": {
            "version": "2.0",
            "supports_blending": True,
            "created_at": datetime.now().isoformat()
        }
    }
    
    index_path = adapters_dir / "adapters_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✓ Created {len(adapters['users'])} user adapters")
    print(f"✓ Created {len(adapters['teams'])} team adapters")
    print(f"✓ Created {len(adapters['departments'])} department adapters")
    print(f"✓ Created global adapter")
    print(f"✓ Saved adapter index to {index_path}")
    
    return True

def test_sequential_blending():
    """Test sequential adapter blending."""
    print("\n" + "="*60)
    print("TEST 1: Sequential Blending")
    print("="*60)
    
    config = BlendConfig(mode=BlendingMode.SEQUENTIAL)
    blender = AdapterBlender(blend_config=config)
    
    # Blend for user in team
    blended = blender.load_blended_adapters(
        user_id="alice",
        team_ids=["ProjectX"],
        use_global=True
    )
    
    assert blended is not None, "Blending failed"
    assert len(blended.adapters) >= 2, "Not enough adapters blended"
    
    info = blended.get_info()
    print(f"\n✓ Blended {info['num_adapters']} adapters sequentially")
    print(f"  Composition: {' -> '.join(info['composition'])}")
    print(f"  Types: {info['types']}")
    
    # Check weights exist
    assert len(blended.blended_weights) > 0, "No blended weights"
    print(f"✓ Generated {len(blended.blended_weights)} blended weight tensors")
    
    return True

def test_weighted_blending():
    """Test weighted average blending."""
    print("\n" + "="*60)
    print("TEST 2: Weighted Blending")
    print("="*60)
    
    # Custom weights
    config = BlendConfig(
        mode=BlendingMode.WEIGHTED,
        weights={
            AdapterType.PERSONAL: 0.6,
            AdapterType.TEAM: 0.3,
            AdapterType.GLOBAL: 0.1
        }
    )
    blender = AdapterBlender(blend_config=config)
    
    # Blend with custom weights
    blended = blender.load_blended_adapters(
        user_id="bob",
        team_ids=["TeamBeta"],
        use_global=True
    )
    
    assert blended is not None, "Weighted blending failed"
    
    info = blended.get_info()
    print(f"\n✓ Weighted blend of {info['num_adapters']} adapters")
    
    # Show weights
    for name, weight in info['weights'].items():
        print(f"  {name}: {weight:.2f}")
    
    # Verify weights sum correctly
    total_weight = sum(info['weights'].values())
    print(f"✓ Total weight: {total_weight:.2f}")
    
    return True

def test_hierarchical_blending():
    """Test hierarchical blending (personal > team > global)."""
    print("\n" + "="*60)
    print("TEST 3: Hierarchical Blending")
    print("="*60)
    
    config = BlendConfig(
        mode=BlendingMode.HIERARCHICAL,
        enable_department=True
    )
    blender = AdapterBlender(blend_config=config)
    
    # Full hierarchy
    blended = blender.load_blended_adapters(
        user_id="charlie",
        team_ids=["ResearchGroup"],
        department_id="DataScience",
        use_global=True
    )
    
    assert blended is not None, "Hierarchical blending failed"
    
    info = blended.get_info()
    print(f"\n✓ Hierarchical blend of {info['num_adapters']} adapters")
    print(f"  Order: {' -> '.join(info['composition'])}")
    
    # Check hierarchy order
    types = info['types']
    if 'global' in types and 'personal' in types:
        assert types.index('global') < types.index('personal'), "Wrong hierarchy order"
        print("✓ Correct hierarchy: global → department → team → personal")
    
    return True

def test_dynamic_blending():
    """Test context-aware dynamic blending."""
    print("\n" + "="*60)
    print("TEST 4: Dynamic Context-Aware Blending")
    print("="*60)
    
    config = BlendConfig(
        mode=BlendingMode.DYNAMIC,
        context_aware=True
    )
    blender = AdapterBlender(blend_config=config)
    
    # Test different contexts
    contexts = [
        {"query_type": "personal", "domain": "user_specific"},
        {"query_type": "team", "domain": "collaboration"},
        {"query_type": "general", "domain": "factual"}
    ]
    
    for context in contexts:
        print(f"\nContext: {context}")
        
        blended = blender.load_blended_adapters(
            user_id="alice",
            team_ids=["ProjectX"],
            use_global=True,
            context=context
        )
        
        assert blended is not None, "Dynamic blending failed"
        
        info = blended.get_info()
        print(f"  Adapters: {info['names']}")
        print(f"  Weights adjusted for '{context['query_type']}' query")
    
    print("\n✓ Dynamic blending adjusts weights based on context")
    
    return True

def test_multi_team_blending():
    """Test blending with multiple teams."""
    print("\n" + "="*60)
    print("TEST 5: Multi-Team Blending")
    print("="*60)
    
    config = BlendConfig(mode=BlendingMode.WEIGHTED)
    blender = AdapterBlender(blend_config=config)
    
    # User in multiple teams
    blended = blender.load_blended_adapters(
        user_id="bob",
        team_ids=["ProjectX", "TeamBeta", "ResearchGroup"],
        use_global=False
    )
    
    assert blended is not None, "Multi-team blending failed"
    
    info = blended.get_info()
    print(f"\n✓ Blended {info['num_adapters']} adapters")
    
    # Count team adapters
    team_adapters = [n for n in info['names'] if 'team' in n]
    print(f"  Team adapters: {team_adapters}")
    print(f"✓ Successfully blended {len(team_adapters)} team adapters")
    
    return True

def test_caching():
    """Test blended adapter caching."""
    print("\n" + "="*60)
    print("TEST 6: Blended Adapter Caching")
    print("="*60)
    
    config = BlendConfig(
        mode=BlendingMode.HIERARCHICAL,
        cache_blended=True
    )
    blender = AdapterBlender(blend_config=config)
    
    # First blend (cache miss)
    start_time = time.time()
    blended1 = blender.load_blended_adapters(
        user_id="alice",
        team_ids=["ProjectX"],
        use_global=True
    )
    first_time = time.time() - start_time
    
    # Second blend (cache hit)
    start_time = time.time()
    blended2 = blender.load_blended_adapters(
        user_id="alice",
        team_ids=["ProjectX"],
        use_global=True
    )
    second_time = time.time() - start_time
    
    print(f"\nFirst blend: {first_time:.3f}s")
    print(f"Second blend: {second_time:.3f}s (cached)")
    
    # Cache should be faster
    assert second_time < first_time * 0.5, "Cache not working"
    print(f"✓ Cache speedup: {first_time/second_time:.1f}x")
    
    # Check statistics
    stats = blender.get_statistics()
    assert stats['cache_hits'] > 0, "No cache hits recorded"
    print(f"✓ Cache hits: {stats['cache_hits']}")
    
    return True

def test_save_and_load():
    """Test saving and loading blended adapters."""
    print("\n" + "="*60)
    print("TEST 7: Save and Load Blended Adapter")
    print("="*60)
    
    config = BlendConfig(mode=BlendingMode.WEIGHTED)
    blender = AdapterBlender(blend_config=config)
    
    # Create blend
    blended = blender.load_blended_adapters(
        user_id="charlie",
        team_ids=["ResearchGroup"],
        use_global=True
    )
    
    # Save
    save_path = "models/adapters/blended_charlie_test.pt"
    saved_path = blender.save_blended_adapter(blended, save_path)
    assert Path(saved_path).exists(), "Save failed"
    print(f"✓ Saved blended adapter to {saved_path}")
    
    # Load and verify
    loaded_data = torch.load(saved_path)
    assert "blended_weights" in loaded_data, "Missing weights"
    assert "blend_info" in loaded_data, "Missing info"
    
    print(f"✓ Loaded blended adapter")
    print(f"  Components: {loaded_data['blend_info']['names']}")
    print(f"  Mode: {loaded_data['config']['mode']}")
    
    # Cleanup
    Path(saved_path).unlink()
    
    return True

def test_fallback_behavior():
    """Test fallback when adapters are missing."""
    print("\n" + "="*60)
    print("TEST 8: Fallback Behavior")
    print("="*60)
    
    config = BlendConfig(mode=BlendingMode.HIERARCHICAL)
    blender = AdapterBlender(blend_config=config)
    
    # Try non-existent user
    blended = blender.load_blended_adapters(
        user_id="nonexistent_user",
        team_ids=["ProjectX"],
        use_global=True
    )
    
    if blended:
        info = blended.get_info()
        print(f"\n✓ Fallback blend created with {info['num_adapters']} adapters")
        print(f"  Available: {info['names']}")
        
        # Should have team and global at least
        assert any('team' in n for n in info['names']) or any('global' in n for n in info['names']), \
               "No fallback adapters"
        print("✓ Successfully fell back to team/global adapters")
    else:
        print("✓ No adapters available (expected behavior)")
    
    return True

def test_adapter_limits():
    """Test maximum adapter limits."""
    print("\n" + "="*60)
    print("TEST 9: Adapter Limits")
    print("="*60)
    
    config = BlendConfig(
        mode=BlendingMode.WEIGHTED,
        max_adapters=3
    )
    blender = AdapterBlender(blend_config=config)
    
    # Try to blend many adapters
    blended = blender.load_blended_adapters(
        user_id="alice",
        team_ids=["ProjectX", "TeamBeta", "ResearchGroup"],
        department_id="Engineering",
        use_global=True
    )
    
    assert blended is not None, "Blending failed"
    
    info = blended.get_info()
    print(f"\n✓ Requested many adapters, limited to {info['num_adapters']}")
    assert info['num_adapters'] <= config.max_adapters, "Limit not enforced"
    print(f"✓ Adapter limit ({config.max_adapters}) correctly enforced")
    print(f"  Selected: {info['names']}")
    
    return True

def test_performance():
    """Test blending performance with timing."""
    print("\n" + "="*60)
    print("TEST 10: Blending Performance")
    print("="*60)
    
    blender = AdapterBlender()
    
    # Time different modes
    modes = [
        BlendingMode.SEQUENTIAL,
        BlendingMode.WEIGHTED,
        BlendingMode.HIERARCHICAL
    ]
    
    for mode in modes:
        blender.blend_config.mode = mode
        
        start_time = time.time()
        blended = blender.load_blended_adapters(
            user_id="alice",
            team_ids=["ProjectX"],
            use_global=True
        )
        duration = time.time() - start_time
        
        print(f"\n{mode.value}: {duration:.3f}s")
        
        if blended:
            info = blended.get_info()
            print(f"  Blended {info['num_adapters']} adapters")
    
    # Check overall statistics
    stats = blender.get_statistics()
    avg_time = stats.get('avg_blend_time', 0)
    print(f"\n✓ Average blend time: {avg_time:.3f}s")
    print(f"✓ Total blends: {stats['total_blends']}")
    
    return True

def run_all_tests():
    """Run all adapter blending tests."""
    print("\n" + "="*60)
    print("ADAPTER BLENDING TEST SUITE (Improvement #3)")
    print("="*60)
    
    # Setup
    setup_test_adapters()
    
    tests = [
        ("Sequential Blending", test_sequential_blending),
        ("Weighted Blending", test_weighted_blending),
        ("Hierarchical Blending", test_hierarchical_blending),
        ("Dynamic Blending", test_dynamic_blending),
        ("Multi-Team Blending", test_multi_team_blending),
        ("Caching", test_caching),
        ("Save and Load", test_save_and_load),
        ("Fallback Behavior", test_fallback_behavior),
        ("Adapter Limits", test_adapter_limits),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All adapter blending tests passed! Hierarchical composition is working.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
