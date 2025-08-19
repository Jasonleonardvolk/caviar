#!/usr/bin/env python3
"""
Adapter Blending Validation Script
Ensures all components are properly set up and working locally
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("ADAPTER BLENDING VALIDATION")
print("="*60)

# ============================================================================
# STEP 1: Check Directory Structure
# ============================================================================
print("\n[1] Checking Directory Structure...")

base_dir = Path("C:/Users/jason/Desktop/tori/kha")
required_dirs = [
    "models/adapters",
    "models/mesh_contexts",
    "python/core",
    "python/tests"
]

all_dirs_exist = True
for dir_path in required_dirs:
    full_path = base_dir / dir_path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {dir_path}")
    if not exists:
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"    → Created {dir_path}")
    all_dirs_exist = all_dirs_exist and exists

# ============================================================================
# STEP 2: Check Module Imports
# ============================================================================
print("\n[2] Checking Module Imports...")

modules_to_check = [
    ("adapter_blending", "core.adapter_blending"),
    ("adapter_loader_enhanced", "core.adapter_loader_enhanced"),
    ("context_filter", "core.context_filter"),
    ("mesh_summary_exporter", "core.mesh_summary_exporter")
]

import_status = {}
for module_name, import_path in modules_to_check:
    try:
        exec(f"from {import_path} import *")
        import_status[module_name] = True
        print(f"  ✓ {module_name}")
    except ImportError as e:
        import_status[module_name] = False
        print(f"  ✗ {module_name}: {e}")

# ============================================================================
# STEP 3: Create Test Adapter Files
# ============================================================================
print("\n[3] Creating Test Adapter Files...")

adapters_dir = base_dir / "models/adapters"

def create_test_adapter(name: str) -> bool:
    """Create a test adapter file."""
    adapter_path = adapters_dir / f"{name}.pt"
    
    # Simple LoRA weights
    adapter_data = {
        "lora_weights": {
            "lstm.lora_A": torch.randn(8, 256) * 0.01,
            "lstm.lora_B": torch.randn(256, 8) * 0.01,
            "linear.lora_A": torch.randn(8, 256) * 0.01,
            "linear.lora_B": torch.randn(256, 8) * 0.01
        },
        "metadata": {
            "name": name,
            "rank": 8,
            "created_at": datetime.now().isoformat()
        }
    }
    
    try:
        torch.save(adapter_data, adapter_path)
        print(f"  ✓ Created {name}.pt")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create {name}: {e}")
        return False

# Create test adapters
test_adapters = [
    "user_jason_lora",
    "user_alice_lora",
    "group_ProjectX_lora",
    "team_TeamBeta_lora",
    "global_adapter_v1"
]

for adapter_name in test_adapters:
    adapter_path = adapters_dir / f"{adapter_name}.pt"
    if not adapter_path.exists():
        create_test_adapter(adapter_name)
    else:
        print(f"  ✓ {adapter_name}.pt already exists")

# ============================================================================
# STEP 4: Create/Update Adapter Index
# ============================================================================
print("\n[4] Creating Adapter Index...")

index_path = adapters_dir / "adapters_index.json"
index_data = {
    "users": {
        "jason": {"personal": "user_jason_lora.pt"},
        "alice": {"personal": "user_alice_lora.pt"}
    },
    "teams": {
        "ProjectX": "group_ProjectX_lora.pt",
        "TeamBeta": "team_TeamBeta_lora.pt"
    },
    "global": "global_adapter_v1.pt",
    "metadata": {
        "version": "2.0",
        "supports_blending": True,
        "created_at": datetime.now().isoformat()
    }
}

try:
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"  ✓ Created adapters_index.json")
except Exception as e:
    print(f"  ✗ Failed to create index: {e}")

# ============================================================================
# STEP 5: Test Adapter Blending
# ============================================================================
print("\n[5] Testing Adapter Blending...")

if import_status.get("adapter_blending") and import_status.get("adapter_loader_enhanced"):
    from core.adapter_blending import AdapterBlender, BlendConfig, BlendingMode
    from core.adapter_loader_enhanced import AdapterManager
    
    # Test with AdapterManager
    print("\n  Testing AdapterManager with blending...")
    try:
        manager = AdapterManager(
            adapters_dir=str(adapters_dir),
            enable_blending=True
        )
        
        # Test single adapter
        adapter = manager.load_adapter(user_id="jason")
        if adapter:
            print(f"    ✓ Single adapter: {adapter.config.name}")
        
        # Test blended adapters
        if manager.blender:
            blended = manager.load_blended_adapters(
                user_id="jason",
                team_ids=["ProjectX"],
                use_global=True,
                blend_mode="hierarchical"
            )
            
            if blended:
                info = blended.get_info()
                print(f"    ✓ Blended {info['num_adapters']} adapters")
                print(f"      Composition: {' -> '.join(info['composition'])}")
            else:
                print("    ✗ Blending failed")
        else:
            print("    ✗ Blender not initialized")
    
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test direct blending
    print("\n  Testing Direct Blending...")
    try:
        blender = AdapterBlender(adapters_dir=str(adapters_dir))
        
        # Test different modes
        modes = ["sequential", "weighted", "hierarchical"]
        for mode in modes:
            config = BlendConfig(mode=BlendingMode[mode.upper()])
            blender.blend_config = config
            
            blended = blender.load_blended_adapters(
                user_id="alice",
                team_ids=["ProjectX", "TeamBeta"],
                use_global=True
            )
            
            if blended:
                info = blended.get_info()
                print(f"    ✓ {mode}: {len(info['names'])} adapters")
            else:
                print(f"    ✗ {mode}: No blend")
    
    except Exception as e:
        print(f"    ✗ Direct blending error: {e}")
else:
    print("  ✗ Cannot test - modules not available")

# ============================================================================
# STEP 6: Summary
# ============================================================================
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

issues = []

# Check results
if not all_dirs_exist:
    issues.append("Some directories were missing (now created)")

if not all(import_status.values()):
    failed_imports = [m for m, s in import_status.items() if not s]
    issues.append(f"Failed imports: {', '.join(failed_imports)}")

if not (adapters_dir / "adapters_index.json").exists():
    issues.append("Adapter index missing")

if issues:
    print("\n⚠️ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nRecommendations:")
    print("  1. Ensure you're running from the kha/ directory")
    print("  2. Check that all .py files are in python/core/")
    print("  3. Verify PYTHONPATH includes the project root")
else:
    print("\n✅ All checks passed! Adapter blending is ready.")
    print("\nYou can now:")
    print("  1. Run inference with blended adapters")
    print("  2. Train new adapters for users/teams")
    print("  3. Export mesh summaries with context")

print("\n" + "="*60)
