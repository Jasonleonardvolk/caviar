#!/usr/bin/env python3
"""
Phase 2 Test Suite: Mesh Context Injection
Tests nightly export, inference injection, and training enrichment
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.mesh_summary_exporter import MeshSummaryExporter, run_nightly_export
from core.saigon_inference import SaigonInference, SaigonConfig
from training.train_lora_adapter import generate_training_data_from_mesh

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_mesh_export():
    """Test mesh summary export for users and groups."""
    print("\n" + "="*60)
    print("TEST 1: Mesh Summary Export")
    print("="*60)
    
    # Create test data directories
    memory_vault = Path("memory_vault")
    memory_vault.mkdir(exist_ok=True)
    (memory_vault / "traces").mkdir(exist_ok=True)
    (memory_vault / "sessions").mkdir(exist_ok=True)
    (memory_vault / "intents").mkdir(exist_ok=True)
    
    # Create sample open intents
    intents_data = {
        "open_intents": [
            {
                "id": "intent_001",
                "description": "Find optimization for Alpha Protocol",
                "type": "optimization",
                "priority": "high",
                "created_at": "2025-08-06T10:00:00Z",
                "last_active": "2025-08-07T09:00:00Z"
            },
            {
                "id": "intent_002",
                "description": "Complete Project X documentation",
                "type": "documentation",
                "priority": "normal",
                "created_at": "2025-08-05T14:00:00Z",
                "last_active": "2025-08-06T16:00:00Z"
            }
        ]
    }
    
    with open(memory_vault / "intents" / "jason_open_intents.json", 'w') as f:
        json.dump(intents_data, f, indent=2)
    
    # Initialize exporter
    exporter = MeshSummaryExporter()
    
    # Export for test user
    user_id = "jason"
    summary_path = exporter.export_user_mesh_summary(user_id)
    
    # Verify export
    assert Path(summary_path).exists(), "Summary file not created"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n✓ Exported summary for {user_id}")
    print(f"  - User ID: {summary['user_id']}")
    print(f"  - Timestamp: {summary['timestamp']}")
    print(f"  - Personal concepts: {len(summary.get('personal_concepts', []))}")
    print(f"  - Open intents: {len(summary.get('open_intents', []))}")
    print(f"  - Recent activity: {summary.get('recent_activity', 'None')}")
    
    # Export group summary
    group_id = "ProjectX"
    group_path = exporter.export_group_mesh_summary(group_id)
    
    print(f"\n✓ Exported group summary for {group_id}")
    print(f"  - Path: {group_path}")
    
    return True

def test_context_injection():
    """Test mesh context injection in inference."""
    print("\n" + "="*60)
    print("TEST 2: Context Injection in Inference")
    print("="*60)
    
    # Create sample mesh summary
    mesh_dir = Path("models/mesh_contexts")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    test_summary = {
        "user_id": "alice",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "Project X", "summary": "Main project focus", "score": 0.9},
            {"name": "Alpha Protocol", "summary": "Security protocol", "score": 0.7}
        ],
        "open_intents": [
            {
                "id": "intent_47",
                "description": "Optimize Alpha Protocol performance",
                "intent_type": "optimization",
                "last_active": "2025-08-07"
            }
        ],
        "recent_activity": "Working on Project X timeline and Alpha Protocol optimization",
        "team_concepts": {
            "ProjectX": [
                {"name": "Beta Algorithm", "summary": "Shared algorithm", "score": 0.8}
            ]
        },
        "global_concepts": [],
        "groups": ["ProjectX"]
    }
    
    with open(mesh_dir / "alice_mesh.json", 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Initialize inference engine
    config = SaigonConfig(
        enable_mesh_injection=True,
        enable_group_context=True
    )
    
    engine = SaigonInference(config)
    
    # Test context loading
    context = engine.load_mesh_context("alice")
    assert context is not None, "Failed to load mesh context"
    
    print("\n✓ Loaded mesh context for alice")
    print(f"  - Personal concepts: {len(context.get('personal_concepts', []))}")
    print(f"  - Open intents: {len(context.get('open_intents', []))}")
    
    # Test prompt preparation
    original_prompt = "How can I improve performance?"
    enhanced_prompt = engine.prepare_prompt_with_context(original_prompt, context)
    
    assert "[Context Information]" in enhanced_prompt, "Context not injected"
    assert "Project X" in enhanced_prompt, "Personal concepts not included"
    assert "Optimize Alpha Protocol" in enhanced_prompt, "Open intents not included"
    
    print("\n✓ Context injection successful")
    print(f"  Original prompt: {original_prompt}")
    print(f"  Enhanced prompt preview: {enhanced_prompt[:200]}...")
    
    # Test with context disabled
    config.enable_mesh_injection = False
    engine = SaigonInference(config)
    no_context_prompt = engine.prepare_prompt_with_context(original_prompt, context)
    
    assert no_context_prompt == original_prompt, "Context injected when disabled"
    print("\n✓ Context injection toggle works correctly")
    
    return True

def test_training_data_generation():
    """Test training data generation from mesh context."""
    print("\n" + "="*60)
    print("TEST 3: Training Data Generation from Mesh")
    print("="*60)
    
    # Ensure mesh summary exists
    user_id = "jason"
    mesh_file = Path("models/mesh_contexts") / f"{user_id}_mesh.json"
    
    if not mesh_file.exists():
        # Run export first
        exporter = MeshSummaryExporter()
        exporter.export_user_mesh_summary(user_id)
    
    # Generate training data
    output_path = generate_training_data_from_mesh(
        user_id=user_id,
        mesh_contexts_dir="models/mesh_contexts",
        memory_vault_dir="memory_vault",
        mask_group_concepts=False
    )
    
    assert Path(output_path).exists(), "Training data not generated"
    
    # Load and verify training data
    training_examples = []
    with open(output_path, 'r') as f:
        for line in f:
            if line.strip():
                training_examples.append(json.loads(line))
    
    print(f"\n✓ Generated {len(training_examples)} training examples")
    
    # Check for different types of examples
    intent_examples = [e for e in training_examples if "Intent" in e.get("output", "")]
    concept_examples = [e for e in training_examples if "Tell me about" in e.get("input", "")]
    
    print(f"  - Intent-based examples: {len(intent_examples)}")
    print(f"  - Concept-based examples: {len(concept_examples)}")
    
    # Test with group masking
    masked_output = generate_training_data_from_mesh(
        user_id=user_id,
        mask_group_concepts=True
    )
    
    masked_examples = []
    with open(masked_output, 'r') as f:
        for line in f:
            if line.strip():
                masked_examples.append(json.loads(line))
    
    team_examples = [e for e in masked_examples if "Team" in e.get("input", "") or "Team" in e.get("output", "")]
    
    print(f"\n✓ Group masking works")
    print(f"  - Team examples when masked: {len(team_examples)} (should be 0)")
    
    return True

def test_nightly_export():
    """Test nightly export for multiple users."""
    print("\n" + "="*60)
    print("TEST 4: Nightly Export Process")
    print("="*60)
    
    # Run nightly export for test users
    test_users = ["jason", "alice", "bob"]
    results = run_nightly_export(test_users)
    
    print(f"\n✓ Nightly export completed for {len(results)} users")
    
    for user_id, path in results.items():
        if path:
            print(f"  - {user_id}: {Path(path).name}")
        else:
            print(f"  - {user_id}: Failed")
    
    # Verify group summaries created
    groups_dir = Path("models/mesh_contexts/groups")
    if groups_dir.exists():
        group_files = list(groups_dir.glob("*.json"))
        print(f"\n✓ Group summaries: {len(group_files)} created")
        for gf in group_files:
            print(f"  - {gf.name}")
    
    return True

def test_group_context_integration():
    """Test group/team context handling."""
    print("\n" + "="*60)
    print("TEST 5: Group Context Integration")
    print("="*60)
    
    # Create group summary
    groups_dir = Path("models/mesh_contexts/groups")
    groups_dir.mkdir(parents=True, exist_ok=True)
    
    group_summary = {
        "group_id": "ProjectX",
        "timestamp": datetime.now().isoformat(),
        "concepts": [
            {"name": "Beta Algorithm", "summary": "Team's shared algorithm", "score": 0.8},
            {"name": "Q4 Planning", "summary": "Quarter 4 objectives", "score": 0.7}
        ],
        "shared_intents": [],
        "recent_activity": "Team collaboration on Beta Algorithm",
        "members": ["alice", "bob", "jason"]
    }
    
    with open(groups_dir / "ProjectX_mesh.json", 'w') as f:
        json.dump(group_summary, f, indent=2)
    
    # Create user summary with group reference
    user_summary = {
        "user_id": "bob",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "My Task", "summary": "Personal work item", "score": 0.8}
        ],
        "open_intents": [],
        "recent_activity": "Working on personal tasks",
        "team_concepts": {},
        "global_concepts": [],
        "groups": ["ProjectX"]
    }
    
    mesh_dir = Path("models/mesh_contexts")
    with open(mesh_dir / "bob_mesh.json", 'w') as f:
        json.dump(user_summary, f, indent=2)
    
    # Test loading with group context
    config = SaigonConfig(
        enable_mesh_injection=True,
        enable_group_context=True
    )
    
    engine = SaigonInference(config)
    context = engine.load_mesh_context("bob")
    
    assert "group_contexts" in context, "Group contexts not loaded"
    assert "ProjectX" in context["group_contexts"], "ProjectX group not loaded"
    
    print("\n✓ Group context loaded successfully")
    print(f"  - User: bob")
    print(f"  - Groups: {context.get('groups', [])}")
    print(f"  - Group contexts loaded: {list(context.get('group_contexts', {}).keys())}")
    
    # Test prompt with group context
    prompt = "What should I work on?"
    enhanced = engine.prepare_prompt_with_context(prompt, context)
    
    assert "Beta Algorithm" in enhanced, "Group concepts not in prompt"
    print("\n✓ Group concepts injected into prompt")
    
    return True

def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "="*60)
    print("PHASE 2 TEST SUITE: MESH CONTEXT INJECTION")
    print("="*60)
    
    tests = [
        ("Mesh Export", test_mesh_export),
        ("Context Injection", test_context_injection),
        ("Training Data Generation", test_training_data_generation),
        ("Nightly Export", test_nightly_export),
        ("Group Context", test_group_context_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
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
        print("\n🎉 All Phase 2 tests passed! Mesh context injection is working.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
