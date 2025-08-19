#!/usr/bin/env python3
"""
Test Suite for Mesh Schema Versioning (Improvement #4)
Tests schema evolution, migrations, and backward compatibility
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.mesh_schema_versioning import (
    MeshSchemaManager,
    BackwardCompatibilityHandler,
    SchemaCompatibility,
    MigrationStatus,
    SchemaVersion,
    get_global_schema_manager,
    add_version_to_mesh,
    migrate_mesh_if_needed,
    read_mesh_safely,
    write_mesh_safely,
    CURRENT_SCHEMA_VERSION,
    MINIMUM_SUPPORTED_VERSION
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_schemas():
    """Create test mesh files with different schema versions."""
    print("\n" + "="*60)
    print("SETUP: Creating Test Schema Files")
    print("="*60)
    
    test_dir = Path("tests/mesh_schemas")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Version 1.0.0 (original)
    v1_0_data = {
        "user_id": "test_user",
        "timestamp": "2024-01-01T10:00:00Z",
        "personal_concepts": [
            {"name": "Concept A", "summary": "Description A"},
            {"name": "Concept B", "summary": "Description B"}
        ],
        "open_intents": [
            {"id": "intent_001", "description": "Do something", "type": "task"}
        ],
        "recent_activity": "Working on projects"
    }
    
    with open(test_dir / "v1.0.0_sample.json", 'w') as f:
        json.dump(v1_0_data, f, indent=2)
    
    # Version 1.1.0 (with teams)
    v1_1_data = v1_0_data.copy()
    v1_1_data["team_concepts"] = {"TeamX": [{"name": "Team Concept", "summary": "Shared"}]}
    v1_1_data["groups"] = ["TeamX"]
    
    with open(test_dir / "v1.1.0_sample.json", 'w') as f:
        json.dump(v1_1_data, f, indent=2)
    
    # Version 1.2.0 (with global)
    v1_2_data = v1_1_data.copy()
    v1_2_data["global_concepts"] = [{"name": "Global Concept", "summary": "Universal"}]
    
    with open(test_dir / "v1.2.0_sample.json", 'w') as f:
        json.dump(v1_2_data, f, indent=2)
    
    # Version 2.0.0 (current)
    v2_0_data = {
        "schema_version": "2.0.0",
        "user_id": "test_user",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "Concept A", "summary": "Description A", "score": 0.9},
            {"name": "Concept B", "summary": "Description B", "score": 0.7}
        ],
        "open_intents": [
            {"id": "intent_001", "description": "Do something", "intent_type": "task", "priority": "high"}
        ],
        "recent_activity": "Working on projects",
        "team_concepts": {"TeamX": [{"name": "Team Concept", "summary": "Shared", "score": 0.5}]},
        "global_concepts": [],
        "groups": ["TeamX"],
        "starred_items": ["Concept A"]
    }
    
    with open(test_dir / "v2.0.0_sample.json", 'w') as f:
        json.dump(v2_0_data, f, indent=2)
    
    print(f"✓ Created test files in {test_dir}")
    return test_dir

def test_version_detection():
    """Test schema version detection."""
    print("\n" + "="*60)
    print("TEST 1: Version Detection")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Test data with different versions
    test_cases = [
        ({}, "1.0.0"),  # No version field
        ({"schema_version": "1.0.0"}, "1.0.0"),
        ({"schema_version": "1.1.0"}, "1.1.0"),
        ({"schema_version": "2.0.0"}, "2.0.0"),
        ({"schema_version": "3.0.0"}, "3.0.0")  # Future version
    ]
    
    for data, expected in test_cases:
        version = data.get("schema_version", "none")
        compatibility = manager.check_compatibility(data)
        print(f"\nVersion {version}: {compatibility.value}")
        
        if version == "none":
            assert compatibility == SchemaCompatibility.REQUIRES_MIGRATION
        elif version == expected and version == CURRENT_SCHEMA_VERSION:
            assert compatibility == SchemaCompatibility.COMPATIBLE
        elif version == "3.0.0":
            assert compatibility == SchemaCompatibility.INCOMPATIBLE
    
    print("\n✓ Version detection working correctly")
    return True

def test_migration_1_0_to_2_0():
    """Test migration from 1.0.0 to 2.0.0."""
    print("\n" + "="*60)
    print("TEST 2: Migration 1.0.0 → 2.0.0")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Old format data
    old_data = {
        "user_id": "alice",
        "timestamp": "2024-01-01T10:00:00Z",
        "personal_concepts": [
            {"name": "Project X", "summary": "Main project"}
        ],
        "open_intents": [
            {"id": "001", "description": "Optimize", "type": "optimization"}
        ],
        "recent_activity": "Working"
    }
    
    # Migrate
    result = manager.migrate(old_data, "2.0.0")
    
    assert result.status == MigrationStatus.SUCCESS, "Migration failed"
    assert result.data["schema_version"] == "2.0.0", "Version not updated"
    
    # Check migrations applied
    concept = result.data["personal_concepts"][0]
    assert "score" in concept, "Score not added to concept"
    assert concept["score"] == 0.5, "Default score incorrect"
    
    intent = result.data["open_intents"][0]
    assert "intent_type" in intent, "Intent type not renamed"
    assert "type" not in intent, "Old type field still present"
    assert intent["priority"] == "normal", "Default priority not added"
    
    print(f"\n✓ Migration successful")
    print(f"  Added {len(result.warnings)} default values")
    print(f"  Warnings: {result.warnings[:3]}")
    
    return True

def test_backward_compatibility():
    """Test reading old schema versions."""
    print("\n" + "="*60)
    print("TEST 3: Backward Compatibility")
    print("="*60)
    
    test_dir = create_test_schemas()
    handler = BackwardCompatibilityHandler()
    
    # Test reading each version
    for version_file in ["v1.0.0_sample.json", "v1.1.0_sample.json", "v2.0.0_sample.json"]:
        file_path = test_dir / version_file
        print(f"\nReading {version_file}...")
        
        data = handler.read_mesh_with_compatibility(str(file_path), auto_migrate=True)
        assert data is not None, f"Failed to read {version_file}"
        
        # Should be migrated to current version
        if "schema_version" in data:
            print(f"  Loaded as version: {data['schema_version']}")
        else:
            print(f"  No version (assumed 1.0.0)")
        
        # Check if migrated
        if version_file.startswith("v1"):
            # Should have been migrated
            assert data.get("schema_version") == CURRENT_SCHEMA_VERSION, "Not migrated to current"
            print(f"  ✓ Auto-migrated to {CURRENT_SCHEMA_VERSION}")
    
    return True

def test_validation():
    """Test schema validation."""
    print("\n" + "="*60)
    print("TEST 4: Schema Validation")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Valid v2.0.0 data
    valid_data = {
        "schema_version": "2.0.0",
        "user_id": "test",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "Concept", "summary": "Description", "score": 0.5}
        ]
    }
    
    is_valid, errors = manager.validate_schema(valid_data)
    assert is_valid, f"Valid data marked invalid: {errors}"
    print("✓ Valid schema passes validation")
    
    # Invalid data (missing score in v2.0.0)
    invalid_data = {
        "schema_version": "2.0.0",
        "user_id": "test",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [
            {"name": "Concept", "summary": "Description"}  # Missing score
        ]
    }
    
    is_valid, errors = manager.validate_schema(invalid_data)
    assert not is_valid, "Invalid data passed validation"
    assert any("score" in e for e in errors), "Score error not detected"
    print(f"✓ Invalid schema detected: {errors[0]}")
    
    return True

def test_write_with_version():
    """Test writing mesh with automatic versioning."""
    print("\n" + "="*60)
    print("TEST 5: Write with Versioning")
    print("="*60)
    
    # Create test data without version
    data = {
        "user_id": "write_test",
        "timestamp": datetime.now().isoformat(),
        "personal_concepts": [],
        "open_intents": []
    }
    
    # Add version
    versioned_data = add_version_to_mesh(data)
    
    assert "schema_version" in versioned_data, "Version not added"
    assert versioned_data["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "schema_metadata" in versioned_data, "Metadata not added"
    
    print(f"✓ Version {CURRENT_SCHEMA_VERSION} added to mesh")
    print(f"✓ Metadata: {versioned_data['schema_metadata']}")
    
    # Write to file
    temp_file = Path("test_mesh_write.json")
    success = write_mesh_safely(versioned_data, str(temp_file))
    
    assert success, "Write failed"
    assert temp_file.exists(), "File not created"
    
    # Read back and verify
    loaded = read_mesh_safely(str(temp_file))
    assert loaded["schema_version"] == CURRENT_SCHEMA_VERSION
    
    print(f"✓ Successfully wrote and read versioned mesh")
    
    # Cleanup
    temp_file.unlink()
    
    return True

def test_migration_path():
    """Test finding migration paths between versions."""
    print("\n" + "="*60)
    print("TEST 6: Migration Paths")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Test migration path finding
    paths = [
        ("1.0.0", "1.1.0", True),
        ("1.0.0", "2.0.0", True),  # Direct migration
        ("1.1.0", "2.0.0", True),
        ("2.0.0", "1.0.0", False),  # No backward migration
        ("1.0.0", "3.0.0", False)  # No path to future
    ]
    
    for from_v, to_v, should_exist in paths:
        has_path = manager._has_migration_path(from_v, to_v)
        print(f"\n{from_v} → {to_v}: {'✓ Path exists' if has_path else '✗ No path'}")
        
        if should_exist:
            assert has_path, f"Migration path {from_v} → {to_v} should exist"
        else:
            assert not has_path, f"Migration path {from_v} → {to_v} should not exist"
    
    return True

def test_safe_functions():
    """Test convenience functions for safe reading/writing."""
    print("\n" + "="*60)
    print("TEST 7: Safe Read/Write Functions")
    print("="*60)
    
    # Create old format file
    old_data = {
        "user_id": "safe_test",
        "personal_concepts": [{"name": "Test", "summary": "Test concept"}]
    }
    
    temp_file = Path("test_safe_mesh.json")
    with open(temp_file, 'w') as f:
        json.dump(old_data, f)
    
    # Read with auto-migration
    migrated = read_mesh_safely(str(temp_file))
    
    assert migrated is not None, "Read failed"
    assert migrated.get("schema_version") == CURRENT_SCHEMA_VERSION, "Not migrated"
    print(f"✓ Old format auto-migrated to v{CURRENT_SCHEMA_VERSION}")
    
    # Check backup was created
    backup_file = temp_file.with_suffix('.backup.json')
    if backup_file.exists():
        print(f"✓ Backup created: {backup_file}")
        backup_file.unlink()
    
    # Cleanup
    temp_file.unlink()
    
    return True

def test_statistics():
    """Test statistics tracking."""
    print("\n" + "="*60)
    print("TEST 8: Statistics Tracking")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Perform various operations
    manager.check_compatibility({"schema_version": "1.0.0"})
    manager.check_compatibility({"schema_version": "2.0.0"})
    manager.migrate({"user_id": "test"}, "2.0.0")
    
    stats = manager.get_statistics()
    
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert stats["compatibility_checks"] >= 2, "Checks not counted"
    assert stats["migrations_performed"] >= 1, "Migrations not counted"
    
    print("\n✓ Statistics tracking working")
    
    return True

def test_version_comparison():
    """Test version comparison logic."""
    print("\n" + "="*60)
    print("TEST 9: Version Comparison")
    print("="*60)
    
    v1 = SchemaVersion("1.0.0")
    v1_1 = SchemaVersion("1.1.0")
    v2 = SchemaVersion("2.0.0")
    
    assert v1 < v1_1, "1.0.0 should be less than 1.1.0"
    assert v1_1 < v2, "1.1.0 should be less than 2.0.0"
    assert v1 == SchemaVersion("1.0.0"), "Same versions should be equal"
    
    print("✓ Version comparison working correctly")
    print(f"  1.0.0 < 1.1.0: {v1 < v1_1}")
    print(f"  1.1.0 < 2.0.0: {v1_1 < v2}")
    print(f"  1.0.0 == 1.0.0: {v1 == SchemaVersion('1.0.0')}")
    
    return True

def test_partial_migration():
    """Test partial migration with warnings."""
    print("\n" + "="*60)
    print("TEST 10: Partial Migration")
    print("="*60)
    
    manager = MeshSchemaManager()
    
    # Data that will generate warnings
    data = {
        "user_id": "partial_test",
        "personal_concepts": [
            {"name": "Concept without score", "summary": "Test"}
        ],
        "open_intents": [
            {"id": "001", "description": "Task", "type": "old_format"}
        ]
    }
    
    result = manager.migrate(data, "2.0.0")
    
    assert result.status in [MigrationStatus.SUCCESS, MigrationStatus.PARTIAL]
    assert len(result.warnings) > 0, "No warnings generated"
    
    print(f"✓ Migration status: {result.status.value}")
    print(f"✓ Warnings generated: {len(result.warnings)}")
    for warning in result.warnings[:3]:
        print(f"  - {warning}")
    
    return True

def run_all_tests():
    """Run all schema versioning tests."""
    print("\n" + "="*60)
    print("MESH SCHEMA VERSIONING TEST SUITE (Improvement #4)")
    print("="*60)
    
    tests = [
        ("Version Detection", test_version_detection),
        ("Migration 1.0→2.0", test_migration_1_0_to_2_0),
        ("Backward Compatibility", test_backward_compatibility),
        ("Schema Validation", test_validation),
        ("Write with Version", test_write_with_version),
        ("Migration Paths", test_migration_path),
        ("Safe Functions", test_safe_functions),
        ("Statistics", test_statistics),
        ("Version Comparison", test_version_comparison),
        ("Partial Migration", test_partial_migration)
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
    
    # Cleanup test directory
    test_dir = Path("tests/mesh_schemas")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    if passed == total:
        print("\n🎉 All schema versioning tests passed! Future-proof versioning is working.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
