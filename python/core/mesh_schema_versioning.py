#!/usr/bin/env python3
"""
Mesh Schema Versioning System for TORI
Handles schema evolution, migrations, and backward compatibility
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy
import semantic_version

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & ENUMS
# ============================================================================

CURRENT_SCHEMA_VERSION = "2.0.0"
MINIMUM_SUPPORTED_VERSION = "1.0.0"

class SchemaCompatibility(Enum):
    """Schema compatibility levels."""
    COMPATIBLE = "compatible"          # No changes needed
    BACKWARD_COMPATIBLE = "backward"   # Can read old, writes new
    FORWARD_COMPATIBLE = "forward"     # Can read new, writes old
    REQUIRES_MIGRATION = "migration"   # Needs migration
    INCOMPATIBLE = "incompatible"      # Cannot handle

class MigrationStatus(Enum):
    """Migration operation status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

# Schema version history with changes
SCHEMA_VERSIONS = {
    "1.0.0": {
        "released": "2024-01-01",
        "description": "Initial schema version",
        "fields": [
            "user_id",
            "timestamp", 
            "personal_concepts",
            "open_intents",
            "recent_activity"
        ]
    },
    "1.1.0": {
        "released": "2024-06-01",
        "description": "Added team concepts and groups",
        "fields_added": [
            "team_concepts",
            "groups"
        ],
        "backward_compatible": True
    },
    "1.2.0": {
        "released": "2024-09-01",
        "description": "Added global concepts",
        "fields_added": [
            "global_concepts"
        ],
        "backward_compatible": True
    },
    "2.0.0": {
        "released": "2025-01-01",
        "description": "Major update with filtering metadata and starred items",
        "fields_added": [
            "schema_version",
            "filtering_applied",
            "filter_mode",
            "filter_stats",
            "starred_items"
        ],
        "fields_modified": {
            "personal_concepts": "Added 'starred' and 'embedding' fields",
            "open_intents": "Added 'priority' and 'deadline' fields"
        },
        "breaking_changes": [
            "Changed intent 'type' to 'intent_type'",
            "Concepts now require 'score' field"
        ]
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SchemaVersion:
    """Represents a schema version."""
    version: str
    major: int = 0
    minor: int = 0
    patch: int = 0
    
    def __post_init__(self):
        """Parse version string."""
        try:
            v = semantic_version.Version(self.version)
            self.major = v.major
            self.minor = v.minor
            self.patch = v.patch
        except:
            # Fallback to simple parsing
            parts = self.version.split('.')
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
    
    def __str__(self):
        return self.version
    
    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

@dataclass
class MigrationResult:
    """Result of a migration operation."""
    source_version: str
    target_version: str
    status: MigrationStatus
    data: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "status": self.status.value,
            "warnings": self.warnings,
            "errors": self.errors
        }

# ============================================================================
# SCHEMA VERSIONING MANAGER
# ============================================================================

class MeshSchemaManager:
    """
    Manages mesh schema versions, migrations, and compatibility.
    """
    
    def __init__(self, 
                 current_version: str = CURRENT_SCHEMA_VERSION,
                 min_supported: str = MINIMUM_SUPPORTED_VERSION):
        """
        Initialize schema manager.
        
        Args:
            current_version: Current schema version
            min_supported: Minimum supported version
        """
        self.current_version = SchemaVersion(current_version)
        self.min_supported = SchemaVersion(min_supported)
        
        # Migration functions registry
        self.migrations: Dict[Tuple[str, str], Callable] = {}
        self._register_migrations()
        
        # Statistics
        self.stats = {
            "checks": 0,
            "migrations": 0,
            "failures": 0
        }
        
        logger.info(f"MeshSchemaManager initialized: v{current_version}")
    
    def add_version_to_mesh(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add schema version to mesh data.
        
        Args:
            mesh_data: Mesh summary data
            
        Returns:
            Updated mesh data with version
        """
        mesh_data["schema_version"] = str(self.current_version)
        mesh_data["schema_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "generator": "MeshSummaryExporter",
            "compatibility": "backward"
        }
        return mesh_data
    
    def check_compatibility(self, mesh_data: Dict[str, Any]) -> SchemaCompatibility:
        """
        Check schema compatibility of mesh data.
        
        Args:
            mesh_data: Mesh summary data
            
        Returns:
            Compatibility level
        """
        self.stats["checks"] += 1
        
        # Get version from data
        version_str = mesh_data.get("schema_version")
        
        if not version_str:
            # No version means legacy 1.0.0
            logger.warning("No schema version found, assuming 1.0.0")
            version_str = "1.0.0"
        
        version = SchemaVersion(version_str)
        
        # Check compatibility
        if version == self.current_version:
            return SchemaCompatibility.COMPATIBLE
        
        if version < self.min_supported:
            return SchemaCompatibility.INCOMPATIBLE
        
        if version < self.current_version:
            # Check if migration exists
            if self._has_migration_path(version_str, str(self.current_version)):
                return SchemaCompatibility.REQUIRES_MIGRATION
            else:
                return SchemaCompatibility.BACKWARD_COMPATIBLE
        
        if version > self.current_version:
            # Newer version - check if we can handle
            if version.major == self.current_version.major:
                return SchemaCompatibility.FORWARD_COMPATIBLE
            else:
                return SchemaCompatibility.INCOMPATIBLE
        
        return SchemaCompatibility.COMPATIBLE
    
    def migrate(self, mesh_data: Dict[str, Any], 
                target_version: Optional[str] = None) -> MigrationResult:
        """
        Migrate mesh data to target version.
        
        Args:
            mesh_data: Mesh summary data
            target_version: Target version (default: current)
            
        Returns:
            Migration result
        """
        source_version = mesh_data.get("schema_version", "1.0.0")
        target_version = target_version or str(self.current_version)
        
        if source_version == target_version:
            return MigrationResult(
                source_version=source_version,
                target_version=target_version,
                status=MigrationStatus.SKIPPED,
                data=mesh_data
            )
        
        # Find migration path
        migration_path = self._find_migration_path(source_version, target_version)
        
        if not migration_path:
            return MigrationResult(
                source_version=source_version,
                target_version=target_version,
                status=MigrationStatus.FAILED,
                errors=[f"No migration path from {source_version} to {target_version}"]
            )
        
        # Apply migrations
        migrated_data = copy.deepcopy(mesh_data)
        warnings = []
        errors = []
        
        for from_v, to_v in migration_path:
            try:
                migration_func = self.migrations.get((from_v, to_v))
                if migration_func:
                    migrated_data, migration_warnings = migration_func(migrated_data)
                    warnings.extend(migration_warnings)
                    logger.info(f"Migrated {from_v} -> {to_v}")
            except Exception as e:
                errors.append(f"Migration {from_v} -> {to_v} failed: {str(e)}")
                self.stats["failures"] += 1
                return MigrationResult(
                    source_version=source_version,
                    target_version=target_version,
                    status=MigrationStatus.FAILED,
                    errors=errors
                )
        
        # Update version
        migrated_data["schema_version"] = target_version
        self.stats["migrations"] += 1
        
        return MigrationResult(
            source_version=source_version,
            target_version=target_version,
            status=MigrationStatus.SUCCESS if not warnings else MigrationStatus.PARTIAL,
            data=migrated_data,
            warnings=warnings
        )
    
    def validate_schema(self, mesh_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate mesh data against its declared schema version.
        
        Args:
            mesh_data: Mesh summary data
            
        Returns:
            (is_valid, list of validation errors)
        """
        version = mesh_data.get("schema_version", "1.0.0")
        errors = []
        
        # Check required fields based on version
        if version >= "2.0.0":
            required = ["schema_version", "user_id", "timestamp", "personal_concepts"]
            for field in required:
                if field not in mesh_data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate concept structure
            for concept in mesh_data.get("personal_concepts", []):
                if "name" not in concept:
                    errors.append("Concept missing 'name' field")
                if "score" not in concept:
                    errors.append("Concept missing 'score' field (required in v2.0.0+)")
        
        elif version >= "1.1.0":
            required = ["user_id", "timestamp", "personal_concepts"]
            for field in required:
                if field not in mesh_data:
                    errors.append(f"Missing required field: {field}")
        
        return len(errors) == 0, errors
    
    def _register_migrations(self):
        """Register migration functions."""
        # 1.0.0 -> 1.1.0
        self.migrations[("1.0.0", "1.1.0")] = self._migrate_1_0_to_1_1
        
        # 1.1.0 -> 1.2.0
        self.migrations[("1.1.0", "1.2.0")] = self._migrate_1_1_to_1_2
        
        # 1.2.0 -> 2.0.0
        self.migrations[("1.2.0", "2.0.0")] = self._migrate_1_2_to_2_0
        
        # Direct migrations for common cases
        self.migrations[("1.0.0", "2.0.0")] = self._migrate_1_0_to_2_0
    
    def _migrate_1_0_to_1_1(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from 1.0.0 to 1.1.0."""
        warnings = []
        
        # Add new fields with defaults
        if "team_concepts" not in data:
            data["team_concepts"] = {}
            warnings.append("Added empty team_concepts")
        
        if "groups" not in data:
            data["groups"] = []
            warnings.append("Added empty groups list")
        
        return data, warnings
    
    def _migrate_1_1_to_1_2(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from 1.1.0 to 1.2.0."""
        warnings = []
        
        # Add global concepts
        if "global_concepts" not in data:
            data["global_concepts"] = []
            warnings.append("Added empty global_concepts")
        
        return data, warnings
    
    def _migrate_1_2_to_2_0(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from 1.2.0 to 2.0.0."""
        warnings = []
        
        # Add scores to concepts if missing
        for concept in data.get("personal_concepts", []):
            if "score" not in concept:
                concept["score"] = 0.5
                warnings.append(f"Added default score to concept: {concept.get('name', 'unknown')}")
        
        # Fix intent type field
        for intent in data.get("open_intents", []):
            if "type" in intent and "intent_type" not in intent:
                intent["intent_type"] = intent.pop("type")
                warnings.append(f"Renamed 'type' to 'intent_type' for intent: {intent.get('id', 'unknown')}")
            
            # Add priority if missing
            if "priority" not in intent:
                intent["priority"] = "normal"
        
        # Add new metadata fields
        if "starred_items" not in data:
            data["starred_items"] = []
        
        return data, warnings
    
    def _migrate_1_0_to_2_0(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Direct migration from 1.0.0 to 2.0.0."""
        # Apply all intermediate migrations
        data, w1 = self._migrate_1_0_to_1_1(data)
        data, w2 = self._migrate_1_1_to_1_2(data)
        data, w3 = self._migrate_1_2_to_2_0(data)
        
        return data, w1 + w2 + w3
    
    def _has_migration_path(self, from_version: str, to_version: str) -> bool:
        """Check if migration path exists."""
        return (from_version, to_version) in self.migrations or \
               self._find_migration_path(from_version, to_version) is not None
    
    def _find_migration_path(self, from_version: str, to_version: str) -> Optional[List[Tuple[str, str]]]:
        """Find migration path between versions."""
        # Direct migration
        if (from_version, to_version) in self.migrations:
            return [(from_version, to_version)]
        
        # Step-by-step migration
        path = []
        current = from_version
        
        # Simplified path finding for common versions
        version_sequence = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
        
        if from_version in version_sequence and to_version in version_sequence:
            from_idx = version_sequence.index(from_version)
            to_idx = version_sequence.index(to_version)
            
            if from_idx < to_idx:
                for i in range(from_idx, to_idx):
                    path.append((version_sequence[i], version_sequence[i + 1]))
                return path
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "current_version": str(self.current_version),
            "min_supported": str(self.min_supported),
            "compatibility_checks": self.stats["checks"],
            "migrations_performed": self.stats["migrations"],
            "migration_failures": self.stats["failures"]
        }

# ============================================================================
# BACKWARD COMPATIBILITY HANDLERS
# ============================================================================

class BackwardCompatibilityHandler:
    """
    Handles reading and writing mesh data with backward compatibility.
    """
    
    def __init__(self, schema_manager: Optional[MeshSchemaManager] = None):
        """Initialize handler."""
        self.schema_manager = schema_manager or MeshSchemaManager()
    
    def read_mesh_with_compatibility(self, 
                                    file_path: str,
                                    auto_migrate: bool = True) -> Optional[Dict[str, Any]]:
        """
        Read mesh file with backward compatibility handling.
        
        Args:
            file_path: Path to mesh JSON file
            auto_migrate: Automatically migrate if needed
            
        Returns:
            Mesh data (possibly migrated) or None
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Mesh file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check compatibility
            compatibility = self.schema_manager.check_compatibility(data)
            
            if compatibility == SchemaCompatibility.COMPATIBLE:
                logger.debug(f"Mesh schema compatible: {file_path}")
                return data
            
            elif compatibility == SchemaCompatibility.BACKWARD_COMPATIBLE:
                logger.info(f"Reading older schema version from {file_path}")
                return data
            
            elif compatibility == SchemaCompatibility.REQUIRES_MIGRATION:
                if auto_migrate:
                    logger.info(f"Migrating mesh schema: {file_path}")
                    result = self.schema_manager.migrate(data)
                    
                    if result.status == MigrationStatus.SUCCESS:
                        # Optionally save migrated version
                        backup_path = file_path.with_suffix('.backup.json')
                        with open(backup_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        with open(file_path, 'w') as f:
                            json.dump(result.data, f, indent=2)
                        
                        logger.info(f"Migration successful, backup saved to {backup_path}")
                        return result.data
                    else:
                        logger.error(f"Migration failed: {result.errors}")
                        return data  # Return original
                else:
                    logger.warning(f"Migration needed but auto_migrate=False")
                    return data
            
            elif compatibility == SchemaCompatibility.INCOMPATIBLE:
                logger.error(f"Incompatible schema version in {file_path}")
                return None
            
            else:
                return data
        
        except Exception as e:
            logger.error(f"Failed to read mesh file {file_path}: {e}")
            return None
    
    def write_mesh_with_version(self,
                               data: Dict[str, Any],
                               file_path: str) -> bool:
        """
        Write mesh data with proper versioning.
        
        Args:
            data: Mesh data
            file_path: Output path
            
        Returns:
            Success flag
        """
        try:
            # Add version if not present
            if "schema_version" not in data:
                data = self.schema_manager.add_version_to_mesh(data)
            
            # Validate before writing
            is_valid, errors = self.schema_manager.validate_schema(data)
            if not is_valid:
                logger.warning(f"Schema validation warnings: {errors}")
            
            # Write to file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Wrote mesh with schema v{data['schema_version']} to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to write mesh file: {e}")
            return False

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_global_schema_manager: Optional[MeshSchemaManager] = None

def get_global_schema_manager() -> MeshSchemaManager:
    """Get or create global schema manager."""
    global _global_schema_manager
    if _global_schema_manager is None:
        _global_schema_manager = MeshSchemaManager()
    return _global_schema_manager

def add_version_to_mesh(mesh_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add current schema version to mesh data."""
    manager = get_global_schema_manager()
    return manager.add_version_to_mesh(mesh_data)

def migrate_mesh_if_needed(mesh_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate mesh data if needed."""
    manager = get_global_schema_manager()
    compatibility = manager.check_compatibility(mesh_data)
    
    if compatibility == SchemaCompatibility.REQUIRES_MIGRATION:
        result = manager.migrate(mesh_data)
        if result.status in [MigrationStatus.SUCCESS, MigrationStatus.PARTIAL]:
            return result.data
    
    return mesh_data

def read_mesh_safely(file_path: str) -> Optional[Dict[str, Any]]:
    """Read mesh file with automatic compatibility handling."""
    handler = BackwardCompatibilityHandler()
    return handler.read_mesh_with_compatibility(file_path, auto_migrate=True)

def write_mesh_safely(data: Dict[str, Any], file_path: str) -> bool:
    """Write mesh file with proper versioning."""
    handler = BackwardCompatibilityHandler()
    return handler.write_mesh_with_version(data, file_path)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "SchemaVersion",
    "SchemaCompatibility",
    "MigrationStatus",
    "MigrationResult",
    "MeshSchemaManager",
    "BackwardCompatibilityHandler",
    "get_global_schema_manager",
    "add_version_to_mesh",
    "migrate_mesh_if_needed",
    "read_mesh_safely",
    "write_mesh_safely",
    "CURRENT_SCHEMA_VERSION",
    "MINIMUM_SUPPORTED_VERSION"
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
    
    # Test schema versioning
    manager = MeshSchemaManager()
    
    # Test with different versions
    test_data = [
        {"user_id": "test", "timestamp": "2024-01-01"},  # No version (1.0.0)
        {"schema_version": "1.0.0", "user_id": "test"},  # Old version
        {"schema_version": "2.0.0", "user_id": "test", "personal_concepts": []},  # Current
        {"schema_version": "3.0.0", "user_id": "test"}  # Future version
    ]
    
    for data in test_data:
        print(f"\n{'='*60}")
        print(f"Testing data with version: {data.get('schema_version', 'none')}")
        
        compatibility = manager.check_compatibility(data)
        print(f"Compatibility: {compatibility.value}")
        
        if compatibility == SchemaCompatibility.REQUIRES_MIGRATION:
            result = manager.migrate(data)
            print(f"Migration status: {result.status.value}")
            if result.warnings:
                print(f"Warnings: {result.warnings}")
    
    # Show statistics
    print(f"\n{'='*60}")
    print("Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
