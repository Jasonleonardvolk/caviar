# Mesh Schema Changelog

## Version History

### Version 2.0.0 (2025-01-01)
**Breaking Changes**
- Changed `type` field to `intent_type` in open_intents
- All concepts now require a `score` field (0.0-1.0)
- Added mandatory `schema_version` field

**New Features**
- Added `schema_version` field for version tracking
- Added `schema_metadata` with generation info
- Added `starred_items` list for user-starred concepts/intents
- Added `filtering_applied` and `filter_mode` for context filtering
- Added `filter_stats` with filtering metrics
- Added `embedding` field to concepts (optional)
- Added `priority` and `deadline` fields to intents

**Improvements**
- Concepts can now include pre-computed embeddings
- Enhanced metadata tracking for audit trails

**Migration Path from 1.x**
- Automatic migration available via `mesh_schema_versioning.py`
- Backup created before migration
- Missing scores default to 0.5
- `type` fields renamed to `intent_type`

---

### Version 1.2.0 (2024-09-01)
**New Features**
- Added `global_concepts` array for organization-wide concepts

**Backward Compatible**
- Old files without global_concepts will have empty array added

---

### Version 1.1.0 (2024-06-01)
**New Features**
- Added `team_concepts` dictionary for team/group concepts
- Added `groups` array for user group memberships

**Backward Compatible**
- Old files without these fields will have defaults added

---

### Version 1.0.0 (2024-01-01)
**Initial Release**
- Basic schema with core fields:
  - `user_id`: User identifier
  - `timestamp`: Generation timestamp
  - `personal_concepts`: User's personal concepts
  - `open_intents`: Unresolved intents
  - `recent_activity`: Recent activity summary

---

## Schema Structure

### Current Schema (v2.0.0)

```json
{
  "schema_version": "2.0.0",
  "user_id": "string",
  "timestamp": "ISO 8601 datetime",
  "personal_concepts": [
    {
      "name": "string",
      "summary": "string",
      "score": 0.0-1.0,
      "keywords": ["string"],
      "embedding": [float array] (optional),
      "starred": boolean (optional)
    }
  ],
  "open_intents": [
    {
      "id": "string",
      "description": "string",
      "intent_type": "string",
      "priority": "low|normal|high|critical",
      "created_at": "ISO 8601",
      "last_active": "ISO 8601",
      "deadline": "ISO 8601" (optional)
    }
  ],
  "recent_activity": "string",
  "team_concepts": {
    "team_name": [concept objects]
  },
  "global_concepts": [concept objects],
  "groups": ["string"],
  "starred_items": ["string"],
  "filtering_applied": boolean (optional),
  "filter_mode": "string" (optional),
  "filter_stats": object (optional),
  "schema_metadata": {
    "generated_at": "ISO 8601",
    "generator": "string",
    "version_info": "string"
  }
}
```

---

## Migration Examples

### Migrating from 1.0.0 to 2.0.0

```python
from mesh_schema_versioning import MeshSchemaManager

manager = MeshSchemaManager()

# Load old data
old_data = {
    "user_id": "alice",
    "timestamp": "2024-01-01T10:00:00Z",
    "personal_concepts": [
        {"name": "Project X", "summary": "Main project"}
    ],
    "open_intents": [
        {"id": "001", "description": "Optimize", "type": "optimization"}
    ]
}

# Migrate
result = manager.migrate(old_data, target_version="2.0.0")

if result.status == "success":
    new_data = result.data
    # Now has:
    # - schema_version: "2.0.0"
    # - concepts have score: 0.5 (default)
    # - intents have intent_type instead of type
    # - intents have priority: "normal" (default)
```

### Reading with Automatic Migration

```python
from mesh_schema_versioning import read_mesh_safely

# Automatically handles any version
mesh_data = read_mesh_safely("models/mesh_contexts/alice_mesh.json")
# Returns migrated data if needed
```

---

## Compatibility Matrix

| Reader Version | File Version | Compatibility | Action Required |
|---------------|--------------|---------------|-----------------|
| 2.0.0 | 2.0.0 | ✅ Full | None |
| 2.0.0 | 1.x.x | ✅ Compatible | Auto-migration available |
| 2.0.0 | 0.x.x | ❌ Incompatible | Manual migration needed |
| 1.x.x | 2.0.0 | ⚠️ Partial | May ignore new fields |
| 1.0.0 | 1.1.0+ | ⚠️ Partial | Will ignore team/global concepts |

---

## Best Practices

### When Adding New Fields

1. **Non-breaking additions**: Add as optional fields with defaults
   - Increment minor version (e.g., 2.0.0 → 2.1.0)
   - Ensure backward compatibility

2. **Breaking changes**: Require migration
   - Increment major version (e.g., 2.0.0 → 3.0.0)
   - Provide migration function
   - Document in changelog

### When Reading Mesh Files

Always use the compatibility handler:
```python
from mesh_schema_versioning import read_mesh_safely

mesh_data = read_mesh_safely(file_path)
# Handles all versions automatically
```

### When Writing Mesh Files

Always include version:
```python
from mesh_schema_versioning import write_mesh_safely

write_mesh_safely(mesh_data, file_path)
# Adds version and validates schema
```

---

## Version Deprecation Schedule

| Version | Status | Support Until | Notes |
|---------|--------|--------------|-------|
| 2.0.0 | Current | - | Active development |
| 1.2.0 | Supported | 2026-01-01 | Auto-migration available |
| 1.1.0 | Supported | 2026-01-01 | Auto-migration available |
| 1.0.0 | Deprecated | 2025-07-01 | Migration recommended |
| < 1.0.0 | Unsupported | - | Manual migration required |

---

## Testing Migration

### Test Files

Test files for each version are available in `tests/mesh_schemas/`:
- `v1.0.0_sample.json` - Original schema
- `v1.1.0_sample.json` - With team concepts
- `v1.2.0_sample.json` - With global concepts
- `v2.0.0_sample.json` - Current schema

### Running Migration Tests

```bash
python python/tests/test_schema_migration.py
```

---

## Troubleshooting

### "Incompatible schema version"
- File version is too old (< 1.0.0) or too new (> current)
- Solution: Check version, update code, or manually migrate

### "Migration failed"
- Data corruption or missing required fields
- Solution: Check migration warnings, fix data manually

### "Schema validation failed"
- Data doesn't match declared version schema
- Solution: Validate required fields, fix structure

---

## Future Versions (Planned)

### Version 2.1.0 (Planned)
- Add `confidence_scores` to concepts
- Add `related_intents` cross-references
- Add `source_documents` tracking

### Version 3.0.0 (Planned)
- Major restructure for graph-based relationships
- Native support for multi-modal embeddings
- Time-series concept evolution

---

*Last Updated: 2025-01-07*
