# VaultInspector Advanced Features 🚀

## New Features Added:

### 1. 🕸️ Vault ↔ Mesh Consistency Checker

Compare your UnifiedMemoryVault entries with ConceptMesh structure to ensure cognitive integrity.

**Usage:**
```bash
# Compare vault with mesh
python tools/vault_inspector.py --compare-mesh concept_mesh_data.json

# Or use the convenience script
tools\vault_mesh_compare.bat concept_mesh_data.json

# JSON output for automation
python tools/vault_inspector.py --compare-mesh mesh.json --json > comparison.json
```

**What it checks:**
- 🔹 **Vault entries referencing missing concepts** - Memory entries that reference concept IDs not present in the mesh
- 🔹 **Orphaned mesh nodes** - Concepts in the mesh that are never referenced by any vault entry
- 🔹 **Invalid concept bindings** - Entries with concept_binding but no concept_ids
- 🔹 **Hash conflicts** (future) - Same concept ID with different data

**Example Output:**
```
🕸️ Vault ↔ Mesh Comparison
──────────────────────────────────────────────────
• Vault entries: 148
• Mesh nodes: 92
• Entries with concepts: 43
• Orphaned concepts: 12

❌ Vault entries referencing missing concepts: 3
   - Entry mem_003 → missing concept concept_missing
   - Entry mem_007 → missing concept old_concept_42
   - Entry mem_015 → missing concept deleted_concept

⚠️ Mesh nodes never referenced in vault: 12
   - Concept concept_orphan: orphaned
   - Concept concept_unused: experimental
   ... and 10 more

✅ Exit code 1 if inconsistencies found (for CI)
```

### 2. 📊 Delta Tracker Between Snapshots

Compare two vault snapshots or session logs to track changes over time.

**Usage:**
```bash
# Compare snapshots
python tools/vault_inspector.py --delta old_snapshot.json new_snapshot.json

# Compare session logs
python tools/vault_inspector.py --delta session_20250701.jsonl session_20250702.jsonl

# Or use the convenience script
tools\vault_delta.bat vault_20250701.json vault_20250702.json

# JSON output
python tools/vault_inspector.py --delta old.json new.json --json
```

**What it tracks:**
- 🆕 **New entries** - Entries added in the new snapshot
- ♻️ **Unchanged entries** - Entries that remain the same
- 🧬 **Modified entries** - Entries with same ID but different content/metadata
- 🗑️ **Deleted entries** - Entries removed in the new snapshot
- 🔀 **Hash changes** - File and entry-level hash tracking

**Example Output:**
```
📊 Snapshot Delta Analysis
──────────────────────────────────────────────────
Old snapshot: 4c77b4b3a2e9f1d6...
New snapshot: 8d92f3e1b7c4a5d2...

🆕 New entries: 17
♻️ Unchanged entries: 125
🧬 Modified entries: 6
🗑️ Deleted entries: 3

🆕 New Entries (17):
   - mem_150 (semantic): New research on quantum...
   - mem_151 (episodic): Meeting with Dr. Smith...
   - mem_152 (procedural): Updated algorithm for...
   ... and 14 more

🧬 Modified Entries (6):
   - mem_001 (semantic): changed metadata, importance
     Hash: 4c77b4b3a2e9f1d6 → 8d92f3e1b7c4a5d2
   - mem_042 (working): changed content
     Hash: 1a2b3c4d5e6f7890 → 9f8e7d6c5b4a3210
   ... and 4 more

🗑️ Deleted Entries (3):
   - mem_002 (semantic): Water is wet...
   - mem_099 (ghost): Temporary calculation...
   - mem_145 (working): Draft response...
```

## Integration Examples:

### CI/CD Pipeline Integration
```yaml
# .github/workflows/vault_check.yml
- name: Check Vault-Mesh Consistency
  run: |
    python tools/vault_inspector.py --compare-mesh data/concept_mesh.json
  continue-on-error: false
```

### Daily Health Check Script
```python
#!/usr/bin/env python3
"""Daily vault health check"""

import subprocess
import json
from datetime import datetime

# Check consistency
result = subprocess.run(
    ["python", "tools/vault_inspector.py", "--compare-mesh", "mesh.json", "--json"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    comparison = json.loads(result.stdout)
    send_alert(f"Vault inconsistencies: {comparison['statistics']}")

# Track daily changes
today = datetime.now().strftime("%Y%m%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

subprocess.run([
    "python", "tools/vault_inspector.py", 
    "--delta", f"vault_{yesterday}.json", f"vault_{today}.json"
])
```

### Automated Orphan Cleanup
```python
# Find orphaned concepts
result = subprocess.run(
    ["python", "tools/vault_inspector.py", "--compare-mesh", "mesh.json", "--json"],
    capture_output=True,
    text=True
)

comparison = json.loads(result.stdout)
orphaned = comparison["mesh_nodes_without_vault"]

if len(orphaned) > 10:
    print(f"Found {len(orphaned)} orphaned concepts")
    # Archive or remove orphaned concepts
```

## Advanced Usage:

### 1. **Pre-Training Validation**
```bash
# Ensure vault-mesh consistency before training
python tools/vault_inspector.py --compare-mesh mesh.json || exit 1
python tools/vault_inspector.py --bundle pre_training_backup.tar.zst
```

### 2. **Session Evolution Tracking**
```bash
# Track how a session evolved
for session in logs/session_*.jsonl; do
    python tools/vault_inspector.py --delta baseline.json $session --json >> evolution.jsonl
done
```

### 3. **Concept Drift Detection**
```python
# Monitor concept usage over time
import json
from collections import defaultdict

concept_usage = defaultdict(list)

for day in range(1, 31):
    result = subprocess.run([
        "python", "tools/vault_inspector.py", 
        "--compare-mesh", "mesh.json", 
        "--vault-path", f"vault_day_{day}",
        "--json"
    ], capture_output=True, text=True)
    
    data = json.loads(result.stdout)
    concept_usage[day] = data["statistics"]["entries_with_concepts"]

# Plot concept usage trends
```

## Error Codes:

- **Exit 0**: All checks passed
- **Exit 1**: Inconsistencies found (for --compare-mesh) or general error
- **Exit 2**: File not found errors

## Best Practices:

1. **Run mesh comparison before major operations**
   ```bash
   python tools/vault_inspector.py --compare-mesh mesh.json && python train.py
   ```

2. **Archive deltas for audit trail**
   ```bash
   python tools/vault_inspector.py --delta old.json new.json --json > deltas/$(date +%Y%m%d).json
   ```

3. **Monitor orphaned concepts**
   - Set thresholds (e.g., >10% orphaned = warning)
   - Periodically clean up unused concepts
   - Track concept lifecycle

4. **Use in pre-commit hooks**
   ```bash
   #!/bin/bash
   # .git/hooks/pre-commit
   python tools/vault_inspector.py --check-consistency || exit 1
   ```

## Troubleshooting:

### "Mesh file not found"
- Check path is correct
- Ensure mesh file exists
- Use absolute paths if needed

### Large number of orphaned concepts
- Normal after concept pruning
- May indicate stale mesh data
- Consider archiving old concepts

### Delta shows all entries as new
- Check file format compatibility
- Ensure IDs are consistent
- Verify both files are valid JSON/JSONL

Your VaultInspector now provides complete cognitive integrity checking and evolution tracking! 🎯
