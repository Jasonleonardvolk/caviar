# 🧪 VaultInspector - Complete Guide

## Overview

VaultInspector is a CLI tool for analyzing and maintaining your UnifiedMemoryVault. It provides insights into memory patterns, session analytics, and data integrity.

## Installation

The tool is already in your `tools/` directory. No installation needed!

## Usage Examples

### 1. Quick Summary
```bash
# From project root
python tools/vault_inspector.py --summary

# Or use the convenience script
tools\vault_summary.bat
```

Output:
```
🧠 Vault Summary
────────────────────────────────
• Entries:        148
• Unique hashes:  145
• Types:
   - persistent: 92
   - ghost: 41
   - working: 15
• Last entry:     2025-07-02 17:57:01
• Corrupt lines:  0
• Sessions:       3
• Total size:     2.4 MB
• Actions:
   - store: 148
   - accessed: 43
   - evicted: 12
```

### 2. Per-Session Analysis
```bash
python tools/vault_inspector.py --per-session
```

Shows detailed breakdown for each session:
- Entry counts
- Duration
- Types distribution
- Actions performed

### 3. Export to CSV
```bash
python tools/vault_inspector.py --export vault_export.csv
```

Creates a CSV with columns:
- session_id
- timestamp
- action
- id
- type
- content (truncated)
- importance
- access_count

### 4. Rebuild Hash Cache
```bash
python tools/vault_inspector.py --rebuild-hash-cache
```

Useful after:
- Manual log edits
- Importing old data
- Recovering from corruption

### 5. Consistency Check
```bash
python tools/vault_inspector.py --check-consistency
```

Detects:
- Duplicate IDs
- Type mismatches
- Orphaned blobs
- Missing files
- Index mismatches

### 6. JSON Output
Add `--json` to any command for machine-readable output:
```bash
python tools/vault_inspector.py --summary --json > vault_stats.json
```

## Advanced Usage

### Custom Vault Path
```bash
python tools/vault_inspector.py --vault-path /path/to/vault --summary
```

### Combined Operations
```bash
# Check everything at once
python tools/vault_inspector.py --summary --check-consistency --per-session
```

### Automated Health Checks
Add to your startup script:
```python
# In enhanced_launcher.py
import subprocess
result = subprocess.run(
    ["python", "tools/vault_inspector.py", "--check-consistency", "--json"],
    capture_output=True,
    text=True
)
issues = json.loads(result.stdout)
if any(issues.values()):
    logger.warning(f"Vault consistency issues: {issues}")
```

## Integration with enhanced_launcher.py

Add to the `--self-test` flag:
```python
def run_vault_health_check():
    """Run VaultInspector as part of self-test"""
    from tools.vault_inspector import VaultInspector
    
    inspector = VaultInspector()
    summary = inspector.summary()
    issues = inspector.check_consistency()
    
    print(f"✅ Vault entries: {summary['entries']}")
    print(f"✅ Unique hashes: {summary['unique_hashes']}")
    
    if any(issues.values()):
        print("❌ Consistency issues found:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"  - {issue_type}: {len(issue_list)} issues")
        return False
    
    return True
```

## Troubleshooting

### "No live log found"
- The vault hasn't been used yet
- Check vault path is correct
- Ensure UnifiedMemoryVault has dual-mode logging enabled

### "Corrupt lines" > 0
- Check for disk space issues
- Look for incomplete writes (system crash)
- Use `--rebuild-hash-cache` to recover

### Large orphaned blobs
- Run consistency check
- Manually remove blobs older than N days
- Check blob compression settings

## Next Steps

1. **Schedule regular checks**: Add to cron/Task Scheduler
2. **Monitor growth**: Track size_mb over time
3. **Export for analysis**: Use CSV export for data science
4. **Automate cleanup**: Script to remove old sessions

The VaultInspector gives you complete visibility into your memory vault's health and contents! 🔍
