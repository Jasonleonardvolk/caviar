#!/usr/bin/env python3
"""
Restore the canonical concept database from backup
The migration was already done but got accidentally reset
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

print("🔄 RESTORING CANONICAL CONCEPT DATABASE")
print("="*60)

# Find the most recent backup
archive_dir = Path("data/archive")
backup_files = list(archive_dir.glob("concept_db_backup_*.json"))

if not backup_files:
    print("❌ No backup files found!")
    exit(1)

# Sort by modification time and get the most recent
backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
latest_backup = backup_files[0]

print(f"📁 Found backup: {latest_backup}")

# Load the backup
with open(latest_backup, 'r', encoding='utf-8') as f:
    backup_data = json.load(f)

print(f"📊 Backup contains {len(backup_data.get('concepts', {}))} concepts")

# Check current state
current_db = Path("data/concept_db.json")
if current_db.exists():
    with open(current_db, 'r', encoding='utf-8') as f:
        current_data = json.load(f)
    current_count = len(current_data.get('concepts', {}))
    print(f"⚠️  Current database has {current_count} concepts")
    
    if current_count > 0:
        response = input("Current database is not empty. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Restoration cancelled.")
            exit(0)

# Restore from backup
print("\n🔄 Restoring from backup...")
with open(current_db, 'w', encoding='utf-8') as f:
    json.dump(backup_data, f, indent=2, ensure_ascii=False)

print(f"✅ Restored {len(backup_data.get('concepts', {}))} concepts")
print(f"✅ Canonical database restored to: {current_db}")

# Show some sample concepts
print("\n📋 Sample concepts restored:")
concepts = backup_data.get('concepts', {})
for i, (cid, concept) in enumerate(list(concepts.items())[:5]):
    print(f"  • {concept['name']} (score: {concept.get('score', 'N/A')}, method: {concept.get('method', 'N/A')})")

print(f"\n✅ Total concepts: {len(concepts)}")
print("🎉 Your canonical concept database has been restored!")
