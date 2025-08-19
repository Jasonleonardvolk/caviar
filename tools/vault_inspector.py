#!/usr/bin/env python3
"""
🧪 VaultInspector - CLI tool for UnifiedMemoryVault analysis
Provides insights into memory vault health, statistics, and contents
"""

import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import hashlib
import sys

class VaultInspector:
    """Inspector for UnifiedMemoryVault logs and snapshots"""
    
    def __init__(self, vault_path: str = "data/memory_vault"):
        self.vault_path = Path(vault_path)
        self.logs_dir = self.vault_path / "logs"
        self.memories_dir = self.vault_path / "memories"
        self.index_dir = self.vault_path / "index"
        
        # Paths to key files
        self.live_log_path = self.logs_dir / "vault_live.jsonl"
        self.snapshot_path = self.logs_dir / "vault_snapshot.json"
        self.seen_hashes_path = self.logs_dir / "seen_hashes.json"
        
    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive vault summary"""
        summary_data = {
            "entries": 0,
            "unique_hashes": 0,
            "types": defaultdict(int),
            "last_entry": None,
            "corrupt_lines": 0,
            "sessions": set(),
            "actions": defaultdict(int),
            "size_mb": 0
        }
        
        # Analyze live log
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        summary_data["entries"] += 1
                        
                        # Track sessions
                        if "session_id" in entry:
                            summary_data["sessions"].add(entry["session_id"])
                        
                        # Track actions
                        action = entry.get("action", "unknown")
                        summary_data["actions"][action] += 1
                        
                        # Track types
                        if "entry" in entry and "type" in entry["entry"]:
                            memory_type = entry["entry"]["type"]
                            summary_data["types"][memory_type] += 1
                        
                        # Track last entry
                        timestamp = entry.get("timestamp")
                        if timestamp:
                            summary_data["last_entry"] = timestamp
                            
                    except json.JSONDecodeError:
                        summary_data["corrupt_lines"] += 1
                    except Exception as e:
                        print(f"Error on line {line_num}: {e}")
                        summary_data["corrupt_lines"] += 1
        
        # Count unique hashes
        if self.seen_hashes_path.exists():
            with open(self.seen_hashes_path, 'r') as f:
                hashes = json.load(f)
                summary_data["unique_hashes"] = len(hashes)
        
        # Calculate total size
        total_size = 0
        for file_path in self.vault_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        summary_data["size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Convert sessions set to count
        summary_data["session_count"] = len(summary_data["sessions"])
        del summary_data["sessions"]
        
        return summary_data
    
    def per_session_analysis(self) -> List[Dict[str, Any]]:
        """Analyze vault data per session"""
        sessions = defaultdict(lambda: {
            "entries": 0,
            "types": defaultdict(int),
            "actions": defaultdict(int),
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0
        })
        
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        session_id = entry.get("session_id", "unknown")
                        
                        session = sessions[session_id]
                        session["entries"] += 1
                        
                        # Track timestamps
                        timestamp = entry.get("timestamp")
                        if timestamp:
                            if session["start_time"] is None or timestamp < session["start_time"]:
                                session["start_time"] = timestamp
                            if session["end_time"] is None or timestamp > session["end_time"]:
                                session["end_time"] = timestamp
                        
                        # Track actions
                        action = entry.get("action", "unknown")
                        session["actions"][action] += 1
                        
                        # Track types
                        if "entry" in entry and "type" in entry["entry"]:
                            memory_type = entry["entry"]["type"]
                            session["types"][memory_type] += 1
                            
                    except Exception:
                        continue
        
        # Calculate durations and format results
        results = []
        for session_id, data in sessions.items():
            if data["start_time"] and data["end_time"]:
                data["duration_seconds"] = round(data["end_time"] - data["start_time"], 2)
            
            # Convert defaultdicts to regular dicts
            data["types"] = dict(data["types"])
            data["actions"] = dict(data["actions"])
            
            # Format timestamps
            if data["start_time"]:
                data["start_time"] = datetime.fromtimestamp(data["start_time"]).isoformat()
            if data["end_time"]:
                data["end_time"] = datetime.fromtimestamp(data["end_time"]).isoformat()
            
            results.append({
                "session_id": session_id,
                **data
            })
        
        # Sort by start time
        results.sort(key=lambda x: x.get("start_time") or "")
        
        return results
    
    def export_to_csv(self, output_path: str):
        """Export vault entries to CSV"""
        entries = []
        
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "entry" in log_entry:
                            entry = log_entry["entry"]
                            flat_entry = {
                                "session_id": log_entry.get("session_id"),
                                "timestamp": log_entry.get("timestamp"),
                                "action": log_entry.get("action"),
                                "id": entry.get("id"),
                                "type": entry.get("type"),
                                "content": str(entry.get("content", ""))[:100],  # Truncate long content
                                "importance": entry.get("importance"),
                                "access_count": entry.get("access_count", 0)
                            }
                            entries.append(flat_entry)
                    except Exception:
                        continue
        
        # Write to CSV
        if entries:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                writer.writeheader()
                writer.writerows(entries)
            
            return len(entries)
        return 0
    
    def rebuild_hash_cache(self):
        """Rebuild the seen_hashes.json from live log"""
        seen_hashes = {}
        entries_processed = 0
        
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("action") == "store" and "entry" in log_entry:
                            entry = log_entry["entry"]
                            
                            # Recalculate hash
                            entry_dict = {
                                'type': entry.get('type'),
                                'content': str(entry.get('content', '')),
                                'metadata': entry.get('metadata', {})
                            }
                            entry_str = json.dumps(entry_dict, sort_keys=True)
                            entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
                            
                            # Store with timestamp
                            timestamp = log_entry.get("timestamp", 0)
                            seen_hashes[entry_hash] = timestamp
                            entries_processed += 1
                            
                    except Exception:
                        continue
        
        # Save rebuilt cache
        with open(self.seen_hashes_path, 'w') as f:
            json.dump(seen_hashes, f, indent=2)
        
        return {
            "entries_processed": entries_processed,
            "unique_hashes": len(seen_hashes),
            "cache_path": str(self.seen_hashes_path)
        }
    
    def check_consistency(self) -> Dict[str, List[str]]:
        """Check for consistency issues"""
        issues = {
            "duplicate_ids": [],
            "type_mismatches": [],
            "orphaned_blobs": [],
            "missing_files": [],
            "index_mismatches": []
        }
        
        # Track IDs and types
        id_occurrences = defaultdict(int)
        id_types = defaultdict(set)
        
        # Scan live log
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "entry" in log_entry:
                            entry = log_entry["entry"]
                            entry_id = entry.get("id")
                            entry_type = entry.get("type")
                            
                            if entry_id:
                                id_occurrences[entry_id] += 1
                                if entry_type:
                                    id_types[entry_id].add(entry_type)
                    except Exception:
                        continue
        
        # Find duplicates
        for entry_id, count in id_occurrences.items():
            if count > 1:
                issues["duplicate_ids"].append(f"{entry_id} (seen {count} times)")
        
        # Find type mismatches
        for entry_id, types in id_types.items():
            if len(types) > 1:
                issues["type_mismatches"].append(f"{entry_id}: {list(types)}")
        
        # Check for orphaned blobs
        blobs_dir = self.vault_path / "blobs"
        if blobs_dir.exists():
            for blob_file in blobs_dir.glob("*.pkl.gz"):
                blob_id = blob_file.stem.split('.')[0]
                if blob_id not in id_occurrences:
                    issues["orphaned_blobs"].append(str(blob_file))
        
        # Check index consistency
        main_index_file = self.index_dir / "main_index.json"
        if main_index_file.exists():
            with open(main_index_file, 'r') as f:
                main_index = json.load(f)
            
            for memory_id, file_path in main_index.items():
                full_path = self.vault_path / file_path
                if not full_path.exists():
                    issues["missing_files"].append(f"{memory_id} -> {file_path}")
                
                if memory_id not in id_occurrences:
                    issues["index_mismatches"].append(f"{memory_id} in index but not in log")
        
        return issues
    
    def generate_fingerprint(self) -> Dict[str, str]:
        """Generate SHA-256 fingerprint of vault snapshot"""
        fingerprint_data = {
            "timestamp": datetime.now().isoformat(),
            "snapshot_exists": self.snapshot_path.exists(),
            "snapshot_sha256": None,
            "live_log_sha256": None,
            "vault_size_bytes": 0,
            "files_hashed": []
        }
        
        # Hash snapshot file
        if self.snapshot_path.exists():
            sha256_hash = hashlib.sha256()
            with open(self.snapshot_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            fingerprint_data["snapshot_sha256"] = sha256_hash.hexdigest()
            fingerprint_data["files_hashed"].append(str(self.snapshot_path))
            fingerprint_data["vault_size_bytes"] += self.snapshot_path.stat().st_size
        
        # Hash live log
        if self.live_log_path.exists():
            sha256_hash = hashlib.sha256()
            with open(self.live_log_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            fingerprint_data["live_log_sha256"] = sha256_hash.hexdigest()
            fingerprint_data["files_hashed"].append(str(self.live_log_path))
            fingerprint_data["vault_size_bytes"] += self.live_log_path.stat().st_size
        
        # Combined fingerprint
        combined = f"{fingerprint_data['snapshot_sha256'] or 'none'}:{fingerprint_data['live_log_sha256'] or 'none'}"
        fingerprint_data["combined_sha256"] = hashlib.sha256(combined.encode()).hexdigest()
        
        # Save fingerprint
        fingerprint_path = self.logs_dir / f"vault_fingerprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fingerprint_path, 'w') as f:
            json.dump(fingerprint_data, f, indent=2)
        
        fingerprint_data["fingerprint_file"] = str(fingerprint_path)
        
        return fingerprint_data
    
    def create_bundle(self, output_path: str) -> Dict[str, Any]:
        """Create compressed archive bundle of vault data"""
        import tarfile
        import zstandard as zstd
        
        bundle_info = {
            "created": datetime.now().isoformat(),
            "files_included": [],
            "total_size_bytes": 0,
            "compressed_size_bytes": 0,
            "compression_ratio": 0,
            "bundle_sha256": None
        }
        
        # Create tar archive
        temp_tar = Path(output_path).with_suffix('.tar')
        
        with tarfile.open(temp_tar, 'w') as tar:
            # Add key files
            files_to_bundle = [
                (self.live_log_path, "vault_live.jsonl"),
                (self.snapshot_path, "vault_snapshot.json"),
                (self.seen_hashes_path, "seen_hashes.json")
            ]
            
            # Add index files
            if self.index_dir.exists():
                for index_file in self.index_dir.glob("*.json"):
                    files_to_bundle.append((index_file, f"index/{index_file.name}"))
            
            # Add session summaries
            for summary_file in self.logs_dir.glob("session_*_summary.json"):
                files_to_bundle.append((summary_file, f"sessions/{summary_file.name}"))
            
            # Bundle files
            for file_path, arc_name in files_to_bundle:
                if file_path.exists():
                    tar.add(file_path, arcname=arc_name)
                    bundle_info["files_included"].append(arc_name)
                    bundle_info["total_size_bytes"] += file_path.stat().st_size
            
            # Add inspector report
            report_data = {
                "summary": self.summary(),
                "consistency_check": self.check_consistency(),
                "fingerprint": self.generate_fingerprint()
            }
            report_json = json.dumps(report_data, indent=2).encode('utf-8')
            
            from io import BytesIO
            report_buffer = BytesIO(report_json)
            report_info = tarfile.TarInfo(name="vault_inspector_report.json")
            report_info.size = len(report_json)
            tar.addfile(report_info, report_buffer)
            bundle_info["files_included"].append("vault_inspector_report.json")
        
        # Compress with zstd
        cctx = zstd.ZstdCompressor(level=3)
        with open(temp_tar, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                cctx.copy_stream(f_in, f_out)
        
        # Get compressed size
        bundle_info["compressed_size_bytes"] = Path(output_path).stat().st_size
        bundle_info["compression_ratio"] = round(
            bundle_info["total_size_bytes"] / bundle_info["compressed_size_bytes"], 2
        ) if bundle_info["compressed_size_bytes"] > 0 else 0
        
        # Calculate bundle SHA256
        sha256_hash = hashlib.sha256()
        with open(output_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        bundle_info["bundle_sha256"] = sha256_hash.hexdigest()
        
        # Clean up temp tar
        temp_tar.unlink()
        
        # Save bundle info
        info_path = Path(output_path).with_suffix('.info.json')
        with open(info_path, 'w') as f:
            json.dump(bundle_info, f, indent=2)
        
        return bundle_info
    
    def compare_mesh(self, mesh_file: str) -> Dict[str, Any]:
        """Compare vault entries with ConceptMesh structure"""
        comparison = {
            "vault_entries_without_mesh": [],
            "mesh_nodes_without_vault": [],
            "conflicting_hashes": [],
            "invalid_concept_bindings": [],
            "statistics": {
                "total_vault_entries": 0,
                "total_mesh_nodes": 0,
                "entries_with_concepts": 0,
                "orphaned_concepts": 0,
                "hash_conflicts": 0
            }
        }
        
        # Load mesh data
        mesh_path = Path(mesh_file)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        with open(mesh_path, 'r', encoding='utf-8') as f:
            mesh_data = json.load(f)
        
        # Extract mesh concepts
        mesh_concepts = {}
        mesh_hashes = {}
        
        # Handle different mesh formats
        if isinstance(mesh_data, dict) and "concepts" in mesh_data:
            concepts = mesh_data["concepts"]
        elif isinstance(mesh_data, list):
            concepts = mesh_data
        else:
            concepts = []
        
        for concept in concepts:
            if isinstance(concept, dict):
                concept_id = concept.get("id")
                if concept_id:
                    mesh_concepts[concept_id] = concept
                    # Calculate concept hash
                    concept_str = json.dumps(concept, sort_keys=True)
                    concept_hash = hashlib.sha256(concept_str.encode()).hexdigest()
                    mesh_hashes[concept_id] = concept_hash
        
        comparison["statistics"]["total_mesh_nodes"] = len(mesh_concepts)
        
        # Track vault concepts
        vault_concepts = {}
        vault_concept_refs = set()
        
        # Analyze vault entries
        if self.live_log_path.exists():
            with open(self.live_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "entry" in log_entry:
                            entry = log_entry["entry"]
                            entry_id = entry.get("id")
                            comparison["statistics"]["total_vault_entries"] += 1
                            
                            # Check for concept references
                            metadata = entry.get("metadata", {})
                            concept_ids = metadata.get("concept_ids", [])
                            
                            if concept_ids:
                                comparison["statistics"]["entries_with_concepts"] += 1
                                vault_concept_refs.update(concept_ids)
                                
                                # Check each referenced concept
                                for concept_id in concept_ids:
                                    if concept_id not in mesh_concepts:
                                        comparison["vault_entries_without_mesh"].append({
                                            "entry_id": entry_id,
                                            "missing_concept_id": concept_id,
                                            "entry_type": entry.get("type")
                                        })
                            
                            # Check for invalid bindings
                            if metadata.get("concept_binding") and not metadata.get("concept_ids"):
                                comparison["invalid_concept_bindings"].append({
                                    "entry_id": entry_id,
                                    "issue": "concept_binding without concept_ids"
                                })
                    
                    except Exception:
                        continue
        
        # Find mesh nodes not referenced in vault
        for concept_id in mesh_concepts:
            if concept_id not in vault_concept_refs:
                comparison["mesh_nodes_without_vault"].append({
                    "concept_id": concept_id,
                    "concept_name": mesh_concepts[concept_id].get("name", "unknown")
                })
                comparison["statistics"]["orphaned_concepts"] += 1
        
        # Check for hash conflicts (if vault stores concept data)
        # This would require vault entries to store concept hashes
        
        return comparison
    
    def compare_snapshots(self, old_file: str, new_file: str) -> Dict[str, Any]:
        """Compare two vault snapshots or session logs"""
        delta = {
            "new_entries": [],
            "unchanged_entries": [],
            "modified_entries": [],
            "deleted_entries": [],
            "hash_summary": {
                "old_snapshot_hash": None,
                "new_snapshot_hash": None,
                "entries_added": 0,
                "entries_modified": 0,
                "entries_deleted": 0,
                "entries_unchanged": 0
            }
        }
        
        # Load old snapshot/log
        old_path = Path(old_file)
        if not old_path.exists():
            raise FileNotFoundError(f"Old file not found: {old_file}")
        
        # Load new snapshot/log
        new_path = Path(new_file)
        if not new_path.exists():
            raise FileNotFoundError(f"New file not found: {new_file}")
        
        # Parse files based on format
        old_entries = self._load_entries_from_file(old_path)
        new_entries = self._load_entries_from_file(new_path)
        
        # Calculate file hashes
        delta["hash_summary"]["old_snapshot_hash"] = self._calculate_file_hash(old_path)
        delta["hash_summary"]["new_snapshot_hash"] = self._calculate_file_hash(new_path)
        
        # Create entry maps by ID
        old_map = {entry["id"]: entry for entry in old_entries if "id" in entry}
        new_map = {entry["id"]: entry for entry in new_entries if "id" in entry}
        
        # Find new entries
        for entry_id, entry in new_map.items():
            if entry_id not in old_map:
                delta["new_entries"].append({
                    "id": entry_id,
                    "type": entry.get("type"),
                    "content_preview": str(entry.get("content", ""))[:50]
                })
                delta["hash_summary"]["entries_added"] += 1
        
        # Find deleted entries
        for entry_id, entry in old_map.items():
            if entry_id not in new_map:
                delta["deleted_entries"].append({
                    "id": entry_id,
                    "type": entry.get("type"),
                    "content_preview": str(entry.get("content", ""))[:50]
                })
                delta["hash_summary"]["entries_deleted"] += 1
        
        # Find modified and unchanged entries
        for entry_id in old_map:
            if entry_id in new_map:
                old_entry = old_map[entry_id]
                new_entry = new_map[entry_id]
                
                # Calculate entry hashes
                old_hash = self._calculate_entry_hash_simple(old_entry)
                new_hash = self._calculate_entry_hash_simple(new_entry)
                
                if old_hash == new_hash:
                    delta["unchanged_entries"].append(entry_id)
                    delta["hash_summary"]["entries_unchanged"] += 1
                else:
                    # Find what changed
                    changes = []
                    if old_entry.get("content") != new_entry.get("content"):
                        changes.append("content")
                    if old_entry.get("type") != new_entry.get("type"):
                        changes.append("type")
                    if old_entry.get("importance") != new_entry.get("importance"):
                        changes.append("importance")
                    if old_entry.get("metadata") != new_entry.get("metadata"):
                        changes.append("metadata")
                    
                    delta["modified_entries"].append({
                        "id": entry_id,
                        "type": new_entry.get("type"),
                        "changes": changes,
                        "old_hash": old_hash[:16],
                        "new_hash": new_hash[:16]
                    })
                    delta["hash_summary"]["entries_modified"] += 1
        
        return delta
    
    def _load_entries_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load entries from snapshot JSON or NDJSON log"""
        entries = []
        
        if file_path.suffix == ".json":
            # Load snapshot format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different snapshot formats
                if isinstance(data, dict):
                    if "memories" in data:
                        # Extract from memories sections
                        memories = data["memories"]
                        if isinstance(memories, dict):
                            for memory_type, memory_list in memories.items():
                                if isinstance(memory_list, list):
                                    entries.extend(memory_list)
                    elif "entries" in data:
                        entries = data["entries"]
                elif isinstance(data, list):
                    entries = data
        
        elif file_path.suffix == ".jsonl":
            # Load NDJSON log format
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "entry" in log_entry:
                            entries.append(log_entry["entry"])
                    except Exception:
                        continue
        
        return entries
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _calculate_entry_hash_simple(self, entry: Dict[str, Any]) -> str:
        """Calculate simple hash for entry comparison"""
        # Create normalized representation
        normalized = {
            "type": entry.get("type"),
            "content": str(entry.get("content", "")),
            "metadata": entry.get("metadata", {}),
            "importance": entry.get("importance", 1.0)
        }
        entry_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()
    
    def print_summary(self, summary_data: Dict[str, Any]):
        """Pretty print summary"""
        print("\n🧠 Vault Summary")
        print("─" * 40)
        print(f"• Entries:        {summary_data['entries']}")
        print(f"• Unique hashes:  {summary_data['unique_hashes']}")
        
        if summary_data['types']:
            print("• Types:")
            for memory_type, count in sorted(summary_data['types'].items()):
                print(f"   - {memory_type}: {count}")
        
        if summary_data['last_entry']:
            last_entry_time = datetime.fromtimestamp(summary_data['last_entry'])
            print(f"• Last entry:     {last_entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"• Corrupt lines:  {summary_data['corrupt_lines']}")
        print(f"• Sessions:       {summary_data['session_count']}")
        print(f"• Total size:     {summary_data['size_mb']} MB")
        
        if summary_data['actions']:
            print("• Actions:")
            for action, count in sorted(summary_data['actions'].items()):
                print(f"   - {action}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="🧪 VaultInspector - Analyze UnifiedMemoryVault data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--vault-path",
        default="data/memory_vault",
        help="Path to memory vault directory (default: data/memory_vault)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show vault summary"
    )
    
    parser.add_argument(
        "--per-session",
        action="store_true",
        help="Show per-session analysis"
    )
    
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export entries to CSV file"
    )
    
    parser.add_argument(
        "--rebuild-hash-cache",
        action="store_true",
        help="Rebuild seen_hashes.json from live log"
    )
    
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Check for consistency issues"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Generate SHA-256 fingerprint of vault snapshot"
    )
    
    parser.add_argument(
        "--bundle",
        metavar="FILE",
        help="Create compressed archive bundle (requires zstandard: pip install zstandard)"
    )
    
    parser.add_argument(
        "--compare-mesh",
        metavar="MESH_FILE",
        help="Compare vault entries with ConceptMesh structure"
    )
    
    parser.add_argument(
        "--delta",
        nargs=2,
        metavar=("OLD_FILE", "NEW_FILE"),
        help="Compare two vault snapshots or session logs"
    )
    
    args = parser.parse_args()
    
    # Require at least one action
    if not any([args.summary, args.per_session, args.export, 
                args.rebuild_hash_cache, args.check_consistency,
                args.fingerprint, args.bundle, args.compare_mesh, args.delta]):
        parser.print_help()
        sys.exit(1)
    
    inspector = VaultInspector(args.vault_path)
    
    if args.summary:
        summary_data = inspector.summary()
        if args.json:
            print(json.dumps(summary_data, indent=2))
        else:
            inspector.print_summary(summary_data)
    
    if args.per_session:
        sessions = inspector.per_session_analysis()
        if args.json:
            print(json.dumps(sessions, indent=2))
        else:
            print("\n📊 Per-Session Analysis")
            print("─" * 60)
            for session in sessions:
                print(f"\nSession: {session['session_id']}")
                print(f"  Entries: {session['entries']}")
                print(f"  Duration: {session['duration_seconds']}s")
                if session['types']:
                    print(f"  Types: {session['types']}")
                if session['actions']:
                    print(f"  Actions: {session['actions']}")
    
    if args.export:
        count = inspector.export_to_csv(args.export)
        print(f"✅ Exported {count} entries to {args.export}")
    
    if args.rebuild_hash_cache:
        result = inspector.rebuild_hash_cache()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✅ Hash cache rebuilt")
            print(f"  Entries processed: {result['entries_processed']}")
            print(f"  Unique hashes: {result['unique_hashes']}")
            print(f"  Cache saved to: {result['cache_path']}")
    
    if args.check_consistency:
        issues = inspector.check_consistency()
        if args.json:
            print(json.dumps(issues, indent=2))
        else:
            print("\n🔍 Consistency Check")
            print("─" * 40)
            
            has_issues = False
            for issue_type, issue_list in issues.items():
                if issue_list:
                    has_issues = True
                    print(f"\n❌ {issue_type.replace('_', ' ').title()}:")
                    for issue in issue_list[:10]:  # Show first 10
                        print(f"   - {issue}")
                    if len(issue_list) > 10:
                        print(f"   ... and {len(issue_list) - 10} more")
            
            if not has_issues:
                print("✅ No consistency issues found!")
    
    if args.fingerprint:
        fingerprint_data = inspector.generate_fingerprint()
        if args.json:
            print(json.dumps(fingerprint_data, indent=2))
        else:
            inspector.print_fingerprint(fingerprint_data)
    
    if args.bundle:
        try:
            result = inspector.create_bundle(args.bundle)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n📦 Bundle created: {args.bundle}")
                print(f"   Files included: {len(result['files_included'])}")
                print(f"   Original size: {result['total_size_bytes'] / 1024 / 1024:.2f} MB")
                print(f"   Compressed size: {result['compressed_size_bytes'] / 1024 / 1024:.2f} MB")
                print(f"   Compression ratio: {result['compression_ratio']}x")
                print(f"   Bundle SHA-256: {result['bundle_sha256'][:32]}...")
        except ImportError:
            print("❌ Bundle creation requires zstandard: pip install zstandard")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Bundle creation failed: {e}")
            sys.exit(1)
    
    if args.compare_mesh:
        try:
            comparison = inspector.compare_mesh(args.compare_mesh)
            if args.json:
                print(json.dumps(comparison, indent=2))
            else:
                inspector.print_mesh_comparison(comparison)
            
            # Exit with error code if inconsistencies found
            has_issues = (
                len(comparison["vault_entries_without_mesh"]) > 0 or
                len(comparison["invalid_concept_bindings"]) > 0
            )
            if has_issues:
                sys.exit(1)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Mesh comparison failed: {e}")
            sys.exit(1)
    
    if args.delta:
        try:
            old_file, new_file = args.delta
            delta = inspector.compare_snapshots(old_file, new_file)
            if args.json:
                print(json.dumps(delta, indent=2))
            else:
                inspector.print_delta(delta)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Delta comparison failed: {e}")
            sys.exit(1)
    
    def print_mesh_comparison(self, comparison: Dict[str, Any]):
        """Pretty print mesh comparison results"""
        print("\n🕸️ Vault ↔ Mesh Comparison")
        print("─" * 50)
        
        stats = comparison["statistics"]
        print(f"• Vault entries: {stats['total_vault_entries']}")
        print(f"• Mesh nodes: {stats['total_mesh_nodes']}")
        print(f"• Entries with concepts: {stats['entries_with_concepts']}")
        print(f"• Orphaned concepts: {stats['orphaned_concepts']}")
        
        # Show issues
        has_issues = False
        
        if comparison["vault_entries_without_mesh"]:
            has_issues = True
            print(f"\n❌ Vault entries referencing missing concepts: {len(comparison['vault_entries_without_mesh'])}")
            for issue in comparison["vault_entries_without_mesh"][:5]:
                print(f"   - Entry {issue['entry_id']} → missing concept {issue['missing_concept_id']}")
            if len(comparison["vault_entries_without_mesh"]) > 5:
                print(f"   ... and {len(comparison['vault_entries_without_mesh']) - 5} more")
        
        if comparison["mesh_nodes_without_vault"]:
            has_issues = True
            print(f"\n⚠️ Mesh nodes never referenced in vault: {len(comparison['mesh_nodes_without_vault'])}")
            for issue in comparison["mesh_nodes_without_vault"][:5]:
                print(f"   - Concept {issue['concept_id']}: {issue['concept_name']}")
            if len(comparison["mesh_nodes_without_vault"]) > 5:
                print(f"   ... and {len(comparison['mesh_nodes_without_vault']) - 5} more")
        
        if comparison["invalid_concept_bindings"]:
            has_issues = True
            print(f"\n❌ Invalid concept bindings: {len(comparison['invalid_concept_bindings'])}")
            for issue in comparison["invalid_concept_bindings"][:5]:
                print(f"   - Entry {issue['entry_id']}: {issue['issue']}")
        
        if not has_issues:
            print("\n✅ Vault and Mesh are consistent!")
    
    def print_delta(self, delta: Dict[str, Any]):
        """Pretty print snapshot delta"""
        print("\n📊 Snapshot Delta Analysis")
        print("─" * 50)
        
        summary = delta["hash_summary"]
        print(f"Old snapshot: {summary['old_snapshot_hash'][:16]}...")
        print(f"New snapshot: {summary['new_snapshot_hash'][:16]}...")
        print()
        
        # Summary stats
        print(f"🆕 New entries: {summary['entries_added']}")
        print(f"♻️ Unchanged entries: {summary['entries_unchanged']}")
        print(f"🧬 Modified entries: {summary['entries_modified']}")
        print(f"🗑️ Deleted entries: {summary['entries_deleted']}")
        
        # Show details
        if delta["new_entries"]:
            print(f"\n🆕 New Entries ({len(delta['new_entries'])}):") 
            for entry in delta["new_entries"][:5]:
                print(f"   - {entry['id']} ({entry['type']}): {entry['content_preview']}...")
            if len(delta["new_entries"]) > 5:
                print(f"   ... and {len(delta['new_entries']) - 5} more")
        
        if delta["modified_entries"]:
            print(f"\n🧬 Modified Entries ({len(delta['modified_entries'])}):") 
            for entry in delta["modified_entries"][:5]:
                changes_str = ", ".join(entry['changes'])
                print(f"   - {entry['id']} ({entry['type']}): changed {changes_str}")
                print(f"     Hash: {entry['old_hash']} → {entry['new_hash']}")
            if len(delta["modified_entries"]) > 5:
                print(f"   ... and {len(delta['modified_entries']) - 5} more")
        
        if delta["deleted_entries"]:
            print(f"\n🗑️ Deleted Entries ({len(delta['deleted_entries'])}):") 
            for entry in delta["deleted_entries"][:5]:
                print(f"   - {entry['id']} ({entry['type']}): {entry['content_preview']}...")
            if len(delta["deleted_entries"]) > 5:
                print(f"   ... and {len(delta['deleted_entries']) - 5} more")

if __name__ == "__main__":
    main()
