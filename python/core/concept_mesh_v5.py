#!/usr/bin/env python3
"""
Concept Mesh Module - Phase 5
==============================
Exports mesh summary for use in prompt/context at inference time.
Handles multi-user mesh contexts with versioning and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import hashlib
import shutil

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MESH_DIR = "data/mesh_contexts"
MESH_BACKUP_DIR = "data/mesh_contexts/backups"

# ============================================================================
# MESH EXPORT AND MANAGEMENT
# ============================================================================

class MeshManager:
    """Manages concept mesh contexts for multiple users."""
    
    def __init__(self, mesh_dir: str = DEFAULT_MESH_DIR):
        self.mesh_dir = Path(mesh_dir)
        self.backup_dir = Path(MESH_BACKUP_DIR)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def export_summary(self, 
                      user_id: str,
                      mesh: Dict[str, Any],
                      validate: bool = True) -> str:
        """
        Export mesh summary JSON for a user.
        
        Args:
            user_id: User identifier
            mesh: Mesh data dictionary
            validate: Whether to validate mesh structure
            
        Returns:
            Path to exported mesh file
        """
        if validate and not self._validate_mesh(mesh):
            raise ValueError("Invalid mesh structure")
        
        # Add metadata
        mesh["metadata"] = {
            "user_id": user_id,
            "exported_at": datetime.now().isoformat(),
            "version": mesh.get("version", "1.0.0"),
            "node_count": len(mesh.get("nodes", [])),
            "edge_count": len(mesh.get("edges", [])),
            "sha256": self._calculate_mesh_hash(mesh)
        }
        
        # Backup existing mesh if it exists
        out_path = self.mesh_dir / f"user_{user_id}_mesh.json"
        if out_path.exists():
            self._backup_mesh(user_id, out_path)
        
        # Write new mesh
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mesh, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported mesh for user {user_id} to {out_path}")
        return str(out_path)
    
    def load_mesh(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load mesh context for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Mesh dictionary or None if not found
        """
        mesh_path = self.mesh_dir / f"user_{user_id}_mesh.json"
        
        if not mesh_path.exists():
            # Try alternative naming
            alt_path = self.mesh_dir / f"{user_id}_mesh.json"
            if alt_path.exists():
                mesh_path = alt_path
            else:
                return None
        
        try:
            with open(mesh_path, "r", encoding="utf-8") as f:
                mesh = json.load(f)
            
            # Verify integrity if hash present
            if "metadata" in mesh and "sha256" in mesh["metadata"]:
                expected_hash = mesh["metadata"]["sha256"]
                # Remove metadata for hash calculation
                mesh_copy = dict(mesh)
                del mesh_copy["metadata"]
                actual_hash = self._calculate_mesh_hash(mesh_copy)
                
                if actual_hash != expected_hash:
                    logger.warning(f"Mesh integrity check failed for user {user_id}")
            
            return mesh
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load mesh for user {user_id}: {e}")
            return None
    
    def _validate_mesh(self, mesh: Dict[str, Any]) -> bool:
        """
        Validate mesh structure.
        
        Args:
            mesh: Mesh dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["summary", "nodes", "edges"]
        
        for field in required_fields:
            if field not in mesh:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate nodes
        if not isinstance(mesh["nodes"], list):
            logger.error("Nodes must be a list")
            return False
        
        for node in mesh["nodes"]:
            if not isinstance(node, dict):
                logger.error("Each node must be a dictionary")
                return False
            if "id" not in node or "label" not in node:
                logger.error("Each node must have 'id' and 'label' fields")
                return False
        
        # Validate edges
        if not isinstance(mesh["edges"], list):
            logger.error("Edges must be a list")
            return False
        
        for edge in mesh["edges"]:
            if not isinstance(edge, dict):
                logger.error("Each edge must be a dictionary")
                return False
            if "source" not in edge or "target" not in edge:
                logger.error("Each edge must have 'source' and 'target' fields")
                return False
        
        return True
    
    def _calculate_mesh_hash(self, mesh: Dict[str, Any]) -> str:
        """
        Calculate SHA256 hash of mesh content.
        
        Args:
            mesh: Mesh dictionary
            
        Returns:
            SHA256 hash as hex string
        """
        # Serialize deterministically
        mesh_str = json.dumps(mesh, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(mesh_str.encode()).hexdigest()
    
    def _backup_mesh(self, user_id: str, mesh_path: Path):
        """
        Create backup of existing mesh.
        
        Args:
            user_id: User identifier
            mesh_path: Path to current mesh file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"user_{user_id}_mesh_backup_{timestamp}.json"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(mesh_path, backup_path)
        logger.info(f"Backed up mesh to {backup_path}")
    
    def get_mesh_summary(self, user_id: str) -> Optional[str]:
        """
        Get just the summary text from a user's mesh.
        
        Args:
            user_id: User identifier
            
        Returns:
            Summary string or None
        """
        mesh = self.load_mesh(user_id)
        return mesh.get("summary") if mesh else None
    
    def get_relevant_nodes(self, 
                          user_id: str,
                          query: str,
                          max_nodes: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant nodes for a query.
        
        Args:
            user_id: User identifier
            query: Query string
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of relevant nodes
        """
        mesh = self.load_mesh(user_id)
        if not mesh:
            return []
        
        nodes = mesh.get("nodes", [])
        
        # Simple relevance: nodes with confidence > threshold
        # In production, use semantic similarity
        relevant = []
        for node in nodes:
            if node.get("confidence", 0) > 0.7:
                relevant.append(node)
        
        # Sort by confidence and return top N
        relevant.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return relevant[:max_nodes]
    
    def update_mesh_node(self,
                        user_id: str,
                        node_id: str,
                        updates: Dict[str, Any]) -> bool:
        """
        Update a specific node in user's mesh.
        
        Args:
            user_id: User identifier
            node_id: Node identifier
            updates: Dictionary of updates
            
        Returns:
            Success flag
        """
        mesh = self.load_mesh(user_id)
        if not mesh:
            return False
        
        # Find and update node
        updated = False
        for node in mesh.get("nodes", []):
            if node.get("id") == node_id:
                node.update(updates)
                node["updated_at"] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            # Re-export mesh
            self.export_summary(user_id, mesh)
            logger.info(f"Updated node {node_id} for user {user_id}")
        
        return updated
    
    def merge_meshes(self,
                    user_id: str,
                    new_mesh: Dict[str, Any],
                    strategy: str = "union") -> Dict[str, Any]:
        """
        Merge new mesh with existing mesh.
        
        Args:
            user_id: User identifier
            new_mesh: New mesh to merge
            strategy: Merge strategy ("union", "replace", "append")
            
        Returns:
            Merged mesh
        """
        existing = self.load_mesh(user_id)
        
        if not existing or strategy == "replace":
            return new_mesh
        
        if strategy == "union":
            # Merge nodes and edges, avoiding duplicates
            existing_node_ids = {n["id"] for n in existing.get("nodes", [])}
            existing_edge_ids = {(e["source"], e["target"]) for e in existing.get("edges", [])}
            
            # Add new nodes
            for node in new_mesh.get("nodes", []):
                if node["id"] not in existing_node_ids:
                    existing["nodes"].append(node)
            
            # Add new edges
            for edge in new_mesh.get("edges", []):
                edge_id = (edge["source"], edge["target"])
                if edge_id not in existing_edge_ids:
                    existing["edges"].append(edge)
            
            # Update summary
            existing["summary"] = new_mesh.get("summary", existing.get("summary", ""))
            
        elif strategy == "append":
            # Simply append new nodes/edges
            existing["nodes"].extend(new_mesh.get("nodes", []))
            existing["edges"].extend(new_mesh.get("edges", []))
        
        return existing
    
    def get_mesh_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's mesh.
        
        Args:
            user_id: User identifier
            
        Returns:
            Statistics dictionary
        """
        mesh = self.load_mesh(user_id)
        if not mesh:
            return {"exists": False}
        
        nodes = mesh.get("nodes", [])
        edges = mesh.get("edges", [])
        
        # Calculate statistics
        stats = {
            "exists": True,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "average_confidence": sum(n.get("confidence", 0) for n in nodes) / max(len(nodes), 1),
            "domains": list(set(n.get("domain", "general") for n in nodes)),
            "last_updated": mesh.get("metadata", {}).get("exported_at"),
            "version": mesh.get("version", "unknown"),
            "size_bytes": len(json.dumps(mesh).encode())
        }
        
        # Node degree statistics
        node_degrees = {}
        for edge in edges:
            node_degrees[edge["source"]] = node_degrees.get(edge["source"], 0) + 1
            node_degrees[edge["target"]] = node_degrees.get(edge["target"], 0) + 1
        
        if node_degrees:
            stats["max_degree"] = max(node_degrees.values())
            stats["avg_degree"] = sum(node_degrees.values()) / len(node_degrees)
        else:
            stats["max_degree"] = 0
            stats["avg_degree"] = 0
        
        return stats
    
    def export_all_users(self) -> List[str]:
        """
        Export all user meshes (for backup).
        
        Returns:
            List of exported user IDs
        """
        exported = []
        
        for mesh_file in self.mesh_dir.glob("user_*_mesh.json"):
            # Extract user_id from filename
            user_id = mesh_file.stem.replace("user_", "").replace("_mesh", "")
            mesh = self.load_mesh(user_id)
            
            if mesh:
                # Re-export with updated metadata
                self.export_summary(user_id, mesh)
                exported.append(user_id)
        
        logger.info(f"Exported {len(exported)} user meshes")
        return exported

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def export_summary(user_id: str, 
                  mesh: Dict[str, Any],
                  mesh_dir: str = DEFAULT_MESH_DIR) -> str:
    """
    Convenience function to export mesh summary.
    
    Args:
        user_id: User identifier
        mesh: Mesh dictionary
        mesh_dir: Directory for mesh contexts
        
    Returns:
        Path to exported file
    """
    manager = MeshManager(mesh_dir)
    return manager.export_summary(user_id, mesh)

def load_mesh_context(user_id: str,
                     mesh_dir: str = DEFAULT_MESH_DIR) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load mesh context.
    
    Args:
        user_id: User identifier
        mesh_dir: Directory for mesh contexts
        
    Returns:
        Mesh dictionary or None
    """
    manager = MeshManager(mesh_dir)
    return manager.load_mesh(user_id)

def create_empty_mesh(user_id: str, 
                     summary: str = "",
                     domains: List[str] = None) -> Dict[str, Any]:
    """
    Create an empty mesh structure.
    
    Args:
        user_id: User identifier
        summary: Initial summary
        domains: List of domains
        
    Returns:
        Empty mesh dictionary
    """
    return {
        "user_id": user_id,
        "summary": summary or f"Knowledge graph for user {user_id}",
        "nodes": [],
        "edges": [],
        "domains": domains or ["general"],
        "version": "1.0.0",
        "created_at": datetime.now().isoformat()
    }

def create_sample_mesh(user_id: str) -> Dict[str, Any]:
    """
    Create a sample mesh for testing.
    
    Args:
        user_id: User identifier
        
    Returns:
        Sample mesh dictionary
    """
    return {
        "user_id": user_id,
        "summary": f"User {user_id} is interested in quantum computing, kagome lattices, and soliton memory systems.",
        "nodes": [
            {
                "id": "node_1",
                "label": "Kagome Lattice",
                "domain": "physics",
                "confidence": 0.95,
                "description": "Geometric pattern in condensed matter physics"
            },
            {
                "id": "node_2",
                "label": "Soliton Memory",
                "domain": "computing",
                "confidence": 0.88,
                "description": "Memory system using soliton waves"
            },
            {
                "id": "node_3",
                "label": "Quantum Computing",
                "domain": "computing",
                "confidence": 0.92,
                "description": "Computation using quantum mechanical phenomena"
            }
        ],
        "edges": [
            {
                "source": "node_1",
                "target": "node_2",
                "relationship": "enables",
                "weight": 0.8
            },
            {
                "source": "node_2",
                "target": "node_3",
                "relationship": "implements",
                "weight": 0.7
            }
        ],
        "domains": ["physics", "computing"],
        "version": "1.0.0",
        "created_at": datetime.now().isoformat()
    }

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for mesh management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mesh Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export mesh")
    export_parser.add_argument("--user_id", required=True, help="User ID")
    export_parser.add_argument("--sample", action="store_true", help="Use sample mesh")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load mesh")
    load_parser.add_argument("--user_id", required=True, help="User ID")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get mesh statistics")
    stats_parser.add_argument("--user_id", required=True, help="User ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all user meshes")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = MeshManager()
    
    if args.command == "export":
        if args.sample:
            mesh = create_sample_mesh(args.user_id)
        else:
            mesh = create_empty_mesh(args.user_id)
        
        path = manager.export_summary(args.user_id, mesh)
        print(f"Exported mesh to: {path}")
    
    elif args.command == "load":
        mesh = manager.load_mesh(args.user_id)
        if mesh:
            print(f"Loaded mesh for user '{args.user_id}':")
            print(f"  Summary: {mesh.get('summary', 'N/A')}")
            print(f"  Nodes: {len(mesh.get('nodes', []))}")
            print(f"  Edges: {len(mesh.get('edges', []))}")
        else:
            print(f"No mesh found for user '{args.user_id}'")
    
    elif args.command == "stats":
        stats = manager.get_mesh_statistics(args.user_id)
        if stats.get("exists"):
            print(f"Mesh statistics for user '{args.user_id}':")
            for key, value in stats.items():
                if key != "exists":
                    print(f"  {key}: {value}")
        else:
            print(f"No mesh found for user '{args.user_id}'")
    
    elif args.command == "list":
        mesh_files = list(Path(DEFAULT_MESH_DIR).glob("user_*_mesh.json"))
        print(f"Found {len(mesh_files)} user meshes:")
        for mesh_file in mesh_files:
            user_id = mesh_file.stem.replace("user_", "").replace("_mesh", "")
            stats = manager.get_mesh_statistics(user_id)
            print(f"  - {user_id}: {stats.get('node_count', 0)} nodes, {stats.get('edge_count', 0)} edges")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
