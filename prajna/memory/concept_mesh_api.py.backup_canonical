"""
concept_mesh_api.py — Internal Mesh Mutators (LOCKDOWN)
-------------------------------------------------------
All mesh mutation methods are internal/private and may only be called by Prajna API controller.
Do not call these from anywhere else. All direct mesh writes outside this file must be deleted/refactored.
"""

import logging
import networkx as nx
import inspect
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("prajna.memory.concept_mesh")

def internal_only(func):
    """Decorator: ensures only Prajna API may call mesh mutators."""
    def wrapper(self, *args, **kwargs):
        caller = inspect.stack()[1]
        if 'prajna_api' not in caller.filename:
            raise PermissionError(f"Mesh mutators may only be called from prajna_api.py. Called from: {caller.filename}")
        return func(self, *args, **kwargs)
    return wrapper

class ConceptMeshAPI:
    """LOCKED-DOWN Concept Mesh with only internal mutation access."""

    def __init__(self, *args, **kwargs):
        self.mesh = nx.Graph()
        self.node_registry = {}
        self.edge_registry = {}
        self.mutation_log = []
        
        # Load existing mesh data if available
        self._load_existing_mesh()
        
        logger.info("🔒 Concept Mesh LOCKDOWN initialized.")

    def _load_existing_mesh(self):
        """Load existing mesh data from concept_mesh_data.json if available."""
        try:
            mesh_file = Path("concept_mesh_data.json")
            if mesh_file.exists():
                with open(mesh_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load nodes
                for node_data in data.get('nodes', []):
                    concept = node_data.get('concept', '')
                    if concept:
                        node_id = f"migrated_{concept}_{len(self.mesh.nodes)}"
                        self.mesh.add_node(node_id, **node_data)
                        self.node_registry[node_id] = node_data
                
                logger.info(f"📂 Loaded {len(self.mesh.nodes)} existing concepts into locked mesh")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing mesh data: {e}")

    def _save_mesh_backup(self):
        """Save mesh state for backup (optional)."""
        try:
            backup_data = {
                "nodes": [dict(data, node_id=node_id) for node_id, data in self.mesh.nodes(data=True)],
                "edges": [{"source": u, "target": v, **data} for u, v, data in self.mesh.edges(data=True)],
                "timestamp": datetime.utcnow().isoformat(),
                "total_nodes": len(self.mesh.nodes),
                "total_edges": len(self.mesh.edges)
            }
            
            backup_file = Path("concept_mesh_backup.json")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save mesh backup: {e}")

    @internal_only
    async def _add_node_locked(self, concept: str, context: str, provenance: dict):
        """
        Add node (concept) to mesh. Only callable by Prajna API.
        """
        try:
            # Generate unique node ID
            node_id = f"{concept}_{len(self.mesh.nodes)}_{int(datetime.now().timestamp())}"
            
            # Add provenance metadata
            full_provenance = {
                **provenance,
                "added_by": "prajna_api",
                "added_at": datetime.utcnow().isoformat(),
                "lockdown_enforced": True
            }
            
            # Add to mesh
            self.mesh.add_node(node_id, 
                concept=concept, 
                context=context, 
                provenance=full_provenance
            )
            
            # Update registry
            self.node_registry[node_id] = {
                "concept": concept, 
                "context": context, 
                "provenance": full_provenance
            }
            
            # Log mutation
            mutation_entry = {
                "action": "add_node",
                "node_id": node_id,
                "concept": concept,
                "timestamp": datetime.utcnow().isoformat(),
                "caller": "prajna_api"
            }
            self.mutation_log.append(mutation_entry)
            
            logger.info(f"🔒 [LOCKDOWN] Mesh node added: {node_id} (concept: {concept})")
            
            # Optional: Save backup
            self._save_mesh_backup()
            
            return {
                "node_id": node_id,
                "concept": concept,
                "total_nodes": len(self.mesh.nodes),
                "lockdown_enforced": True
            }
            
        except Exception as e:
            logger.error(f"❌ [LOCKDOWN] Failed to add node: {e}")
            raise

    @internal_only
    async def _add_edge_locked(self, source: str, target: str, relationship: str = "related"):
        """
        Add edge between concepts. Only callable by Prajna API.
        """
        try:
            if source in self.mesh and target in self.mesh:
                edge_data = {
                    "relationship": relationship,
                    "created_at": datetime.utcnow().isoformat(),
                    "created_by": "prajna_api",
                    "lockdown_enforced": True
                }
                
                self.mesh.add_edge(source, target, **edge_data)
                
                # Update registry
                edge_key = f"{source}--{target}"
                self.edge_registry[edge_key] = edge_data
                
                # Log mutation
                mutation_entry = {
                    "action": "add_edge",
                    "source": source,
                    "target": target,
                    "relationship": relationship,
                    "timestamp": datetime.utcnow().isoformat(),
                    "caller": "prajna_api"
                }
                self.mutation_log.append(mutation_entry)
                
                logger.info(f"🔒 [LOCKDOWN] Mesh edge added: {source} —[{relationship}]— {target}")
                
                return {
                    "edge": (source, target),
                    "relationship": relationship,
                    "total_edges": len(self.mesh.edges),
                    "lockdown_enforced": True
                }
            else:
                raise ValueError(f"Source or target node not found: {source}, {target}")
                
        except Exception as e:
            logger.error(f"❌ [LOCKDOWN] Failed to add edge: {e}")
            raise

    # READ-ONLY methods (safe for external use)
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Read-only node access - safe for external use."""
        return dict(self.mesh.nodes.get(node_id, {}))
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Read-only neighbor access - safe for external use."""
        if node_id in self.mesh:
            return list(self.mesh.neighbors(node_id))
        return []
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict]:
        """Read-only concept search - safe for external use."""
        results = []
        query_lower = query.lower()
        
        for node_id, data in self.mesh.nodes(data=True):
            concept = data.get('concept', '')
            context = data.get('context', '')
            
            if (query_lower in concept.lower() or 
                query_lower in context.lower()):
                results.append({
                    "node_id": node_id,
                    "concept": concept,
                    "context": context,
                    "relevance_score": 1.0  # Simple scoring
                })
                
            if len(results) >= limit:
                break
                
        return results
    
    def get_mesh_stats(self) -> Dict[str, Any]:
        """Read-only mesh statistics - safe for external use."""
        return {
            "total_nodes": len(self.mesh.nodes),
            "total_edges": len(self.mesh.edges),
            "mutation_count": len(self.mutation_log),
            "lockdown_active": True,
            "last_mutation": self.mutation_log[-1] if self.mutation_log else None
        }
    
    def get_recent_mutations(self, limit: int = 10) -> List[Dict]:
        """Read-only mutation log - safe for external use."""
        return self.mutation_log[-limit:] if self.mutation_log else []

    # FORBIDDEN METHODS - These would normally be public but are now blocked
    def add_node(self, *args, **kwargs):
        """BLOCKED: Use /api/prajna/propose endpoint instead."""
        raise PermissionError("Direct mesh writes forbidden. Use /api/prajna/propose endpoint.")
    
    def add_edge(self, *args, **kwargs):
        """BLOCKED: Use /api/prajna/propose endpoint instead."""
        raise PermissionError("Direct mesh writes forbidden. Use /api/prajna/propose endpoint.")
        
    def update_node(self, *args, **kwargs):
        """BLOCKED: Use /api/prajna/propose endpoint instead."""
        raise PermissionError("Direct mesh writes forbidden. Use /api/prajna/propose endpoint.")
        
    def delete_node(self, *args, **kwargs):
        """BLOCKED: Use /api/prajna/propose endpoint instead."""
        raise PermissionError("Direct mesh writes forbidden. Use /api/prajna/propose endpoint.")
