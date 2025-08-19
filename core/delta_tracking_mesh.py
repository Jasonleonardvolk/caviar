# core/delta_tracking_mesh.py - Delta tracking concept mesh
import logging
from typing import Dict, Any, List, Optional
import time
import json

logger = logging.getLogger(__name__)

class DeltaTrackingConceptMesh:
    """Concept mesh with delta tracking for versioning and provenance"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.current_delta = {
            "nodes_added": [],
            "nodes_removed": [],
            "edges_added": [],
            "edges_removed": [],
            "timestamp": time.time()
        }
        logger.info("Delta Tracking Concept Mesh initialized")
    
    def clear_delta(self):
        """Clear the current delta to start fresh"""
        self.current_delta = {
            "nodes_added": [],
            "nodes_removed": [],
            "edges_added": [],
            "edges_removed": [],
            "timestamp": time.time()
        }
    
    def add_node(self, node_id: str, **attributes):
        """Add a node to the mesh with attributes"""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "created_at": time.time(),
                **attributes
            }
            self.current_delta["nodes_added"].append(node_id)
            logger.debug(f"Added node: {node_id}")
    
    def add_edge(self, source: str, target: str, edge_type: str = "related", **attributes):
        """Add an edge between nodes"""
        edge_id = f"{source}->{target}"
        if edge_id not in self.edges:
            self.edges[edge_id] = {
                "source": source,
                "target": target,
                "type": edge_type,
                "created_at": time.time(),
                **attributes
            }
            self.current_delta["edges_added"].append(edge_id)
            logger.debug(f"Added edge: {edge_id}")
    
    def get_last_delta(self) -> Dict[str, Any]:
        """Get the last delta summary"""
        return {
            "nodes_added": len(self.current_delta["nodes_added"]),
            "nodes_removed": len(self.current_delta["nodes_removed"]),
            "edges_added": len(self.current_delta["edges_added"]),
            "edges_removed": len(self.current_delta["edges_removed"]),
            "timestamp": self.current_delta["timestamp"]
        }
    
    def get_node_count(self) -> int:
        """Get total number of nodes"""
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges"""
        return len(self.edges)
    
    def export_delta(self) -> str:
        """Export current delta as JSON"""
        return json.dumps(self.current_delta, indent=2)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edges_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all edges connected to a node"""
        connected_edges = []
        for edge_id, edge in self.edges.items():
            if edge["source"] == node_id or edge["target"] == node_id:
                connected_edges.append(edge)
        return connected_edges
