"""
Integration module to merge temporal reasoning with existing system
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from python.core.reasoning_traversal import (
    ConceptMesh as BaseMesh,
    ConceptNode as BaseNode,
    ConceptEdge as BaseEdge,
    EdgeType,
    ReasoningPath
)

# Import the new temporal components
try:
    from reasoning_traversal import (
        ConceptEdge as TemporalEdge,
        ConceptMesh as TemporalMesh,
        evaluate_temporal_drift
    )
except ImportError:
    # If reasoning_traversal is not available, use base components
    from python.core.reasoning_traversal import (
        ConceptEdge as TemporalEdge,
        ConceptMesh as TemporalMesh
    )
    
    # Provide a default implementation of evaluate_temporal_drift
    def evaluate_temporal_drift(concept_drift, semantic_shift):
        """Default implementation when temporal module not available"""
        return {
            'concept_drift': concept_drift,
            'semantic_shift': semantic_shift,
            'temporal_confidence': 0.5
        }

class EnhancedConceptEdge(BaseEdge):
    """Enhanced edge with temporal support"""
    def __init__(self, target: str, relation: EdgeType, weight: float = 1.0, 
                 justification: Optional[str] = None, sources: List[str] = None,
                 timestamp: Optional[str] = None):
        super().__init__(target, relation, weight, justification, sources or [])
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

class TemporalConceptMesh(BaseMesh):
    """Enhanced ConceptMesh with temporal reasoning"""
    
    def traverse_temporal(self, anchor_id: str, max_depth: int = 3, 
                         after: Optional[str] = None) -> List[ReasoningPath]:
        """Traverse with temporal awareness"""
        paths = []
        now = datetime.now(timezone.utc)
        after_dt = datetime.fromisoformat(after) if after else None
        
        def time_decay(edge: EnhancedConceptEdge) -> float:
            if not hasattr(edge, 'timestamp'):
                return 0.5  # Default for edges without timestamps
            
            try:
                edge_time = datetime.fromisoformat(edge.timestamp)
                if after_dt and edge_time < after_dt:
                    return 0.0  # Filter out
                
                delta = (now - edge_time).days
                return max(0.0, 1.0 - delta / 365)  # 1-year decay
            except:
                return 0.5
        
        # Use parent traverse but apply temporal scoring
        all_paths = self.traverse(anchor_id, max_depth)
        
        # Re-score paths based on temporal factors
        for path in all_paths:
            temporal_score = 1.0
            
            # Score based on edge timestamps
            for i, node in enumerate(path.chain[:-1]):
                for edge in node.edges:
                    if i < len(path.chain) - 1 and edge.target == path.chain[i+1].id:
                        temporal_score *= time_decay(edge)
            
            path.score *= temporal_score
        
        # Filter and sort
        temporal_paths = [p for p in all_paths if p.score > 0.01]
        temporal_paths.sort(key=lambda p: p.score, reverse=True)
        
        return temporal_paths
    
    def add_temporal_edge(self, from_id: str, to_id: str, relation: EdgeType,
                         weight: float = 1.0, justification: Optional[str] = None,
                         timestamp: Optional[str] = None):
        """Add edge with timestamp"""
        if from_id in self.nodes:
            edge = EnhancedConceptEdge(
                target=to_id,
                relation=relation,
                weight=weight,
                justification=justification,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat()
            )
            self.nodes[from_id].edges.append(edge)
            self._index_edges()

class TrainingDataExporter:
    """Export reasoning paths for model training"""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_path(self, path: ReasoningPath, query: str, 
                   additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert reasoning path to training sample"""
        sample = {
            "query": query,
            "path_type": path.path_type,
            "chain": [
                {
                    "id": node.id,
                    "name": node.name,
                    "text": node.description,
                    "sources": node.sources
                }
                for node in path.chain
            ],
            "justifications": path.edge_justifications,
            "confidence": path.confidence,
            "score": path.score,
            "narration": self._generate_narration(path),
            "source_ids": list({src for node in path.chain for src in node.sources}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if additional_metadata:
            sample["metadata"] = additional_metadata
        
        return sample
    
    def _generate_narration(self, path: ReasoningPath) -> str:
        """Generate natural language narration of reasoning"""
        narration = []
        
        for i, node in enumerate(path.chain):
            # Add node description
            narration.append(node.description)
            
            # Add edge explanation
            if i < len(path.edge_justifications):
                justification = path.edge_justifications[i]
                if i + 1 < len(path.chain):
                    next_node = path.chain[i + 1]
                    narration.append(f"This {justification} that {next_node.description.lower()}")
        
        return " ".join(narration)
    
    def save_training_batch(self, samples: List[Dict[str, Any]], 
                           filename: str = "reasoning_paths.jsonl"):
        """Save batch of training samples to JSONL"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        return output_path

class TemporalReasoningAnalyzer:
    """Analyze temporal aspects of reasoning"""
    
    def analyze_knowledge_drift(self, mesh: TemporalConceptMesh, 
                              window_days: int = 30) -> Dict[str, Any]:
        """Analyze how knowledge has drifted over time"""
        now = datetime.now(timezone.utc)
        
        # Collect all timestamps
        edge_timestamps = []
        node_timestamps = []
        
        for node in mesh.nodes.values():
            # Node creation times (from sources)
            for source in node.sources:
                if "T" in source:  # ISO timestamp
                    try:
                        dt = datetime.fromisoformat(source)
                        node_timestamps.append((node.id, dt))
                    except:
                        pass
            
            # Edge creation times
            for edge in node.edges:
                if hasattr(edge, 'timestamp'):
                    try:
                        dt = datetime.fromisoformat(edge.timestamp)
                        edge_timestamps.append((f"{node.id}->{edge.target}", dt))
                    except:
                        pass
        
        # Calculate drift metrics
        recent_nodes = sum(1 for _, dt in node_timestamps 
                          if (now - dt).days <= window_days)
        recent_edges = sum(1 for _, dt in edge_timestamps 
                          if (now - dt).days <= window_days)
        
        # Find stale areas
        stale_nodes = [(nid, dt) for nid, dt in node_timestamps 
                      if (now - dt).days > 365]
        
        return {
            "total_nodes": len(mesh.nodes),
            "recent_nodes": recent_nodes,
            "recent_edges": recent_edges,
            "recent_ratio": recent_nodes / len(mesh.nodes) if mesh.nodes else 0,
            "stale_nodes": len(stale_nodes),
            "stale_node_ids": [nid for nid, _ in stale_nodes[:10]],
            "analysis_timestamp": now.isoformat(),
            "window_days": window_days
        }
    
    def find_temporal_inconsistencies(self, mesh: TemporalConceptMesh) -> List[Dict[str, Any]]:
        """Find edges that point to future nodes"""
        inconsistencies = []
        
        for node_id, node in mesh.nodes.items():
            for edge in node.edges:
                if hasattr(edge, 'timestamp') and edge.target in mesh.nodes:
                    target_node = mesh.nodes[edge.target]
                    
                    # Check if edge predates target node
                    try:
                        edge_time = datetime.fromisoformat(edge.timestamp)
                        for target_source in target_node.sources:
                            if "T" in target_source:
                                target_time = datetime.fromisoformat(target_source)
                                if edge_time < target_time:
                                    inconsistencies.append({
                                        "from": node_id,
                                        "to": edge.target,
                                        "edge_time": edge.timestamp,
                                        "target_time": target_source,
                                        "issue": "Edge predates target node creation"
                                    })
                    except:
                        pass
        
        return inconsistencies

# Example usage
def demonstrate_temporal_reasoning():
    """Show temporal reasoning in action"""
    
    # Create temporal mesh
    mesh = TemporalConceptMesh()
    
    # Add nodes
    entropy = BaseNode("entropy", "Entropy", 
                      "Measure of uncertainty in information theory",
                      ["Shannon1948", "PDF_2023-01-15T10:00:00Z"])
    
    information = BaseNode("information", "Information",
                          "Data that reduces uncertainty", 
                          ["PDF_2024-06-20T14:30:00Z"])
    
    compression = BaseNode("compression", "Compression",
                          "Process of encoding data efficiently",
                          ["PDF_2025-01-10T09:00:00Z"])
    
    mesh.add_node(entropy)
    mesh.add_node(information)
    mesh.add_node(compression)
    
    # Add temporal edges
    mesh.add_temporal_edge("entropy", "information", EdgeType.IMPLIES,
                          timestamp="2024-06-20T15:00:00Z",
                          justification="entropy theory implies information measurement")
    
    mesh.add_temporal_edge("information", "compression", EdgeType.ENABLES,
                          timestamp="2025-01-10T10:00:00Z",
                          justification="information theory enables compression algorithms")
    
    # Test temporal traversal
    print("🕐 Temporal Traversal Demo")
    print("=" * 50)
    
    # All paths
    all_paths = mesh.traverse_temporal("entropy")
    print(f"\nAll paths from entropy: {len(all_paths)}")
    
    # Recent paths only (after 2024)
    recent_paths = mesh.traverse_temporal("entropy", after="2024-01-01")
    print(f"Paths after 2024: {len(recent_paths)}")
    
    # Export for training
    exporter = TrainingDataExporter()
    training_samples = []
    
    for path in recent_paths[:3]:
        sample = exporter.export_path(
            path, 
            "How does entropy relate to compression?",
            {"session_id": "demo_001", "user": "test"}
        )
        training_samples.append(sample)
    
    # Save training data
    output_file = exporter.save_training_batch(training_samples)
    print(f"\n📦 Training data saved to: {output_file}")
    
    # Analyze drift
    analyzer = TemporalReasoningAnalyzer()
    drift_analysis = analyzer.analyze_knowledge_drift(mesh, window_days=180)
    
    print("\n📊 Knowledge Drift Analysis:")
    print(f"  Recent nodes (last 180 days): {drift_analysis['recent_nodes']}")
    print(f"  Recent ratio: {drift_analysis['recent_ratio']:.2%}")
    print(f"  Stale nodes: {drift_analysis['stale_nodes']}")
    
    # Check inconsistencies
    inconsistencies = analyzer.find_temporal_inconsistencies(mesh)
    if inconsistencies:
        print(f"\n⚠️ Found {len(inconsistencies)} temporal inconsistencies")

if __name__ == "__main__":
    demonstrate_temporal_reasoning()
