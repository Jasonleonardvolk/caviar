#!/usr/bin/env python3
"""
Reasoning Traversal System for Prajna
Implements true graph traversal with causal chain reconstruction and inline attribution
"""

from typing import List, Optional, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ========== Core Data Structures ==========

class EdgeType(Enum):
    """Types of relationships between concepts"""
    IMPLIES = "implies"
    SUPPORTS = "supports"
    BECAUSE = "because"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    PREVENTS = "prevents"

@dataclass
class ConceptNode:
    """Enhanced concept node with metadata and sources"""
    id: str
    name: str
    description: str
    sources: List[str] = field(default_factory=list)
    edges: List["ConceptEdge"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance: float = 1.0
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class ConceptEdge:
    """Edge representing relationship between concepts"""
    target: str
    relation: EdgeType
    weight: float = 1.0
    justification: Optional[str] = None
    sources: List[str] = field(default_factory=list)

@dataclass
class ReasoningPath:
    """Complete reasoning path with scoring and justifications"""
    chain: List[ConceptNode]
    edge_justifications: List[str]
    score: float = 0.0
    path_type: str = "inference"  # inference, support, causal, etc.
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chain": [{"id": n.id, "name": n.name} for n in self.chain],
            "justifications": self.edge_justifications,
            "score": self.score,
            "type": self.path_type,
            "confidence": self.confidence
        }

# ========== Enhanced Concept Mesh with Traversal ==========

class ConceptMesh:
    """Enhanced concept mesh with reasoning traversal capabilities"""
    
    def __init__(self, nodes: Optional[Dict[str, ConceptNode]] = None):
        self.nodes: Dict[str, ConceptNode] = nodes or {}
        self._index_edges()
    
    def _index_edges(self):
        """Build reverse edge index for efficient traversal"""
        self.reverse_edges: Dict[str, List[Tuple[str, ConceptEdge]]] = {}
        for node_id, node in self.nodes.items():
            for edge in node.edges:
                if edge.target not in self.reverse_edges:
                    self.reverse_edges[edge.target] = []
                self.reverse_edges[edge.target].append((node_id, edge))
    
    def add_node(self, node: ConceptNode):
        """Add a node to the mesh"""
        self.nodes[node.id] = node
        self._index_edges()
    
    def add_edge(self, from_id: str, to_id: str, relation: EdgeType, 
                 weight: float = 1.0, justification: Optional[str] = None):
        """Add an edge between nodes"""
        if from_id in self.nodes:
            edge = ConceptEdge(
                target=to_id,
                relation=relation,
                weight=weight,
                justification=justification
            )
            self.nodes[from_id].edges.append(edge)
            self._index_edges()
    
    def traverse(self, anchor_id: str, max_depth: int = 3, 
                 min_score: float = 0.1) -> List[ReasoningPath]:
        """
        Traverse mesh from anchor using multiple strategies
        Returns list of reasoning paths sorted by score
        """
        paths = []
        
        # Strategy 1: Forward chaining (follow implications)
        forward_paths = self._forward_traverse(anchor_id, max_depth, min_score)
        paths.extend(forward_paths)
        
        # Strategy 2: Support gathering (find supporting evidence)
        support_paths = self._support_traverse(anchor_id, max_depth, min_score)
        paths.extend(support_paths)
        
        # Strategy 3: Causal chains (follow cause-effect)
        causal_paths = self._causal_traverse(anchor_id, max_depth, min_score)
        paths.extend(causal_paths)
        
        # Sort by score and return unique paths
        paths = self._deduplicate_paths(paths)
        paths.sort(key=lambda p: p.score, reverse=True)
        
        return paths
    
    def _forward_traverse(self, anchor_id: str, max_depth: int, 
                         min_score: float) -> List[ReasoningPath]:
        """Forward chaining traversal (implications and enables)"""
        paths = []
        
        def dfs(node_id: str, depth: int, path: List[ConceptNode], 
                justifications: List[str], visited: Set[str], score: float):
            
            if depth > max_depth or node_id in visited or score < min_score:
                return
            
            node = self.nodes.get(node_id)
            if not node:
                return
            
            visited.add(node_id)
            path.append(node)
            
            # Create path if we have a chain
            if len(path) > 1:
                reasoning_path = ReasoningPath(
                    chain=path.copy(),
                    edge_justifications=justifications.copy(),
                    score=score,
                    path_type="inference",
                    confidence=score / len(path)
                )
                paths.append(reasoning_path)
            
            # Follow forward edges
            for edge in node.edges:
                if edge.relation in [EdgeType.IMPLIES, EdgeType.ENABLES, EdgeType.CAUSES]:
                    new_score = score * edge.weight * 0.9  # Decay factor
                    edge_just = edge.justification or f"{edge.relation.value}"
                    
                    dfs(edge.target, depth + 1, path, 
                        justifications + [edge_just], 
                        visited.copy(), new_score)
            
            path.pop()
        
        # Start traversal
        dfs(anchor_id, 0, [], [], set(), 1.0)
        return paths
    
    def _support_traverse(self, anchor_id: str, max_depth: int, 
                         min_score: float) -> List[ReasoningPath]:
        """Find supporting evidence paths"""
        paths = []
        
        # Find nodes that support the anchor
        supporting_nodes = []
        if anchor_id in self.reverse_edges:
            for source_id, edge in self.reverse_edges[anchor_id]:
                if edge.relation == EdgeType.SUPPORTS:
                    supporting_nodes.append((source_id, edge))
        
        # Build paths from each supporter
        for source_id, edge in supporting_nodes:
            source_node = self.nodes.get(source_id)
            if source_node:
                path = ReasoningPath(
                    chain=[source_node, self.nodes[anchor_id]],
                    edge_justifications=[edge.justification or "supports"],
                    score=edge.weight,
                    path_type="support",
                    confidence=edge.weight
                )
                paths.append(path)
        
        return paths
    
    def _causal_traverse(self, anchor_id: str, max_depth: int, 
                        min_score: float) -> List[ReasoningPath]:
        """Trace causal chains backward and forward"""
        paths = []
        
        # Find causal antecedents
        if anchor_id in self.reverse_edges:
            for source_id, edge in self.reverse_edges[anchor_id]:
                if edge.relation in [EdgeType.CAUSES, EdgeType.BECAUSE]:
                    # Build causal chain backward
                    chain = self._build_causal_chain_backward(
                        source_id, anchor_id, max_depth - 1, set()
                    )
                    if chain:
                        paths.extend(chain)
        
        return paths
    
    def _build_causal_chain_backward(self, start_id: str, end_id: str, 
                                    remaining_depth: int, visited: Set[str]) -> List[ReasoningPath]:
        """Build causal chain from start to end"""
        if remaining_depth <= 0 or start_id in visited:
            return []
        
        visited.add(start_id)
        paths = []
        
        start_node = self.nodes.get(start_id)
        end_node = self.nodes.get(end_id)
        
        if not start_node or not end_node:
            return []
        
        # Direct path
        for edge in start_node.edges:
            if edge.target == end_id and edge.relation in [EdgeType.CAUSES, EdgeType.BECAUSE]:
                path = ReasoningPath(
                    chain=[start_node, end_node],
                    edge_justifications=[edge.justification or edge.relation.value],
                    score=edge.weight,
                    path_type="causal",
                    confidence=edge.weight
                )
                paths.append(path)
        
        # Recursive paths
        if start_id in self.reverse_edges:
            for prev_id, prev_edge in self.reverse_edges[start_id]:
                if prev_edge.relation in [EdgeType.CAUSES, EdgeType.BECAUSE]:
                    sub_paths = self._build_causal_chain_backward(
                        prev_id, start_id, remaining_depth - 1, visited.copy()
                    )
                    for sub_path in sub_paths:
                        # Extend the chain
                        extended_chain = sub_path.chain + [end_node]
                        extended_just = sub_path.edge_justifications + [
                            f"{start_node.name} causes {end_node.name}"
                        ]
                        extended_path = ReasoningPath(
                            chain=extended_chain,
                            edge_justifications=extended_just,
                            score=sub_path.score * 0.9,
                            path_type="causal",
                            confidence=sub_path.confidence * 0.9
                        )
                        paths.append(extended_path)
        
        return paths
    
    def _deduplicate_paths(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """Remove duplicate paths based on node sequence"""
        seen = set()
        unique_paths = []
        
        for path in paths:
            # Create hash of node sequence
            path_hash = tuple(node.id for node in path.chain)
            if path_hash not in seen:
                seen.add(path_hash)
                unique_paths.append(path)
        
        return unique_paths

# ========== Reasoning Engine ==========

class ReasoningEngine:
    """Orchestrates reasoning traversal and path selection"""
    
    def __init__(self, concept_mesh: ConceptMesh):
        self.mesh = concept_mesh
    
    def plan_causal_chain(self, query_concepts: List[str], 
                         max_paths: int = 5) -> List[ReasoningPath]:
        """Plan reasoning chains from query concepts"""
        all_paths = []
        
        # Find paths from each query concept
        for concept_id in query_concepts:
            if concept_id in self.mesh.nodes:
                paths = self.mesh.traverse(concept_id, max_depth=4)
                all_paths.extend(paths)
        
        # Combine and rank paths
        ranked_paths = self._rank_paths(all_paths)
        
        return ranked_paths[:max_paths]
    
    def _rank_paths(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """Rank paths by relevance, coherence, and confidence"""
        for path in paths:
            # Adjust score based on path characteristics
            length_penalty = 1.0 / (1.0 + len(path.chain) * 0.1)
            type_bonus = 1.2 if path.path_type == "causal" else 1.0
            
            path.score = path.score * length_penalty * type_bonus * path.confidence
        
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths
    
    def select_best_path(self, paths: List[ReasoningPath], 
                        context: Optional[Dict[str, Any]] = None) -> Optional[ReasoningPath]:
        """Select the best reasoning path given context"""
        if not paths:
            return None
        
        # Could incorporate context-specific scoring here
        return paths[0]

# ========== Enhanced Explanation Generator ==========

class ExplanationGenerator:
    """Generates natural language explanations with inline attribution"""
    
    def __init__(self, enable_inline_attribution: bool = True):
        self.enable_inline_attribution = enable_inline_attribution
    
    def explain_path(self, path: ReasoningPath, 
                    verbose: bool = False) -> str:
        """Convert reasoning path to natural language with attribution"""
        output = []
        
        for i, node in enumerate(path.chain):
            # Generate node explanation
            sentence = self._explain_node(node, verbose)
            
            # Add inline attribution
            if self.enable_inline_attribution and node.sources:
                sentence += f" [source: {', '.join(node.sources[:2])}]"
            
            output.append(sentence)
            
            # Add edge explanation
            if i < len(path.edge_justifications):
                edge_explanation = self._explain_edge(
                    path.edge_justifications[i], 
                    node, 
                    path.chain[i + 1] if i + 1 < len(path.chain) else None
                )
                output.append(edge_explanation)
        
        return "\n".join(output)
    
    def _explain_node(self, node: ConceptNode, verbose: bool) -> str:
        """Generate explanation for a single node"""
        if verbose:
            return f"{node.name}: {node.description}"
        else:
            # More natural phrasing
            return f"{node.description}"
    
    def _explain_edge(self, justification: str, 
                     from_node: ConceptNode, 
                     to_node: Optional[ConceptNode]) -> str:
        """Generate explanation for edge relationship"""
        if not to_node:
            return ""
        
        # Natural language connectors
        connectors = {
            "implies": "This implies that",
            "supports": "This is supported by the fact that",
            "because": "This is because",
            "enables": "This enables",
            "causes": "This causes",
            "related_to": "This relates to"
        }
        
        connector = connectors.get(justification, f"↳ {justification} →")
        return f" {connector}"
    
    def explain_multiple_paths(self, paths: List[ReasoningPath], 
                             max_paths: int = 3) -> str:
        """Explain multiple reasoning paths"""
        explanations = []
        
        for i, path in enumerate(paths[:max_paths]):
            explanation = f"\n**Reasoning Path {i+1}** (confidence: {path.confidence:.2f}):\n"
            explanation += self.explain_path(path)
            explanations.append(explanation)
        
        return "\n".join(explanations)

# ========== Enhanced Prajna Response ==========

@dataclass
class PrajnaResponsePlus:
    """Enhanced response with reasoning chains and attribution"""
    text: str
    reasoning_paths: List[ReasoningPath]
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "reasoning_paths": [p.to_dict() for p in self.reasoning_paths],
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    def to_graphviz(self) -> str:
        """Generate Graphviz representation of reasoning paths"""
        dot_lines = ["digraph reasoning {", "  rankdir=LR;"]
        
        # Add nodes
        seen_nodes = set()
        for path in self.reasoning_paths:
            for node in path.chain:
                if node.id not in seen_nodes:
                    seen_nodes.add(node.id)
                    label = node.name.replace('"', '\\"')
                    dot_lines.append(f'  "{node.id}" [label="{label}"];')
        
        # Add edges
        for path in self.reasoning_paths:
            for i in range(len(path.chain) - 1):
                from_id = path.chain[i].id
                to_id = path.chain[i + 1].id
                label = path.edge_justifications[i] if i < len(path.edge_justifications) else ""
                dot_lines.append(f'  "{from_id}" -> "{to_id}" [label="{label}"];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)

# ========== Integration with Prajna ==========

class PrajnaReasoningIntegration:
    """Integrates reasoning traversal with Prajna generation"""
    
    def __init__(self, concept_mesh: ConceptMesh, 
                 enable_inline_attribution: bool = True):
        self.mesh = concept_mesh
        self.reasoning_engine = ReasoningEngine(concept_mesh)
        self.explanation_generator = ExplanationGenerator(enable_inline_attribution)
    
    def generate_reasoned_response(self, query: str, 
                                 anchor_concepts: List[str],
                                 context: Optional[Dict[str, Any]] = None) -> PrajnaResponsePlus:
        """Generate response with full reasoning traversal"""
        
        # Step 1: Plan reasoning chains
        reasoning_paths = self.reasoning_engine.plan_causal_chain(
            anchor_concepts, 
            max_paths=5
        )
        
        # Step 2: Select best paths
        if not reasoning_paths:
            return PrajnaResponsePlus(
                text="No reasoning paths found for the given concepts.",
                reasoning_paths=[],
                sources=[],
                confidence=0.0
            )
        
        best_path = self.reasoning_engine.select_best_path(reasoning_paths, context)
        
        # Step 3: Generate explanation
        explanation = self.explanation_generator.explain_path(best_path, verbose=False)
        
        # Step 4: Collect all sources
        all_sources = []
        for node in best_path.chain:
            all_sources.extend(node.sources)
        
        # Step 5: Create enhanced response
        response = PrajnaResponsePlus(
            text=explanation,
            reasoning_paths=reasoning_paths[:3],  # Include top 3 paths
            sources=list(set(all_sources)),  # Unique sources
            confidence=best_path.confidence,
            metadata={
                "query": query,
                "anchor_concepts": anchor_concepts,
                "path_type": best_path.path_type,
                "traversal_depth": len(best_path.chain)
            }
        )
        
        return response

# ========== Example Usage and Testing ==========

def create_test_mesh() -> ConceptMesh:
    """Create a test concept mesh"""
    mesh = ConceptMesh()
    
    # Create nodes
    entropy = ConceptNode("entropy", "Entropy", 
                         "Measure of uncertainty in information theory", 
                         ["PDF_001", "arxiv_2023_045"])
    
    information = ConceptNode("information", "Information", 
                            "Encoded knowledge that reduces uncertainty", 
                            ["PDF_002", "textbook_ch3"])
    
    compression = ConceptNode("compression", "Compression", 
                            "Reduction of data size while preserving information", 
                            ["PDF_003", "wiki_compression"])
    
    redundancy = ConceptNode("redundancy", "Redundancy", 
                           "Repetitive or predictable patterns in data", 
                           ["PDF_004"])
    
    # Add nodes to mesh
    mesh.add_node(entropy)
    mesh.add_node(information)
    mesh.add_node(compression)
    mesh.add_node(redundancy)
    
    # Add edges
    mesh.add_edge("entropy", "information", EdgeType.IMPLIES, 
                 weight=0.9, justification="high entropy implies need for information")
    
    mesh.add_edge("information", "compression", EdgeType.ENABLES, 
                 weight=0.85, justification="information theory enables compression algorithms")
    
    mesh.add_edge("redundancy", "compression", EdgeType.SUPPORTS, 
                 weight=0.95, justification="redundancy in data supports better compression")
    
    mesh.add_edge("entropy", "redundancy", EdgeType.CONTRADICTS, 
                 weight=0.7, justification="high entropy means low redundancy")
    
    return mesh

def test_reasoning_traversal():
    """Test the reasoning traversal system"""
    print("🧪 Testing Reasoning Traversal System\n")
    print("=" * 60)
    
    # Create test mesh
    mesh = create_test_mesh()
    
    # Create integration
    prajna_reasoning = PrajnaReasoningIntegration(mesh)
    
    # Test query
    query = "How does entropy relate to data compression?"
    anchor_concepts = ["entropy"]
    
    print(f"Query: {query}")
    print(f"Anchor concepts: {anchor_concepts}")
    print("=" * 60)
    
    # Generate reasoned response
    response = prajna_reasoning.generate_reasoned_response(
        query, 
        anchor_concepts
    )
    
    # Display results
    print("\n📝 Generated Explanation:")
    print(response.text)
    
    print("\n🔗 Reasoning Paths Found:")
    for i, path in enumerate(response.reasoning_paths):
        print(f"\nPath {i+1} ({path.path_type}, score: {path.score:.3f}):")
        chain_str = " → ".join([n.name for n in path.chain])
        print(f"  {chain_str}")
    
    print("\n📚 Sources:")
    for source in response.sources:
        print(f"  - {source}")
    
    print("\n🎯 Confidence:", response.confidence)
    
    # Generate Graphviz
    print("\n📊 Graphviz Representation:")
    print(response.to_graphviz())

if __name__ == "__main__":
    test_reasoning_traversal()
