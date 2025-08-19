"""
Prajna Reasoning Engine: Multi-Hop Cognitive Pathfinding
========================================================

This module implements Prajna's core reasoning capabilities - the cognitive bridge
between retrieval and synthesis. It performs phase-aware traversal through the
concept mesh to find coherent explanatory paths between ideas.

Key Features:
- Phase-stable multi-hop reasoning
- Resonance-guided pathfinding  
- Semantic drift minimization
- Explanatory narrative generation
- Integration with existing Prajna pipeline

This is where Prajna truly learns to REASON, not just respond.
"""

import asyncio
import heapq
import logging
import time
import math
import re
from typing import Union, Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("prajna.reasoning")

class ReasoningMode(Enum):
    """Different modes of reasoning for different query types"""
    EXPLANATORY = "explanatory"      # Explain connections between concepts
    INFERENTIAL = "inferential"      # Derive new conclusions
    ANALOGICAL = "analogical"        # Find similar patterns
    CAUSAL = "causal"               # Trace cause-effect relationships
    COMPARATIVE = "comparative"      # Compare and contrast concepts

@dataclass
class ConceptNode:
    """
    Enhanced concept node with phase vectors and semantic properties
    """
    concept_id: Union[str, int]
    name: str
    phase_vector: List[float]  # [semantic_density, abstraction_level, temporal_proximity]
    content_summary: str
    source: str
    node_type: str = "concept"  # concept, entity, relation, fact
    
    # Connections and metadata
    links: Dict[Union[str, int], float] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    confidence: float = 1.0
    
    @property
    def phase(self) -> float:
        """Unified phase value for compatibility"""
        return sum(self.phase_vector) / len(self.phase_vector) if self.phase_vector else 0.0
    
    def add_link(self, neighbor_id: Union[str, int], resonance: float) -> None:
        """Add a resonant link to another concept"""
        self.links[neighbor_id] = max(0.0, min(1.0, resonance))
    
    def calculate_phase_drift(self, other: 'ConceptNode') -> float:
        """Calculate phase drift to another concept"""
        if not self.phase_vector or not other.phase_vector:
            return abs(self.phase - other.phase)
        
        # L2 distance between phase vectors
        drift = sum((a - b) ** 2 for a, b in zip(self.phase_vector, other.phase_vector))
        return math.sqrt(drift)
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class ReasoningPath:
    """
    A complete reasoning path through the concept mesh
    """
    path: List[Union[str, int]]
    nodes: List[ConceptNode]
    edges: List[Tuple[str, str, float]]  # (from, to, resonance)
    
    # Path metrics
    total_drift: float
    total_resonance: float
    coherence_score: float
    path_length: int
    reasoning_mode: ReasoningMode
    
    # Explanatory content
    narrative: str = ""
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def __post_init__(self):
        self.path_length = len(self.path)
        if self.path_length > 1:
            self.confidence = max(0.0, min(1.0, 
                (self.total_resonance - self.total_drift) / self.path_length
            ))

@dataclass
class ReasoningRequest:
    """
    Request for reasoning operation
    """
    query: str
    start_concepts: List[str]
    target_concepts: List[str]
    mode: ReasoningMode = ReasoningMode.EXPLANATORY
    max_hops: int = 5
    min_confidence: float = 0.3
    focus_domains: List[str] = field(default_factory=list)
    enable_narrative: bool = True

@dataclass
class ReasoningResult:
    """
    Complete result of reasoning operation
    """
    paths: List[ReasoningPath]
    best_path: Optional[ReasoningPath]
    reasoning_time: float
    concepts_explored: int
    narrative_explanation: str
    confidence: float
    metadata: Dict[str, any] = field(default_factory=dict)

class ConceptMeshInterface:
    """
    Abstract interface for concept mesh - production implementation
    """
    
    def get_node(self, concept_id: Union[str, int]) -> Optional[ConceptNode]:
        raise NotImplementedError
    
    def get_neighbors(self, concept_id: Union[str, int]) -> List[Union[str, int]]:
        raise NotImplementedError
    
    def get_resonance(self, id1: Union[str, int], id2: Union[str, int]) -> Optional[float]:
        raise NotImplementedError
    
    def search_concepts(self, query: str, limit: int = 10) -> List[ConceptNode]:
        raise NotImplementedError
    
    def get_domain_concepts(self, domain: str) -> List[ConceptNode]:
        raise NotImplementedError

class PrajnaReasoningEngine:
    """
    Production reasoning engine that performs multi-hop cognitive pathfinding
    
    This is where Prajna gains true reasoning abilities - finding coherent paths
    through the concept mesh that explain relationships and generate insights.
    """
    
    def __init__(self, concept_mesh: ConceptMeshInterface):
        self.mesh = concept_mesh
        self.reasoning_cache: Dict[str, ReasoningResult] = {}
        self.performance_stats = {
            "queries_processed": 0,
            "paths_found": 0,
            "total_reasoning_time": 0.0,
            "cache_hits": 0
        }
    
    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """
        Main reasoning entry point - finds explanatory paths between concepts
        """
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Prajna reasoning: {request.query[:100]}...")
            logger.info(f"Mode: {request.mode.value}, Max hops: {request.max_hops}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.reasoning_cache:
                self.performance_stats["cache_hits"] += 1
                return self.reasoning_cache[cache_key]
            
            # Find start and target concepts
            start_nodes = await self._identify_concepts(request.start_concepts)
            target_nodes = await self._identify_concepts(request.target_concepts)
            
            if not start_nodes or not target_nodes:
                return ReasoningResult(
                    paths=[],
                    best_path=None,
                    reasoning_time=time.time() - start_time,
                    concepts_explored=0,
                    narrative_explanation="Could not identify start or target concepts",
                    confidence=0.0
                )
            
            # Find reasoning paths
            all_paths = []
            concepts_explored = 0
            
            for start_node in start_nodes:
                for target_node in target_nodes:
                    paths, explored = await self._find_reasoning_paths(
                        start_node.concept_id,
                        target_node.concept_id,
                        request
                    )
                    all_paths.extend(paths)
                    concepts_explored += explored
            
            # Select best path and generate narrative
            best_path = self._select_best_path(all_paths, request)
            narrative = await self._generate_narrative_explanation(best_path, request) if best_path else ""
            
            # Calculate overall confidence
            confidence = best_path.confidence if best_path else 0.0
            
            result = ReasoningResult(
                paths=all_paths,
                best_path=best_path,
                reasoning_time=time.time() - start_time,
                concepts_explored=concepts_explored,
                narrative_explanation=narrative,
                confidence=confidence,
                metadata={
                    "reasoning_mode": request.mode.value,
                    "max_hops": request.max_hops,
                    "paths_considered": len(all_paths)
                }
            )
            
            # Cache result
            self.reasoning_cache[cache_key] = result
            
            # Update stats
            self.performance_stats["queries_processed"] += 1
            self.performance_stats["paths_found"] += len(all_paths)
            self.performance_stats["total_reasoning_time"] += result.reasoning_time
            
            logger.info(f"🧠 Reasoning complete: {len(all_paths)} paths, confidence {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            return ReasoningResult(
                paths=[],
                best_path=None,
                reasoning_time=time.time() - start_time,
                concepts_explored=0,
                narrative_explanation=f"Reasoning error: {str(e)}",
                confidence=0.0
            )
    
    async def _identify_concepts(self, concept_hints: List[str]) -> List[ConceptNode]:
        """Identify concrete concept nodes from textual hints"""
        identified = []
        
        for hint in concept_hints:
            # Direct lookup first
            node = self.mesh.get_node(hint)
            if node:
                identified.append(node)
                continue
            
            # Search for matches
            matches = self.mesh.search_concepts(hint, limit=3)
            if matches:
                identified.extend(matches[:1])  # Take best match
        
        return identified
    
    async def _find_reasoning_paths(
        self, 
        start_id: Union[str, int], 
        target_id: Union[str, int],
        request: ReasoningRequest
    ) -> Tuple[List[ReasoningPath], int]:
        """
        Find all viable reasoning paths between two concepts using enhanced Dijkstra
        """
        
        if start_id == target_id:
            # Direct concept - create trivial path
            start_node = self.mesh.get_node(start_id)
            if start_node:
                path = ReasoningPath(
                    path=[start_id],
                    nodes=[start_node],
                    edges=[],
                    total_drift=0.0,
                    total_resonance=1.0,
                    coherence_score=1.0,
                    path_length=1,
                    reasoning_mode=request.mode
                )
                return [path], 1
            return [], 0
        
        # Priority queue: (cost, current_id, path, total_drift, total_resonance)
        pq = [(0.0, start_id, [start_id], 0.0, 0.0)]
        
        # Best cost to reach each node
        best_cost: Dict[Union[str, int], float] = {start_id: 0.0}
        
        # Track all viable paths found
        viable_paths = []
        concepts_explored = 0
        
        while pq:
            current_cost, current_id, path, total_drift, total_resonance = heapq.heappop(pq)
            
            # Skip if we've found a better path to this node
            if current_cost > best_cost.get(current_id, float('inf')):
                continue
            
            concepts_explored += 1
            
            # Check if we've reached target
            if current_id == target_id:
                reasoning_path = await self._create_reasoning_path(
                    path, total_drift, total_resonance, request
                )
                if reasoning_path and reasoning_path.confidence >= request.min_confidence:
                    viable_paths.append(reasoning_path)
                continue
            
            # Stop if max hops reached
            if len(path) >= request.max_hops + 1:
                continue
            
            # Explore neighbors
            current_node = self.mesh.get_node(current_id)
            if not current_node:
                continue
            
            for neighbor_id in self.mesh.get_neighbors(current_id):
                if neighbor_id in path:  # Avoid cycles
                    continue
                
                neighbor_node = self.mesh.get_node(neighbor_id)
                if not neighbor_node:
                    continue
                
                # Calculate edge cost
                phase_drift = current_node.calculate_phase_drift(neighbor_node)
                resonance = self.mesh.get_resonance(current_id, neighbor_id) or 0.0
                
                # Enhanced cost function based on reasoning mode
                edge_cost = self._calculate_edge_cost(
                    phase_drift, resonance, request.mode, len(path)
                )
                
                new_cost = current_cost + edge_cost
                new_drift = total_drift + phase_drift
                new_resonance = total_resonance + resonance
                
                # Update if this is a better path to neighbor
                if new_cost < best_cost.get(neighbor_id, float('inf')):
                    best_cost[neighbor_id] = new_cost
                    heapq.heappush(pq, (
                        new_cost, 
                        neighbor_id, 
                        path + [neighbor_id],
                        new_drift,
                        new_resonance
                    ))
        
        return viable_paths, concepts_explored
    
    def _calculate_edge_cost(
        self, 
        phase_drift: float, 
        resonance: float, 
        mode: ReasoningMode,
        path_length: int
    ) -> float:
        """
        Calculate edge cost with mode-specific adjustments
        """
        base_cost = phase_drift + (1.0 - resonance)
        
        # Mode-specific adjustments
        if mode == ReasoningMode.EXPLANATORY:
            # Favor high resonance for clear explanations
            base_cost += (1.0 - resonance) * 0.5
            
        elif mode == ReasoningMode.CAUSAL:
            # Penalize high drift more for causal chains
            base_cost += phase_drift * 0.3
            
        elif mode == ReasoningMode.ANALOGICAL:
            # Allow more drift for analogies
            base_cost -= phase_drift * 0.2
            
        elif mode == ReasoningMode.COMPARATIVE:
            # Balance drift and resonance equally
            pass  # Use base cost
        
        # Slight penalty for longer paths
        base_cost += path_length * 0.05
        
        return max(0.0, base_cost)
    
    async def _create_reasoning_path(
        self,
        path_ids: List[Union[str, int]],
        total_drift: float,
        total_resonance: float,
        request: ReasoningRequest
    ) -> Optional[ReasoningPath]:
        """Create a complete ReasoningPath object"""
        
        # Get all nodes
        nodes = []
        for path_id in path_ids:
            node = self.mesh.get_node(path_id)
            if not node:
                return None
            node.update_access()
            nodes.append(node)
        
        # Create edges
        edges = []
        for i in range(len(path_ids) - 1):
            resonance = self.mesh.get_resonance(path_ids[i], path_ids[i + 1]) or 0.0
            edges.append((str(path_ids[i]), str(path_ids[i + 1]), resonance))
        
        # Calculate coherence score
        path_length = len(path_ids)
        if path_length > 1:
            coherence_score = (total_resonance - total_drift) / (path_length - 1)
        else:
            coherence_score = 1.0
        
        return ReasoningPath(
            path=path_ids,
            nodes=nodes,
            edges=edges,
            total_drift=total_drift,
            total_resonance=total_resonance,
            coherence_score=coherence_score,
            path_length=path_length,
            reasoning_mode=request.mode
        )
    
    def _select_best_path(self, paths: List[ReasoningPath], request: ReasoningRequest) -> Optional[ReasoningPath]:
        """Select the best reasoning path based on multiple criteria"""
        
        if not paths:
            return None
        
        # Score each path
        scored_paths = []
        for path in paths:
            score = self._score_path(path, request)
            scored_paths.append((score, path))
        
        # Sort by score (descending)
        scored_paths.sort(reverse=True, key=lambda x: x[0])
        
        return scored_paths[0][1]
    
    def _score_path(self, path: ReasoningPath, request: ReasoningRequest) -> float:
        """Score a reasoning path based on multiple factors"""
        
        # Base score from coherence and confidence
        score = path.coherence_score * 0.4 + path.confidence * 0.4
        
        # Penalize very long paths
        length_penalty = max(0, (path.path_length - 3) * 0.05)
        score -= length_penalty
        
        # Bonus for high total resonance
        resonance_bonus = min(0.2, path.total_resonance / path.path_length * 0.1)
        score += resonance_bonus
        
        # Mode-specific adjustments
        if request.mode == ReasoningMode.EXPLANATORY:
            # Favor paths with good narrative potential
            score += path.coherence_score * 0.1
            
        elif request.mode == ReasoningMode.CAUSAL:
            # Favor shorter, more direct paths
            score += max(0, (5 - path.path_length) * 0.05)
        
        return max(0.0, min(1.0, score))
    
    async def _generate_narrative_explanation(self, path: ReasoningPath, request: ReasoningRequest) -> str:
        """
        Generate a natural language explanation of the reasoning path
        """
        
        if not path or len(path.nodes) < 2:
            return "No clear reasoning path found."
        
        # Build narrative based on reasoning mode
        if request.mode == ReasoningMode.EXPLANATORY:
            return self._generate_explanatory_narrative(path)
        elif request.mode == ReasoningMode.CAUSAL:
            return self._generate_causal_narrative(path)
        elif request.mode == ReasoningMode.ANALOGICAL:
            return self._generate_analogical_narrative(path)
        elif request.mode == ReasoningMode.COMPARATIVE:
            return self._generate_comparative_narrative(path)
        else:
            return self._generate_explanatory_narrative(path)
    
    def _generate_explanatory_narrative(self, path: ReasoningPath) -> str:
        """Generate explanatory narrative connecting concepts"""
        
        if len(path.nodes) == 1:
            return f"The concept '{path.nodes[0].name}' directly addresses this query."
        
        narrative_parts = []
        narrative_parts.append(f"Starting with '{path.nodes[0].name}' ({path.nodes[0].content_summary})")
        
        for i in range(1, len(path.nodes)):
            prev_node = path.nodes[i-1]
            curr_node = path.nodes[i]
            edge_resonance = path.edges[i-1][2] if i-1 < len(path.edges) else 0.0
            
            connection_strength = "strongly" if edge_resonance > 0.8 else "clearly" if edge_resonance > 0.6 else "potentially"
            
            narrative_parts.append(
                f"This {connection_strength} connects to '{curr_node.name}' ({curr_node.content_summary})"
            )
        
        narrative_parts.append(f"This reasoning path has a coherence score of {path.coherence_score:.2f} and demonstrates the conceptual relationship through {len(path.nodes)} key ideas.")
        
        return ". ".join(narrative_parts) + "."
    
    def _generate_causal_narrative(self, path: ReasoningPath) -> str:
        """Generate causal narrative showing cause-effect relationships"""
        
        if len(path.nodes) == 1:
            return f"'{path.nodes[0].name}' appears to be the direct cause or effect in this scenario."
        
        narrative_parts = []
        narrative_parts.append(f"The causal chain begins with '{path.nodes[0].name}'")
        
        for i in range(1, len(path.nodes)):
            curr_node = path.nodes[i]
            narrative_parts.append(f"which leads to '{curr_node.name}'")
        
        narrative_parts.append(f"This causal pathway shows how the initial concept ultimately connects to the target through {len(path.nodes)-1} causal steps")
        
        return " ".join(narrative_parts) + "."
    
    def _generate_analogical_narrative(self, path: ReasoningPath) -> str:
        """Generate analogical narrative highlighting similarities"""
        
        return f"The analogy between '{path.nodes[0].name}' and '{path.nodes[-1].name}' can be understood through their shared conceptual framework, connecting via {len(path.nodes)-2} intermediate concepts that demonstrate similar patterns or structures."
    
    def _generate_comparative_narrative(self, path: ReasoningPath) -> str:
        """Generate comparative narrative showing differences and similarities"""
        
        return f"Comparing '{path.nodes[0].name}' with '{path.nodes[-1].name}' reveals both similarities and differences, with the key distinguishing factors emerging through {len(path.nodes)-2} intermediate comparative points."
    
    def _generate_cache_key(self, request: ReasoningRequest) -> str:
        """Generate cache key for reasoning request"""
        key_parts = [
            request.query.lower().strip(),
            "_".join(sorted(request.start_concepts)),
            "_".join(sorted(request.target_concepts)),
            request.mode.value,
            str(request.max_hops),
            str(request.min_confidence)
        ]
        return "|".join(key_parts)
    
    async def get_stats(self) -> Dict[str, any]:
        """Get reasoning engine performance statistics"""
        stats = self.performance_stats.copy()
        stats["cache_size"] = len(self.reasoning_cache)
        if stats["queries_processed"] > 0:
            stats["average_reasoning_time"] = stats["total_reasoning_time"] / stats["queries_processed"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["queries_processed"]
        return stats
    
    def clear_cache(self):
        """Clear reasoning cache"""
        self.reasoning_cache.clear()
        logger.info("🧹 Reasoning cache cleared")

# Integration functions for Prajna pipeline

async def detect_reasoning_triggers(query: str) -> Optional[ReasoningRequest]:
    """
    Detect if a query requires reasoning and create appropriate request
    """
    
    query_lower = query.lower()
    
    # Bridge words indicating reasoning needed
    bridge_patterns = [
        r'\b(between|connects?|relation|relationship)\b',
        r'\b(how does.*affect|why does.*cause|what leads to)\b', 
        r'\b(explain.*connection|show.*relationship)\b',
        r'\b(compare|contrast|difference|similarity)\b',
        r'\b(because|therefore|thus|hence|consequently)\b'
    ]
    
    reasoning_mode = ReasoningMode.EXPLANATORY  # Default
    
    # Detect reasoning mode
    if any(word in query_lower for word in ["cause", "effect", "leads to", "results in"]):
        reasoning_mode = ReasoningMode.CAUSAL
    elif any(word in query_lower for word in ["similar", "like", "analogy", "analogous"]):
        reasoning_mode = ReasoningMode.ANALOGICAL
    elif any(word in query_lower for word in ["compare", "contrast", "difference", "versus"]):
        reasoning_mode = ReasoningMode.COMPARATIVE
    elif any(word in query_lower for word in ["infer", "conclude", "deduce", "implies"]):
        reasoning_mode = ReasoningMode.INFERENTIAL
    
    # Check for trigger patterns
    for pattern in bridge_patterns:
        if re.search(pattern, query_lower):
            # Extract concepts from query (simplified)
            words = query.split()
            concepts = [word.strip(".,!?") for word in words if len(word) > 3]
            
            return ReasoningRequest(
                query=query,
                start_concepts=concepts[:3],  # First few concepts as start
                target_concepts=concepts[-3:],  # Last few as targets
                mode=reasoning_mode,
                max_hops=5,
                min_confidence=0.3
            )
    
    return None

async def enhance_context_with_reasoning(
    context_text: str,
    reasoning_result: ReasoningResult
) -> str:
    """
    Enhance retrieved context with reasoning narrative
    """
    
    if not reasoning_result.best_path or not reasoning_result.narrative_explanation:
        return context_text
    
    reasoning_section = f"""

REASONING PATH ANALYSIS:
{reasoning_result.narrative_explanation}

Path confidence: {reasoning_result.confidence:.2f}
Concepts explored: {reasoning_result.concepts_explored}
Reasoning mode: {reasoning_result.metadata.get('reasoning_mode', 'explanatory')}

This reasoning analysis provides additional context for understanding the relationships between key concepts.
"""
    
    return context_text + reasoning_section

# Production concept mesh adapter
class ProductionConceptMeshAdapter(ConceptMeshInterface):
    """
    Production adapter for existing TORI concept mesh systems
    """
    
    def __init__(self, concept_mesh_api, soliton_memory=None):
        self.concept_mesh_api = concept_mesh_api
        self.soliton_memory = soliton_memory
        self.node_cache = {}
        
    def get_node(self, concept_id: Union[str, int]) -> Optional[ConceptNode]:
        """Get concept node from production concept mesh"""
        
        # Check cache first
        if concept_id in self.node_cache:
            return self.node_cache[concept_id]
        
        try:
            # Adapt from production concept mesh
            mesh_concept = self.concept_mesh_api.get_concept(concept_id)
            if not mesh_concept:
                return None
            
            # Convert to reasoning engine format
            node = ConceptNode(
                concept_id=concept_id,
                name=getattr(mesh_concept, 'name', str(concept_id)),
                phase_vector=self._extract_phase_vector(mesh_concept),
                content_summary=getattr(mesh_concept, 'summary', ''),
                source=getattr(mesh_concept, 'source', 'unknown'),
                node_type=getattr(mesh_concept, 'type', 'concept')
            )
            
            # Cache and return
            self.node_cache[concept_id] = node
            return node
            
        except Exception as e:
            logger.warning(f"Failed to get concept node {concept_id}: {e}")
            return None
    
    def get_neighbors(self, concept_id: Union[str, int]) -> List[Union[str, int]]:
        """Get concept neighbors from production mesh"""
        try:
            return self.concept_mesh_api.get_neighbors(concept_id)
        except Exception as e:
            logger.warning(f"Failed to get neighbors for {concept_id}: {e}")
            return []
    
    def get_resonance(self, id1: Union[str, int], id2: Union[str, int]) -> Optional[float]:
        """Get resonance between concepts from production mesh"""
        try:
            return self.concept_mesh_api.get_edge_weight(id1, id2)
        except Exception as e:
            logger.warning(f"Failed to get resonance {id1}-{id2}: {e}")
            return None
    
    def search_concepts(self, query: str, limit: int = 10) -> List[ConceptNode]:
        """Search concepts in production mesh"""
        try:
            results = self.concept_mesh_api.search(query, limit)
            nodes = []
            for result in results:
                node = self.get_node(result.concept_id)
                if node:
                    nodes.append(node)
            return nodes
        except Exception as e:
            logger.warning(f"Failed to search concepts for '{query}': {e}")
            return []
    
    def get_domain_concepts(self, domain: str) -> List[ConceptNode]:
        """Get concepts from specific domain"""
        try:
            results = self.concept_mesh_api.get_by_domain(domain)
            nodes = []
            for result in results:
                node = self.get_node(result.concept_id)
                if node:
                    nodes.append(node)
            return nodes
        except Exception as e:
            logger.warning(f"Failed to get domain concepts for '{domain}': {e}")
            return []
    
    def _extract_phase_vector(self, mesh_concept) -> List[float]:
        """Extract phase vector from production concept"""
        try:
            # Try to get existing phase data
            if hasattr(mesh_concept, 'phase_vector'):
                return mesh_concept.phase_vector
            
            # Calculate basic phase vector from concept properties
            semantic_density = len(getattr(mesh_concept, 'content', '')) / 1000.0
            abstraction_level = getattr(mesh_concept, 'abstraction', 0.5)
            temporal_proximity = getattr(mesh_concept, 'recency', 0.5)
            
            return [
                min(1.0, semantic_density),
                min(1.0, abstraction_level), 
                min(1.0, temporal_proximity)
            ]
            
        except Exception:
            # Default phase vector
            return [0.5, 0.5, 0.5]
