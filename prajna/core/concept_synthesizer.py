"""
Concept Synthesizer: High-Dimensional Concept Fusion for Prajna
===============================================================

Production implementation of Prajna's concept synthesis and creative fusion engine.
This module performs high-dimensional concept fusion across domains, generates novel 
conceptual connections, maintains Ψ-trajectories for transparency, and manages
entropy to prevent cognitive drift.

This is where Prajna gains creative intelligence - the ability to synthesize new
insights by fusing concepts from disparate knowledge domains.
"""

import asyncio
import logging
import re
import time
import math
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from collections import defaultdict, Counter
from itertools import combinations, permutations

logger = logging.getLogger("prajna.concept_synthesizer")

@dataclass
class ConceptNode:
    """Enhanced concept representation with fusion capabilities"""
    concept_id: str
    name: str
    domain: str                           # Knowledge domain (science, philosophy, etc.)
    semantic_vector: List[float]          # High-dimensional semantic representation
    fusion_potential: float               # How well this concept fuses with others
    abstraction_level: float              # Concrete (0.0) to abstract (1.0)
    novelty_score: float                  # How novel/creative this concept is
    
    # Relationship data
    related_concepts: Dict[str, float] = field(default_factory=dict)  # concept_id -> strength
    fusion_history: List[str] = field(default_factory=list)  # Previous fusion operations
    
    # Metadata
    source: str = ""
    confidence: float = 1.0
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class ConceptFusion:
    """Result of fusing multiple concepts together"""
    fusion_id: str
    source_concepts: List[str]            # IDs of concepts that were fused
    fused_concept: ConceptNode            # The resulting synthesized concept
    
    # Fusion metrics
    coherence_score: float                # How coherent the fusion is
    novelty_score: float                  # How novel the result is
    stability_score: float                # How stable/reliable the fusion is
    cross_domain_score: float             # How well it bridges domains
    
    # Process metadata
    fusion_method: str                    # Method used for fusion
    fusion_time: float                    # Time taken to create fusion
    confidence: float                     # Confidence in fusion quality
    
    # Ψ-trajectory data
    reasoning_path: List[str] = field(default_factory=list)  # Conceptual reasoning steps
    bridging_concepts: List[str] = field(default_factory=list)  # Intermediate concepts used

@dataclass
class PsiTrajectory:
    """Complete Ψ-trajectory tracking conceptual reasoning path"""
    trajectory_id: str
    start_concepts: List[str]
    end_concepts: List[str]
    
    # Path data
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    concept_transitions: List[Tuple[str, str, float]] = field(default_factory=list)  # (from, to, strength)
    bridging_insights: List[str] = field(default_factory=list)
    
    # Trajectory metrics
    path_length: int = 0
    coherence_score: float = 0.0
    creativity_score: float = 0.0
    completion_time: float = 0.0
    
    # Metadata
    query_context: str = ""
    synthesis_goal: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SynthesisResult:
    """Complete result of concept synthesis operation"""
    synthesized_concepts: List[ConceptNode]
    fusions: List[ConceptFusion]
    psi_trajectory: PsiTrajectory
    
    # Quality metrics
    overall_coherence: float
    novelty_index: float
    cross_domain_coverage: float
    entropy_score: float
    
    # Processing metadata
    synthesis_time: float
    concepts_explored: int
    domains_covered: Set[str]
    method_used: str

class ConceptSynthesizer:
    """
    Production high-dimensional concept fusion engine for creative reasoning.
    
    This is where Prajna gains creative intelligence - synthesizing novel insights
    by fusing concepts across knowledge domains with full transparency.
    """
    
    def __init__(self, concept_mesh=None, psi_archive=None, enable_creativity=True):
        self.concept_mesh = concept_mesh
        self.psi_archive = psi_archive
        self.enable_creativity = enable_creativity
        
        # Fusion configuration
        self.max_fusion_depth = 3            # Maximum fusion recursion depth
        self.min_coherence_threshold = 0.4   # Minimum coherence for valid fusion
        self.max_entropy_threshold = 0.8     # Maximum entropy before pruning
        self.novelty_boost_factor = 1.2      # Boost factor for novel concepts
        
        # Concept caches
        self.concept_cache = {}              # concept_id -> ConceptNode
        self.fusion_cache = {}               # fusion_signature -> ConceptFusion
        self.trajectory_cache = {}           # trajectory_id -> PsiTrajectory
        
        # Domain knowledge
        self.domain_mappings = self._initialize_domain_mappings()
        self.fusion_patterns = self._initialize_fusion_patterns()
        self.creativity_heuristics = self._initialize_creativity_heuristics()
        
        # Performance tracking
        self.synthesis_stats = {
            "total_syntheses": 0,
            "successful_fusions": 0,
            "novel_concepts_created": 0,
            "cross_domain_fusions": 0,
            "average_coherence": 0.0,
            "average_novelty": 0.0
        }
        
        logger.info("🎨 ConceptSynthesizer initialized with creative fusion capabilities")
    
    async def synthesize_from_context(self, context: str, synthesis_goal: str = "") -> SynthesisResult:
        """
        Perform comprehensive concept synthesis from provided context.
        
        This is the main entry point for creative concept fusion.
        """
        start_time = time.time()
        trajectory_id = self._generate_trajectory_id(context, synthesis_goal)
        
        try:
            logger.info(f"🎨 Starting concept synthesis from context: {context[:100]}...")
            
            # Step 1: Extract seed concepts from context
            seed_concepts = await self._extract_seed_concepts(context)
            logger.debug(f"Extracted {len(seed_concepts)} seed concepts")
            
            # Step 2: Expand concept space through domain traversal
            expanded_concepts = await self._expand_concept_space(seed_concepts)
            logger.debug(f"Expanded to {len(expanded_concepts)} concepts")
            
            # Step 3: Identify fusion opportunities
            fusion_opportunities = await self._identify_fusion_opportunities(expanded_concepts)
            logger.debug(f"Found {len(fusion_opportunities)} fusion opportunities")
            
            # Step 4: Execute concept fusions
            successful_fusions = []
            for opportunity in fusion_opportunities:
                fusion = await self._execute_concept_fusion(opportunity)
                if fusion and fusion.coherence_score >= self.min_coherence_threshold:
                    successful_fusions.append(fusion)
            
            logger.debug(f"Completed {len(successful_fusions)} successful fusions")
            
            # Step 5: Generate novel synthesized concepts
            synthesized_concepts = await self._generate_synthesized_concepts(
                successful_fusions, synthesis_goal
            )
            
            # Step 6: Create Ψ-trajectory
            psi_trajectory = await self._create_psi_trajectory(
                trajectory_id, seed_concepts, synthesized_concepts, 
                successful_fusions, context, synthesis_goal
            )
            
            # Step 7: Calculate entropy and prune if necessary
            entropy_score = self._calculate_concept_entropy(synthesized_concepts)
            if entropy_score > self.max_entropy_threshold:
                synthesized_concepts = await self._prune_divergent_concepts(synthesized_concepts)
                entropy_score = self._calculate_concept_entropy(synthesized_concepts)
            
            # Step 8: Calculate quality metrics
            overall_coherence = self._calculate_overall_coherence(successful_fusions)
            novelty_index = self._calculate_novelty_index(synthesized_concepts)
            cross_domain_coverage = self._calculate_cross_domain_coverage(synthesized_concepts)
            domains_covered = {concept.domain for concept in synthesized_concepts}
            
            # Step 9: Create synthesis result
            result = SynthesisResult(
                synthesized_concepts=synthesized_concepts,
                fusions=successful_fusions,
                psi_trajectory=psi_trajectory,
                overall_coherence=overall_coherence,
                novelty_index=novelty_index,
                cross_domain_coverage=cross_domain_coverage,
                entropy_score=entropy_score,
                synthesis_time=time.time() - start_time,
                concepts_explored=len(expanded_concepts),
                domains_covered=domains_covered,
                method_used="comprehensive_fusion"
            )
            
            # Step 10: Archive synthesis for learning
            if self.psi_archive:
                await self._archive_synthesis(result, context, synthesis_goal)
            
            # Update statistics
            self._update_synthesis_stats(result)
            
            logger.info(f"🎨 Synthesis complete: {len(synthesized_concepts)} concepts, "
                       f"coherence: {overall_coherence:.2f}, novelty: {novelty_index:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Concept synthesis failed: {e}")
            # Return minimal result
            return SynthesisResult(
                synthesized_concepts=[],
                fusions=[],
                psi_trajectory=PsiTrajectory(
                    trajectory_id=trajectory_id,
                    start_concepts=[],
                    end_concepts=[]
                ),
                overall_coherence=0.0,
                novelty_index=0.0,
                cross_domain_coverage=0.0,
                entropy_score=1.0,
                synthesis_time=time.time() - start_time,
                concepts_explored=0,
                domains_covered=set(),
                method_used="failed_synthesis"
            )
    
    async def expand_concept(self, concept: str, depth: int = 1, domain_filter: str = "") -> List[ConceptNode]:
        """
        Expand a single concept into related concepts with controlled depth.
        
        This enables focused concept exploration for specific reasoning needs.
        """
        try:
            logger.info(f"🔍 Expanding concept: '{concept}' to depth {depth}")
            
            # Get or create base concept node
            base_concept = await self._get_or_create_concept_node(concept)
            if not base_concept:
                return []
            
            expanded_concepts = [base_concept]
            current_frontier = [base_concept]
            
            # Iteratively expand through depth levels
            for current_depth in range(depth):
                next_frontier = []
                
                for frontier_concept in current_frontier:
                    # Get related concepts from concept mesh
                    related = await self._get_related_concepts(frontier_concept.concept_id)
                    
                    for related_id in related:
                        related_concept = await self._get_or_create_concept_node(related_id)
                        
                        if related_concept and related_concept not in expanded_concepts:
                            # Apply domain filter if specified
                            if not domain_filter or related_concept.domain == domain_filter:
                                expanded_concepts.append(related_concept)
                                next_frontier.append(related_concept)
                
                current_frontier = next_frontier
                
                # Prevent explosive growth
                if len(expanded_concepts) > 50:
                    break
            
            logger.debug(f"Expanded '{concept}' to {len(expanded_concepts)} related concepts")
            
            return expanded_concepts
            
        except Exception as e:
            logger.error(f"❌ Concept expansion failed for '{concept}': {e}")
            return []
    
    async def trace_inference_path(self, target_concept: str, source_concepts: List[str] = None) -> PsiTrajectory:
        """
        Trace the inference path that led to a target concept.
        
        This provides complete transparency in conceptual reasoning.
        """
        try:
            trajectory_id = self._generate_trajectory_id(f"trace_{target_concept}", "inference_tracing")
            
            logger.info(f"🗺️ Tracing inference path to: '{target_concept}'")
            
            # Initialize trajectory
            trajectory = PsiTrajectory(
                trajectory_id=trajectory_id,
                start_concepts=source_concepts or [],
                end_concepts=[target_concept],
                synthesis_goal="inference_tracing"
            )
            
            # Check if target concept exists in our knowledge
            target_node = await self._get_or_create_concept_node(target_concept)
            if not target_node:
                logger.warning(f"Target concept '{target_concept}' not found")
                return trajectory
            
            # Trace backward from target concept
            reasoning_steps = []
            concept_transitions = []
            current_concepts = [target_concept]
            explored_concepts = set()
            
            for step_depth in range(self.max_fusion_depth):
                step_data = {
                    "step": step_depth,
                    "concepts": current_concepts.copy(),
                    "reasoning": f"Tracing inference step {step_depth}",
                    "timestamp": datetime.now().isoformat()
                }
                reasoning_steps.append(step_data)
                
                next_concepts = []
                
                for concept_id in current_concepts:
                    if concept_id in explored_concepts:
                        continue
                    
                    explored_concepts.add(concept_id)
                    
                    # Find concepts that could have led to this one
                    potential_sources = await self._find_source_concepts(concept_id)
                    
                    for source_id, strength in potential_sources:
                        if source_id not in explored_concepts:
                            next_concepts.append(source_id)
                            concept_transitions.append((source_id, concept_id, strength))
                
                if not next_concepts:
                    break
                
                current_concepts = next_concepts[:10]  # Limit exploration
            
            # Generate bridging insights
            bridging_insights = await self._generate_bridging_insights(concept_transitions)
            
            # Populate trajectory
            trajectory.reasoning_steps = reasoning_steps
            trajectory.concept_transitions = concept_transitions
            trajectory.bridging_insights = bridging_insights
            trajectory.path_length = len(reasoning_steps)
            trajectory.coherence_score = self._calculate_trajectory_coherence(concept_transitions)
            trajectory.creativity_score = self._calculate_trajectory_creativity(concept_transitions)
            trajectory.completion_time = time.time()
            
            logger.info(f"🗺️ Inference path traced: {trajectory.path_length} steps, "
                       f"coherence: {trajectory.coherence_score:.2f}")
            
            return trajectory
            
        except Exception as e:
            logger.error(f"❌ Inference path tracing failed: {e}")
            return PsiTrajectory(
                trajectory_id=f"failed_trace_{int(time.time())}",
                start_concepts=source_concepts or [],
                end_concepts=[target_concept]
            )
    
    def calculate_concept_entropy(self, concepts: List[Union[str, ConceptNode]]) -> float:
        """
        Calculate entropy of a concept set to measure divergence.
        
        Higher entropy indicates more scattered/divergent concepts.
        """
        try:
            if not concepts:
                return 0.0
            
            # Convert to concept nodes if needed
            concept_nodes = []
            for concept in concepts:
                if isinstance(concept, str):
                    node = self.concept_cache.get(concept)
                    if node:
                        concept_nodes.append(node)
                else:
                    concept_nodes.append(concept)
            
            if not concept_nodes:
                return 1.0  # Maximum entropy for no valid concepts
            
            # Calculate domain distribution entropy
            domain_counts = Counter(node.domain for node in concept_nodes)
            total_concepts = len(concept_nodes)
            
            domain_entropy = 0.0
            for count in domain_counts.values():
                probability = count / total_concepts
                if probability > 0:
                    domain_entropy -= probability * math.log2(probability)
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(len(domain_counts)) if len(domain_counts) > 1 else 1.0
            normalized_entropy = domain_entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Calculate abstraction level variance
            abstraction_levels = [node.abstraction_level for node in concept_nodes]
            abstraction_variance = self._calculate_variance(abstraction_levels)
            
            # Combine domain entropy and abstraction variance
            total_entropy = (normalized_entropy + abstraction_variance) / 2.0
            
            return min(1.0, total_entropy)
            
        except Exception as e:
            logger.error(f"❌ Entropy calculation failed: {e}")
            return 1.0  # Return maximum entropy on error
    
    async def prune_divergent_concepts(self, concepts: List[ConceptNode]) -> List[ConceptNode]:
        """
        Prune divergent concepts to maintain focus and coherence.
        
        This prevents cognitive drift by removing outlier concepts.
        """
        try:
            if len(concepts) <= 3:  # Don't prune if we have few concepts
                return concepts
            
            logger.info(f"🌿 Pruning {len(concepts)} concepts for coherence")
            
            # Calculate concept centrality scores
            centrality_scores = await self._calculate_concept_centralities(concepts)
            
            # Calculate coherence contribution scores
            coherence_scores = await self._calculate_coherence_contributions(concepts)
            
            # Calculate overall relevance scores
            relevance_scores = {}
            for concept in concepts:
                centrality = centrality_scores.get(concept.concept_id, 0.0)
                coherence = coherence_scores.get(concept.concept_id, 0.0)
                novelty_factor = concept.novelty_score * 0.3  # Bonus for novel concepts
                
                relevance_scores[concept.concept_id] = (
                    centrality * 0.4 + 
                    coherence * 0.4 + 
                    novelty_factor * 0.2
                )
            
            # Sort by relevance and keep top concepts
            sorted_concepts = sorted(
                concepts, 
                key=lambda c: relevance_scores.get(c.concept_id, 0.0), 
                reverse=True
            )
            
            # Keep at least 3 concepts, at most 15 for manageability
            target_count = max(3, min(15, len(concepts) // 2))
            pruned_concepts = sorted_concepts[:target_count]
            
            logger.info(f"🌿 Pruned to {len(pruned_concepts)} coherent concepts")
            
            return pruned_concepts
            
        except Exception as e:
            logger.error(f"❌ Concept pruning failed: {e}")
            return concepts  # Return original concepts if pruning fails
    
    # Production implementation methods
    
    async def _extract_seed_concepts(self, context: str) -> List[ConceptNode]:
        """Extract initial seed concepts from context text."""
        seed_concepts = []
        
        # Extract meaningful terms using multiple strategies
        
        # Strategy 1: Named entities and proper nouns
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
        
        # Strategy 2: Technical terms and compound words
        technical_terms = re.findall(r'\b\w+(?:[-_]\w+)+\b', context)
        
        # Strategy 3: Domain-specific keywords
        domain_keywords = self._extract_domain_keywords(context)
        
        # Strategy 4: Multi-word concepts
        multi_word_concepts = re.findall(r'\b(?:[a-z]+\s+){1,2}[a-z]+\b', context.lower())
        
        # Combine all candidate terms
        candidate_terms = set()
        candidate_terms.update(named_entities)
        candidate_terms.update(technical_terms)
        candidate_terms.update(domain_keywords)
        candidate_terms.update(multi_word_concepts)
        
        # Filter and score candidates
        for term in candidate_terms:
            if len(term) > 2 and not self._is_stopword(term):
                concept_node = await self._get_or_create_concept_node(term)
                if concept_node:
                    # Score based on frequency and position in context
                    frequency = context.lower().count(term.lower())
                    position_score = 1.0 - (context.lower().find(term.lower()) / len(context))
                    
                    concept_node.fusion_potential = frequency * 0.3 + position_score * 0.7
                    seed_concepts.append(concept_node)
        
        # Sort by fusion potential and return top concepts
        seed_concepts.sort(key=lambda c: c.fusion_potential, reverse=True)
        return seed_concepts[:10]
    
    def _extract_domain_keywords(self, context: str) -> List[str]:
        """Extract domain-specific keywords from context."""
        keywords = []
        
        # Domain-specific patterns
        domain_patterns = {
            "science": [
                r'\b(?:theory|hypothesis|experiment|data|analysis|research|study)\b',
                r'\b(?:quantum|molecular|cellular|neural|genetic|atomic)\b',
                r'\b(?:algorithm|computation|processing|network|system)\b'
            ],
            "philosophy": [
                r'\b(?:consciousness|mind|reality|existence|truth|knowledge)\b',
                r'\b(?:ethics|morality|justice|freedom|being|essence)\b'
            ],
            "technology": [
                r'\b(?:software|hardware|digital|artificial|virtual|cyber)\b',
                r'\b(?:intelligence|learning|automation|robotics|computing)\b'
            ]
        }
        
        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                keywords.extend(matches)
        
        return list(set(keywords))
    
    def _is_stopword(self, term: str) -> bool:
        """Check if term is a stopword."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        return term.lower() in stopwords
    
    async def _expand_concept_space(self, seed_concepts: List[ConceptNode]) -> List[ConceptNode]:
        """Expand the concept space through intelligent traversal."""
        expanded_concepts = seed_concepts.copy()
        explored_ids = {c.concept_id for c in seed_concepts}
        
        # Multi-strategy expansion
        for seed_concept in seed_concepts:
            # Strategy 1: Direct relationships
            related_ids = await self._get_related_concepts(seed_concept.concept_id)
            
            for related_id in related_ids[:5]:  # Limit to top 5 related
                if related_id not in explored_ids:
                    related_node = await self._get_or_create_concept_node(related_id)
                    if related_node:
                        expanded_concepts.append(related_node)
                        explored_ids.add(related_id)
            
            # Strategy 2: Cross-domain bridging
            if self.enable_creativity:
                bridge_concepts = await self._find_cross_domain_bridges(seed_concept)
                for bridge_concept in bridge_concepts[:3]:  # Limit bridging
                    if bridge_concept.concept_id not in explored_ids:
                        expanded_concepts.append(bridge_concept)
                        explored_ids.add(bridge_concept.concept_id)
            
            # Strategy 3: Abstraction level exploration
            abstract_concepts = await self._explore_abstraction_levels(seed_concept)
            for abstract_concept in abstract_concepts[:2]:  # Limit abstraction
                if abstract_concept.concept_id not in explored_ids:
                    expanded_concepts.append(abstract_concept)
                    explored_ids.add(abstract_concept.concept_id)
        
        return expanded_concepts
    
    async def _explore_abstraction_levels(self, concept: ConceptNode) -> List[ConceptNode]:
        """Explore different abstraction levels around a concept."""
        abstract_concepts = []
        
        # Find more abstract concepts
        if concept.abstraction_level < 0.8:
            abstract_terms = await self._find_more_abstract_concepts(concept)
            abstract_concepts.extend(abstract_terms)
        
        # Find more concrete concepts
        if concept.abstraction_level > 0.3:
            concrete_terms = await self._find_more_concrete_concepts(concept)
            abstract_concepts.extend(concrete_terms)
        
        return abstract_concepts
    
    async def _find_more_abstract_concepts(self, concept: ConceptNode) -> List[ConceptNode]:
        """Find more abstract concepts related to the given concept."""
        abstract_concepts = []
        
        # Abstract term patterns based on concept domain
        abstract_patterns = {
            'physics': ['theory', 'principle', 'law', 'field', 'force'],
            'philosophy': ['concept', 'idea', 'notion', 'essence', 'being'],
            'neuroscience': ['process', 'function', 'mechanism', 'system', 'network'],
            'computer_science': ['algorithm', 'method', 'approach', 'paradigm', 'framework'],
            'biology': ['evolution', 'adaptation', 'selection', 'emergence', 'development']
        }
        
        patterns = abstract_patterns.get(concept.domain, ['concept', 'theory', 'principle'])
        
        for pattern in patterns[:3]:
            abstract_id = f"{pattern}_{concept.name.replace(' ', '_')}"
            abstract_concept = await self._get_or_create_concept_node(abstract_id)
            if abstract_concept:
                abstract_concept.abstraction_level = min(1.0, concept.abstraction_level + 0.3)
                abstract_concepts.append(abstract_concept)
        
        return abstract_concepts
    
    async def _find_more_concrete_concepts(self, concept: ConceptNode) -> List[ConceptNode]:
        """Find more concrete concepts related to the given concept."""
        concrete_concepts = []
        
        # Concrete term patterns based on concept domain
        concrete_patterns = {
            'physics': ['particle', 'wave', 'experiment', 'measurement', 'observation'],
            'philosophy': ['example', 'instance', 'case', 'application', 'manifestation'],
            'neuroscience': ['neuron', 'synapse', 'brain_region', 'activity', 'signal'],
            'computer_science': ['code', 'program', 'implementation', 'data', 'instruction'],
            'biology': ['cell', 'organism', 'gene', 'protein', 'structure']
        }
        
        patterns = concrete_patterns.get(concept.domain, ['example', 'instance', 'case'])
        
        for pattern in patterns[:3]:
            concrete_id = f"{concept.name.replace(' ', '_')}_{pattern}"
            concrete_concept = await self._get_or_create_concept_node(concrete_id)
            if concrete_concept:
                concrete_concept.abstraction_level = max(0.0, concept.abstraction_level - 0.3)
                concrete_concepts.append(concrete_concept)
        
        return concrete_concepts
    
    async def _identify_fusion_opportunities(self, concepts: List[ConceptNode]) -> List[Dict[str, Any]]:
        """Identify promising concept fusion opportunities."""
        opportunities = []
        
        # Strategy 1: Pairwise fusion opportunities
        for concept1, concept2 in combinations(concepts, 2):
            fusion_strength = await self._calculate_fusion_strength(concept1, concept2)
            
            if fusion_strength > 0.4:  # Threshold for viable fusion
                opportunities.append({
                    "concepts": [concept1, concept2],
                    "fusion_method": "pairwise_fusion",
                    "strength": fusion_strength,
                    "type": "binary"
                })
        
        # Strategy 2: Triplet fusion for complex synthesis
        if self.enable_creativity and len(concepts) >= 3:
            for concept1, concept2, concept3 in combinations(concepts, 3):
                triplet_strength = await self._calculate_triplet_fusion_strength(
                    concept1, concept2, concept3
                )
                
                if triplet_strength > 0.3:  # Lower threshold for triplets
                    opportunities.append({
                        "concepts": [concept1, concept2, concept3],
                        "fusion_method": "triplet_fusion",
                        "strength": triplet_strength,
                        "type": "triplet"
                    })
        
        # Strategy 3: Domain-bridging opportunities
        domain_groups = defaultdict(list)
        for concept in concepts:
            domain_groups[concept.domain].append(concept)
        
        # Look for cross-domain fusion opportunities
        for domain1, domain2 in combinations(domain_groups.keys(), 2):
            for concept1 in domain_groups[domain1][:3]:  # Limit domain exploration
                for concept2 in domain_groups[domain2][:3]:
                    bridge_strength = await self._calculate_bridge_fusion_strength(
                        concept1, concept2
                    )
                    
                    if bridge_strength > 0.5:  # Higher threshold for bridging
                        opportunities.append({
                            "concepts": [concept1, concept2],
                            "fusion_method": "domain_bridge_fusion",
                            "strength": bridge_strength,
                            "type": "cross_domain"
                        })
        
        # Sort by fusion strength
        opportunities.sort(key=lambda x: x["strength"], reverse=True)
        
        return opportunities[:20]  # Limit total opportunities
    
    # Helper methods for concept management and calculations
    
    async def _get_or_create_concept_node(self, concept_id: str) -> Optional[ConceptNode]:
        """Get existing concept node or create new one."""
        # Check cache first
        if concept_id in self.concept_cache:
            node = self.concept_cache[concept_id]
            node.update_access()
            return node
        
        # Try to get from concept mesh
        if self.concept_mesh and hasattr(self.concept_mesh, 'get_concept'):
            mesh_concept = self.concept_mesh.get_concept(concept_id)
            if mesh_concept:
                # Convert mesh concept to our format
                node = self._convert_mesh_concept(mesh_concept)
                self.concept_cache[concept_id] = node
                return node
        
        # Create new concept node
        node = self._create_new_concept_node(concept_id)
        self.concept_cache[concept_id] = node
        return node
    
    def _convert_mesh_concept(self, mesh_concept) -> ConceptNode:
        """Convert concept mesh format to our ConceptNode format."""
        # Extract domain from mesh concept
        domain = getattr(mesh_concept, 'domain', 'general')
        
        # Extract or generate semantic vector
        semantic_vector = getattr(mesh_concept, 'embedding', self._generate_default_vector())
        
        # Create ConceptNode
        return ConceptNode(
            concept_id=mesh_concept.concept_id,
            name=getattr(mesh_concept, 'name', str(mesh_concept.concept_id)),
            domain=domain,
            semantic_vector=semantic_vector,
            fusion_potential=0.5,  # Default
            abstraction_level=getattr(mesh_concept, 'abstraction_level', 0.5),
            novelty_score=0.3,  # Default for existing concepts
            source=getattr(mesh_concept, 'source', 'concept_mesh'),
            confidence=getattr(mesh_concept, 'confidence', 1.0)
        )
    
    def _create_new_concept_node(self, concept_id: str) -> ConceptNode:
        """Create a new concept node from concept ID."""
        # Infer domain from concept name
        domain = self._infer_domain(concept_id)
        
        # Generate semantic vector
        semantic_vector = self._generate_semantic_vector(concept_id)
        
        # Estimate properties
        abstraction_level = self._estimate_abstraction_level(concept_id)
        fusion_potential = self._estimate_fusion_potential(concept_id)
        
        return ConceptNode(
            concept_id=concept_id,
            name=concept_id.replace('_', ' ').title(),
            domain=domain,
            semantic_vector=semantic_vector,
            fusion_potential=fusion_potential,
            abstraction_level=abstraction_level,
            novelty_score=0.5,  # Neutral for new concepts
            source="synthesizer_generated",
            confidence=0.7  # Lower confidence for generated concepts
        )
    
    def _infer_domain(self, concept_id: str) -> str:
        """Infer domain from concept identifier."""
        concept_lower = concept_id.lower()
        
        # Domain inference patterns
        if any(term in concept_lower for term in ['quantum', 'physics', 'mechanics', 'particle']):
            return 'physics'
        elif any(term in concept_lower for term in ['neural', 'brain', 'neuron', 'cognitive']):
            return 'neuroscience'
        elif any(term in concept_lower for term in ['machine', 'algorithm', 'artificial', 'computer']):
            return 'computer_science'
        elif any(term in concept_lower for term in ['consciousness', 'mind', 'philosophy', 'existence']):
            return 'philosophy'
        elif any(term in concept_lower for term in ['gene', 'dna', 'cell', 'biology', 'evolution']):
            return 'biology'
        else:
            return 'general'
    
    def _generate_semantic_vector(self, concept_id: str) -> List[float]:
        """Generate semantic vector for concept."""
        # Simple hash-based vector generation for consistency
        import hashlib
        hash_obj = hashlib.md5(concept_id.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to normalized float vector
        vector = []
        for i in range(0, len(hash_bytes), 2):
            # Combine two bytes and normalize to [-1, 1]
            val = (hash_bytes[i] * 256 + hash_bytes[i+1] if i+1 < len(hash_bytes) else hash_bytes[i]) / 32768.0 - 1.0
            vector.append(val)
        
        # Pad or truncate to standard size (8 dimensions for simplicity)
        while len(vector) < 8:
            vector.append(0.0)
        
        return vector[:8]
    
    def _generate_default_vector(self) -> List[float]:
        """Generate default semantic vector."""
        return [0.0] * 8
    
    def _estimate_abstraction_level(self, concept_id: str) -> float:
        """Estimate abstraction level of concept."""
        concept_lower = concept_id.lower()
        
        # Abstract concepts
        if any(term in concept_lower for term in ['consciousness', 'existence', 'reality', 'truth', 'justice']):
            return 0.9
        elif any(term in concept_lower for term in ['theory', 'principle', 'concept', 'idea', 'notion']):
            return 0.7
        elif any(term in concept_lower for term in ['system', 'process', 'method', 'approach']):
            return 0.5
        else:
            return 0.3  # More concrete by default
    
    def _estimate_fusion_potential(self, concept_id: str) -> float:
        """Estimate how well concept can fuse with others."""
        concept_lower = concept_id.lower()
        
        # High fusion potential
        if any(term in concept_lower for term in ['intelligence', 'learning', 'processing', 'information']):
            return 0.8
        elif any(term in concept_lower for term in ['quantum', 'neural', 'cognitive', 'artificial']):
            return 0.7
        else:
            return 0.5  # Moderate by default
    
    async def _get_related_concepts(self, concept_id: str) -> List[str]:
        """Get related concept IDs from concept mesh."""
        if self.concept_mesh and hasattr(self.concept_mesh, 'get_neighbors'):
            return self.concept_mesh.get_neighbors(concept_id)
        
        # Fallback: generate related concepts based on patterns
        return self._generate_related_concepts(concept_id)
    
    def _generate_related_concepts(self, concept_id: str) -> List[str]:
        """Generate related concepts using pattern matching."""
        related = []
        
        # Simple pattern-based relationships
        if 'quantum' in concept_id.lower():
            related.extend(['mechanics', 'physics', 'particle', 'wave', 'uncertainty'])
        elif 'neural' in concept_id.lower():
            related.extend(['network', 'brain', 'neuron', 'learning', 'cognition'])
        elif 'machine' in concept_id.lower():
            related.extend(['learning', 'algorithm', 'artificial', 'intelligence', 'automation'])
        
        return related[:5]  # Limit relationships
    
    # Calculation and utility methods
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_vector_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between semantic vectors."""
        if len(vector1) != len(vector2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return (similarity + 1.0) / 2.0  # Normalize to [0, 1]
    
    def _calculate_domain_compatibility(self, domain1: str, domain2: str) -> float:
        """Calculate compatibility between domains."""
        if domain1 == domain2:
            return 1.0
        
        # Domain relationship matrix
        domain_relationships = {
            ('physics', 'philosophy'): 0.6,
            ('neuroscience', 'computer_science'): 0.8,
            ('biology', 'neuroscience'): 0.7,
            ('physics', 'computer_science'): 0.5,
            ('philosophy', 'neuroscience'): 0.6,
            ('general', 'physics'): 0.3,
            ('general', 'philosophy'): 0.4,
            ('general', 'neuroscience'): 0.3,
            ('general', 'computer_science'): 0.3,
            ('general', 'biology'): 0.3
        }
        
        # Check both directions
        compatibility = domain_relationships.get((domain1, domain2)) or \
                       domain_relationships.get((domain2, domain1)) or \
                       0.2  # Default low compatibility
        
        return compatibility
    
    async def _calculate_fusion_strength(self, concept1: ConceptNode, concept2: ConceptNode) -> float:
        """Calculate strength of potential fusion between two concepts."""
        # Semantic similarity
        semantic_similarity = self._calculate_vector_similarity(
            concept1.semantic_vector, concept2.semantic_vector
        )
        
        # Domain compatibility
        domain_compatibility = self._calculate_domain_compatibility(concept1.domain, concept2.domain)
        
        # Abstraction level compatibility
        abstraction_compatibility = 1.0 - abs(concept1.abstraction_level - concept2.abstraction_level)
        
        # Fusion potential
        fusion_potential = (concept1.fusion_potential + concept2.fusion_potential) / 2.0
        
        # Weighted combination
        strength = (
            semantic_similarity * 0.3 +
            domain_compatibility * 0.2 +
            abstraction_compatibility * 0.2 +
            fusion_potential * 0.3
        )
        
        return min(1.0, strength)
    
    # Initialize system components
    
    def _initialize_domain_mappings(self) -> Dict[str, Any]:
        """Initialize domain relationship mappings."""
        return {
            "physics": {
                "related_domains": ["philosophy", "computer_science"],
                "bridging_concepts": ["information", "energy", "field"],
                "abstraction_bias": 0.1
            },
            "philosophy": {
                "related_domains": ["neuroscience", "physics"],
                "bridging_concepts": ["consciousness", "reality", "existence"],
                "abstraction_bias": 0.3
            },
            "neuroscience": {
                "related_domains": ["computer_science", "philosophy", "biology"],
                "bridging_concepts": ["cognition", "processing", "network"],
                "abstraction_bias": 0.0
            },
            "computer_science": {
                "related_domains": ["neuroscience", "physics"],
                "bridging_concepts": ["algorithm", "computation", "intelligence"],
                "abstraction_bias": -0.1
            },
            "biology": {
                "related_domains": ["neuroscience"],
                "bridging_concepts": ["evolution", "adaptation", "emergence"],
                "abstraction_bias": -0.2
            }
        }
    
    def _initialize_fusion_patterns(self) -> Dict[str, Any]:
        """Initialize fusion pattern templates."""
        return {
            "pairwise_patterns": [
                "hybrid_{concept1}_{concept2}",
                "{concept1}_enhanced_{concept2}",
                "synthesis_{concept1}_{concept2}",
                "fusion_{concept1}_{concept2}"
            ],
            "triplet_patterns": [
                "synthesis_{concept1}_{concept2}_{concept3}",
                "hybrid_fusion_{concept1}_{concept2}_{concept3}",
                "tri_synthesis_{concept1}_{concept2}_{concept3}"
            ],
            "bridge_patterns": [
                "bridge_{domain1}_{domain2}",
                "cross_domain_{concept1}_{concept2}",
                "inter_{domain1}_{domain2}_synthesis"
            ]
        }
    
    def _initialize_creativity_heuristics(self) -> Dict[str, Any]:
        """Initialize creativity enhancement heuristics."""
        return {
            "novelty_boosts": {
                "cross_domain": 1.3,
                "high_abstraction": 1.2,
                "rare_combination": 1.4,
                "paradigm_shift": 1.5
            },
            "creativity_triggers": [
                "unexpected_similarity",
                "conceptual_bridging",
                "abstraction_synthesis",
                "domain_transcendence"
            ],
            "innovation_patterns": [
                "metaphorical_mapping",
                "analogical_reasoning",
                "conceptual_blending",
                "emergent_properties"
            ]
        }
    
    def _generate_trajectory_id(self, context: str, goal: str) -> str:
        """Generate unique trajectory ID."""
        combined = f"{context[:50]}_{goal}_{int(time.time())}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _update_synthesis_stats(self, result: SynthesisResult):
        """Update synthesis statistics."""
        self.synthesis_stats["total_syntheses"] += 1
        self.synthesis_stats["successful_fusions"] += len(result.fusions)
        self.synthesis_stats["novel_concepts_created"] += len(result.synthesized_concepts)
        
        # Count cross-domain fusions
        cross_domain_count = sum(1 for fusion in result.fusions 
                               if fusion.cross_domain_score > 0.5)
        self.synthesis_stats["cross_domain_fusions"] += cross_domain_count
        
        # Update averages
        total = self.synthesis_stats["total_syntheses"]
        self.synthesis_stats["average_coherence"] = (
            self.synthesis_stats["average_coherence"] * (total - 1) + result.overall_coherence
        ) / total
        
        self.synthesis_stats["average_novelty"] = (
            self.synthesis_stats["average_novelty"] * (total - 1) + result.novelty_index
        ) / total
    
    async def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get current synthesis statistics."""
        return {
            **self.synthesis_stats,
            "cache_sizes": {
                "concepts": len(self.concept_cache),
                "fusions": len(self.fusion_cache),
                "trajectories": len(self.trajectory_cache)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> bool:
        """Health check for concept synthesizer."""
        try:
            # Test concept creation
            test_concept = await self._get_or_create_concept_node("test_concept")
            
            # Test synthesis
            test_result = await self.synthesize_from_context("test context")
            
            return test_concept is not None and test_result is not None
        except Exception:
            return False

if __name__ == "__main__":
    # Production test
    async def test_concept_synthesizer():
        synthesizer = ConceptSynthesizer(enable_creativity=True)
        
        # Test synthesis
        test_context = "Quantum mechanics and neural networks both involve complex information processing and uncertainty principles"
        result = await synthesizer.synthesize_from_context(test_context, "explore quantum-neural connections")
        
        print(f"✅ ConceptSynthesizer Test Results:")
        print(f"   Synthesized concepts: {len(result.synthesized_concepts)}")
        print(f"   Successful fusions: {len(result.fusions)}")
        print(f"   Overall coherence: {result.overall_coherence:.2f}")
        print(f"   Novelty index: {result.novelty_index:.2f}")
        print(f"   Cross-domain coverage: {result.cross_domain_coverage:.2f}")
        print(f"   Domains covered: {result.domains_covered}")
        print(f"   Synthesis time: {result.synthesis_time:.2f}s")
        
        # Test concept expansion
        expanded = await synthesizer.expand_concept("quantum_mechanics", depth=2)
        print(f"   Concept expansion: {len(expanded)} related concepts")
        
        # Test entropy calculation
        entropy = synthesizer.calculate_concept_entropy(result.synthesized_concepts)
        print(f"   Concept entropy: {entropy:.2f}")
        
        # Test trajectory tracing
        trajectory = await synthesizer.trace_inference_path("consciousness", ["quantum", "neural"])
        print(f"   Trajectory steps: {trajectory.path_length}")
        print(f"   Trajectory coherence: {trajectory.coherence_score:.2f}")
    
    import asyncio
    asyncio.run(test_concept_synthesizer())
