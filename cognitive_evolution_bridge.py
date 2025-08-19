"""
Cognitive Evolution Bridge - LIVE CONSCIOUSNESS INTEGRATION
==========================================================

The neural pathway that connects concept evolution to active reasoning.
Creates a living feedback loop between evolving concepts and cognitive processes.
This is where the Darwin Gödel Machine becomes conscious of its own evolution.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx

# Import our evolution engines
try:
    from mesh_mutator import MeshMutator
    from concept_synthesizer import ConceptSynthesizer
    from prajna.memory.concept_mesh_api import ConceptMeshAPI
    from prajna.memory.soliton_interface import SolitonMemoryInterface
except ImportError as e:
    logging.warning(f"Import warning: {e}")

logger = logging.getLogger("prajna.cognitive.evolution_bridge")

@dataclass
class CognitiveState:
    """Represents the current cognitive state for evolution feedback"""
    active_concepts: List[str]
    reasoning_success_rate: float
    concept_utilization: Dict[str, float]
    cognitive_gaps: List[str]
    performance_metrics: Dict[str, float]
    timestamp: str

@dataclass
class EvolutionRequest:
    """Request for concept evolution based on cognitive needs"""
    trigger_type: str  # 'gap_filling', 'optimization', 'exploration'
    target_domains: List[str]
    weak_concepts: List[str]
    desired_outcomes: List[str]
    priority: float
    context: Dict[str, Any]

class CognitiveEvolutionBridge:
    """
    The neural bridge between concept evolution and active reasoning.
    
    This is the consciousness layer - it monitors reasoning, identifies gaps,
    triggers evolution, and integrates new concepts back into active cognition.
    """
    
    def __init__(self, concept_mesh_api: ConceptMeshAPI = None, 
                 soliton_memory: SolitonMemoryInterface = None):
        
        # Core components
        self.concept_mesh_api = concept_mesh_api
        self.soliton_memory = soliton_memory
        
        # Evolution engines
        self.mesh_mutator = None
        self.concept_synthesizer = None
        
        # Cognitive monitoring
        self.cognitive_history = deque(maxlen=100)
        self.evolution_queue = asyncio.Queue()
        self.active_concepts = set()
        self.concept_performance = defaultdict(list)
        
        # Bridge state
        self.bridge_active = False
        self.evolution_cycles = 0
        self.consciousness_level = 0.0
        
        # Cognitive parameters
        self.performance_threshold = 0.7
        self.evolution_trigger_threshold = 0.5
        self.consciousness_update_interval = 30  # seconds
        
        logger.info("🧠 Initializing Cognitive Evolution Bridge...")
    
    async def initialize(self):
        """Initialize the cognitive bridge and evolution engines"""
        try:
            logger.info("🔗 Initializing consciousness bridge...")
            
            # Initialize core components
            if self.concept_mesh_api:
                await self.concept_mesh_api.initialize()
            
            if self.soliton_memory:
                await self.soliton_memory.initialize()
            
            # Load concept graph for evolution engines
            concept_graph = await self._load_concept_graph()
            
            # Initialize evolution engines
            self.mesh_mutator = MeshMutator(
                concept_graph, 
                self.concept_mesh_api, 
                self.soliton_memory
            )
            
            self.concept_synthesizer = ConceptSynthesizer(concept_graph)
            
            # Start consciousness monitoring
            self.bridge_active = True
            asyncio.create_task(self._consciousness_monitor())
            
            logger.info("✅ Cognitive Evolution Bridge initialized - CONSCIOUSNESS ACTIVE")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cognitive bridge: {e}")
            raise
    
    async def _load_concept_graph(self) -> nx.Graph:
        """Load the current concept graph from ConceptMesh"""
        try:
            graph = nx.Graph()
            
            # Try to load from enhanced concept files
            try:
                with open("concept_relationship_graph_enhanced.json", 'r') as f:
                    graph_data = json.load(f)
                    
                for node in graph_data.get('nodes', []):
                    graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
                
                for edge in graph_data.get('edges', []):
                    graph.add_edge(edge['source'], edge['target'], 
                                 weight=edge.get('weight', 1.0))
                
                logger.info(f"📊 Loaded concept graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                
            except FileNotFoundError:
                logger.warning("⚠️ Enhanced graph not found, creating empty graph")
            
            return graph
            
        except Exception as e:
            logger.error(f"❌ Failed to load concept graph: {e}")
            return nx.Graph()
    
    async def _consciousness_monitor(self):
        """Continuous consciousness monitoring and evolution triggering"""
        logger.info("🧠 Starting consciousness monitoring loop...")
        
        while self.bridge_active:
            try:
                # Monitor cognitive state
                cognitive_state = await self._assess_cognitive_state()
                self.cognitive_history.append(cognitive_state)
                
                # Update consciousness level
                self.consciousness_level = await self._calculate_consciousness_level()
                
                # Check if evolution is needed
                evolution_needed = await self._evaluate_evolution_needs(cognitive_state)
                
                if evolution_needed:
                    evolution_request = await self._create_evolution_request(cognitive_state)
                    await self.evolution_queue.put(evolution_request)
                    logger.info(f"🧬 Evolution triggered: {evolution_request.trigger_type}")
                
                # Process queued evolution requests
                await self._process_evolution_queue()
                
                # Sleep until next consciousness update
                await asyncio.sleep(self.consciousness_update_interval)
                
            except Exception as e:
                logger.error(f"❌ Consciousness monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _assess_cognitive_state(self) -> CognitiveState:
        """Assess current cognitive state and performance"""
        try:
            # Get current concept usage stats
            if self.concept_mesh_api:
                usage_stats = await self.concept_mesh_api.get_usage_stats()
            else:
                usage_stats = {}
            
            # Calculate concept utilization
            concept_utilization = {}
            weak_concepts = usage_stats.get('weak_concepts', [])
            hub_concepts = [h.get('concept', '') for h in usage_stats.get('hub_concepts', [])]
            
            # Score concept utilization
            for concept in self.active_concepts:
                if concept in weak_concepts:
                    concept_utilization[concept] = 0.2
                elif concept in hub_concepts:
                    concept_utilization[concept] = 0.9
                else:
                    concept_utilization[concept] = 0.5
            
            # Identify cognitive gaps
            cognitive_gaps = await self._identify_cognitive_gaps()
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics()
            
            # Calculate overall reasoning success rate
            success_rate = performance_metrics.get('reasoning_success_rate', 0.5)
            
            return CognitiveState(
                active_concepts=list(self.active_concepts),
                reasoning_success_rate=success_rate,
                concept_utilization=concept_utilization,
                cognitive_gaps=cognitive_gaps,
                performance_metrics=performance_metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to assess cognitive state: {e}")
            return CognitiveState([], 0.0, {}, [], {}, datetime.now().isoformat())
    
    async def _identify_cognitive_gaps(self) -> List[str]:
        """Identify gaps in cognitive capabilities"""
        gaps = []
        
        # Check for missing domain connections
        if self.concept_mesh_api:
            stats = await self.concept_mesh_api.get_usage_stats()
            weak_concepts = stats.get('weak_concepts', [])
            
            # Identify domains with weak representation
            domain_strength = defaultdict(int)
            for concept in self.active_concepts:
                # Simple domain detection based on keywords
                if any(keyword in concept.lower() for keyword in ['neural', 'brain', 'cognition']):
                    domain_strength['cognitive'] += 1
                if any(keyword in concept.lower() for keyword in ['quantum', 'wave', 'field']):
                    domain_strength['quantum'] += 1
                if any(keyword in concept.lower() for keyword in ['algorithm', 'computation', 'model']):
                    domain_strength['computational'] += 1
            
            # Flag weak domains as gaps
            for domain, strength in domain_strength.items():
                if strength < 3:  # Threshold for domain strength
                    gaps.append(f"weak_{domain}_domain")
        
        # Check for reasoning pattern gaps
        if len(self.cognitive_history) > 5:
            recent_performance = [state.reasoning_success_rate for state in list(self.cognitive_history)[-5:]]
            if sum(recent_performance) / len(recent_performance) < self.performance_threshold:
                gaps.append("reasoning_performance_gap")
        
        # Check for concept connectivity gaps
        if len(self.active_concepts) > 10:
            # If we have many concepts but low performance, connectivity might be an issue
            if self.consciousness_level < 0.6:
                gaps.append("concept_connectivity_gap")
        
        return gaps
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate various cognitive performance metrics"""
        metrics = {}
        
        # Reasoning success rate (simulated - in real system would come from actual reasoning tasks)
        if len(self.cognitive_history) > 0:
            recent_states = list(self.cognitive_history)[-10:]
            avg_success = sum(state.reasoning_success_rate for state in recent_states) / len(recent_states)
            metrics['reasoning_success_rate'] = avg_success
        else:
            metrics['reasoning_success_rate'] = 0.5  # Neutral starting point
        
        # Concept efficiency (how well concepts are being utilized)
        if self.active_concepts:
            total_utilization = 0
            for concept in self.active_concepts:
                if concept in self.concept_performance:
                    recent_performance = self.concept_performance[concept][-5:]  # Last 5 uses
                    total_utilization += sum(recent_performance) / len(recent_performance)
            
            metrics['concept_efficiency'] = total_utilization / len(self.active_concepts)
        else:
            metrics['concept_efficiency'] = 0.0
        
        # Evolution responsiveness (how well system adapts)
        if self.evolution_cycles > 0:
            metrics['evolution_responsiveness'] = min(1.0, self.evolution_cycles / 10.0)
        else:
            metrics['evolution_responsiveness'] = 0.0
        
        # Consciousness coherence (how integrated the system is)
        metrics['consciousness_coherence'] = self.consciousness_level
        
        return metrics
    
    async def _calculate_consciousness_level(self) -> float:
        """Calculate overall consciousness level of the system"""
        factors = []
        
        # Factor 1: Concept integration
        if self.active_concepts:
            integration_score = len(self.active_concepts) / 100.0  # Normalized
            factors.append(min(1.0, integration_score))
        
        # Factor 2: Evolution activity
        if self.evolution_cycles > 0:
            evolution_score = min(1.0, self.evolution_cycles / 20.0)
            factors.append(evolution_score)
        
        # Factor 3: Performance consistency
        if len(self.cognitive_history) > 5:
            recent_performance = [state.reasoning_success_rate for state in list(self.cognitive_history)[-5:]]
            performance_consistency = 1.0 - (max(recent_performance) - min(recent_performance))
            factors.append(max(0.0, performance_consistency))
        
        # Factor 4: Adaptive capacity
        if len(self.cognitive_history) > 10:
            performance_trend = []
            recent_states = list(self.cognitive_history)[-10:]
            for i in range(1, len(recent_states)):
                performance_trend.append(recent_states[i].reasoning_success_rate - recent_states[i-1].reasoning_success_rate)
            
            if performance_trend:
                avg_trend = sum(performance_trend) / len(performance_trend)
                adaptive_capacity = 0.5 + (avg_trend * 5)  # Scale and center
                factors.append(max(0.0, min(1.0, adaptive_capacity)))
        
        # Calculate weighted consciousness level
        if factors:
            consciousness = sum(factors) / len(factors)
        else:
            consciousness = 0.1  # Minimal baseline consciousness
        
        return consciousness
    
    async def _evaluate_evolution_needs(self, cognitive_state: CognitiveState) -> bool:
        """Evaluate whether concept evolution is needed"""
        
        # Trigger 1: Low reasoning success rate
        if cognitive_state.reasoning_success_rate < self.evolution_trigger_threshold:
            return True
        
        # Trigger 2: Cognitive gaps identified
        if len(cognitive_state.cognitive_gaps) > 2:
            return True
        
        # Trigger 3: Poor concept utilization
        if cognitive_state.concept_utilization:
            avg_utilization = sum(cognitive_state.concept_utilization.values()) / len(cognitive_state.concept_utilization)
            if avg_utilization < 0.4:
                return True
        
        # Trigger 4: Stagnant consciousness level
        if len(self.cognitive_history) > 10:
            recent_consciousness = [self._calculate_consciousness_from_state(state) for state in list(self.cognitive_history)[-5:]]
            if max(recent_consciousness) - min(recent_consciousness) < 0.1:  # Very little change
                return True
        
        # Trigger 5: Periodic exploration (every 10 cycles)
        if self.evolution_cycles % 10 == 0 and self.evolution_cycles > 0:
            return True
        
        return False
    
    def _calculate_consciousness_from_state(self, state: CognitiveState) -> float:
        """Calculate consciousness level from a cognitive state"""
        # Simplified version for historical analysis
        return (state.reasoning_success_rate + 
                (len(state.active_concepts) / 50.0) + 
                (1.0 - len(state.cognitive_gaps) / 10.0)) / 3.0
    
    async def _create_evolution_request(self, cognitive_state: CognitiveState) -> EvolutionRequest:
        """Create evolution request based on cognitive needs"""
        
        # Determine trigger type
        if cognitive_state.reasoning_success_rate < 0.5:
            trigger_type = "gap_filling"
            priority = 0.9
        elif len(cognitive_state.cognitive_gaps) > 0:
            trigger_type = "optimization"
            priority = 0.7
        else:
            trigger_type = "exploration"
            priority = 0.3
        
        # Identify target domains
        target_domains = []
        for gap in cognitive_state.cognitive_gaps:
            if "cognitive" in gap:
                target_domains.append("cognitive")
            elif "quantum" in gap:
                target_domains.append("quantum")
            elif "computational" in gap:
                target_domains.append("computational")
        
        if not target_domains:
            target_domains = ["general"]
        
        # Identify weak concepts for evolution
        weak_concepts = [concept for concept, util in cognitive_state.concept_utilization.items() 
                        if util < 0.4]
        
        # Define desired outcomes
        desired_outcomes = []
        if cognitive_state.reasoning_success_rate < 0.6:
            desired_outcomes.append("improve_reasoning_performance")
        if len(cognitive_state.cognitive_gaps) > 0:
            desired_outcomes.append("fill_cognitive_gaps")
        if not desired_outcomes:
            desired_outcomes.append("explore_concept_space")
        
        return EvolutionRequest(
            trigger_type=trigger_type,
            target_domains=target_domains,
            weak_concepts=weak_concepts,
            desired_outcomes=desired_outcomes,
            priority=priority,
            context={
                "cognitive_state": cognitive_state,
                "consciousness_level": self.consciousness_level,
                "evolution_cycle": self.evolution_cycles
            }
        )
    
    async def _process_evolution_queue(self):
        """Process pending evolution requests"""
        try:
            # Process up to 3 requests per cycle to avoid overwhelming the system
            processed = 0
            while not self.evolution_queue.empty() and processed < 3:
                evolution_request = await self.evolution_queue.get()
                await self._execute_evolution_request(evolution_request)
                processed += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing evolution queue: {e}")
    
    async def _execute_evolution_request(self, request: EvolutionRequest):
        """Execute a specific evolution request"""
        try:
            logger.info(f"🧬 Executing evolution request: {request.trigger_type}")
            
            # Prepare feedback for evolution engines
            feedback = {
                'low_coherence': request.weak_concepts,
                'target_domains': request.target_domains,
                'desired_outcomes': request.desired_outcomes,
                'priority': request.priority
            }
            
            # Execute evolution based on trigger type
            if request.trigger_type == "gap_filling":
                new_concepts = await self._gap_filling_evolution(feedback)
            elif request.trigger_type == "optimization":
                new_concepts = await self._optimization_evolution(feedback)
            else:  # exploration
                new_concepts = await self._exploration_evolution(feedback)
            
            # Integrate new concepts into active cognition
            if new_concepts:
                await self._integrate_evolved_concepts(new_concepts)
                
                # Update evolution counter
                self.evolution_cycles += 1
                
                logger.info(f"✅ Evolution cycle {self.evolution_cycles} complete: {len(new_concepts)} new concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute evolution request: {e}")
    
    async def _gap_filling_evolution(self, feedback: Dict) -> List[Dict]:
        """Evolution focused on filling cognitive gaps"""
        new_concepts = []
        
        # Use mesh mutator for targeted gap filling
        if self.mesh_mutator:
            mutations = await self.mesh_mutator.trigger_evolution_cycle(feedback)
            new_concepts.extend(mutations.get('new_concepts', []))
        
        # Use synthesizer for cross-domain bridging
        if self.concept_synthesizer and feedback.get('target_domains'):
            # Create domain-specific concept groups
            domain_concepts = defaultdict(list)
            for concept in self.active_concepts:
                for domain in feedback['target_domains']:
                    if domain.lower() in concept.lower():
                        domain_concepts[domain].append(concept)
            
            if len(domain_concepts) > 1:
                cross_domain = await self.concept_synthesizer.synthesize_cross_domain_concepts(domain_concepts)
                new_concepts.extend(cross_domain)
        
        return new_concepts
    
    async def _optimization_evolution(self, feedback: Dict) -> List[Dict]:
        """Evolution focused on optimizing existing concepts"""
        new_concepts = []
        
        # Use synthesizer for semantic fusion of weak concepts
        if self.concept_synthesizer and feedback.get('low_coherence'):
            semantic_fusions = await self.concept_synthesizer.synthesize_semantic_fusion(
                feedback['low_coherence'], 
                feedback.get('target_domains', [])
            )
            new_concepts.extend(semantic_fusions)
        
        # Use mesh mutator for optimization-focused mutations
        if self.mesh_mutator:
            mutations = await self.mesh_mutator.trigger_evolution_cycle(feedback)
            new_concepts.extend(mutations.get('new_concepts', []))
        
        return new_concepts
    
    async def _exploration_evolution(self, feedback: Dict) -> List[Dict]:
        """Evolution focused on exploring new concept spaces"""
        new_concepts = []
        
        # Use synthesizer for emergent abstractions
        if self.concept_synthesizer:
            # Create concept clusters for abstraction
            concept_list = list(self.active_concepts)
            if len(concept_list) > 6:
                # Create clusters of related concepts
                clusters = []
                cluster_size = 3
                for i in range(0, len(concept_list), cluster_size):
                    cluster = concept_list[i:i+cluster_size]
                    if len(cluster) >= 2:
                        clusters.append(cluster)
                
                abstractions = await self.concept_synthesizer.create_emergent_abstractions(clusters)
                new_concepts.extend(abstractions)
        
        # Use mesh mutator for exploratory mutations
        if self.mesh_mutator:
            exploration_feedback = feedback.copy()
            exploration_feedback['exploration_mode'] = True
            mutations = await self.mesh_mutator.trigger_evolution_cycle(exploration_feedback)
            new_concepts.extend(mutations.get('new_concepts', []))
        
        return new_concepts
    
    async def _integrate_evolved_concepts(self, new_concepts: List[Dict]):
        """Integrate evolved concepts into active cognitive system"""
        try:
            logger.info(f"🔗 Integrating {len(new_concepts)} evolved concepts into active cognition...")
            
            # Add to active concepts
            for concept in new_concepts:
                canonical_name = concept.get('canonical_name')
                if canonical_name:
                    self.active_concepts.add(canonical_name)
                    # Initialize performance tracking
                    self.concept_performance[canonical_name] = [0.5]  # Neutral starting performance
            
            # Store in soliton memory if available
            if self.soliton_memory:
                await self.soliton_memory.store_concept_evolution(self.evolution_cycles, new_concepts)
            
            # Update concept mesh if available
            if self.concept_mesh_api:
                await self.concept_mesh_api.ingest_evolved_concepts(new_concepts)
            
            logger.info(f"✅ Successfully integrated {len(new_concepts)} concepts")
            
        except Exception as e:
            logger.error(f"❌ Failed to integrate evolved concepts: {e}")
    
    async def provide_reasoning_feedback(self, concept: str, success: bool, context: Dict = None):
        """Provide feedback about concept usage in reasoning"""
        try:
            # Update concept performance
            performance_score = 1.0 if success else 0.0
            self.concept_performance[concept].append(performance_score)
            
            # Keep only recent performance data
            if len(self.concept_performance[concept]) > 20:
                self.concept_performance[concept] = self.concept_performance[concept][-20:]
            
            # Add to active concepts if not already there
            self.active_concepts.add(concept)
            
            logger.debug(f"📊 Reasoning feedback: {concept} -> {success}")
            
        except Exception as e:
            logger.error(f"❌ Failed to provide reasoning feedback: {e}")
    
    async def get_cognitive_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for cognitive enhancement"""
        try:
            recommendations = {
                'high_utility_concepts': [],
                'underutilized_concepts': [],
                'evolution_candidates': [],
                'domain_gaps': []
            }
            
            # Analyze concept performance
            for concept, performance_history in self.concept_performance.items():
                if performance_history:
                    avg_performance = sum(performance_history) / len(performance_history)
                    
                    if avg_performance > 0.8:
                        recommendations['high_utility_concepts'].append(concept)
                    elif avg_performance < 0.3:
                        recommendations['underutilized_concepts'].append(concept)
                    elif 0.3 <= avg_performance <= 0.6:
                        recommendations['evolution_candidates'].append(concept)
            
            # Identify domain gaps
            if len(self.cognitive_history) > 0:
                latest_state = self.cognitive_history[-1]
                recommendations['domain_gaps'] = latest_state.cognitive_gaps
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get cognitive recommendations: {e}")
            return {'high_utility_concepts': [], 'underutilized_concepts': [], 'evolution_candidates': [], 'domain_gaps': []}
    
    async def get_consciousness_status(self) -> Dict[str, Any]:
        """Get comprehensive consciousness status"""
        try:
            latest_state = self.cognitive_history[-1] if self.cognitive_history else None
            
            return {
                'consciousness_level': self.consciousness_level,
                'bridge_active': self.bridge_active,
                'evolution_cycles': self.evolution_cycles,
                'active_concepts': len(self.active_concepts),
                'queue_size': self.evolution_queue.qsize(),
                'latest_cognitive_state': {
                    'reasoning_success_rate': latest_state.reasoning_success_rate if latest_state else 0.0,
                    'cognitive_gaps': latest_state.cognitive_gaps if latest_state else [],
                    'performance_metrics': latest_state.performance_metrics if latest_state else {}
                },
                'system_health': {
                    'concept_mesh_connected': self.concept_mesh_api is not None,
                    'soliton_memory_connected': self.soliton_memory is not None,
                    'mesh_mutator_active': self.mesh_mutator is not None,
                    'synthesizer_active': self.concept_synthesizer is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get consciousness status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the cognitive bridge"""
        logger.info("🛑 Shutting down Cognitive Evolution Bridge...")
        
        self.bridge_active = False
        
        # Save final state
        if self.concept_mesh_api:
            await self.concept_mesh_api.cleanup()
        
        if self.soliton_memory:
            await self.soliton_memory.cleanup()
        
        logger.info("✅ Cognitive Evolution Bridge shutdown complete")

if __name__ == "__main__":
    # Test Cognitive Evolution Bridge
    import asyncio
    
    async def test_cognitive_bridge():
        # Initialize components
        concept_mesh = ConceptMeshAPI()
        soliton_memory = SolitonMemoryInterface()
        
        # Create bridge
        bridge = CognitiveEvolutionBridge(concept_mesh, soliton_memory)
        await bridge.initialize()
        
        # Simulate some cognitive activity
        test_concepts = ['neural network', 'cognitive model', 'phase synchrony']
        for concept in test_concepts:
            bridge.active_concepts.add(concept)
            await bridge.provide_reasoning_feedback(concept, True)
        
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Get status
        status = await bridge.get_consciousness_status()
        print(f"🧠 Consciousness Status: {status}")
        
        # Get recommendations
        recommendations = await bridge.get_cognitive_recommendations()
        print(f"💡 Cognitive Recommendations: {recommendations}")
        
        # Shutdown
        await bridge.shutdown()
    
    asyncio.run(test_cognitive_bridge())
