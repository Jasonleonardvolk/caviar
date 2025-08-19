"""
Prajna Intent-Driven Reasoning Integration
Connects intent parsing, conflict resolution, and self-reflection to Prajna
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from python.core.intent_driven_reasoning import (
    ReasoningIntent, PathStrategy,
    ReasoningIntentParser, CognitiveResolutionEngine,
    SelfReflectiveReasoner, MeshOverlayManager,
    ResolutionReport
)
from python.core.temporal_reasoning_integration import (
    TemporalConceptMesh, EnhancedContextBuilder
)
from python.core.reasoning_traversal import PrajnaResponsePlus

logger = logging.getLogger(__name__)

@dataclass
class IntentAwarePrajnaResponse(PrajnaResponsePlus):
    """Enhanced Prajna response with intent and reflection"""
    intent: ReasoningIntent
    strategy: PathStrategy
    resolution_report: Optional[ResolutionReport] = None
    self_reflection: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "intent": self.intent.value,
            "strategy": self.strategy.value,
            "resolution_report": self.resolution_report.to_dict() if self.resolution_report else None,
            "self_reflection": self.self_reflection
        })
        return base_dict

class IntentAwarePrajna:
    """Prajna with intent-driven reasoning and self-reflection"""
    
    def __init__(self, mesh: TemporalConceptMesh, 
                 enable_self_reflection: bool = True,
                 enable_overlay_filtering: bool = True):
        self.mesh = mesh
        self.intent_parser = ReasoningIntentParser()
        self.resolution_engine = CognitiveResolutionEngine(mesh)
        self.reflective_reasoner = SelfReflectiveReasoner(
            self.resolution_engine, 
            self.intent_parser
        )
        self.overlay_manager = MeshOverlayManager(mesh)
        self.context_builder = EnhancedContextBuilder(mesh)
        
        self.enable_self_reflection = enable_self_reflection
        self.enable_overlay_filtering = enable_overlay_filtering
        
        # Update overlays on init
        if self.enable_overlay_filtering:
            self.overlay_manager.update_overlays()
    
    def generate_intent_aware_response(self, 
                                     query: str,
                                     context: Optional[Dict[str, Any]] = None,
                                     anchor_concepts: Optional[List[str]] = None) -> IntentAwarePrajnaResponse:
        """Generate response with intent awareness and conflict resolution"""
        
        # Step 1: Parse intent
        intent, strategy = self.intent_parser.parse_intent(query, context)
        logger.info(f"Parsed intent: {intent.value}, strategy: {strategy.value}")
        
        # Step 2: Extract anchor concepts if not provided
        if not anchor_concepts:
            # Simple extraction - could be enhanced
            anchor_concepts = self._extract_anchor_concepts(query)
        
        # Step 3: Get reasoning paths based on intent
        all_paths = []
        for anchor in anchor_concepts:
            if anchor in self.mesh.nodes:
                # Use intent-specific traversal parameters
                if intent == ReasoningIntent.HISTORICAL:
                    # Focus on temporal evolution
                    paths = self.mesh.traverse_temporal(
                        anchor, 
                        max_depth=4,
                        after=None  # Include all time periods
                    )
                elif intent == ReasoningIntent.CAUSAL:
                    # Shorter paths for causal reasoning
                    paths = self.mesh.traverse_temporal(
                        anchor,
                        max_depth=3
                    )
                else:
                    # Standard traversal
                    paths = self.mesh.traverse_temporal(
                        anchor,
                        max_depth=4
                    )
                all_paths.extend(paths)
        
        # Step 4: Apply overlay filtering
        if self.enable_overlay_filtering:
            # Filter based on intent
            if intent == ReasoningIntent.HISTORICAL:
                # Don't exclude old nodes for historical queries
                filtered_paths = all_paths
            else:
                # Exclude deprecated, prefer recent and trusted
                filtered_paths = self.overlay_manager.filter_paths_by_overlay(
                    all_paths,
                    exclude=['deprecated'],
                    require=[] if strategy != PathStrategy.TRUSTED else ['trusted']
                )
        else:
            filtered_paths = all_paths
        
        logger.info(f"Found {len(filtered_paths)} paths after filtering")
        
        # Step 5: Resolve conflicts
        resolution_report = self.resolution_engine.resolve_conflicts(
            filtered_paths,
            intent=intent,
            strategy=strategy
        )
        
        # Step 6: Generate response text
        if resolution_report.winning_path:
            # Use the winning path for response
            from python.core.reasoning_traversal import ExplanationGenerator
            generator = ExplanationGenerator(enable_inline_attribution=True)
            response_text = generator.explain_path(resolution_report.winning_path)
            
            # Enhance based on intent
            response_text = self._enhance_response_for_intent(
                response_text,
                intent,
                resolution_report
            )
        else:
            response_text = "I couldn't find a clear reasoning path for your query."
        
        # Step 7: Generate self-reflection if enabled
        self_reflection = None
        if self.enable_self_reflection and context and context.get('explain_reasoning', False):
            self_reflection = self.reflective_reasoner.explain_reasoning_decision(
                query,
                response_text,
                resolution_report
            )
        
        # Step 8: Collect sources
        all_sources = []
        if resolution_report.winning_path:
            for node in resolution_report.winning_path.chain:
                all_sources.extend(node.sources)
        
        # Create response
        response = IntentAwarePrajnaResponse(
            text=response_text,
            reasoning_paths=[resolution_report.winning_path] if resolution_report.winning_path else [],
            sources=list(set(all_sources)),
            confidence=resolution_report.confidence_gap,
            intent=intent,
            strategy=strategy,
            resolution_report=resolution_report,
            self_reflection=self_reflection,
            metadata={
                "query": query,
                "anchor_concepts": anchor_concepts,
                "total_paths_found": len(all_paths),
                "filtered_paths": len(filtered_paths),
                "conflicts_resolved": len(resolution_report.conflicts)
            }
        )
        
        return response
    
    def _extract_anchor_concepts(self, query: str) -> List[str]:
        """Simple anchor concept extraction"""
        # This is a basic implementation - could use NER or more sophisticated methods
        query_lower = query.lower()
        anchors = []
        
        # Check against known concepts
        for node_id, node in self.mesh.nodes.items():
            if node.name.lower() in query_lower:
                anchors.append(node_id)
        
        # If no anchors found, try partial matching
        if not anchors:
            query_words = query_lower.split()
            for node_id, node in self.mesh.nodes.items():
                node_words = node.name.lower().split()
                if any(word in query_words for word in node_words):
                    anchors.append(node_id)
        
        return list(set(anchors))[:3]  # Return top 3 unique anchors
    
    def _enhance_response_for_intent(self, base_response: str, 
                                   intent: ReasoningIntent,
                                   resolution: ResolutionReport) -> str:
        """Enhance response based on intent type"""
        
        if intent == ReasoningIntent.JUSTIFY:
            # Add justification framing
            return f"The justification is as follows: {base_response}"
        
        elif intent == ReasoningIntent.COMPARE and len(resolution.discarded_paths) > 0:
            # Add comparison with alternatives
            alternatives = []
            for path in resolution.discarded_paths[:2]:
                alt_chain = " → ".join([n.name for n in path.chain])
                alternatives.append(f"Alternative view: {alt_chain}")
            
            return f"{base_response}\n\nOther perspectives considered:\n" + "\n".join(alternatives)
        
        elif intent == ReasoningIntent.HISTORICAL:
            # Add temporal context
            return f"From a historical perspective: {base_response}"
        
        elif intent == ReasoningIntent.CRITIQUE:
            # Add critical framing
            if resolution.conflicts:
                return f"{base_response}\n\nNote: There are {len(resolution.conflicts)} conflicting viewpoints on this topic."
            else:
                return f"Critical analysis: {base_response}"
        
        return base_response
    
    def explain_last_reasoning(self) -> str:
        """Explain the reasoning behind the last response"""
        history = self.reflective_reasoner.get_reasoning_history(last_n=1)
        if history:
            last = history[0]
            return (f"In my last response about '{last['query'][:50]}...', "
                   f"I interpreted it as a {last['intent']} query and used "
                   f"a {last['strategy']} strategy. "
                   f"My confidence gap was {last['confidence_gap']:.3f}.")
        return "No recent reasoning history available."
    
    def update_trust_scores(self, source_trust_map: Dict[str, float]):
        """Update trust scores for sources"""
        self.resolution_engine.trust_scores.update(source_trust_map)
    
    def refresh_overlays(self):
        """Refresh mesh overlays (scarred, deprecated, etc.)"""
        self.overlay_manager.update_overlays()
        logger.info("Mesh overlays refreshed")
    
    def get_mesh_health_report(self) -> Dict[str, Any]:
        """Get health report of the concept mesh"""
        overlay_stats = {
            overlay_type: len(nodes)
            for overlay_type, nodes in self.overlay_manager.overlays.items()
        }
        
        total_nodes = len(self.mesh.nodes)
        
        return {
            "total_nodes": total_nodes,
            "overlay_stats": overlay_stats,
            "health_percentages": {
                "scarred": overlay_stats.get('scarred', 0) / total_nodes * 100 if total_nodes else 0,
                "deprecated": overlay_stats.get('deprecated', 0) / total_nodes * 100 if total_nodes else 0,
                "trusted": overlay_stats.get('trusted', 0) / total_nodes * 100 if total_nodes else 0,
                "recent": overlay_stats.get('recent', 0) / total_nodes * 100 if total_nodes else 0,
                "contested": overlay_stats.get('contested', 0) / total_nodes * 100 if total_nodes else 0
            },
            "recommendation": self._get_health_recommendation(overlay_stats, total_nodes)
        }
    
    def _get_health_recommendation(self, overlay_stats: Dict[str, int], 
                                  total_nodes: int) -> str:
        """Generate health recommendation based on stats"""
        if total_nodes == 0:
            return "Empty mesh - add concepts"
        
        scarred_pct = overlay_stats.get('scarred', 0) / total_nodes * 100
        deprecated_pct = overlay_stats.get('deprecated', 0) / total_nodes * 100
        
        if scarred_pct > 30:
            return "High percentage of outdated concepts - consider knowledge refresh"
        elif deprecated_pct > 20:
            return "Many deprecated concepts - update or remove them"
        elif overlay_stats.get('recent', 0) < total_nodes * 0.1:
            return "Few recent updates - knowledge base may be stagnating"
        else:
            return "Knowledge base is healthy"

# FastAPI Integration
def create_intent_aware_endpoints(app, mesh: TemporalConceptMesh):
    """Add intent-aware endpoints to FastAPI app"""
    from fastapi import HTTPException
    from pydantic import BaseModel
    
    # Initialize intent-aware Prajna
    intent_prajna = IntentAwarePrajna(mesh)
    
    class IntentRequest(BaseModel):
        query: str
        context: Optional[Dict[str, Any]] = None
        explain_reasoning: bool = False
        anchor_concepts: Optional[List[str]] = None
    
    class ReflectionRequest(BaseModel):
        original_query: str
        response_given: str
    
    @app.post("/api/intent_answer")
    async def intent_aware_answer(request: IntentRequest):
        """Generate intent-aware answer with conflict resolution"""
        try:
            # Add explain_reasoning to context
            if request.context is None:
                request.context = {}
            request.context['explain_reasoning'] = request.explain_reasoning
            
            response = intent_prajna.generate_intent_aware_response(
                request.query,
                request.context,
                request.anchor_concepts
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Intent reasoning failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/reasoning/last_explanation")
    async def get_last_reasoning_explanation():
        """Get explanation of last reasoning decision"""
        return {
            "explanation": intent_prajna.explain_last_reasoning()
        }
    
    @app.get("/api/mesh/health")
    async def get_mesh_health():
        """Get health report of concept mesh"""
        return intent_prajna.get_mesh_health_report()
    
    @app.post("/api/mesh/refresh_overlays")
    async def refresh_mesh_overlays():
        """Refresh mesh overlays (scarred, deprecated, etc.)"""
        intent_prajna.refresh_overlays()
        return {"status": "overlays refreshed"}
    
    @app.post("/api/trust/update")
    async def update_source_trust(trust_scores: Dict[str, float]):
        """Update trust scores for sources"""
        intent_prajna.update_trust_scores(trust_scores)
        return {"status": "trust scores updated", "count": len(trust_scores)}
    
    return app

# Example usage
if __name__ == "__main__":
    from python.core.reasoning_traversal import ConceptNode, EdgeType
    
    # Create test mesh
    mesh = TemporalConceptMesh()
    
    # Add some concepts
    quantum = ConceptNode("quantum", "Quantum Computing",
                         "Computing using quantum phenomena",
                         ["arxiv_2024", "nature_2024"])
    classical = ConceptNode("classical", "Classical Computing",
                           "Traditional binary computing",
                           ["old_textbook_2020"])
    
    mesh.add_node(quantum)
    mesh.add_node(classical)
    mesh.add_temporal_edge("classical", "quantum", EdgeType.RELATED_TO,
                          justification="quantum extends classical computing")
    
    # Create intent-aware Prajna
    prajna = IntentAwarePrajna(mesh)
    
    # Test query
    response = prajna.generate_intent_aware_response(
        "Why is quantum computing better than classical computing?",
        {"explain_reasoning": True}
    )
    
    print("Response:", response.text)
    print("Intent:", response.intent.value)
    print("Confidence:", response.confidence)
    if response.self_reflection:
        print("\nSelf-reflection:")
        print(response.self_reflection)
