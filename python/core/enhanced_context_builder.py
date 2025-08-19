"""
Enhanced Context Builder with Reasoning Traversal Integration
Bridges the gap between context building and reasoning chains
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .reasoning_traversal import (
    ConceptMesh, ConceptNode, EdgeType, 
    PrajnaReasoningIntegration, PrajnaResponsePlus
)
from .context_builder import ContextBuilder, QueryContext

logger = logging.getLogger(__name__)

class EnhancedContextBuilder(ContextBuilder):
    """Context builder with reasoning traversal capabilities"""
    
    def __init__(self, mesh: Any = None):
        super().__init__(mesh)
        
        # Initialize reasoning components
        self.concept_mesh = ConceptMesh()
        self.reasoning_integration = PrajnaReasoningIntegration(
            self.concept_mesh,
            enable_inline_attribution=True
        )
        
        # Convert existing mesh to ConceptMesh format if provided
        if mesh:
            self._convert_mesh_to_concept_mesh(mesh)
    
    def _convert_mesh_to_concept_mesh(self, old_mesh: Any):
        """Convert existing mesh format to new ConceptMesh"""
        # This would depend on your existing mesh structure
        # Example conversion:
        try:
            if hasattr(old_mesh, 'nodes'):
                for node_id, node_data in old_mesh.nodes.items():
                    concept_node = ConceptNode(
                        id=str(node_id),
                        name=node_data.get('name', str(node_id)),
                        description=node_data.get('description', ''),
                        sources=node_data.get('sources', [])
                    )
                    self.concept_mesh.add_node(concept_node)
                
                # Convert edges
                if hasattr(old_mesh, 'edges'):
                    for edge in old_mesh.edges:
                        self.concept_mesh.add_edge(
                            from_id=str(edge.get('from')),
                            to_id=str(edge.get('to')),
                            relation=self._map_edge_type(edge.get('type', 'related_to')),
                            weight=edge.get('weight', 1.0)
                        )
        except Exception as e:
            logger.error(f"Failed to convert mesh: {e}")
    
    def _map_edge_type(self, edge_type_str: str) -> EdgeType:
        """Map string edge types to EdgeType enum"""
        mapping = {
            'implies': EdgeType.IMPLIES,
            'supports': EdgeType.SUPPORTS,
            'because': EdgeType.BECAUSE,
            'enables': EdgeType.ENABLES,
            'causes': EdgeType.CAUSES,
            'contradicts': EdgeType.CONTRADICTS,
            'part_of': EdgeType.PART_OF,
            'prevents': EdgeType.PREVENTS
        }
        return mapping.get(edge_type_str.lower(), EdgeType.RELATED_TO)
    
    def build_context_with_reasoning(self, query: str, 
                                   params: Optional[Dict[str, Any]] = None) -> Tuple[QueryContext, PrajnaResponsePlus]:
        """Build context and generate reasoning chains"""
        
        # Step 1: Build traditional context
        context = self.build_context(query, params)
        
        # Step 2: Extract anchor concepts from query
        anchor_concepts = self._extract_anchor_concepts(query, context)
        
        # Step 3: Update concept mesh with context information
        self._update_concept_mesh_from_context(context)
        
        # Step 4: Generate reasoned response
        reasoned_response = self.reasoning_integration.generate_reasoned_response(
            query=query,
            anchor_concepts=anchor_concepts,
            context={
                'query_context': context,
                'parameters': params
            }
        )
        
        # Step 5: Enhance context with reasoning paths
        context.metadata['reasoning_paths'] = [
            path.to_dict() for path in reasoned_response.reasoning_paths
        ]
        context.metadata['reasoning_confidence'] = reasoned_response.confidence
        
        return context, reasoned_response
    
    def _extract_anchor_concepts(self, query: str, context: QueryContext) -> List[str]:
        """Extract anchor concepts from query and context"""
        anchor_concepts = []
        
        # Extract from context nodes if available
        if hasattr(context, 'relevant_nodes'):
            for node in context.relevant_nodes[:3]:  # Top 3 relevant nodes
                if hasattr(node, 'id'):
                    anchor_concepts.append(str(node.id))
        
        # Extract from query keywords
        # This is a simple implementation - could use NER or more sophisticated methods
        query_lower = query.lower()
        for node_id, node in self.concept_mesh.nodes.items():
            if node.name.lower() in query_lower:
                anchor_concepts.append(node_id)
        
        # Deduplicate
        return list(set(anchor_concepts))
    
    def _update_concept_mesh_from_context(self, context: QueryContext):
        """Update concept mesh with information from context"""
        # Add new nodes discovered in context
        if hasattr(context, 'blocks'):
            for i, block in enumerate(context.blocks):
                # Create node for each context block if it contains concepts
                if hasattr(block, 'concepts'):
                    for concept in block.concepts:
                        node_id = f"context_{i}_{concept.get('id', i)}"
                        if node_id not in self.concept_mesh.nodes:
                            node = ConceptNode(
                                id=node_id,
                                name=concept.get('name', f'Concept {i}'),
                                description=concept.get('description', block.text[:100]),
                                sources=[block.source] if hasattr(block, 'source') else []
                            )
                            self.concept_mesh.add_node(node)

class PrajnaWithReasoning:
    """Enhanced Prajna that includes reasoning traversal"""
    
    def __init__(self, existing_prajna=None):
        self.prajna = existing_prajna
        self.enhanced_context_builder = EnhancedContextBuilder()
    
    def generate_with_reasoning(self, query: str, 
                              persona: Optional[Dict[str, Any]] = None,
                              **kwargs) -> PrajnaResponsePlus:
        """Generate response with full reasoning traversal"""
        
        # Build context with reasoning
        context, reasoned_response = self.enhanced_context_builder.build_context_with_reasoning(
            query, 
            {'persona': persona, **kwargs}
        )
        
        # If we have existing Prajna, enhance its response
        if self.prajna:
            try:
                # Get traditional Prajna response
                traditional_response = self.prajna.generate(
                    query,
                    context=context,
                    persona=persona,
                    **kwargs
                )
                
                # Merge responses
                reasoned_response.text = traditional_response.text
                reasoned_response.metadata['traditional_response'] = traditional_response
                
            except Exception as e:
                logger.error(f"Failed to get traditional Prajna response: {e}")
        
        return reasoned_response
    
    def explain_reasoning(self, response: PrajnaResponsePlus) -> str:
        """Generate detailed explanation of reasoning process"""
        explanation = []
        
        explanation.append("🧠 Reasoning Process Explanation\n")
        explanation.append("=" * 50)
        
        # Explain each reasoning path
        for i, path in enumerate(response.reasoning_paths):
            explanation.append(f"\n📊 Reasoning Path {i+1}:")
            explanation.append(f"Type: {path.path_type}")
            explanation.append(f"Confidence: {path.confidence:.2%}")
            explanation.append(f"Steps:")
            
            for j, node in enumerate(path.chain):
                explanation.append(f"  {j+1}. {node.name}")
                if j < len(path.edge_justifications):
                    explanation.append(f"     ↳ {path.edge_justifications[j]}")
        
        # Explain sources
        explanation.append(f"\n📚 Sources Used: {len(response.sources)}")
        for source in response.sources[:5]:
            explanation.append(f"  - {source}")
        
        return "\n".join(explanation)

# Example integration function for your existing system
def integrate_reasoning_with_prajna(prajna_instance):
    """
    Integrate reasoning traversal with existing Prajna instance
    
    Usage:
        enhanced_prajna = integrate_reasoning_with_prajna(your_prajna)
        response = enhanced_prajna.generate_with_reasoning(
            "How does entropy relate to compression?",
            persona={'name': 'Teacher', 'style': 'explanatory'}
        )
    """
    return PrajnaWithReasoning(prajna_instance)
