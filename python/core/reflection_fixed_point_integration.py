#!/usr/bin/env python3
"""
Reflection Fixed-Point Integration
Enhances self-reflective reasoning with iterative stabilization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from enum import Enum

from python.core.reasoning_traversal import ReasoningPath, PrajnaResponsePlus
from python.core.intent_driven_reasoning import ResolutionReport
from python.core.unified_metacognitive_integration import (
    CognitiveStateManager, ReflectionSystem, MetacognitiveState
)

logger = logging.getLogger(__name__)

# ========== Reflection Types ==========

class ReflectionType(Enum):
    """Types of reflection operations"""
    SHALLOW = "shallow"          # Single-pass reflection
    DEEP = "deep"               # Multi-pass until convergence
    ADVERSARIAL = "adversarial" # Ghost vs base debate
    CONSTRUCTIVE = "constructive" # Building on ideas
    CRITICAL = "critical"       # Finding flaws
    SYNTHESIS = "synthesis"     # Combining perspectives

@dataclass
class ReflectionResult:
    """Result of reflection process"""
    original_response: str
    reflected_response: str
    iterations: int
    converged: bool
    confidence_delta: float
    insights: List[str]
    ghost_critiques: List[str] = field(default_factory=list)
    synthesis: Optional[str] = None

# ========== Ghost Persona System ==========

class GhostPersona:
    """Internal ghost persona for adversarial reflection"""
    
    def __init__(self, base_response: PrajnaResponsePlus,
                 personality: str = "critical"):
        self.base_response = base_response
        self.personality = personality
        self.critique_history = []
    
    def critique(self, content: str, reasoning_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """Ghost persona critiques the base response"""
        
        critiques = []
        suggestions = []
        confidence_adjustments = []
        
        # Personality-based critique style
        if self.personality == "critical":
            critiques, suggestions = self._critical_analysis(content, reasoning_paths)
        elif self.personality == "constructive":
            critiques, suggestions = self._constructive_analysis(content, reasoning_paths)
        elif self.personality == "adversarial":
            critiques, suggestions = self._adversarial_analysis(content, reasoning_paths)
        else:
            critiques = ["No specific critique style selected"]
        
        # Analyze confidence
        for path in reasoning_paths:
            if path.confidence < 0.5:
                confidence_adjustments.append({
                    "path": " → ".join([n.name for n in path.chain]),
                    "issue": "Low confidence path",
                    "adjustment": -0.1
                })
        
        critique_result = {
            "critiques": critiques,
            "suggestions": suggestions,
            "confidence_adjustments": confidence_adjustments,
            "overall_assessment": self._overall_assessment(critiques, suggestions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.critique_history.append(critique_result)
        return critique_result
    
    def _critical_analysis(self, content: str, paths: List[ReasoningPath]) -> Tuple[List[str], List[str]]:
        """Critical personality - finds flaws"""
        critiques = []
        suggestions = []
        
        # Check for logical gaps
        if len(paths) > 0:
            # Look for missing connections
            for i, path in enumerate(paths):
                if len(path.chain) > 3:
                    critiques.append(f"Path {i+1} seems overly complex with {len(path.chain)} steps")
                    suggestions.append("Consider finding more direct reasoning")
                
                # Check justifications
                weak_justifications = [j for j in path.edge_justifications 
                                     if len(j) < 20 or "related" in j.lower()]
                if weak_justifications:
                    critiques.append(f"Weak justifications found: {weak_justifications[0][:50]}...")
                    suggestions.append("Strengthen causal relationships")
        
        # Content analysis
        if len(content) < 100:
            critiques.append("Response lacks depth")
            suggestions.append("Expand with more detailed explanation")
        
        if "however" not in content.lower() and "but" not in content.lower():
            critiques.append("Response lacks nuance or counterpoints")
            suggestions.append("Consider alternative perspectives")
        
        return critiques, suggestions
    
    def _constructive_analysis(self, content: str, paths: List[ReasoningPath]) -> Tuple[List[str], List[str]]:
        """Constructive personality - builds on ideas"""
        critiques = []
        suggestions = []
        
        # Look for expansion opportunities
        if paths:
            for path in paths[:2]:
                if path.path_type == "inference":
                    suggestions.append(f"Could strengthen with supporting evidence for {path.chain[-1].name}")
                elif path.path_type == "causal":
                    suggestions.append("Consider exploring downstream effects")
        
        # Identify gaps to fill
        concepts_mentioned = set()
        for path in paths:
            for node in path.chain:
                concepts_mentioned.add(node.name.lower())
        
        if "implications" not in content.lower():
            critiques.append("Missing discussion of implications")
            suggestions.append("Add section on practical implications")
        
        return critiques, suggestions
    
    def _adversarial_analysis(self, content: str, paths: List[ReasoningPath]) -> Tuple[List[str], List[str]]:
        """Adversarial personality - challenges everything"""
        critiques = []
        suggestions = []
        
        # Challenge core assumptions
        if paths:
            critiques.append(f"Why assume {paths[0].chain[0].name} is the starting point?")
            suggestions.append("Question fundamental assumptions")
            
            # Challenge relationships
            for path in paths:
                for i, justification in enumerate(path.edge_justifications):
                    if "implies" in justification:
                        critiques.append(f"Does {path.chain[i].name} necessarily imply the next step?")
        
        # Challenge conclusions
        critiques.append("The conclusion may be premature")
        suggestions.append("Consider edge cases and exceptions")
        
        return critiques, suggestions
    
    def _overall_assessment(self, critiques: List[str], suggestions: List[str]) -> str:
        """Generate overall assessment"""
        severity = len(critiques)
        
        if severity == 0:
            return "Response appears sound"
        elif severity <= 2:
            return "Minor improvements suggested"
        elif severity <= 4:
            return "Significant enhancements recommended"
        else:
            return "Major revision needed"

# ========== Enhanced Reflection System ==========

class EnhancedReflectionSystem(ReflectionSystem):
    """Extended reflection system with fixed-point convergence"""
    
    def __init__(self, state_manager: CognitiveStateManager):
        super().__init__(state_manager)
        self.convergence_history = []
    
    def iterative_reflection(self, response: PrajnaResponsePlus,
                           reflection_type: ReflectionType = ReflectionType.DEEP,
                           max_iterations: int = 5) -> ReflectionResult:
        """Apply iterative reflection until convergence"""
        
        # Initialize
        current_response = response.text
        current_paths = response.reasoning_paths
        current_confidence = response.confidence
        
        iterations = 0
        converged = False
        insights = []
        ghost_critiques = []
        
        # Create ghost persona
        ghost = GhostPersona(response, 
                           personality="critical" if reflection_type == ReflectionType.CRITICAL else "constructive")
        
        # Get initial state
        state = self.state_manager.compute_state_from_reasoning(current_paths)
        
        for iteration in range(max_iterations):
            iterations += 1
            
            # Ghost critique
            critique = ghost.critique(current_response, current_paths)
            ghost_critiques.extend(critique["critiques"])
            
            # Apply reflection based on type
            if reflection_type == ReflectionType.SHALLOW:
                new_state, change = self.reflect(state, steps=1)
            elif reflection_type == ReflectionType.DEEP:
                new_state, change = self.reflect(state, steps=3, momentum=0.7)
            elif reflection_type == ReflectionType.ADVERSARIAL:
                # Adversarial reflection with higher noise
                new_state, change = self.reflect(state, steps=2, momentum=0.5)
                new_state += np.random.randn(*state.shape) * 0.1  # Add perturbation
            else:
                new_state, change = self.reflect(state, steps=2)
            
            # Check convergence
            if change < 0.01:
                converged = True
                insights.append(f"Converged after {iterations} iterations")
                break
            
            # Generate insights from state change
            insight = self._generate_insight_from_state_change(state, new_state, critique)
            if insight:
                insights.append(insight)
            
            # Update for next iteration
            state = new_state
            
            # Synthesize improvements (simplified for demo)
            if critique["suggestions"]:
                current_response = self._apply_suggestions(current_response, critique["suggestions"])
        
        # Calculate confidence delta
        final_confidence = current_confidence * (0.9 ** len(ghost_critiques))  # Decrease with critiques
        confidence_delta = final_confidence - current_confidence
        
        # Generate synthesis if requested
        synthesis = None
        if reflection_type == ReflectionType.SYNTHESIS:
            synthesis = self._synthesize_perspectives(response.text, current_response, insights)
        
        # Record convergence
        self.convergence_history.append({
            "original_confidence": current_confidence,
            "final_confidence": final_confidence,
            "iterations": iterations,
            "converged": converged,
            "type": reflection_type.value
        })
        
        return ReflectionResult(
            original_response=response.text,
            reflected_response=current_response,
            iterations=iterations,
            converged=converged,
            confidence_delta=confidence_delta,
            insights=insights,
            ghost_critiques=ghost_critiques[:5],  # Top 5 critiques
            synthesis=synthesis
        )
    
    def _generate_insight_from_state_change(self, old_state: np.ndarray,
                                          new_state: np.ndarray,
                                          critique: Dict[str, Any]) -> Optional[str]:
        """Generate insight from state transition"""
        
        # Measure change magnitude and direction
        delta = new_state - old_state
        magnitude = np.linalg.norm(delta)
        
        if magnitude < 0.001:
            return None
        
        # Find dimension with largest change
        max_change_idx = np.argmax(np.abs(delta))
        max_change = delta[max_change_idx]
        
        # Generate insight based on change pattern
        if magnitude > 0.5:
            return f"Significant shift detected (magnitude: {magnitude:.3f}) - reconsidering approach"
        elif max_change > 0:
            return f"Strengthening dimension {max_change_idx} based on critique"
        else:
            return f"Weakening dimension {max_change_idx} to address concerns"
    
    def _apply_suggestions(self, response: str, suggestions: List[str]) -> str:
        """Apply suggestions to response (simplified)"""
        
        # Add a note about suggestions
        if suggestions:
            addition = "\n\nUpon reflection: " + suggestions[0]
            return response + addition
        
        return response
    
    def _synthesize_perspectives(self, original: str, reflected: str, 
                               insights: List[str]) -> str:
        """Synthesize original and reflected perspectives"""
        
        synthesis = "Synthesis of perspectives:\n\n"
        synthesis += f"Original view: {original[:100]}...\n\n"
        synthesis += f"Reflected view: {reflected[:100]}...\n\n"
        
        if insights:
            synthesis += "Key insights:\n"
            for insight in insights[:3]:
                synthesis += f"- {insight}\n"
        
        synthesis += "\nIntegrated understanding: The truth likely incorporates elements from both perspectives."
        
        return synthesis

# ========== Metacognitive Reflection Orchestrator ==========

class MetacognitiveReflectionOrchestrator:
    """Orchestrates reflection processes with metacognitive awareness"""
    
    def __init__(self, state_manager: CognitiveStateManager):
        self.state_manager = state_manager
        self.reflection_system = EnhancedReflectionSystem(state_manager)
        self.reflection_history = []
    
    def should_reflect(self, response: PrajnaResponsePlus, 
                      resolution_report: Optional[ResolutionReport] = None) -> Tuple[bool, ReflectionType]:
        """Determine if reflection is needed and what type"""
        
        # Always reflect if explicitly requested
        if response.metadata.get("deep_reflection", False):
            return True, ReflectionType.DEEP
        
        # Check confidence
        if response.confidence < 0.6:
            return True, ReflectionType.CRITICAL
        
        # Check for conflicts
        if resolution_report and len(resolution_report.conflicts) > 2:
            return True, ReflectionType.ADVERSARIAL
        
        # Check path complexity
        if response.reasoning_paths:
            avg_length = np.mean([len(p.chain) for p in response.reasoning_paths])
            if avg_length > 4:
                return True, ReflectionType.SYNTHESIS
        
        # Random deep reflection (10% chance)
        if np.random.random() < 0.1:
            return True, ReflectionType.CONSTRUCTIVE
        
        return False, ReflectionType.SHALLOW
    
    def orchestrate_reflection(self, response: PrajnaResponsePlus,
                             forced_type: Optional[ReflectionType] = None) -> PrajnaResponsePlus:
        """Orchestrate the reflection process"""
        
        # Determine reflection need and type
        if forced_type:
            should_reflect = True
            reflection_type = forced_type
        else:
            should_reflect, reflection_type = self.should_reflect(response)
        
        if not should_reflect:
            return response
        
        logger.info(f"Initiating {reflection_type.value} reflection")
        
        # Perform reflection
        reflection_result = self.reflection_system.iterative_reflection(
            response, reflection_type
        )
        
        # Update response with reflection
        enhanced_response = self._integrate_reflection(response, reflection_result)
        
        # Record in history
        self.reflection_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": reflection_type.value,
            "iterations": reflection_result.iterations,
            "converged": reflection_result.converged,
            "confidence_delta": reflection_result.confidence_delta
        })
        
        return enhanced_response
    
    def _integrate_reflection(self, original: PrajnaResponsePlus,
                            reflection: ReflectionResult) -> PrajnaResponsePlus:
        """Integrate reflection results into response"""
        
        # Update text
        enhanced_text = reflection.reflected_response
        
        # Add reflection metadata
        if reflection.ghost_critiques:
            enhanced_text += "\n\n**Reflection Notes:**\n"
            for critique in reflection.ghost_critiques[:3]:
                enhanced_text += f"- {critique}\n"
        
        if reflection.synthesis:
            enhanced_text += f"\n\n**Synthesis:**\n{reflection.synthesis}"
        
        # Update confidence
        new_confidence = original.confidence + reflection.confidence_delta
        
        # Create enhanced response
        enhanced = PrajnaResponsePlus(
            text=enhanced_text,
            reasoning_paths=original.reasoning_paths,
            sources=original.sources,
            confidence=new_confidence,
            metadata=original.metadata.copy()
        )
        
        # Add reflection metadata
        enhanced.metadata.update({
            "reflection_type": reflection.iterations,
            "reflection_converged": reflection.converged,
            "reflection_insights": reflection.insights,
            "confidence_delta": reflection.confidence_delta
        })
        
        return enhanced
    
    def get_reflection_analytics(self) -> Dict[str, Any]:
        """Get analytics on reflection performance"""
        
        if not self.reflection_history:
            return {"message": "No reflection history available"}
        
        # Analyze convergence rates by type
        convergence_by_type = {}
        avg_iterations_by_type = {}
        
        for entry in self.reflection_history:
            r_type = entry["type"]
            if r_type not in convergence_by_type:
                convergence_by_type[r_type] = []
                avg_iterations_by_type[r_type] = []
            
            convergence_by_type[r_type].append(entry["converged"])
            avg_iterations_by_type[r_type].append(entry["iterations"])
        
        # Calculate stats
        stats = {}
        for r_type in convergence_by_type:
            stats[r_type] = {
                "convergence_rate": np.mean(convergence_by_type[r_type]),
                "avg_iterations": np.mean(avg_iterations_by_type[r_type]),
                "total_reflections": len(convergence_by_type[r_type])
            }
        
        # Overall stats
        all_iterations = [e["iterations"] for e in self.reflection_history]
        all_converged = [e["converged"] for e in self.reflection_history]
        
        return {
            "total_reflections": len(self.reflection_history),
            "overall_convergence_rate": np.mean(all_converged),
            "average_iterations": np.mean(all_iterations),
            "stats_by_type": stats,
            "recent_reflections": self.reflection_history[-5:]
        }

# ========== Demo and Testing ==========

def demonstrate_reflection_fixed_point():
    """Demonstrate reflection fixed-point system"""
    from python.core.reasoning_traversal import ConceptNode, EdgeType
    
    print("🔄 Reflection Fixed-Point System Demo")
    print("=" * 60)
    
    # Create test response
    test_paths = [
        ReasoningPath(
            chain=[
                ConceptNode("premise", "Premise", "Initial assumption"),
                ConceptNode("inference", "Inference", "Derived conclusion"),
                ConceptNode("conclusion", "Conclusion", "Final result")
            ],
            edge_justifications=["implies", "therefore"],
            score=0.7,
            path_type="inference",
            confidence=0.65
        )
    ]
    
    test_response = PrajnaResponsePlus(
        text="Based on the premise, we can infer the conclusion.",
        reasoning_paths=test_paths,
        sources=["test_source"],
        confidence=0.65,
        metadata={"query": "test query"}
    )
    
    # Initialize systems
    state_manager = CognitiveStateManager()
    orchestrator = MetacognitiveReflectionOrchestrator(state_manager)
    
    # Test different reflection types
    reflection_types = [
        ReflectionType.SHALLOW,
        ReflectionType.DEEP,
        ReflectionType.CRITICAL,
        ReflectionType.ADVERSARIAL
    ]
    
    for r_type in reflection_types:
        print(f"\n{'='*60}")
        print(f"Testing {r_type.value} reflection")
        print("-" * 60)
        
        # Perform reflection
        reflected = orchestrator.orchestrate_reflection(
            test_response, forced_type=r_type
        )
        
        print(f"Original confidence: {test_response.confidence:.2f}")
        print(f"Reflected confidence: {reflected.confidence:.2f}")
        print(f"Iterations: {reflected.metadata.get('reflection_type', 0)}")
        print(f"Converged: {reflected.metadata.get('reflection_converged', False)}")
        
        if reflected.metadata.get('reflection_insights'):
            print("\nInsights:")
            for insight in reflected.metadata['reflection_insights'][:3]:
                print(f"  - {insight}")
    
    # Show analytics
    print(f"\n{'='*60}")
    print("📊 Reflection Analytics")
    print("-" * 60)
    
    analytics = orchestrator.get_reflection_analytics()
    print(f"Total reflections: {analytics['total_reflections']}")
    print(f"Overall convergence rate: {analytics['overall_convergence_rate']:.2%}")
    print(f"Average iterations: {analytics['average_iterations']:.1f}")
    
    print("\nStats by type:")
    for r_type, stats in analytics['stats_by_type'].items():
        print(f"  {r_type}: {stats['convergence_rate']:.2%} convergence, "
              f"{stats['avg_iterations']:.1f} avg iterations")

if __name__ == "__main__":
    demonstrate_reflection_fixed_point()
