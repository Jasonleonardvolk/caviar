"""
Intent Router - COMPLETE IMPLEMENTATION (NO STUBS!)
Full-featured intent routing with advanced reasoning capabilities
"""

import os
import sys
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)

# ============== ROUTING CONTEXT MANAGEMENT ==============

@dataclass
class RoutingContext:
    """Context for intent routing decisions"""
    context_id: str
    priority: int = 0
    rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def matches(self, intent: str, query: str) -> bool:
        """Check if this context matches the given intent/query"""
        # Check intent match
        if "intents" in self.rules:
            if intent not in self.rules["intents"]:
                return False
        
        # Check query patterns
        if "patterns" in self.rules:
            matched = False
            for pattern in self.rules["patterns"]:
                if pattern.lower() in query.lower():
                    matched = True
                    break
            if not matched:
                return False
        
        # Check metadata conditions
        if "conditions" in self.rules:
            for key, value in self.rules["conditions"].items():
                if self.metadata.get(key) != value:
                    return False
        
        return True
    
    def apply_routing(self, base_route: str) -> str:
        """Apply context-specific routing modifications"""
        if "route_prefix" in self.rules:
            return f"{self.rules['route_prefix']}/{base_route}"
        if "route_suffix" in self.rules:
            return f"{base_route}/{self.rules['route_suffix']}"
        return base_route


class IntentRouter:
    """Advanced intent routing engine"""
    
    def __init__(self):
        self.contexts: Dict[str, RoutingContext] = {}
        self.route_history: List[Dict[str, Any]] = []
        self.intent_mappings = {
            "explain": self._route_explain,
            "justify": self._route_justify,
            "causal": self._route_causal,
            "support": self._route_support,
            "historical": self._route_historical,
            "compare": self._route_compare,
            "critique": self._route_critique,
            "speculate": self._route_speculate,
            "analyze": self._route_analyze,
            "synthesize": self._route_synthesize
        }
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        
    def route(self, intent: str, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Route an intent with full context awareness
        
        Returns:
            Dictionary containing:
            - route: The determined routing path
            - confidence: Confidence score (0-1)
            - contexts_applied: List of contexts that influenced routing
            - reasoning_path: Detailed reasoning steps
            - alternatives: Other possible routes considered
        """
        
        # Initialize routing result
        result = {
            "intent": intent,
            "query": query,
            "route": None,
            "confidence": 0.0,
            "contexts_applied": [],
            "reasoning_path": [],
            "alternatives": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Step 1: Normalize intent
        intent_normalized = intent.lower().strip()
        
        # Step 2: Check for matching contexts
        applicable_contexts = []
        for ctx_id, context in self.contexts.items():
            if context.matches(intent_normalized, query):
                applicable_contexts.append(context)
                result["contexts_applied"].append(ctx_id)
        
        # Sort by priority
        applicable_contexts.sort(key=lambda x: x.priority, reverse=True)
        
        # Step 3: Determine base route
        if intent_normalized in self.intent_mappings:
            base_route, confidence, reasoning = self.intent_mappings[intent_normalized](query, metadata)
            result["reasoning_path"] = reasoning
        else:
            # Fallback routing with similarity matching
            base_route, confidence = self._fallback_routing(intent_normalized, query)
            result["reasoning_path"] = [f"No exact match for intent '{intent}', using similarity-based routing"]
        
        result["route"] = base_route
        result["confidence"] = confidence
        
        # Step 4: Apply context modifications
        for context in applicable_contexts:
            modified_route = context.apply_routing(base_route)
            if modified_route != base_route:
                result["reasoning_path"].append(
                    f"Applied context '{context.context_id}': {base_route} -> {modified_route}"
                )
                base_route = modified_route
        
        result["route"] = base_route
        
        # Step 5: Generate alternatives
        result["alternatives"] = self._generate_alternatives(intent_normalized, query, base_route)
        
        # Step 6: Record in history
        self.route_history.append(result.copy())
        
        return result
    
    def _route_explain(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route explanation intent"""
        reasoning = ["Intent: Provide explanation"]
        
        # Analyze query complexity
        complexity = len(query.split()) / 10  # Simple heuristic
        
        if complexity < 0.5:
            route = "explanation/simple"
            confidence = 0.9
            reasoning.append("Query complexity: Low - routing to simple explanation")
        elif complexity < 1.5:
            route = "explanation/detailed"
            confidence = 0.85
            reasoning.append("Query complexity: Medium - routing to detailed explanation")
        else:
            route = "explanation/comprehensive"
            confidence = 0.8
            reasoning.append("Query complexity: High - routing to comprehensive explanation")
        
        # Check for domain-specific terms
        domains = ["quantum", "topology", "soliton", "bps", "energy", "phase"]
        for domain in domains:
            if domain in query.lower():
                route = f"explanation/technical/{domain}"
                confidence = min(0.95, confidence + 0.1)
                reasoning.append(f"Domain-specific term '{domain}' detected")
                break
        
        return route, confidence, reasoning
    
    def _route_justify(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route justification intent"""
        reasoning = ["Intent: Provide justification"]
        
        # Check for causal indicators
        causal_terms = ["why", "because", "reason", "cause", "since"]
        has_causal = any(term in query.lower() for term in causal_terms)
        
        if has_causal:
            route = "justification/causal"
            confidence = 0.85
            reasoning.append("Causal indicators found - routing to causal justification")
        else:
            route = "justification/logical"
            confidence = 0.8
            reasoning.append("No causal indicators - routing to logical justification")
        
        return route, confidence, reasoning
    
    def _route_causal(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route causal analysis intent"""
        reasoning = ["Intent: Causal analysis"]
        
        # Detect causal chain depth
        chain_indicators = ["then", "therefore", "consequently", "results in", "leads to"]
        chain_count = sum(1 for ind in chain_indicators if ind in query.lower())
        
        if chain_count == 0:
            route = "causal/direct"
            confidence = 0.9
            reasoning.append("Single cause-effect relationship")
        elif chain_count <= 2:
            route = "causal/chain"
            confidence = 0.85
            reasoning.append(f"Causal chain with {chain_count} links detected")
        else:
            route = "causal/network"
            confidence = 0.8
            reasoning.append(f"Complex causal network with {chain_count}+ relationships")
        
        return route, confidence, reasoning
    
    def _route_support(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route supporting evidence intent"""
        reasoning = ["Intent: Find supporting evidence"]
        
        # Check evidence type needed
        if "data" in query.lower() or "statistics" in query.lower():
            route = "support/quantitative"
            confidence = 0.85
            reasoning.append("Quantitative evidence requested")
        elif "example" in query.lower() or "case" in query.lower():
            route = "support/examples"
            confidence = 0.85
            reasoning.append("Examples or case studies requested")
        else:
            route = "support/general"
            confidence = 0.8
            reasoning.append("General supporting evidence")
        
        return route, confidence, reasoning
    
    def _route_historical(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route historical perspective intent"""
        reasoning = ["Intent: Historical analysis"]
        
        # Detect time range
        if "recent" in query.lower() or "lately" in query.lower():
            route = "historical/recent"
            confidence = 0.85
            reasoning.append("Recent history focus")
        elif "evolution" in query.lower() or "development" in query.lower():
            route = "historical/evolution"
            confidence = 0.85
            reasoning.append("Evolutionary/developmental perspective")
        else:
            route = "historical/chronological"
            confidence = 0.8
            reasoning.append("Standard chronological history")
        
        return route, confidence, reasoning
    
    def _route_compare(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route comparison intent"""
        reasoning = ["Intent: Comparison analysis"]
        
        # Count comparison subjects
        comparison_terms = ["vs", "versus", "compared to", "difference between", "or"]
        comparisons = sum(1 for term in comparison_terms if term in query.lower())
        
        if comparisons == 1:
            route = "compare/binary"
            confidence = 0.9
            reasoning.append("Binary comparison (2 subjects)")
        elif comparisons > 1:
            route = "compare/multiple"
            confidence = 0.85
            reasoning.append(f"Multi-way comparison ({comparisons+1} subjects)")
        else:
            route = "compare/implicit"
            confidence = 0.75
            reasoning.append("Implicit comparison detected")
        
        return route, confidence, reasoning
    
    def _route_critique(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route critical analysis intent"""
        reasoning = ["Intent: Critical analysis"]
        
        # Detect critique focus
        if "flaw" in query.lower() or "problem" in query.lower():
            route = "critique/weaknesses"
            confidence = 0.85
            reasoning.append("Focus on weaknesses/problems")
        elif "improve" in query.lower() or "better" in query.lower():
            route = "critique/constructive"
            confidence = 0.85
            reasoning.append("Constructive criticism requested")
        else:
            route = "critique/balanced"
            confidence = 0.8
            reasoning.append("Balanced critical analysis")
        
        return route, confidence, reasoning
    
    def _route_speculate(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route speculative reasoning intent"""
        reasoning = ["Intent: Speculative reasoning"]
        
        # Detect speculation type
        if "what if" in query.lower():
            route = "speculate/hypothetical"
            confidence = 0.85
            reasoning.append("Hypothetical scenario")
        elif "future" in query.lower() or "will" in query.lower():
            route = "speculate/predictive"
            confidence = 0.85
            reasoning.append("Future prediction")
        else:
            route = "speculate/exploratory"
            confidence = 0.8
            reasoning.append("Exploratory speculation")
        
        return route, confidence, reasoning
    
    def _route_analyze(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route analysis intent"""
        reasoning = ["Intent: Deep analysis"]
        route = "analyze/comprehensive"
        confidence = 0.85
        return route, confidence, reasoning
    
    def _route_synthesize(self, query: str, metadata: Optional[Dict]) -> Tuple[str, float, List[str]]:
        """Route synthesis intent"""
        reasoning = ["Intent: Synthesis of information"]
        route = "synthesize/integrated"
        confidence = 0.85
        return route, confidence, reasoning
    
    def _fallback_routing(self, intent: str, query: str) -> Tuple[str, float]:
        """Fallback routing using similarity matching"""
        
        # Simple similarity scoring
        best_match = None
        best_score = 0.0
        
        for known_intent in self.intent_mappings.keys():
            # Calculate similarity (simple character overlap)
            common = len(set(intent) & set(known_intent))
            score = common / max(len(intent), len(known_intent))
            
            if score > best_score:
                best_score = score
                best_match = known_intent
        
        if best_match and best_score > 0.5:
            route = f"fallback/{best_match}"
            confidence = best_score * 0.7  # Reduce confidence for fallback
        else:
            route = "fallback/general"
            confidence = 0.3
        
        return route, confidence
    
    def _generate_alternatives(self, intent: str, query: str, primary_route: str) -> List[Dict[str, Any]]:
        """Generate alternative routing options"""
        alternatives = []
        
        # Try other intents
        for alt_intent in ["explain", "analyze", "justify"]:
            if alt_intent != intent and alt_intent in self.intent_mappings:
                alt_route, alt_confidence, _ = self.intent_mappings[alt_intent](query, None)
                if alt_route != primary_route:
                    alternatives.append({
                        "route": alt_route,
                        "confidence": alt_confidence * 0.8,  # Reduce confidence for alternatives
                        "intent": alt_intent
                    })
        
        # Sort by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def add_context(self, context_id: str, priority: int, rules: Dict[str, Any]) -> bool:
        """Add a routing context"""
        try:
            self.contexts[context_id] = RoutingContext(
                context_id=context_id,
                priority=priority,
                rules=rules
            )
            logger.info(f"Added routing context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add context {context_id}: {e}")
            return False
    
    def remove_context(self, context_id: str) -> bool:
        """Remove a routing context"""
        if context_id in self.contexts:
            del self.contexts[context_id]
            logger.info(f"Removed routing context: {context_id}")
            return True
        return False
    
    def get_contexts(self) -> Dict[str, RoutingContext]:
        """Get all routing contexts"""
        return self.contexts.copy()
    
    def clear_contexts(self) -> bool:
        """Clear all routing contexts"""
        self.contexts.clear()
        logger.info("Cleared all routing contexts")
        return True
    
    def get_route_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent routing history"""
        return self.route_history[-limit:]


# ============== GLOBAL ROUTER INSTANCE ==============

_global_router = IntentRouter()

# ============== PUBLIC API ==============

def route_intent(intent: str, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Route an intent based on query and context
    
    Args:
        intent: The intent type (explain, justify, causal, etc.)
        query: The query string
        metadata: Optional metadata dictionary
    
    Returns:
        Routing result dictionary with route, confidence, and reasoning
    """
    return _global_router.route(intent, query, metadata)


def add_routing_context(context_id: str, routing_rule: Dict[str, Any], priority: int = 0) -> bool:
    """
    Add a routing context rule
    
    Args:
        context_id: Unique identifier for the context
        routing_rule: Dictionary defining the routing rules
        priority: Priority level (higher = more important)
    
    Returns:
        True if successful
    """
    return _global_router.add_context(context_id, priority, routing_rule)


def get_routing_contexts() -> Dict[str, Any]:
    """
    Get all active routing contexts
    
    Returns:
        Dictionary of context_id -> context data
    """
    contexts = _global_router.get_contexts()
    return {
        ctx_id: {
            "priority": ctx.priority,
            "rules": ctx.rules,
            "created_at": ctx.created_at.isoformat()
        }
        for ctx_id, ctx in contexts.items()
    }


def clear_routing_contexts() -> bool:
    """
    Clear all routing contexts
    
    Returns:
        True if successful
    """
    return _global_router.clear_contexts()


def is_available() -> bool:
    """
    Check if the intent router is available
    
    Returns:
        True (always available in this implementation)
    """
    return True


def is_intent_router_available() -> bool:
    """
    Check if the intent router is available (alias for compatibility)
    
    Returns:
        True (always available in this implementation)
    """
    return True


def get_route_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent routing history
    
    Args:
        limit: Maximum number of entries to return
    
    Returns:
        List of recent routing results
    """
    return _global_router.get_route_history(limit)


# ============== MODULE INITIALIZATION ==============

logger.info("Intent Router initialized - FULL IMPLEMENTATION (NO STUBS!)")
logger.info("Available intents: explain, justify, causal, support, historical, compare, critique, speculate, analyze, synthesize")

# Export public API
__all__ = [
    'route_intent',
    'add_routing_context',
    'get_routing_contexts',
    'clear_routing_contexts',
    'is_available',
    'is_intent_router_available',
    'get_route_history',
    'IntentRouter',
    'RoutingContext'
]
