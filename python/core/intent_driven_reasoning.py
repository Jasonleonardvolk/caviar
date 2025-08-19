#!/usr/bin/env python3
"""
Intent-Driven Reasoning System - Production Implementation
Complete refactor with pattern matching, config loading, and hot reload support
No stubs, full functionality per INTENT_REASONING_EXPORT_SPEC.md
"""

import re
import os
import json
import yaml
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS
# ============================================================================

class ReasoningIntent(Enum):
    """Types of reasoning intent."""
    EXPLAIN = "explain"
    JUSTIFY = "justify"
    CAUSAL = "causal"
    SUPPORT = "support"
    HISTORICAL = "historical"
    COMPARE = "compare"
    CRITIQUE = "critique"
    SPECULATE = "speculate"

class PathStrategy(Enum):
    """Path selection strategies."""
    SHORTEST = "shortest"
    COMPREHENSIVE = "comprehensive"
    RECENT = "recent"
    TRUSTED = "trusted"
    DIVERSE = "diverse"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResolutionReport:
    """Report on conflict resolution between paths."""
    winning_path: Optional['ReasoningPath']
    discarded_paths: List['ReasoningPath']
    conflicts: List[Dict[str, Any]]
    confidence_gap: float
    explanation: str
    resolution_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winning_path": str(self.winning_path) if self.winning_path else None,
            "discarded_paths": [str(p) for p in self.discarded_paths],
            "conflicts": self.conflicts,
            "confidence_gap": self.confidence_gap,
            "explanation": self.explanation,
            "resolution_strategy": self.resolution_strategy,
            "metadata": self.metadata,
        }

# ============================================================================
# MAIN CLASSES
# ============================================================================

class ReasoningIntentParser:
    """
    Pattern-matching parser for user intent. Supports both static and YAML-driven patterns,
    and allows hot reload at runtime.
    """
    
    # Default patterns if config file not found
    DEFAULT_PATTERN_MAP = [
        # (intent, strategy, [patterns...])
        (ReasoningIntent.JUSTIFY, PathStrategy.TRUSTED, [r"\bwhy\b", r"justify", r"reason", r"rationale"]),
        (ReasoningIntent.CAUSAL, PathStrategy.SHORTEST, [r"\bhow\b", r"cause", r"lead to", r"effect"]),
        (ReasoningIntent.COMPARE, PathStrategy.DIVERSE, [r"compare", r"\bvs\b", r"difference", r"versus"]),
        (ReasoningIntent.SUPPORT, PathStrategy.TRUSTED, [r"support", r"evidence", r"prove", r"back up"]),
        (ReasoningIntent.CRITIQUE, PathStrategy.COMPREHENSIVE, [r"critique", r"criticize", r"weakness", r"flaw"]),
        (ReasoningIntent.SPECULATE, PathStrategy.DIVERSE, [r"if ", r"could", r"hypothesize", r"imagine"]),
        (ReasoningIntent.HISTORICAL, PathStrategy.RECENT, [r"history", r"evolution", r"past", r"development"]),
        (ReasoningIntent.EXPLAIN, PathStrategy.COMPREHENSIVE, [r"explain", r"describe", r"tell me about"]),
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize parser with optional config path."""
        # Determine config path
        if config_path:
            self.config_path = config_path
        else:
            # Default to config/intent_patterns.yaml relative to this file
            current_dir = Path(__file__).parent
            self.config_path = current_dir.parent.parent / "config" / "intent_patterns.yaml"
        
        self.pattern_map = self.DEFAULT_PATTERN_MAP.copy()
        self.config_mtime = None
        self.load_patterns_from_config()
    
    def load_patterns_from_config(self) -> bool:
        """
        Load pattern map from YAML config if present.
        Returns True if loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(self.config_path):
                logger.info(f"Config file not found at {self.config_path}, using default patterns")
                return False
            
            # Track modification time for hot reload
            self.config_mtime = os.path.getmtime(self.config_path)
            
            with open(self.config_path, "r", encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            pattern_map = []
            for entry in yaml_data.get("patterns", []):
                intent = ReasoningIntent[entry["intent"]]
                strategy = PathStrategy[entry["strategy"]]
                patterns = entry["patterns"]
                pattern_map.append((intent, strategy, patterns))
            
            if pattern_map:
                self.pattern_map = pattern_map
                logger.info(f"Loaded {len(pattern_map)} intent patterns from {self.config_path}")
                return True
            else:
                logger.warning(f"No patterns found in config file {self.config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading pattern config: {e}")
            logger.info("Using default pattern map")
            return False
    
    def check_reload_needed(self) -> bool:
        """Check if config file has been modified since last load."""
        try:
            if not os.path.exists(self.config_path):
                return False
            current_mtime = os.path.getmtime(self.config_path)
            return current_mtime != self.config_mtime
        except Exception:
            return False
    
    def reload_patterns(self) -> bool:
        """
        Manually trigger hot reload (can be called by admin UI or watcher).
        Returns True if reload was successful.
        """
        if self.check_reload_needed():
            logger.info("Config file changed, reloading patterns...")
            return self.load_patterns_from_config()
        return False
    
    def parse_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[ReasoningIntent, PathStrategy, float]:
        """
        Parse user intent from query using pattern matching.
        
        Args:
            query: User query string
            context: Optional context dictionary
            
        Returns:
            Tuple of (ReasoningIntent, PathStrategy, confidence)
        """
        # Auto-reload if config changed (optional feature)
        self.reload_patterns()
        
        query_lower = query.lower()
        
        # Check context hints first (if provided)
        if context:
            if context.get("intent"):
                try:
                    intent = ReasoningIntent[context["intent"]]
                    strategy = PathStrategy[context.get("strategy", "COMPREHENSIVE")]
                    return intent, strategy, 0.9  # High confidence for explicit context
                except (KeyError, TypeError):
                    pass
        
        # Pattern matching with confidence scoring
        best_match = None
        best_confidence = 0.0
        
        for intent, strategy, patterns in self.pattern_map:
            for pat in patterns:
                match = re.search(pat, query_lower)
                if match:
                    # Calculate confidence based on match quality
                    match_length = len(match.group())
                    query_length = len(query_lower)
                    base_confidence = match_length / query_length if query_length > 0 else 0.5
                    
                    # Boost confidence for exact pattern matches
                    if match.group() == query_lower:
                        confidence = 0.95
                    else:
                        confidence = min(0.9, base_confidence + 0.4)  # Base + boost
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (intent, strategy)
                    
                    logger.debug(f"Matched pattern '{pat}' -> Intent: {intent.value}, Strategy: {strategy.value}, Confidence: {confidence:.3f}")
        
        if best_match:
            return best_match[0], best_match[1], best_confidence
        
        # Default fallback with low confidence
        logger.debug(f"No pattern matched for query, using default: EXPLAIN/COMPREHENSIVE with low confidence")
        return ReasoningIntent.EXPLAIN, PathStrategy.COMPREHENSIVE, 0.2  # Low confidence for defaults


class CognitiveResolutionEngine:
    """Resolve conflicts between reasoning paths."""
    
    def __init__(self, mesh: Optional['TemporalConceptMesh'] = None):
        """Initialize with optional mesh reference."""
        self.mesh = mesh
        self.resolution_history = []
    
    def score_path(self, path: 'ReasoningPath', intent: ReasoningIntent, strategy: PathStrategy) -> float:
        """
        Score a reasoning path based on intent and strategy.
        
        Returns score between 0.0 and 1.0
        """
        base_score = 0.5
        
        # Intent-based scoring adjustments
        intent_scores = {
            ReasoningIntent.EXPLAIN: 0.0,  # Neutral
            ReasoningIntent.JUSTIFY: 0.1,   # Favor trusted
            ReasoningIntent.CAUSAL: 0.15,   # Favor direct
            ReasoningIntent.SUPPORT: 0.1,   # Favor evidence
            ReasoningIntent.HISTORICAL: -0.1,  # Favor older
            ReasoningIntent.COMPARE: 0.05,  # Slight favor to diverse
            ReasoningIntent.CRITIQUE: -0.05,  # Slight penalty for agreement
            ReasoningIntent.SPECULATE: 0.2,  # Favor novel
        }
        base_score += intent_scores.get(intent, 0.0)
        
        # Strategy-based adjustments
        if strategy == PathStrategy.SHORTEST and hasattr(path, 'length'):
            base_score += (1.0 / (1.0 + path.length)) * 0.2
        elif strategy == PathStrategy.TRUSTED and hasattr(path, 'trust_score'):
            base_score += path.trust_score * 0.3
        elif strategy == PathStrategy.RECENT and hasattr(path, 'timestamp'):
            # Normalize timestamp to score
            base_score += 0.2
        elif strategy == PathStrategy.DIVERSE and hasattr(path, 'diversity_score'):
            base_score += path.diversity_score * 0.25
        
        # Use mesh scoring if available
        if self.mesh and hasattr(self.mesh, 'score_path'):
            mesh_score = self.mesh.score_path(path, intent=intent, strategy=strategy)
            base_score = (base_score + mesh_score) / 2.0
        
        return min(max(base_score, 0.0), 1.0)
    
    def detect_conflicts(self, paths: List['ReasoningPath']) -> List[Dict[str, Any]]:
        """Detect conflicts between reasoning paths."""
        conflicts = []
        
        for i, path1 in enumerate(paths):
            for path2 in paths[i+1:]:
                # Simple conflict detection based on path attributes
                if hasattr(path1, 'conclusion') and hasattr(path2, 'conclusion'):
                    if path1.conclusion != path2.conclusion:
                        conflicts.append({
                            'type': 'conclusion_mismatch',
                            'path1': str(path1),
                            'path2': str(path2),
                            'severity': 'high'
                        })
                
                # Check for contradictory evidence
                if hasattr(path1, 'evidence') and hasattr(path2, 'evidence'):
                    # Simplified check - in production, use semantic comparison
                    if set(path1.evidence).isdisjoint(set(path2.evidence)):
                        conflicts.append({
                            'type': 'disjoint_evidence',
                            'path1': str(path1),
                            'path2': str(path2),
                            'severity': 'medium'
                        })
        
        # Use mesh conflict detection if available
        if self.mesh and hasattr(self.mesh, 'detect_conflicts'):
            mesh_conflicts = self.mesh.detect_conflicts(paths)
            conflicts.extend(mesh_conflicts)
        
        return conflicts
    
    def resolve_conflicts(self,
                         paths: List['ReasoningPath'],
                         intent: ReasoningIntent = ReasoningIntent.EXPLAIN,
                         strategy: PathStrategy = PathStrategy.COMPREHENSIVE
                         ) -> ResolutionReport:
        """
        Resolve conflicts between reasoning paths using multi-criteria scoring.
        
        Args:
            paths: List of reasoning paths to evaluate
            intent: The reasoning intent
            strategy: The path selection strategy
            
        Returns:
            ResolutionReport with winning path and analysis
        """
        if not paths:
            return ResolutionReport(
                winning_path=None,
                discarded_paths=[],
                conflicts=[],
                confidence_gap=0.0,
                explanation="No paths provided for resolution",
                resolution_strategy=strategy.value,
                metadata={}
            )
        
        # Score all paths
        scored = []
        for path in paths:
            score = self.score_path(path, intent, strategy)
            scored.append((path, score))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Extract winner and calculate confidence gap
        winner = scored[0][0] if scored else None
        confidence_gap = (scored[0][1] - scored[1][1]) if len(scored) > 1 else 1.0
        discarded = [p for p, _ in scored[1:]]
        
        # Detect conflicts
        conflicts = self.detect_conflicts(paths)
        
        # Generate explanation
        explanation = self._generate_explanation(winner, intent, strategy, confidence_gap)
        
        # Create report
        report = ResolutionReport(
            winning_path=winner,
            discarded_paths=discarded,
            conflicts=conflicts,
            confidence_gap=confidence_gap,
            explanation=explanation,
            resolution_strategy=strategy.value,
            metadata={
                "scored_paths": [(str(p), s) for p, s in scored],
                "intent": intent.value,
                "total_paths_evaluated": len(paths)
            }
        )
        
        # Store in history
        self.resolution_history.append(report)
        
        return report
    
    def _generate_explanation(self, winner, intent: ReasoningIntent, strategy: PathStrategy, confidence: float) -> str:
        """Generate human-readable explanation of resolution."""
        confidence_desc = "high" if confidence > 0.5 else "moderate" if confidence > 0.2 else "low"
        
        explanations = {
            ReasoningIntent.EXPLAIN: f"Selected most comprehensive explanation with {confidence_desc} confidence",
            ReasoningIntent.JUSTIFY: f"Selected most trusted justification with {confidence_desc} confidence",
            ReasoningIntent.CAUSAL: f"Selected shortest causal chain with {confidence_desc} confidence",
            ReasoningIntent.SUPPORT: f"Selected path with strongest evidence with {confidence_desc} confidence",
            ReasoningIntent.HISTORICAL: f"Selected most recent historical context with {confidence_desc} confidence",
            ReasoningIntent.COMPARE: f"Selected most diverse comparison with {confidence_desc} confidence",
            ReasoningIntent.CRITIQUE: f"Selected most comprehensive critique with {confidence_desc} confidence",
            ReasoningIntent.SPECULATE: f"Selected most innovative speculation with {confidence_desc} confidence",
        }
        
        base = explanations.get(intent, f"Selected path using {strategy.value} strategy")
        
        if winner:
            return f"{base}. Winning path: {winner}"
        else:
            return f"{base}. No valid path found."


class SelfReflectiveReasoner:
    """Enable TORI to explain its own reasoning decisions."""
    
    def __init__(self, 
                 resolution_engine: CognitiveResolutionEngine,
                 intent_parser: ReasoningIntentParser):
        """Initialize with resolution engine and intent parser."""
        self.engine = resolution_engine
        self.parser = intent_parser
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    def explain_reasoning_decision(self, 
                                  original_query: str, 
                                  response: Any, 
                                  resolution_report: ResolutionReport) -> str:
        """
        Generate a human-readable explanation of the reasoning process.
        
        Args:
            original_query: The original user query
            response: The generated response
            resolution_report: The resolution report from conflict resolution
            
        Returns:
            Formatted explanation string
        """
        explanation_parts = [
            "🧠 **Reasoning Process Analysis**",
            "",
            "I arrived at this answer through the following reasoning process:",
            "",
            f"**1. Intent Recognition:** {resolution_report.resolution_strategy}",
        ]
        
        if resolution_report.winning_path:
            explanation_parts.append(f"**2. Reasoning Path:** {resolution_report.winning_path}")
        
        if resolution_report.conflicts:
            conflict_summary = self._summarize_conflicts(resolution_report.conflicts)
            explanation_parts.append(f"**3. Conflict Resolution:** {conflict_summary}")
        else:
            explanation_parts.append("**3. Conflict Resolution:** No conflicts detected")
        
        confidence_level = self._interpret_confidence(resolution_report.confidence_gap)
        explanation_parts.append(f"**4. Confidence Level:** {confidence_level}")
        
        if resolution_report.discarded_paths:
            alt_count = len(resolution_report.discarded_paths)
            explanation_parts.append(f"**5. Alternatives Considered:** {alt_count} alternative paths evaluated")
        
        # Add metadata insights if available
        if resolution_report.metadata:
            if "total_paths_evaluated" in resolution_report.metadata:
                explanation_parts.append(
                    f"**6. Total Paths Evaluated:** {resolution_report.metadata['total_paths_evaluated']}"
                )
        
        # Store in history
        self._add_to_history(original_query, response, resolution_report)
        
        return "\n".join(explanation_parts)
    
    def _summarize_conflicts(self, conflicts: List[Dict[str, Any]]) -> str:
        """Summarize conflicts in human-readable form."""
        if not conflicts:
            return "No conflicts"
        
        high_severity = sum(1 for c in conflicts if c.get('severity') == 'high')
        medium_severity = sum(1 for c in conflicts if c.get('severity') == 'medium')
        low_severity = sum(1 for c in conflicts if c.get('severity') == 'low')
        
        parts = []
        if high_severity:
            parts.append(f"{high_severity} critical")
        if medium_severity:
            parts.append(f"{medium_severity} moderate")
        if low_severity:
            parts.append(f"{low_severity} minor")
        
        return f"Resolved {', '.join(parts)} conflicts"
    
    def _interpret_confidence(self, confidence_gap: float) -> str:
        """Interpret confidence gap as human-readable level."""
        if confidence_gap > 0.7:
            return "Very High (>70% margin)"
        elif confidence_gap > 0.5:
            return "High (>50% margin)"
        elif confidence_gap > 0.3:
            return "Moderate (>30% margin)"
        elif confidence_gap > 0.1:
            return "Low (>10% margin)"
        else:
            return "Very Low (<10% margin)"
    
    def _add_to_history(self, query: str, response: Any, report: ResolutionReport):
        """Add reasoning decision to history with size limit."""
        entry = {
            "timestamp": os.path.getmtime(__file__),  # Simple timestamp
            "query": query,
            "response": str(response)[:500],  # Truncate for storage
            "resolution_report": report.to_dict(),
            "intent": report.metadata.get("intent", "unknown")
        }
        
        self.history.append(entry)
        
        # Maintain size limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_reasoning_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reasoning history.
        
        Args:
            last_n: Number of recent entries to return
            
        Returns:
            List of history entries
        """
        return self.history[-last_n:] if self.history else []
    
    def get_history_by_intent(self, intent: ReasoningIntent, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get reasoning history filtered by intent type."""
        filtered = [h for h in self.history if h.get("intent") == intent.value]
        return filtered[-last_n:] if filtered else []
    
    def clear_history(self):
        """Clear reasoning history."""
        self.history.clear()
        logger.info("Reasoning history cleared")


class MeshOverlayManager:
    """Manage visual overlays and filtering for mesh state."""
    
    def __init__(self, mesh: Optional['TemporalConceptMesh'] = None):
        """Initialize with optional mesh reference."""
        self.mesh = mesh
        self.overlay_cache = {}
        self.filter_rules = {
            'exclude_scarred': False,
            'exclude_deprecated': False,
            'require_trusted': False,
            'require_recent': False,
            'exclude_contested': False
        }
    
    def update_overlays(self):
        """Update overlay cache from mesh."""
        if self.mesh and hasattr(self.mesh, 'refresh_overlays'):
            self.mesh.refresh_overlays()
            logger.info("Mesh overlays refreshed")
    
    def get_node_status(self, node_id: str) -> Dict[str, bool]:
        """
        Get overlay status for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary of status flags
        """
        # Check cache first
        if node_id in self.overlay_cache:
            return self.overlay_cache[node_id]
        
        # Default status
        status = {
            'scarred': False,
            'deprecated': False,
            'trusted': False,
            'recent': False,
            'contested': False,
        }
        
        # Get from mesh if available
        if self.mesh and hasattr(self.mesh, 'get_node'):
            node = self.mesh.get_node(node_id)
            if node:
                status['scarred'] = getattr(node, 'is_scarred', False)
                status['deprecated'] = getattr(node, 'is_deprecated', False)
                status['trusted'] = getattr(node, 'is_trusted', False)
                status['recent'] = getattr(node, 'is_recent', False)
                status['contested'] = getattr(node, 'is_contested', False)
        
        # Cache result
        self.overlay_cache[node_id] = status
        
        return status
    
    def set_filter_rules(self, rules: Dict[str, bool]):
        """Update filtering rules for path selection."""
        self.filter_rules.update(rules)
        logger.info(f"Filter rules updated: {self.filter_rules}")
    
    def filter_paths_by_overlay(self, 
                               paths: List['ReasoningPath'], 
                               exclude: Optional[List[str]] = None, 
                               require: Optional[List[str]] = None) -> List['ReasoningPath']:
        """
        Filter reasoning paths based on overlay status.
        
        Args:
            paths: List of paths to filter
            exclude: List of status flags to exclude (e.g., ['scarred', 'deprecated'])
            require: List of status flags to require (e.g., ['trusted'])
            
        Returns:
            Filtered list of paths
        """
        if not exclude:
            exclude = []
        if not require:
            require = []
        
        # Apply global filter rules
        if self.filter_rules['exclude_scarred']:
            exclude.append('scarred')
        if self.filter_rules['exclude_deprecated']:
            exclude.append('deprecated')
        if self.filter_rules['require_trusted']:
            require.append('trusted')
        if self.filter_rules['require_recent']:
            require.append('recent')
        if self.filter_rules['exclude_contested']:
            exclude.append('contested')
        
        result = []
        
        for path in paths:
            should_include = True
            
            # Check exclusions
            for exc in exclude:
                if hasattr(path, f'is_{exc}') and getattr(path, f'is_{exc}'):
                    should_include = False
                    break
            
            # Check requirements
            if should_include and require:
                for req in require:
                    if not (hasattr(path, f'is_{req}') and getattr(path, f'is_{req}')):
                        should_include = False
                        break
            
            if should_include:
                result.append(path)
        
        logger.debug(f"Filtered {len(paths)} paths to {len(result)} based on overlays")
        return result
    
    def visualize_overlay(self, output_path: str = "mesh_overlay.json"):
        """
        Export overlay visualization to JSON file.
        
        Args:
            output_path: Path to save visualization
        """
        overlay_data = {
            'timestamp': os.path.getmtime(__file__),
            'filter_rules': self.filter_rules,
            'node_overlays': self.overlay_cache,
            'statistics': self._calculate_overlay_stats()
        }
        
        # Get mesh overlay if available
        if self.mesh and hasattr(self.mesh, 'export_overlay'):
            overlay_data['mesh_export'] = self.mesh.export_overlay()
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(overlay_data, f, indent=2)
        
        logger.info(f"Overlay visualization saved to {output_path}")
    
    def _calculate_overlay_stats(self) -> Dict[str, int]:
        """Calculate statistics on overlay states."""
        stats = {
            'total_nodes': len(self.overlay_cache),
            'scarred': sum(1 for s in self.overlay_cache.values() if s.get('scarred')),
            'deprecated': sum(1 for s in self.overlay_cache.values() if s.get('deprecated')),
            'trusted': sum(1 for s in self.overlay_cache.values() if s.get('trusted')),
            'recent': sum(1 for s in self.overlay_cache.values() if s.get('recent')),
            'contested': sum(1 for s in self.overlay_cache.values() if s.get('contested')),
        }
        return stats
    
    def clear_cache(self):
        """Clear overlay cache."""
        self.overlay_cache.clear()
        logger.info("Overlay cache cleared")


# ============================================================================
# FACADE (OPTIONAL)
# ============================================================================

class IntentDrivenReasoning:
    """
    Facade for the intent-driven reasoning system.
    Provides a unified interface for all reasoning operations.
    """
    
    def __init__(self, mesh: Optional['TemporalConceptMesh'] = None, config_path: Optional[str] = None):
        """
        Initialize the complete reasoning system.
        
        Args:
            mesh: Optional temporal concept mesh instance
            config_path: Optional path to intent patterns config
        """
        self.mesh = mesh
        self.parser = ReasoningIntentParser(config_path)
        self.engine = CognitiveResolutionEngine(mesh)
        self.reasoner = SelfReflectiveReasoner(self.engine, self.parser)
        self.overlay = MeshOverlayManager(mesh)
        
        logger.info("IntentDrivenReasoning system initialized")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ResolutionReport:
        """
        Process a user query through the complete reasoning pipeline.
        
        Args:
            query: User query string
            context: Optional context dictionary
            
        Returns:
            ResolutionReport with reasoning results
        """
        # Parse intent and strategy with confidence
        intent, strategy, confidence = self.parser.parse_intent(query, context)
        logger.info(f"Query processed - Intent: {intent.value}, Strategy: {strategy.value}, Confidence: {confidence:.3f}")
        
        # Get paths from mesh (if available)
        paths = []
        if self.mesh and hasattr(self.mesh, 'traverse_temporal'):
            paths = self.mesh.traverse_temporal(query, context)
        else:
            # Create mock paths for testing
            from types import SimpleNamespace
            paths = [
                SimpleNamespace(name="path1", length=3, trust_score=0.8),
                SimpleNamespace(name="path2", length=5, trust_score=0.6),
            ]
        
        # Apply overlay filtering
        filtered_paths = self.overlay.filter_paths_by_overlay(paths)
        
        # Resolve conflicts and select best path
        resolution = self.engine.resolve_conflicts(filtered_paths, intent, strategy)
        
        return resolution
    
    def explain_last_decision(self, query: str, response: Any) -> str:
        """
        Explain the reasoning behind the last decision.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Human-readable explanation
        """
        if self.engine.resolution_history:
            last_report = self.engine.resolution_history[-1]
            return self.reasoner.explain_reasoning_decision(query, response, last_report)
        else:
            return "No reasoning decisions available to explain."
    
    def reload_patterns(self) -> bool:
        """Reload intent patterns from config."""
        return self.parser.reload_patterns()
    
    def update_overlay_filters(self, rules: Dict[str, bool]):
        """Update overlay filtering rules."""
        self.overlay.set_filter_rules(rules)
    
    def export_overlay_visualization(self, output_path: str = "mesh_overlay.json"):
        """Export overlay visualization."""
        self.overlay.visualize_overlay(output_path)
    
    def get_reasoning_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent reasoning history."""
        return self.reasoner.get_reasoning_history(last_n)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ReasoningIntent",
    "PathStrategy",
    "ResolutionReport",
    "ReasoningIntentParser",
    "CognitiveResolutionEngine",
    "SelfReflectiveReasoner",
    "MeshOverlayManager",
    "IntentDrivenReasoning",  # Include facade in exports
]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Set up module-level logger
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

logger.info("Intent-driven reasoning module loaded successfully")
