"""introspection.py - Implements ALAN's self-monitoring and adaptive mechanisms.

This module enables Phase II: Introspective Cognition, allowing ALAN to observe, model,
and regulate itself during operation. It detects friction, modulates rhythm, tracks learning,
and enables self-reflection - transforming ALAN from reactive computation to reflective cognition.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
from pathlib import Path

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from memory_gating import calculate_spectral_entropy
except ImportError:
    # Fallback to relative import
    from .memory_gating import calculate_spectral_entropy

# Configure logger
logger = logging.getLogger("alan_introspection")

class FrictionMonitor:
    """
    Monitors internal processing friction and concept instability.
    
    The FrictionMonitor detects signs of cognitive strain, concept instability,
    phase decoherence, and other indicators that suggest the system is struggling.
    """
    
    def __init__(self):
        self.friction_events = []
        self.concept_friction = defaultdict(int)  # concept_id -> friction count
        self.session_start = datetime.now()
        self.recent_events_window = 50  # Number of events to consider "recent"
        
    def log_friction_event(
        self, 
        concept_id: str,
        event_type: str,
        severity: float,
        context: Dict[str, Any]
    ) -> None:
        """Log a friction event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "concept_id": concept_id,
            "event_type": event_type,
            "severity": severity,
            "context": context
        }
        self.friction_events.append(event)
        self.concept_friction[concept_id] += 1
        
        logger.debug(f"Friction event: {event_type} (severity: {severity:.2f}), "
                    f"concept: {concept_id}")
                    
    def log_phase_desync(
        self,
        concept1: str,
        concept2: str,
        expected_coherence: float,
        actual_coherence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a phase desynchronization event between concepts."""
        severity = abs(expected_coherence - actual_coherence)
        self.log_friction_event(
            concept_id=f"{concept1}:{concept2}",
            event_type="phase_desync",
            severity=severity,
            context={
                "concepts": [concept1, concept2],
                "expected_coherence": expected_coherence,
                "actual_coherence": actual_coherence,
                **(context or {})
            }
        )
        
    def log_concept_instability(
        self,
        concept_id: str,
        stability_metric: float,
        threshold: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an unstable concept event."""
        severity = max(0, threshold - stability_metric) / threshold
        self.log_friction_event(
            concept_id=concept_id,
            event_type="concept_instability",
            severity=severity,
            context={
                "stability_metric": stability_metric,
                "threshold": threshold,
                **(context or {})
            }
        )
        
    def log_entropy_spike(
        self,
        before: float,
        after: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a sudden increase in spectral entropy."""
        if after > before:
            change_pct = (after - before) / before if before > 0 else float('inf')
            severity = min(1.0, change_pct / 2)  # Cap at 1.0, 100% increase → 0.5 severity
            self.log_friction_event(
                concept_id="global",
                event_type="entropy_spike",
                severity=severity,
                context={
                    "before": before,
                    "after": after,
                    "change_percent": change_pct * 100,
                    **(context or {})
                }
            )
    
    def get_recent_friction_rate(self) -> float:
        """Get the rate of friction events in the recent window."""
        if len(self.friction_events) < 2:
            return 0.0
            
        # Look at most recent events
        recent = self.friction_events[-min(len(self.friction_events), self.recent_events_window):]
        if not recent:
            return 0.0
            
        # Calculate time span and event count
        start_time = datetime.fromisoformat(recent[0]["timestamp"])
        end_time = datetime.fromisoformat(recent[-1]["timestamp"])
        time_span = (end_time - start_time).total_seconds()
        
        # Avoid division by zero
        if time_span < 0.1:
            return 0.0
            
        return len(recent) / time_span
    
    def get_troublesome_concepts(self, threshold: int = 3) -> List[str]:
        """Get concepts with friction count above threshold."""
        return [cid for cid, count in self.concept_friction.items() 
                if count >= threshold]
                
    def get_overall_friction(self) -> float:
        """
        Calculate overall friction score (0-1).
        
        Considers the recent event rate and severity.
        """
        if not self.friction_events:
            return 0.0
            
        # Get recent events
        recent = self.friction_events[-min(len(self.friction_events), self.recent_events_window):]
        if not recent:
            return 0.0
            
        # Average severity
        avg_severity = sum(e["severity"] for e in recent) / len(recent)
        
        # Recent event rate, normalized to [0, 1]
        rate = min(1.0, self.get_recent_friction_rate() / 5)  # Cap at 5 events/sec
        
        # Combined score
        return (0.4 * rate + 0.6 * avg_severity)
        
    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """Analyze friction trend over specified window."""
        if len(self.friction_events) < window * 2:
            return {"status": "insufficient_data"}
            
        # Split into earlier and later windows
        mid_point = len(self.friction_events) - window
        earlier = self.friction_events[mid_point - window:mid_point]
        later = self.friction_events[mid_point:]
        
        # Calculate stats for each window
        earlier_avg = sum(e["severity"] for e in earlier) / len(earlier)
        later_avg = sum(e["severity"] for e in later) / len(later)
        
        change = later_avg - earlier_avg
        percent_change = (change / earlier_avg) * 100 if earlier_avg > 0 else float('inf')
        
        # Determine trend
        if abs(percent_change) < 10:
            trend = "stable"
        elif percent_change > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
            
        return {
            "status": "trend_available",
            "trend": trend,
            "earlier_average": earlier_avg,
            "later_average": later_avg,
            "change": change,
            "percent_change": percent_change
        }
        
    def should_trigger_regulation(self, 
                                threshold: float = 0.7, 
                                min_events: int = 5) -> bool:
        """Determine if regulation should be triggered based on friction."""
        if len(self.friction_events) < min_events:
            return False
            
        return self.get_overall_friction() >= threshold
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive friction report."""
        return {
            "event_count": len(self.friction_events),
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "overall_friction": self.get_overall_friction(),
            "trend": self.get_trend(),
            "troublesome_concepts": self.get_troublesome_concepts(),
            "event_types": dict(Counter(e["event_type"] for e in self.friction_events)),
            "recent_rate": self.get_recent_friction_rate()
        }
        
    def save_report(self, path: str) -> None:
        """Save friction report to a file."""
        report = self.generate_report()
        report["events"] = self.friction_events[-min(100, len(self.friction_events)):]  # Last 100 events
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved friction report to {path}")


class RhythmModulator:
    """
    Adjusts processing rhythm based on system state and friction.
    
    The RhythmModulator adapts ALAN's processing tempo, thresholds,
    and behavior in response to detected friction and flow state.
    """
    
    def __init__(self, friction_monitor: FrictionMonitor):
        self.friction_monitor = friction_monitor
        self.baseline_thresholds = {
            "coherence": 0.7,       # Default phase coherence threshold
            "entropy": 0.2,         # Default entropy threshold
            "redundancy": 0.85,     # Default redundancy threshold
            "confidence": 0.6,      # Default confidence threshold
            "processing_delay": 0.0  # Default processing delay (seconds)
        }
        self.current_thresholds = dict(self.baseline_thresholds)
        self.adjustment_history = []
        
    def update_thresholds(self) -> Dict[str, float]:
        """
        Update thresholds based on current friction state.
        
        Returns adjusted thresholds dictionary.
        """
        friction = self.friction_monitor.get_overall_friction()
        trend = self.friction_monitor.get_trend()
        
        # Base adjustments on friction level
        if friction > 0.8:  # Very high friction
            self._adjust_for_high_friction()
        elif friction > 0.5:  # Moderate friction
            self._adjust_for_moderate_friction()
        elif friction < 0.2:  # Low friction, flowing well
            self._adjust_for_flow()
        else:  # Normal state
            self._adjust_for_normal()
            
        # Record adjustment
        self.adjustment_history.append({
            "timestamp": datetime.now().isoformat(),
            "friction": friction,
            "trend": trend.get("trend", "unknown"),
            "thresholds": dict(self.current_thresholds)
        })
        
        # Trim history to last 100 entries
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
            
        return self.current_thresholds
    
    def _adjust_for_high_friction(self) -> None:
        """Adjust thresholds for high-friction state."""
        # Be more conservative, slow down processing
        self.current_thresholds["coherence"] = min(0.9, self.baseline_thresholds["coherence"] * 1.2)
        self.current_thresholds["entropy"] = max(0.1, self.baseline_thresholds["entropy"] * 0.5)
        self.current_thresholds["redundancy"] = min(0.95, self.baseline_thresholds["redundancy"] * 1.1) 
        self.current_thresholds["confidence"] = min(0.9, self.baseline_thresholds["confidence"] * 1.3)
        self.current_thresholds["processing_delay"] = 0.5  # Add delay to slow down
        
        logger.info("High friction detected - adjusting to conservative thresholds")
    
    def _adjust_for_moderate_friction(self) -> None:
        """Adjust thresholds for moderate-friction state."""
        # Slightly more conservative
        self.current_thresholds["coherence"] = min(0.8, self.baseline_thresholds["coherence"] * 1.1)
        self.current_thresholds["entropy"] = self.baseline_thresholds["entropy"]
        self.current_thresholds["redundancy"] = min(0.9, self.baseline_thresholds["redundancy"] * 1.05)
        self.current_thresholds["confidence"] = min(0.8, self.baseline_thresholds["confidence"] * 1.2)
        self.current_thresholds["processing_delay"] = 0.2  # Small delay
        
        logger.info("Moderate friction detected - adjusting thresholds")
    
    def _adjust_for_flow(self) -> None:
        """Adjust thresholds for flow state (low friction)."""
        # Be more permissive, accelerate processing
        self.current_thresholds["coherence"] = max(0.6, self.baseline_thresholds["coherence"] * 0.9)
        self.current_thresholds["entropy"] = min(0.3, self.baseline_thresholds["entropy"] * 1.5)
        self.current_thresholds["redundancy"] = max(0.75, self.baseline_thresholds["redundancy"] * 0.9)
        self.current_thresholds["confidence"] = max(0.5, self.baseline_thresholds["confidence"] * 0.8)
        self.current_thresholds["processing_delay"] = 0.0  # No delay
        
        logger.info("Flow state detected - relaxing thresholds")
    
    def _adjust_for_normal(self) -> None:
        """Reset thresholds to baseline for normal operation."""
        # Gradual return to baseline
        for key in self.current_thresholds:
            if key == "processing_delay":
                # Special case for delay
                self.current_thresholds[key] *= 0.5  # Decay delay
            else:
                # Move 20% closer to baseline
                diff = self.baseline_thresholds[key] - self.current_thresholds[key]
                self.current_thresholds[key] += diff * 0.2
                
        logger.debug("Normal operation - gradually returning to baseline thresholds")
    
    def get_current_rhythm_state(self) -> Dict[str, Any]:
        """Get the current rhythm state of the system."""
        return {
            "thresholds": dict(self.current_thresholds),
            "friction": self.friction_monitor.get_overall_friction(),
            "trend": self.friction_monitor.get_trend().get("trend", "unknown"),
            "mode": self._determine_current_mode()
        }
    
    def _determine_current_mode(self) -> str:
        """Determine the current operating mode based on friction and thresholds."""
        friction = self.friction_monitor.get_overall_friction()
        
        if friction > 0.8:
            return "conservative"
        elif friction > 0.5:
            return "cautious"
        elif friction < 0.2:
            return "flow"
        else:
            return "normal"
            
    def apply_processing_delay(self) -> None:
        """Apply the current processing delay to slow down if needed."""
        delay = self.current_thresholds.get("processing_delay", 0)
        if delay > 0:
            time.sleep(delay)
            
    def save_rhythm_state(self, path: str) -> None:
        """Save current rhythm state to a file."""
        state = {
            "current_state": self.get_current_rhythm_state(),
            "adjustment_history": self.adjustment_history[-20:],  # Last 20 adjustments
            "baseline_thresholds": self.baseline_thresholds
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved rhythm state to {path}")


class LearningTracker:
    """
    Tracks concept learning patterns and stability over time.
    
    The LearningTracker monitors which concepts ALAN revisits, struggles with,
    or overuses, providing insight into the learning process.
    """
    
    def __init__(self):
        self.concept_visits = defaultdict(list)  # concept_id -> list of timestamps
        self.concept_stability = {}  # concept_id -> stability score (0-1)
        self.start_time = datetime.now()
        
    def record_concept_visit(self, concept_id: str) -> None:
        """Record a visit/activation of a concept."""
        self.concept_visits[concept_id].append(datetime.now().isoformat())
        
    def record_concept_stability(
        self, 
        concept_id: str,
        stability: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record stability measurement for a concept.
        
        Args:
            concept_id: Concept identifier
            stability: Stability score (0-1)
            context: Optional context information
        """
        if concept_id not in self.concept_stability:
            self.concept_stability[concept_id] = {
                "current": stability,
                "history": [],
                "contexts": []
            }
        else:
            self.concept_stability[concept_id]["current"] = stability
            
        self.concept_stability[concept_id]["history"].append({
            "timestamp": datetime.now().isoformat(),
            "value": stability
        })
        
        if context:
            self.concept_stability[concept_id]["contexts"].append({
                "timestamp": datetime.now().isoformat(),
                **context
            })
            
    def get_visit_frequency(self, concept_id: str, window_hours: float = 24) -> int:
        """Get number of visits to a concept within the time window."""
        if concept_id not in self.concept_visits:
            return 0
            
        cutoff = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        return sum(1 for ts in self.concept_visits[concept_id] if ts >= cutoff)
        
    def get_most_visited_concepts(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequently visited concepts."""
        all_visits = [(cid, len(visits)) for cid, visits in self.concept_visits.items()]
        return sorted(all_visits, key=lambda x: x[1], reverse=True)[:top_n]
        
    def get_stability_trend(self, concept_id: str) -> Dict[str, Any]:
        """Analyze stability trend for a specific concept."""
        if concept_id not in self.concept_stability:
            return {"status": "unknown"}
            
        history = self.concept_stability[concept_id]["history"]
        if len(history) < 3:
            return {"status": "insufficient_data"}
            
        values = [h["value"] for h in history]
        current = values[-1]
        avg = sum(values) / len(values)
        
        # Get trend direction
        window = min(5, len(values))
        recent = values[-window:]
        
        if len(recent) < 2:
            trend = "unknown"
        elif recent[-1] > recent[0] * 1.1:
            trend = "improving"
        elif recent[-1] < recent[0] * 0.9:
            trend = "declining"
        else:
            trend = "stable"
            
        return {
            "status": "analyzed",
            "current": current,
            "average": avg,
            "trend": trend,
            "measurements": len(history)
        }
        
    def get_unstable_concepts(self, threshold: float = 0.4) -> List[str]:
        """Get concepts with stability below threshold."""
        return [
            cid for cid, data in self.concept_stability.items()
            if data["current"] < threshold
        ]
        
    def get_learning_summary(self) -> Dict[str, Any]:
        """Generate a summary of learning patterns."""
        unstable = self.get_unstable_concepts()
        most_visited = self.get_most_visited_concepts()
        
        # Calculate overall learning state
        avg_stability = (
            sum(data["current"] for data in self.concept_stability.values()) / 
            len(self.concept_stability) if self.concept_stability else 0
        )
        
        return {
            "total_concepts_tracked": len(self.concept_visits),
            "total_stability_measurements": sum(len(data["history"]) for data in self.concept_stability.values()),
            "average_stability": avg_stability,
            "unstable_concept_count": len(unstable),
            "most_visited_concepts": most_visited[:5],  # Top 5
            "session_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
    def save_learning_state(self, path: str) -> None:
        """Save learning state to a file."""
        state = {
            "summary": self.get_learning_summary(),
            "unstable_concepts": [
                {
                    "id": cid,
                    "stability": self.concept_stability[cid]["current"],
                    "trend": self.get_stability_trend(cid).get("trend", "unknown")
                }
                for cid in self.get_unstable_concepts()
            ],
            "most_visited": [
                {
                    "id": cid,
                    "visits": count,
                    "stability": self.concept_stability.get(cid, {}).get("current", "unknown")
                }
                for cid, count in self.get_most_visited_concepts(10)
            ]
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved learning state to {path}")


class SelfReflectionEngine:
    """
    Enables ALAN to reflect on its own performance and state.
    
    The SelfReflectionEngine integrates information from other introspection 
    components to generate assessments of ALAN's overall cognitive state
    and performance.
    """
    
    def __init__(
        self,
        friction_monitor: FrictionMonitor,
        rhythm_modulator: RhythmModulator,
        learning_tracker: LearningTracker
    ):
        self.friction_monitor = friction_monitor
        self.rhythm_modulator = rhythm_modulator
        self.learning_tracker = learning_tracker
        self.reflection_log = []
        self.last_reflection_time = datetime.now()
        self.reflection_interval = 3600  # Default: reflect hourly
        
    def reflect(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate a self-reflection on current system state.
        
        Args:
            force: Force reflection even if interval hasn't elapsed
            
        Returns:
            Dictionary containing reflection data
        """
        now = datetime.now()
        elapsed = (now - self.last_reflection_time).total_seconds()
        
        if not force and elapsed < self.reflection_interval:
            # Not time for reflection yet
            return {
                "status": "deferred",
                "seconds_until_next": self.reflection_interval - elapsed
            }
            
        # Gather data from all monitoring components
        friction_data = self.friction_monitor.generate_report()
        rhythm_state = self.rhythm_modulator.get_current_rhythm_state()
        learning_summary = self.learning_tracker.get_learning_summary()
        
        # Calculate overall cognitive health score (0-100)
        health_score = self._calculate_health_score(
            friction_data, rhythm_state, learning_summary
        )
        
        # Generate descriptive assessment
        assessment = self._generate_assessment(
            health_score, friction_data, rhythm_state, learning_summary
        )
        
        # Create reflection record
        reflection = {
            "timestamp": now.isoformat(),
            "health_score": health_score,
            "assessment": assessment,
            "friction_level": friction_data["overall_friction"],
            "operating_mode": rhythm_state["mode"],
            "average_concept_stability": learning_summary["average_stability"],
            "unstable_concepts": learning_summary["unstable_concept_count"],
            "suggestions": self._generate_suggestions(health_score, friction_data, rhythm_state, learning_summary)
        }
        
        # Record reflection
        self.reflection_log.append(reflection)
        self.last_reflection_time = now
        
        # Trim log if needed
        if len(self.reflection_log) > 100:
            self.reflection_log = self.reflection_log[-100:]
            
        logger.info(f"Self-reflection complete: Health score {health_score}/100")
        return reflection
    
    def _calculate_health_score(
        self,
        friction_data: Dict[str, Any],
        rhythm_state: Dict[str, Any],
        learning_summary: Dict[str, Any]
    ) -> float:
        """Calculate overall cognitive health score (0-100)."""
        # Friction component (0-40 points)
        friction_score = 40 * (1 - friction_data["overall_friction"])
        
        # Rhythm component (0-30 points)
        mode_scores = {
            "flow": 30,
            "normal": 25,
            "cautious": 15,
            "conservative": 5
        }
        rhythm_score = mode_scores.get(rhythm_state["mode"], 20)
        
        # Learning component (0-30 points)
        stability_score = 30 * learning_summary["average_stability"]
        
        # Total score
        total = friction_score + rhythm_score + stability_score
        
        return min(100, max(0, round(total)))
    
    def _generate_assessment(
        self,
        health_score: float,
        friction_data: Dict[str, Any],
        rhythm_state: Dict[str, Any],
        learning_summary: Dict[str, Any]
    ) -> str:
        """Generate descriptive assessment of system state."""
        if health_score >= 90:
            return "Excellent cognitive flow. System is operating with high stability and minimal friction."
        elif health_score >= 75:
            return "Good overall performance. Some minor oscillations but maintaining coherence."
        elif health_score >= 60:
            return "Moderate cognitive health. System is functional but experiencing some friction points."
        elif health_score >= 40:
            return "Experiencing significant cognitive strain. Multiple unstable concepts and friction events."
        else:
            return "System in cognitive distress. High friction, low stability, and poor phase coherence."
    
    def _generate_suggestions(
        self,
        health_score: float,
        friction_data: Dict[str, Any],
        rhythm_state: Dict[str, Any],
        learning_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable suggestions based on system state."""
        suggestions = []
        
        # Friction-based suggestions
        if friction_data["overall_friction"] > 0.6:
            suggestions.append("Reduce processing velocity to allow for reintegration of destabilized concepts.")
            
        if len(friction_data["troublesome_concepts"]) > 0:
            suggestions.append(f"Examine {len(friction_data['troublesome_concepts'])} problematic concepts for phase recalibration.")
        
        # Rhythm-based suggestions
        if rhythm_state["mode"] == "conservative":
            suggestions.append("Consider temporary reduction in concept intake to stabilize existing structure.")
            
        # Learning-based suggestions
        if learning_summary["unstable_concept_count"] > 5:
            suggestions.append("Run targeted memory sculptor cycle focusing on unstable concepts.")
            
        if learning_summary["average_stability"] < 0.5:
            suggestions.append("Perform spectral recalibration to strengthen eigenfunction foundations.")
        
        # If everything is good
        if health_score >= 85 and not suggestions:
            suggestions.append("Maintain current operating parameters. System is in optimal state.")
            
        return suggestions
    
    def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reflection history."""
        return self.reflection_log[-limit:]
    
    def get_health_trend(self, window: int = 5) -> Dict[str, Any]:
        """Analyze trend in cognitive health scores."""
        if len(self.reflection_log) < window:
            return {"status": "insufficient_data"}
            
        recent = self.reflection_log[-window:]
        scores = [r["health_score"] for r in recent]
        
        avg = sum(scores) / len(scores)
        trend = "unknown"
        
        if len(scores) >= 3:
            if scores[-1] > scores[0] * 1.1:
                trend = "improving"
            elif scores[-1] < scores[0] * 0.9:
                trend = "declining"
            else:
                trend = "stable"
                
        return {
            "status": "analyzed",
            "average_score": avg,
            "current_score": scores[-1],
            "trend": trend,
            "data_points": len(scores)
        }
    
    def save_reflection_log(self, path: str) -> None:
        """Save reflection log to a file."""
        data = {
            "log": self.reflection_log,
            "summary": {
                "total_reflections": len(self.reflection_log),
                "latest_score": self.reflection_log[-1]["health_score"] if self.reflection_log else None,
                "trend": self.get_health_trend()
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved reflection log to {path}")


class IntrospectionSystem:
    """
    Main class integrating all introspection components.
    
    The IntrospectionSystem provides a unified interface to ALAN's 
    self-monitoring capabilities, enabling the system to observe 
    and regulate its own cognitive processes.
    """
    
    def __init__(self, log_dir: str = "logs/introspection"):
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        
        # Initialize components
        self.friction_monitor = FrictionMonitor()
        self.rhythm_modulator = RhythmModulator(self.friction_monitor)
        self.learning_tracker = LearningTracker()
        self.self_reflection = SelfReflectionEngine(
            self.friction_monitor,
            self.rhythm_modulator,
            self.learning_tracker
        )
        
        logger.info("Introspection system initialized")
        
    def monitor_concept_activation(
        self, 
        concept: ConceptTuple,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Monitor activation of a concept.
        
        Args:
            concept: The concept being activated
            context: Optional context information
        """
        if not concept or not hasattr(concept, 'eigenfunction_id') or not concept.eigenfunction_id:
            logger.warning("Attempted to monitor activation of invalid concept")
            return
            
        # Track concept activation in learning tracker
        self.learning_tracker.record_concept_visit(concept.eigenfunction_id)
        
        # Calculate concept stability based on predictability and coherence
        stability = 0.5  # Default value
        
        if hasattr(concept, 'predictability_score'):
            # Higher predictability means more stable
            stability = concept.predictability_score
            
        if hasattr(concept, 'cluster_coherence'):
            # Adjust with cluster coherence if available
            stability = 0.7 * stability + 0.3 * concept.cluster_coherence
            
        # Record stability measurement
        self.learning_tracker.record_concept_stability(
            concept.eigenfunction_id,
            stability,
            context
        )
        
        # Check for instability
        if stability < 0.4:
            self.friction_monitor.log_concept_instability(
                concept.eigenfunction_id,
                stability,
                0.4,
                context
            )
            
        logger.debug(f"Monitored activation of concept {concept.name} "
                    f"(eigenfunction: {concept.eigenfunction_id[:8]}...), "
                    f"stability: {stability:.2f}")
    
    def monitor_concept_interaction(
        self,
        concept1: ConceptTuple,
        concept2: ConceptTuple,
        expected_coherence: Optional[float] = None,
        actual_coherence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Monitor interaction between two concepts.
        
        Args:
            concept1, concept2: The interacting concepts
            expected_coherence: Expected phase coherence
            actual_coherence: Actual observed coherence
            context: Additional context information
        """
        if not concept1 or not concept2:
            return
            
        # Extract IDs
        id1 = concept1.eigenfunction_id if hasattr(concept1, 'eigenfunction_id') else "unknown"
        id2 = concept2.eigenfunction_id if hasattr(concept2, 'eigenfunction_id') else "unknown"
        name1 = concept1.name if hasattr(concept1, 'name') else "unnamed"
        name2 = concept2.name if hasattr(concept2, 'name') else "unnamed"
        
        # If coherence values were provided, check for desynchronization
        if expected_coherence is not None and actual_coherence is not None:
            # Threshold for significant difference
            if abs(expected_coherence - actual_coherence) > 0.2:
                self.friction_monitor.log_phase_desync(
                    id1, id2, expected_coherence, actual_coherence,
                    {
                        "concept1_name": name1,
                        "concept2_name": name2,
                        **(context or {})
                    }
                )
                
        logger.debug(f"Monitored interaction between concepts '{name1}' and '{name2}'")
        
    def update_rhythm(self) -> Dict[str, float]:
        """
        Update system rhythm based on current state.
        
        Returns adjusted thresholds dictionary.
        """
        thresholds = self.rhythm_modulator.update_thresholds()
        
        # Apply processing delay if needed
        self.rhythm_modulator.apply_processing_delay()
        
        return thresholds
        
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current system thresholds."""
        return dict(self.rhythm_modulator.current_thresholds)
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        # Force a reflection to get current state
        reflection = self.self_reflection.reflect(force=True)
        
        return {
            "health_score": reflection["health_score"],
            "assessment": reflection["assessment"],
            "operating_mode": reflection["operating_mode"],
            "friction_level": reflection["friction_level"],
            "concept_stability": reflection["average_concept_stability"],
            "suggestions": reflection["suggestions"]
        }
        
    def save_state(self, base_path: Optional[str] = None) -> Dict[str, str]:
        """
        Save all introspection component states to files.
        
        Args:
            base_path: Optional base path for log files
            
        Returns:
            Dictionary mapping component names to saved file paths
        """
        if not base_path:
            base_path = self.log_dir
            
        # Create directory if needed
        os.makedirs(base_path, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each component's state
        paths = {}
        
        # Save friction monitor state
        friction_path = os.path.join(base_path, f"friction_{timestamp}.json")
        self.friction_monitor.save_report(friction_path)
        paths["friction"] = friction_path
        
        # Save rhythm state
        rhythm_path = os.path.join(base_path, f"rhythm_{timestamp}.json")
        self.rhythm_modulator.save_rhythm_state(rhythm_path)
        paths["rhythm"] = rhythm_path
        
        # Save learning state
        learning_path = os.path.join(base_path, f"learning_{timestamp}.json")
        self.learning_tracker.save_learning_state(learning_path)
        paths["learning"] = learning_path
        
        # Save reflection log
        reflection_path = os.path.join(base_path, f"reflection_{timestamp}.json")
        self.self_reflection.save_reflection_log(reflection_path)
        paths["reflection"] = reflection_path
        
        logger.info(f"Saved introspection state to {base_path}")
        return paths

# Create a singleton introspection system
_introspection_system = None

def get_introspection_system(log_dir: str = "logs/introspection") -> IntrospectionSystem:
    """Get or create the singleton introspection system."""
    global _introspection_system
    if _introspection_system is None:
        _introspection_system = IntrospectionSystem(log_dir)
    return _introspection_system
