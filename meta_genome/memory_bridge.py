# /meta_genome/memory_bridge.py
"""
Bridge between self-transformation system and persistent memory vault.
Enables true metacognition through temporal self-awareness.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add python core to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from python.core.memory_vault import MemoryVault
from python.core.soliton_memory_integration import SolitonMemory
from meta_genome.critics.aggregation import aggregate
from meta.energy_budget import EnergyBudget
from audit.logger import log_event

class MetacognitiveMemoryBridge:
    """
    Connects self-transformation components with persistent memory,
    enabling true metacognition across sessions.
    """
    
    def __init__(self, memory_vault: Optional[MemoryVault] = None):
        self.memory_vault = memory_vault or MemoryVault()
        self.soliton_memory = SolitonMemory()
        
        # Initialize metacognitive memories
        self._init_metacognitive_stores()
        
    def _init_metacognitive_stores(self):
        """Initialize persistent stores for metacognitive data"""
        stores = [
            "critic_history",      # Historical critic performance
            "self_reflections",    # Accumulated self-knowledge
            "transformation_log",  # History of self-modifications
            "relationship_graph",  # Persistent relationships (birthdays, preferences)
            "error_patterns",      # Recurring mistakes across sessions
            "strategy_evolution",  # How strategies change over time
        ]
        
        for store in stores:
            if not self.memory_vault.has_store(store):
                self.memory_vault.create_store(store)
    
    def remember_critic_decision(self, critic_id: str, score: float, 
                               reliability: float, decision: str, outcome: Optional[bool] = None):
        """Store critic decisions for long-term reliability tracking"""
        memory = {
            "timestamp": datetime.now().isoformat(),
            "critic_id": critic_id,
            "score": score,
            "reliability": reliability,
            "decision": decision,
            "outcome": outcome,  # Was the decision correct in hindsight?
        }
        
        self.memory_vault.store("critic_history", f"{critic_id}_{datetime.now().timestamp()}", memory)
        
        # Update critic reliability based on historical performance
        if outcome is not None:
            self._update_critic_reliability(critic_id, outcome)
    
    def _update_critic_reliability(self, critic_id: str, was_correct: bool):
        """Bayesian update of critic reliability based on outcome"""
        # Retrieve historical performance
        history = self.memory_vault.query("critic_history", 
                                        filter_func=lambda x: x.get("critic_id") == critic_id)
        
        # Calculate new reliability using Beta-Bernoulli update
        alpha = sum(1 for h in history if h.get("outcome") == True) + 2  # Prior Beta(2,2)
        beta = sum(1 for h in history if h.get("outcome") == False) + 2
        
        new_reliability = alpha / (alpha + beta)
        
        # Store updated reliability
        self.memory_vault.store("critic_history", f"{critic_id}_reliability", {
            "reliability": new_reliability,
            "alpha": alpha,
            "beta": beta,
            "sample_size": len(history),
            "updated": datetime.now().isoformat()
        })
    
    def add_self_reflection(self, reflection_type: str, content: Dict[str, Any]):
        """Store metacognitive self-reflections that accumulate over time"""
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "type": reflection_type,
            "content": content,
            "session_id": self.soliton_memory.current_session_id()
        }
        
        reflection_id = f"reflection_{datetime.now().timestamp()}"
        self.memory_vault.store("self_reflections", reflection_id, reflection)
        
        # Also store in soliton memory for phase-coherent access
        self.soliton_memory.record_thought(content.get("thought", ""), 
                                         metadata={"type": "self_reflection"})
    
    def remember_transformation(self, mutation_id: str, mutation_type: str,
                              success: bool, impact_metrics: Dict[str, float]):
        """Log self-transformations for learning what works"""
        transformation = {
            "timestamp": datetime.now().isoformat(),
            "mutation_id": mutation_id,
            "type": mutation_type,
            "success": success,
            "metrics": impact_metrics,
            "energy_cost": impact_metrics.get("energy_cost", 0),
            "utility_gained": impact_metrics.get("utility", 0)
        }
        
        self.memory_vault.store("transformation_log", mutation_id, transformation)
        
        # Analyze patterns in successful transformations
        self._analyze_transformation_patterns()
    
    def _analyze_transformation_patterns(self):
        """Identify patterns in successful vs failed transformations"""
        transformations = self.memory_vault.get_all("transformation_log")
        
        if len(transformations) < 10:
            return  # Need more data
        
        successful = [t for t in transformations if t["success"]]
        failed = [t for t in transformations if not t["success"]]
        
        patterns = {
            "success_rate": len(successful) / len(transformations),
            "avg_energy_successful": np.mean([t["energy_cost"] for t in successful]) if successful else 0,
            "avg_energy_failed": np.mean([t["energy_cost"] for t in failed]) if failed else 0,
            "best_transformation_types": self._get_best_types(transformations),
            "analyzed_at": datetime.now().isoformat()
        }
        
        self.memory_vault.store("transformation_log", "_patterns", patterns)
    
    def _get_best_types(self, transformations: List[Dict]) -> List[str]:
        """Identify most successful transformation types"""
        type_stats = {}
        
        for t in transformations:
            t_type = t["type"]
            if t_type not in type_stats:
                type_stats[t_type] = {"success": 0, "total": 0}
            
            type_stats[t_type]["total"] += 1
            if t["success"]:
                type_stats[t_type]["success"] += 1
        
        # Calculate success rates
        success_rates = {
            t_type: stats["success"] / stats["total"] 
            for t_type, stats in type_stats.items()
            if stats["total"] > 0
        }
        
        # Return top 3 types by success rate
        return sorted(success_rates.keys(), 
                     key=lambda x: success_rates[x], 
                     reverse=True)[:3]
    
    def remember_entity(self, entity_id: str, attributes: Dict[str, Any]):
        """Remember information about entities (users, systems, etc)"""
        # Check if entity exists
        existing = self.memory_vault.get("relationship_graph", entity_id)
        
        if existing:
            # Merge attributes
            existing["attributes"].update(attributes)
            existing["last_updated"] = datetime.now().isoformat()
            existing["interaction_count"] = existing.get("interaction_count", 0) + 1
        else:
            existing = {
                "entity_id": entity_id,
                "attributes": attributes,
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "interaction_count": 1
            }
        
        self.memory_vault.store("relationship_graph", entity_id, existing)
        
        # Special handling for temporal attributes (birthdays, anniversaries)
        if "birthday" in attributes:
            self._schedule_reminder("birthday", entity_id, attributes["birthday"])
    
    def _schedule_reminder(self, reminder_type: str, entity_id: str, date_str: str):
        """Schedule future reminders for important dates"""
        reminder = {
            "type": reminder_type,
            "entity_id": entity_id,
            "date": date_str,
            "created": datetime.now().isoformat()
        }
        
        self.memory_vault.store("relationship_graph", 
                              f"reminder_{entity_id}_{reminder_type}", 
                              reminder)
    
    def recall_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Recall everything known about an entity"""
        return self.memory_vault.get("relationship_graph", entity_id)
    
    def get_active_reminders(self) -> List[Dict[str, Any]]:
        """Get reminders that should be activated today"""
        today = datetime.now().date()
        reminders = self.memory_vault.query("relationship_graph",
                                          filter_func=lambda x: x.get("type") == "birthday")
        
        active = []
        for reminder in reminders:
            reminder_date = datetime.fromisoformat(reminder["date"]).date()
            if reminder_date.month == today.month and reminder_date.day == today.day:
                active.append(reminder)
        
        return active
    
    def record_error_pattern(self, error_type: str, context: Dict[str, Any]):
        """Track recurring error patterns across sessions"""
        error_id = f"error_{error_type}_{datetime.now().timestamp()}"
        
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "context": context,
            "session_id": self.soliton_memory.current_session_id()
        }
        
        self.memory_vault.store("error_patterns", error_id, error_record)
        
        # Check if this is a recurring pattern
        similar_errors = self.memory_vault.query("error_patterns",
                                               filter_func=lambda x: x.get("type") == error_type)
        
        if len(similar_errors) > 3:
            # This is a recurring pattern - trigger metacognitive analysis
            self._analyze_error_pattern(error_type, similar_errors)
    
    def _analyze_error_pattern(self, error_type: str, errors: List[Dict]):
        """Analyze recurring errors to identify root causes"""
        analysis = {
            "error_type": error_type,
            "frequency": len(errors),
            "first_occurrence": min(e["timestamp"] for e in errors),
            "last_occurrence": max(e["timestamp"] for e in errors),
            "common_context": self._find_common_context(errors),
            "suggested_remedy": self._suggest_remedy(error_type, errors)
        }
        
        self.add_self_reflection("error_analysis", {
            "thought": f"I keep making {error_type} errors. Pattern analysis suggests: {analysis['suggested_remedy']}",
            "analysis": analysis
        })
    
    def _find_common_context(self, errors: List[Dict]) -> Dict[str, Any]:
        """Find common elements in error contexts"""
        if not errors:
            return {}
        
        # Simple implementation - find keys present in all contexts
        common_keys = set(errors[0].get("context", {}).keys())
        for error in errors[1:]:
            common_keys &= set(error.get("context", {}).keys())
        
        return {key: "varies" for key in common_keys}
    
    def _suggest_remedy(self, error_type: str, errors: List[Dict]) -> str:
        """Suggest remediation based on error patterns"""
        # Simple heuristic-based suggestions
        if "timeout" in error_type.lower():
            return "Increase timeout limits or optimize performance"
        elif "memory" in error_type.lower():
            return "Implement better memory management or increase limits"
        elif "type" in error_type.lower():
            return "Add stronger type checking or validation"
        else:
            return f"Implement specific handler for {error_type} errors"
    
    def evolve_strategy(self, strategy_name: str, performance_metrics: Dict[str, float]):
        """Track how strategies evolve over time based on performance"""
        evolution_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name,
            "metrics": performance_metrics,
            "session_id": self.soliton_memory.current_session_id()
        }
        
        # Store in strategy evolution log
        evolution_id = f"{strategy_name}_{datetime.now().timestamp()}"
        self.memory_vault.store("strategy_evolution", evolution_id, evolution_record)
        
        # Analyze strategy performance over time
        history = self.memory_vault.query("strategy_evolution",
                                        filter_func=lambda x: x.get("strategy_name") == strategy_name)
        
        if len(history) > 5:
            trend = self._analyze_strategy_trend(history)
            if trend["direction"] == "declining":
                self.add_self_reflection("strategy_analysis", {
                    "thought": f"Strategy '{strategy_name}' showing declining performance. Time to try something new.",
                    "trend": trend
                })
    
    def _analyze_strategy_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trend of a strategy"""
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        # Simple linear regression on primary metric
        if history and "primary_metric" in history[0].get("metrics", {}):
            values = [h["metrics"]["primary_metric"] for h in history[-5:]]
            
            # Simple trend detection
            recent_avg = np.mean(values[-3:])
            older_avg = np.mean(values[:2])
            
            if recent_avg < older_avg * 0.9:
                direction = "declining"
            elif recent_avg > older_avg * 1.1:
                direction = "improving"
            else:
                direction = "stable"
            
            return {
                "direction": direction,
                "recent_performance": recent_avg,
                "historical_performance": older_avg,
                "samples": len(history)
            }
        
        return {"direction": "unknown", "samples": len(history)}
    
    def get_metacognitive_summary(self) -> Dict[str, Any]:
        """Generate a summary of metacognitive state"""
        return {
            "total_reflections": len(self.memory_vault.get_all("self_reflections")),
            "transformation_success_rate": self._get_transformation_success_rate(),
            "known_entities": len(self.memory_vault.get_all("relationship_graph")),
            "error_patterns_identified": len(set(e["type"] for e in self.memory_vault.get_all("error_patterns"))),
            "strategies_tracked": len(set(s["strategy_name"] for s in self.memory_vault.get_all("strategy_evolution"))),
            "critic_reliabilities": self._get_all_critic_reliabilities(),
            "memory_coherence": self.soliton_memory.check_coherence(),
            "generated_at": datetime.now().isoformat()
        }
    
    def _get_transformation_success_rate(self) -> float:
        """Calculate overall transformation success rate"""
        transformations = self.memory_vault.get_all("transformation_log")
        if not transformations:
            return 0.0
        
        successful = sum(1 for t in transformations if t["success"])
        return successful / len(transformations)
    
    def _get_all_critic_reliabilities(self) -> Dict[str, float]:
        """Get current reliability scores for all critics"""
        reliabilities = {}
        
        # Find all reliability records
        all_critics = set()
        for record in self.memory_vault.get_all("critic_history"):
            if "critic_id" in record:
                all_critics.add(record["critic_id"])
        
        for critic_id in all_critics:
            rel_record = self.memory_vault.get("critic_history", f"{critic_id}_reliability")
            if rel_record:
                reliabilities[critic_id] = rel_record["reliability"]
            else:
                reliabilities[critic_id] = 0.5  # Default prior
        
        return reliabilities
