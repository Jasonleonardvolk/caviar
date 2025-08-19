"""
World Model: Internal Simulation Engine for Prajna
==================================================

Production implementation of Prajna's causal world simulation and consistency validation.
This module maintains an internal representation of the world state, simulates causal
consequences of hypothetical scenarios, validates reasoning consistency, and manages
dynamic knowledge updates.

This is where Prajna gains causal understanding - the ability to simulate "what if"
scenarios and maintain coherent world knowledge across reasoning operations.
"""

import asyncio
import logging
import time
import json
import math
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger("prajna.world_model")

class EntityType(Enum):
    """Types of entities in the world model"""
    CONCEPT = "concept"
    FACT = "fact"
    RELATIONSHIP = "relationship"
    RULE = "rule"
    PROCESS = "process"
    AGENT = "agent"
    EVENT = "event"
    STATE = "state"

class ChangeType(Enum):
    """Types of changes that can occur in the world"""
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    STRENGTHEN = "strengthen"
    WEAKEN = "weaken"

@dataclass
class WorldEntity:
    """Individual entity in the world model"""
    entity_id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Relationship data
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [entity_ids]
    dependencies: Set[str] = field(default_factory=set)  # entities this depends on
    dependents: Set[str] = field(default_factory=set)   # entities that depend on this
    
    # State tracking
    confidence: float = 1.0
    stability: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    
    # Causal properties
    causal_strength: float = 0.5  # How strongly this entity affects others
    volatility: float = 0.1       # How likely this entity is to change
    
    def update_properties(self, new_properties: Dict[str, Any]):
        """Update entity properties"""
        self.properties.update(new_properties)
        self.last_updated = datetime.now()
        self.update_count += 1

@dataclass
class CausalRule:
    """Causal rule in the world model"""
    rule_id: str
    condition: Dict[str, Any]      # Conditions that trigger this rule
    effect: Dict[str, Any]         # Effects when rule is triggered
    strength: float                # How strongly the rule applies (0.0-1.0)
    confidence: float              # Confidence in the rule (0.0-1.0)
    domain: str                    # Domain this rule applies to
    
    # Rule metadata
    source: str = ""
    examples: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def applies_to(self, world_state: 'WorldState') -> Tuple[bool, float]:
        """Check if this rule applies to the current world state"""
        applicability_score = 0.0
        condition_count = 0
        
        for condition_key, condition_value in self.condition.items():
            condition_count += 1
            if world_state.has_property(condition_key, condition_value):
                applicability_score += 1.0
            elif world_state.has_similar_property(condition_key, condition_value):
                applicability_score += 0.5
        
        if condition_count == 0:
            return False, 0.0
        
        applies = applicability_score / condition_count >= 0.5
        score = (applicability_score / condition_count) * self.strength
        
        return applies, score

@dataclass
class SimulationStep:
    """Single step in a world simulation"""
    step_id: int
    timestamp: datetime
    changes: List[Dict[str, Any]]     # Changes made in this step
    triggered_rules: List[str]        # Rules that were triggered
    affected_entities: Set[str]       # Entities affected by changes
    step_description: str             # Human-readable description
    confidence: float                 # Confidence in this simulation step

@dataclass
class SimulationResult:
    """Complete result of world simulation"""
    simulation_id: str
    initial_state: 'WorldState'
    final_state: 'WorldState'
    steps: List[SimulationStep]
    
    # Simulation metrics
    total_changes: int
    cascade_depth: int                # How many levels of causation occurred
    consistency_score: float          # How consistent the simulation is
    plausibility_score: float         # How plausible the result is
    
    # Metadata
    hypothesis: str                   # What was being simulated
    simulation_time: float            # Time taken to simulate
    success: bool                     # Whether simulation completed successfully

class WorldState:
    """Represents the current state of the world as understood by Prajna"""
    
    def __init__(self):
        self.entities: Dict[str, WorldEntity] = {}
        self.facts: Dict[str, Any] = {}
        self.relationships: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (entity1, entity2) -> relationship_data
        self.state_id = self._generate_state_id()
        self.timestamp = datetime.now()
        self.consistency_score = 1.0
        
    def _generate_state_id(self) -> str:
        """Generate unique state ID"""
        return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16]
    
    def add_entity(self, entity: WorldEntity):
        """Add entity to world state"""
        self.entities[entity.entity_id] = entity
        self._update_state_id()
    
    def remove_entity(self, entity_id: str):
        """Remove entity from world state"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Remove relationships involving this entity
            to_remove = []
            for (e1, e2) in self.relationships:
                if e1 == entity_id or e2 == entity_id:
                    to_remove.append((e1, e2))
            
            for rel_key in to_remove:
                del self.relationships[rel_key]
            
            # Update dependencies
            for dependent_id in entity.dependents:
                if dependent_id in self.entities:
                    self.entities[dependent_id].dependencies.discard(entity_id)
            
            del self.entities[entity_id]
            self._update_state_id()
    
    def add_fact(self, fact_key: str, fact_value: Any, confidence: float = 1.0):
        """Add fact to world state"""
        self.facts[fact_key] = {
            "value": fact_value,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
        self._update_state_id()
    
    def get_fact(self, fact_key: str) -> Optional[Any]:
        """Get fact value from world state"""
        fact_data = self.facts.get(fact_key)
        return fact_data["value"] if fact_data else None
    
    def has_property(self, property_key: str, property_value: Any) -> bool:
        """Check if world state has specific property"""
        # Check in facts
        if property_key in self.facts:
            return self.facts[property_key]["value"] == property_value
        
        # Check in entity properties
        for entity in self.entities.values():
            if property_key in entity.properties:
                return entity.properties[property_key] == property_value
        
        return False
    
    def has_similar_property(self, property_key: str, property_value: Any) -> bool:
        """Check if world state has similar property (fuzzy match)"""
        # For numeric values, check if within 20% range
        if isinstance(property_value, (int, float)):
            tolerance = abs(property_value) * 0.2
            
            # Check facts
            if property_key in self.facts:
                fact_value = self.facts[property_key]["value"]
                if isinstance(fact_value, (int, float)):
                    return abs(fact_value - property_value) <= tolerance
            
            # Check entity properties
            for entity in self.entities.values():
                if property_key in entity.properties:
                    entity_value = entity.properties[property_key]
                    if isinstance(entity_value, (int, float)):
                        return abs(entity_value - property_value) <= tolerance
        
        # For string values, check if substring match
        elif isinstance(property_value, str):
            property_lower = property_value.lower()
            
            # Check facts
            if property_key in self.facts:
                fact_value = self.facts[property_key]["value"]
                if isinstance(fact_value, str):
                    return property_lower in fact_value.lower() or fact_value.lower() in property_lower
            
            # Check entity properties
            for entity in self.entities.values():
                if property_key in entity.properties:
                    entity_value = entity.properties[property_key]
                    if isinstance(entity_value, str):
                        return property_lower in entity_value.lower() or entity_value.lower() in property_lower
        
        return False
    
    def add_relationship(self, entity1_id: str, entity2_id: str, relationship_type: str, 
                        strength: float = 1.0, properties: Dict[str, Any] = None):
        """Add relationship between entities"""
        rel_key = (entity1_id, entity2_id)
        self.relationships[rel_key] = {
            "type": relationship_type,
            "strength": strength,
            "properties": properties or {},
            "timestamp": datetime.now()
        }
        
        # Update entity relationship tracking
        if entity1_id in self.entities:
            if relationship_type not in self.entities[entity1_id].relationships:
                self.entities[entity1_id].relationships[relationship_type] = []
            self.entities[entity1_id].relationships[relationship_type].append(entity2_id)
        
        if entity2_id in self.entities:
            # Add reverse relationship if applicable
            reverse_type = f"reverse_{relationship_type}"
            if reverse_type not in self.entities[entity2_id].relationships:
                self.entities[entity2_id].relationships[reverse_type] = []
            self.entities[entity2_id].relationships[reverse_type].append(entity1_id)
        
        self._update_state_id()
    
    def copy(self) -> 'WorldState':
        """Create deep copy of world state"""
        new_state = WorldState()
        
        # Copy entities
        for entity_id, entity in self.entities.items():
            new_entity = WorldEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=entity.properties.copy(),
                relationships={k: v.copy() for k, v in entity.relationships.items()},
                dependencies=entity.dependencies.copy(),
                dependents=entity.dependents.copy(),
                confidence=entity.confidence,
                stability=entity.stability,
                last_updated=entity.last_updated,
                update_count=entity.update_count,
                causal_strength=entity.causal_strength,
                volatility=entity.volatility
            )
            new_state.entities[entity_id] = new_entity
        
        # Copy facts
        for fact_key, fact_data in self.facts.items():
            new_state.facts[fact_key] = fact_data.copy()
        
        # Copy relationships
        for rel_key, rel_data in self.relationships.items():
            new_state.relationships[rel_key] = rel_data.copy()
        
        new_state.consistency_score = self.consistency_score
        return new_state
    
    def _update_state_id(self):
        """Update state ID after changes"""
        self.state_id = self._generate_state_id()
        self.timestamp = datetime.now()

class WorldModel:
    """
    Production internal simulation engine for causal reasoning and consistency validation.
    
    This is where Prajna gains causal understanding - the ability to simulate hypothetical
    scenarios and maintain coherent world knowledge.
    """
    
    def __init__(self, psi_archive=None):
        self.psi_archive = psi_archive
        
        # Current world state
        self.current_state = WorldState()
        
        # Causal rules database
        self.causal_rules: Dict[str, CausalRule] = {}
        
        # Domain-specific rule sets
        self.domain_rules = self._initialize_domain_rules()
        
        # Simulation configuration
        self.max_simulation_steps = 10
        self.max_cascade_depth = 5
        self.min_rule_confidence = 0.3
        self.consistency_threshold = 0.7
        
        # Performance tracking
        self.simulation_stats = {
            "total_simulations": 0,
            "successful_simulations": 0,
            "consistency_checks": 0,
            "rule_applications": 0,
            "average_simulation_time": 0.0
        }
        
        # Initialize basic world knowledge
        self._initialize_base_world_state()
        
        logger.info("🌍 WorldModel initialized with causal simulation capabilities")
    
    async def simulate_effects(self, hypothesis: str, context: Dict[str, Any] = None) -> SimulationResult:
        """
        Simulate the effects of a hypothetical scenario on the world state.
        
        This is the core causal reasoning function - testing "what if" scenarios.
        """
        start_time = time.time()
        simulation_id = self._generate_simulation_id(hypothesis)
        
        try:
            logger.info(f"🌍 Simulating hypothesis: {hypothesis[:100]}...")
            
            # Step 1: Parse hypothesis into actionable changes
            initial_changes = await self._parse_hypothesis(hypothesis, context or {})
            
            # Step 2: Create simulation state (copy of current state)
            simulation_state = self.current_state.copy()
            initial_state = self.current_state.copy()
            
            # Step 3: Execute simulation steps
            simulation_steps = []
            current_step = 0
            
            # Apply initial changes from hypothesis
            step = await self._apply_changes(
                simulation_state, initial_changes, current_step, "Initial hypothesis application"
            )
            simulation_steps.append(step)
            current_step += 1
            
            # Step 4: Simulate cascading effects
            for cascade_level in range(self.max_cascade_depth):
                # Find triggered rules
                triggered_rules = await self._find_triggered_rules(simulation_state)
                
                if not triggered_rules:
                    break  # No more cascading effects
                
                # Apply rule effects
                for rule_id, applicability_score in triggered_rules:
                    if current_step >= self.max_simulation_steps:
                        break
                    
                    rule = self.causal_rules[rule_id]
                    rule_changes = await self._generate_rule_effects(rule, simulation_state, applicability_score)
                    
                    if rule_changes:
                        step = await self._apply_changes(
                            simulation_state, rule_changes, current_step, 
                            f"Applied causal rule: {rule.rule_id}"
                        )
                        step.triggered_rules.append(rule_id)
                        simulation_steps.append(step)
                        current_step += 1
                
                if current_step >= self.max_simulation_steps:
                    break
            
            # Step 5: Calculate simulation metrics
            total_changes = sum(len(step.changes) for step in simulation_steps)
            cascade_depth = len([step for step in simulation_steps if step.triggered_rules])
            consistency_score = await self._calculate_simulation_consistency(simulation_state)
            plausibility_score = await self._calculate_simulation_plausibility(simulation_steps)
            
            # Step 6: Create simulation result
            result = SimulationResult(
                simulation_id=simulation_id,
                initial_state=initial_state,
                final_state=simulation_state,
                steps=simulation_steps,
                total_changes=total_changes,
                cascade_depth=cascade_depth,
                consistency_score=consistency_score,
                plausibility_score=plausibility_score,
                hypothesis=hypothesis,
                simulation_time=time.time() - start_time,
                success=consistency_score >= self.consistency_threshold
            )
            
            # Step 7: Archive simulation for learning
            if self.psi_archive:
                await self._archive_simulation(result, hypothesis, context)
            
            # Update statistics
            self._update_simulation_stats(result)
            
            logger.info(f"🌍 Simulation complete: {total_changes} changes, "
                       f"consistency: {consistency_score:.2f}, plausibility: {plausibility_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ World simulation failed: {e}")
            return SimulationResult(
                simulation_id=simulation_id,
                initial_state=self.current_state.copy(),
                final_state=self.current_state.copy(),
                steps=[],
                total_changes=0,
                cascade_depth=0,
                consistency_score=0.0,
                plausibility_score=0.0,
                hypothesis=hypothesis,
                simulation_time=time.time() - start_time,
                success=False
            )
    
    async def evaluate_model_consistency(self, reasoning_trace: Any) -> Tuple[bool, float, List[str]]:
        """
        Evaluate if reasoning is consistent with the current world model.
        
        This validates reasoning against known facts and relationships.
        """
        try:
            logger.info("🔍 Evaluating reasoning consistency against world model")
            
            consistency_issues = []
            consistency_scores = []
            
            # Extract claims from reasoning trace
            claims = await self._extract_claims_from_reasoning(reasoning_trace)
            
            # Check each claim against world model
            for claim in claims:
                claim_consistent, score, issues = await self._check_claim_consistency(claim)
                consistency_scores.append(score)
                
                if not claim_consistent:
                    consistency_issues.extend(issues)
            
            # Calculate overall consistency
            if consistency_scores:
                overall_consistency = sum(consistency_scores) / len(consistency_scores)
            else:
                overall_consistency = 1.0  # No claims to check
            
            is_consistent = overall_consistency >= self.consistency_threshold and len(consistency_issues) == 0
            
            # Update statistics
            self.simulation_stats["consistency_checks"] += 1
            
            logger.info(f"🔍 Consistency evaluation: {overall_consistency:.2f}, "
                       f"consistent: {is_consistent}, issues: {len(consistency_issues)}")
            
            return is_consistent, overall_consistency, consistency_issues
            
        except Exception as e:
            logger.error(f"❌ Consistency evaluation failed: {e}")
            return False, 0.0, [f"Consistency evaluation error: {str(e)}"]
    
    async def update_world_model(self, change_description: str, confidence: float = 0.8) -> bool:
        """
        Update the world model with new information.
        
        This integrates new knowledge while maintaining consistency.
        """
        try:
            logger.info(f"🔄 Updating world model: {change_description}")
            
            # Parse change description into structured updates
            updates = await self._parse_change_description(change_description)
            
            # Validate updates for consistency
            for update in updates:
                is_valid = await self._validate_update(update, confidence)
                
                if is_valid:
                    await self._apply_update(update, confidence)
                else:
                    logger.warning(f"Rejected inconsistent update: {update}")
                    return False
            
            # Recalculate world state consistency
            self.current_state.consistency_score = await self._calculate_world_consistency()
            
            logger.info(f"🔄 World model updated successfully, consistency: {self.current_state.consistency_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ World model update failed: {e}")
            return False
    
    async def rollback_simulation(self, simulation_id: str) -> bool:
        """
        Rollback effects of a simulation (if it was applied to current state).
        
        This maintains simulation isolation and allows experimentation.
        """
        try:
            # In production, this would maintain a history of state changes
            # For now, we assume simulations don't affect current state
            logger.info(f"🔄 Simulation rollback not needed - simulations are isolated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulation rollback failed: {e}")
            return False
    
    async def get_world_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current world state."""
        return {
            "state_id": self.current_state.state_id,
            "timestamp": self.current_state.timestamp.isoformat(),
            "entities": {
                "total": len(self.current_state.entities),
                "by_type": self._count_entities_by_type()
            },
            "facts": {
                "total": len(self.current_state.facts),
                "high_confidence": len([f for f in self.current_state.facts.values() if f["confidence"] > 0.8])
            },
            "relationships": {
                "total": len(self.current_state.relationships),
                "strong": len([r for r in self.current_state.relationships.values() if r["strength"] > 0.7])
            },
            "consistency_score": self.current_state.consistency_score,
            "causal_rules": len(self.causal_rules),
            "simulation_stats": self.simulation_stats.copy()
        }
    
    # Production implementation methods
    
    async def _parse_hypothesis(self, hypothesis: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse hypothesis into actionable changes."""
        changes = []
        
        # Simple pattern-based parsing for different types of hypotheses
        hypothesis_lower = hypothesis.lower()
        
        # Conditional hypotheses: "if X then Y"
        if "if" in hypothesis_lower and "then" in hypothesis_lower:
            condition_part = hypothesis_lower.split("if")[1].split("then")[0].strip()
            effect_part = hypothesis_lower.split("then")[1].strip()
            
            changes.append({
                "type": ChangeType.ADD,
                "target": "conditional_rule",
                "condition": condition_part,
                "effect": effect_part,
                "confidence": 0.7
            })
        
        # State changes: "X becomes Y" or "X is Y"
        elif any(word in hypothesis_lower for word in ["becomes", "is", "equals", "changes to"]):
            # Extract subject and predicate
            for separator in ["becomes", "is", "equals", "changes to"]:
                if separator in hypothesis_lower:
                    parts = hypothesis_lower.split(separator, 1)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        predicate = parts[1].strip()
                        
                        changes.append({
                            "type": ChangeType.MODIFY,
                            "target": subject,
                            "property": "state",
                            "value": predicate,
                            "confidence": 0.8
                        })
                    break
        
        # Addition hypotheses: "add X" or "introduce Y"
        elif any(word in hypothesis_lower for word in ["add", "introduce", "create", "establish"]):
            for word in ["add", "introduce", "create", "establish"]:
                if word in hypothesis_lower:
                    target = hypothesis_lower.split(word, 1)[1].strip()
                    
                    changes.append({
                        "type": ChangeType.ADD,
                        "target": target,
                        "confidence": 0.7
                    })
                    break
        
        # Removal hypotheses: "remove X" or "eliminate Y"
        elif any(word in hypothesis_lower for word in ["remove", "eliminate", "delete", "stop"]):
            for word in ["remove", "eliminate", "delete", "stop"]:
                if word in hypothesis_lower:
                    target = hypothesis_lower.split(word, 1)[1].strip()
                    
                    changes.append({
                        "type": ChangeType.REMOVE,
                        "target": target,
                        "confidence": 0.7
                    })
                    break
        
        # Strengthening/weakening: "increase X" or "decrease Y"
        elif any(word in hypothesis_lower for word in ["increase", "strengthen", "boost", "enhance"]):
            for word in ["increase", "strengthen", "boost", "enhance"]:
                if word in hypothesis_lower:
                    target = hypothesis_lower.split(word, 1)[1].strip()
                    
                    changes.append({
                        "type": ChangeType.STRENGTHEN,
                        "target": target,
                        "confidence": 0.8
                    })
                    break
        
        elif any(word in hypothesis_lower for word in ["decrease", "weaken", "reduce", "diminish"]):
            for word in ["decrease", "weaken", "reduce", "diminish"]:
                if word in hypothesis_lower:
                    target = hypothesis_lower.split(word, 1)[1].strip()
                    
                    changes.append({
                        "type": ChangeType.WEAKEN,
                        "target": target,
                        "confidence": 0.8
                    })
                    break
        
        # Default: treat as general state change
        else:
            changes.append({
                "type": ChangeType.MODIFY,
                "target": "general_state",
                "property": "hypothesis_applied",
                "value": hypothesis,
                "confidence": 0.6
            })
        
        return changes
    
    async def _apply_changes(self, state: WorldState, changes: List[Dict[str, Any]], 
                           step_id: int, description: str) -> SimulationStep:
        """Apply a set of changes to the world state."""
        affected_entities = set()
        applied_changes = []
        
        for change in changes:
            try:
                change_type = change["type"]
                target = change["target"]
                confidence = change.get("confidence", 0.5)
                
                if change_type == ChangeType.ADD:
                    await self._apply_add_change(state, change, affected_entities)
                    applied_changes.append(change)
                    
                elif change_type == ChangeType.REMOVE:
                    await self._apply_remove_change(state, change, affected_entities)
                    applied_changes.append(change)
                    
                elif change_type == ChangeType.MODIFY:
                    await self._apply_modify_change(state, change, affected_entities)
                    applied_changes.append(change)
                    
                elif change_type == ChangeType.STRENGTHEN:
                    await self._apply_strengthen_change(state, change, affected_entities)
                    applied_changes.append(change)
                    
                elif change_type == ChangeType.WEAKEN:
                    await self._apply_weaken_change(state, change, affected_entities)
                    applied_changes.append(change)
                    
            except Exception as e:
                logger.warning(f"Failed to apply change {change}: {e}")
        
        return SimulationStep(
            step_id=step_id,
            timestamp=datetime.now(),
            changes=applied_changes,
            triggered_rules=[],
            affected_entities=affected_entities,
            step_description=description,
            confidence=sum(c.get("confidence", 0.5) for c in applied_changes) / len(applied_changes) if applied_changes else 0.0
        )
    
    async def _apply_add_change(self, state: WorldState, change: Dict[str, Any], affected_entities: Set[str]):
        """Apply an ADD change to the state."""
        target = change["target"]
        confidence = change.get("confidence", 0.5)
        
        # Create new entity if it doesn't exist
        if target not in state.entities:
            entity = WorldEntity(
                entity_id=target,
                name=target.replace("_", " ").title(),
                entity_type=EntityType.CONCEPT,
                confidence=confidence
            )
            state.add_entity(entity)
            affected_entities.add(target)
        
        # Add properties if specified
        if "properties" in change:
            if target in state.entities:
                state.entities[target].update_properties(change["properties"])
                affected_entities.add(target)
    
    async def _apply_remove_change(self, state: WorldState, change: Dict[str, Any], affected_entities: Set[str]):
        """Apply a REMOVE change to the state."""
        target = change["target"]
        
        if target in state.entities:
            state.remove_entity(target)
            affected_entities.add(target)
        
        # Remove from facts if it's a fact
        facts_to_remove = [key for key in state.facts.keys() if target in key]
        for fact_key in facts_to_remove:
            del state.facts[fact_key]
    
    async def _apply_modify_change(self, state: WorldState, change: Dict[str, Any], affected_entities: Set[str]):
        """Apply a MODIFY change to the state."""
        target = change["target"]
        property_name = change.get("property", "state")
        new_value = change.get("value")
        confidence = change.get("confidence", 0.5)
        
        # Modify entity property
        if target in state.entities:
            state.entities[target].properties[property_name] = new_value
            state.entities[target].confidence = min(state.entities[target].confidence, confidence)
            affected_entities.add(target)
        else:
            # Add as fact
            fact_key = f"{target}_{property_name}"
            state.add_fact(fact_key, new_value, confidence)
    
    async def _apply_strengthen_change(self, state: WorldState, change: Dict[str, Any], affected_entities: Set[str]):
        """Apply a STRENGTHEN change to the state."""
        target = change["target"]
        
        if target in state.entities:
            entity = state.entities[target]
            entity.causal_strength = min(1.0, entity.causal_strength + 0.2)
            entity.confidence = min(1.0, entity.confidence + 0.1)
            affected_entities.add(target)
        
        # Strengthen relationships involving this entity
        for (e1, e2), rel_data in state.relationships.items():
            if e1 == target or e2 == target:
                rel_data["strength"] = min(1.0, rel_data["strength"] + 0.1)
    
    async def _apply_weaken_change(self, state: WorldState, change: Dict[str, Any], affected_entities: Set[str]):
        """Apply a WEAKEN change to the state."""
        target = change["target"]
        
        if target in state.entities:
            entity = state.entities[target]
            entity.causal_strength = max(0.0, entity.causal_strength - 0.2)
            entity.confidence = max(0.1, entity.confidence - 0.1)
            affected_entities.add(target)
        
        # Weaken relationships involving this entity
        for (e1, e2), rel_data in state.relationships.items():
            if e1 == target or e2 == target:
                rel_data["strength"] = max(0.1, rel_data["strength"] - 0.1)
    
    async def _find_triggered_rules(self, state: WorldState) -> List[Tuple[str, float]]:
        """Find causal rules triggered by current state."""
        triggered_rules = []
        
        for rule_id, rule in self.causal_rules.items():
            applies, score = rule.applies_to(state)
            
            if applies and score >= self.min_rule_confidence:
                triggered_rules.append((rule_id, score))
        
        # Sort by applicability score
        triggered_rules.sort(key=lambda x: x[1], reverse=True)
        
        return triggered_rules[:5]  # Limit to top 5 rules per step
    
    async def _generate_rule_effects(self, rule: CausalRule, state: WorldState, applicability_score: float) -> List[Dict[str, Any]]:
        """Generate effects from applying a causal rule."""
        effects = []
        
        for effect_key, effect_value in rule.effect.items():
            # Scale effect by applicability and rule strength
            effect_strength = applicability_score * rule.strength
            
            effect = {
                "type": ChangeType.MODIFY,
                "target": effect_key,
                "property": "causal_effect",
                "value": effect_value,
                "confidence": effect_strength,
                "source_rule": rule.rule_id
            }
            effects.append(effect)
        
        return effects
    
    async def _calculate_simulation_consistency(self, state: WorldState) -> float:
        """Calculate consistency score for simulation state."""
        consistency_scores = []
        
        # Check entity consistency
        for entity in state.entities.values():
            entity_consistency = self._check_entity_consistency(entity, state)
            consistency_scores.append(entity_consistency)
        
        # Check fact consistency
        for fact_key, fact_data in state.facts.items():
            fact_consistency = self._check_fact_consistency(fact_key, fact_data, state)
            consistency_scores.append(fact_consistency)
        
        # Check relationship consistency
        for rel_key, rel_data in state.relationships.items():
            rel_consistency = self._check_relationship_consistency(rel_key, rel_data, state)
            consistency_scores.append(rel_consistency)
        
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 1.0
    
    def _check_entity_consistency(self, entity: WorldEntity, state: WorldState) -> float:
        """Check consistency of an entity within the state."""
        consistency = 1.0
        
        # Check if entity properties are mutually consistent
        properties = entity.properties
        
        # Simple consistency checks
        if "temperature" in properties and "state" in properties:
            temp = properties["temperature"]
            state_val = properties["state"]
            
            if isinstance(temp, (int, float)):
                if temp < 0 and state_val == "liquid":
                    consistency *= 0.3  # Water shouldn't be liquid below 0°C
                elif temp > 100 and state_val == "liquid":
                    consistency *= 0.3  # Water shouldn't be liquid above 100°C
        
        # Check dependency consistency
        for dep_id in entity.dependencies:
            if dep_id not in state.entities:
                consistency *= 0.5  # Missing dependency reduces consistency
        
        return consistency
    
    def _check_fact_consistency(self, fact_key: str, fact_data: Dict[str, Any], state: WorldState) -> float:
        """Check consistency of a fact within the state."""
        consistency = fact_data["confidence"]
        
        # Check for contradictory facts
        fact_value = fact_data["value"]
        
        for other_key, other_data in state.facts.items():
            if other_key != fact_key and fact_key in other_key:
                other_value = other_data["value"]
                
                # Simple contradiction check
                if isinstance(fact_value, bool) and isinstance(other_value, bool):
                    if fact_value != other_value:
                        consistency *= 0.5
        
        return consistency
    
    def _check_relationship_consistency(self, rel_key: Tuple[str, str], rel_data: Dict[str, Any], state: WorldState) -> float:
        """Check consistency of a relationship within the state."""
        entity1_id, entity2_id = rel_key
        
        # Check if both entities exist
        if entity1_id not in state.entities or entity2_id not in state.entities:
            return 0.3
        
        return rel_data["strength"]
    
    async def _calculate_simulation_plausibility(self, steps: List[SimulationStep]) -> float:
        """Calculate plausibility score for simulation steps."""
        if not steps:
            return 1.0
        
        plausibility_scores = []
        
        for step in steps:
            # Base plausibility from step confidence
            step_plausibility = step.confidence
            
            # Reduce plausibility for steps with many changes
            if len(step.changes) > 3:
                step_plausibility *= 0.8
            
            # Increase plausibility for rule-based steps
            if step.triggered_rules:
                step_plausibility *= 1.1
            
            plausibility_scores.append(step_plausibility)
        
        return sum(plausibility_scores) / len(plausibility_scores)
    
    async def _extract_claims_from_reasoning(self, reasoning_trace: Any) -> List[Dict[str, Any]]:
        """Extract factual claims from reasoning trace."""
        claims = []
        
        # Handle different reasoning trace formats
        if hasattr(reasoning_trace, 'best_path') and reasoning_trace.best_path:
            for node in reasoning_trace.best_path.nodes:
                if hasattr(node, 'content_summary'):
                    claim = {
                        "text": node.content_summary,
                        "source": getattr(node, 'source', 'unknown'),
                        "confidence": getattr(node, 'confidence', 1.0)
                    }
                    claims.append(claim)
        
        elif hasattr(reasoning_trace, 'narrative_explanation'):
            # Parse narrative for claims
            narrative = reasoning_trace.narrative_explanation
            sentences = narrative.split('.')
            
            for sentence in sentences:
                if sentence.strip() and len(sentence.strip()) > 10:
                    claim = {
                        "text": sentence.strip(),
                        "source": "narrative",
                        "confidence": 0.8
                    }
                    claims.append(claim)
        
        return claims
    
    async def _check_claim_consistency(self, claim: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Check if a claim is consistent with world model."""
        claim_text = claim["text"].lower()
        issues = []
        consistency_score = 1.0
        
        # Check against known facts
        for fact_key, fact_data in self.current_state.facts.items():
            fact_value = str(fact_data["value"]).lower()
            
            if fact_key.lower() in claim_text or fact_value in claim_text:
                # Found related fact - check for contradiction
                if self._detect_contradiction(claim_text, fact_key, fact_value):
                    issues.append(f"Claim contradicts known fact: {fact_key} = {fact_value}")
                    consistency_score *= 0.3
        
        # Check against entity properties
        for entity in self.current_state.entities.values():
            entity_name = entity.name.lower()
            
            if entity_name in claim_text:
                for prop_key, prop_value in entity.properties.items():
                    prop_value_str = str(prop_value).lower()
                    
                    if self._detect_contradiction(claim_text, prop_key, prop_value_str):
                        issues.append(f"Claim contradicts entity property: {entity_name}.{prop_key} = {prop_value}")
                        consistency_score *= 0.5
        
        is_consistent = consistency_score >= self.consistency_threshold
        
        return is_consistent, consistency_score, issues
    
    def _detect_contradiction(self, claim_text: str, fact_key: str, fact_value: str) -> bool:
        """Detect if claim contradicts a known fact."""
        # Simple contradiction detection
        
        # Check for direct negation
        if f"not {fact_value}" in claim_text or f"{fact_value} is false" in claim_text:
            return True
        
        # Check for opposite values
        opposites = {
            "true": "false",
            "false": "true",
            "hot": "cold",
            "cold": "hot",
            "large": "small",
            "small": "large",
            "fast": "slow",
            "slow": "fast"
        }
        
        fact_value_lower = fact_value.lower()
        if fact_value_lower in opposites:
            opposite = opposites[fact_value_lower]
            if opposite in claim_text and fact_key.lower() in claim_text:
                return True
        
        return False
    
    async def _parse_change_description(self, description: str) -> List[Dict[str, Any]]:
        """Parse change description into structured updates."""
        updates = []
        description_lower = description.lower()
        
        # Extract facts from description
        if "=" in description:
            # Handle "X = Y" format
            parts = description.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                updates.append({
                    "type": "fact",
                    "key": key,
                    "value": value
                })
        
        # Extract entity updates
        elif "learned" in description_lower or "discovered" in description_lower:
            # Extract learning updates
            if "learned" in description_lower:
                content = description_lower.split("learned", 1)[1].strip()
            else:
                content = description_lower.split("discovered", 1)[1].strip()
            
            updates.append({
                "type": "entity",
                "content": content,
                "entity_type": EntityType.FACT
            })
        
        # Extract relationship updates
        elif any(word in description_lower for word in ["connects", "relates", "associated", "linked"]):
            updates.append({
                "type": "relationship",
                "content": description
            })
        
        return updates
    
    async def _validate_update(self, update: Dict[str, Any], confidence: float) -> bool:
        """Validate update for consistency."""
        # Basic validation - check if update conflicts with existing knowledge
        
        if update["type"] == "fact":
            key = update["key"]
            value = update["value"]
            
            # Check if contradicts existing fact
            if key in self.current_state.facts:
                existing_value = self.current_state.facts[key]["value"]
                if str(existing_value).lower() != str(value).lower():
                    existing_confidence = self.current_state.facts[key]["confidence"]
                    
                    # Only reject if existing fact has higher confidence
                    if existing_confidence > confidence:
                        return False
        
        return True
    
    async def _apply_update(self, update: Dict[str, Any], confidence: float):
        """Apply validated update to world model."""
        if update["type"] == "fact":
            key = update["key"]
            value = update["value"]
            self.current_state.add_fact(key, value, confidence)
        
        elif update["type"] == "entity":
            content = update["content"]
            entity_id = f"learned_{int(time.time())}"
            
            entity = WorldEntity(
                entity_id=entity_id,
                name=content[:50],  # Truncate name
                entity_type=update.get("entity_type", EntityType.FACT),
                confidence=confidence
            )
            entity.properties["content"] = content
            
            self.current_state.add_entity(entity)
        
        elif update["type"] == "relationship":
            # Simple relationship extraction
            content = update["content"]
            # This would be more sophisticated in a full implementation
            pass
    
    async def _calculate_world_consistency(self) -> float:
        """Calculate overall world state consistency."""
        return await self._calculate_simulation_consistency(self.current_state)
    
    def _count_entities_by_type(self) -> Dict[str, int]:
        """Count entities by type."""
        counts = defaultdict(int)
        for entity in self.current_state.entities.values():
            counts[entity.entity_type.value] += 1
        return dict(counts)
    
    def _initialize_base_world_state(self):
        """Initialize basic world knowledge."""
        # Add fundamental concepts
        concepts = [
            ("physics", EntityType.CONCEPT, {"domain": "physics", "abstraction": 0.8}),
            ("consciousness", EntityType.CONCEPT, {"domain": "philosophy", "abstraction": 0.9}),
            ("intelligence", EntityType.CONCEPT, {"domain": "cognitive_science", "abstraction": 0.8}),
            ("learning", EntityType.PROCESS, {"domain": "cognitive_science", "abstraction": 0.6}),
            ("reasoning", EntityType.PROCESS, {"domain": "cognitive_science", "abstraction": 0.7})
        ]
        
        for concept_id, entity_type, properties in concepts:
            entity = WorldEntity(
                entity_id=concept_id,
                name=concept_id.title(),
                entity_type=entity_type,
                properties=properties,
                confidence=0.9
            )
            self.current_state.add_entity(entity)
        
        # Add basic facts
        facts = [
            ("gravity_acceleration", 9.8, 0.95),
            ("speed_of_light", 299792458, 0.99),
            ("water_boiling_point", 100, 0.9),
            ("water_freezing_point", 0, 0.9)
        ]
        
        for fact_key, fact_value, confidence in facts:
            self.current_state.add_fact(fact_key, fact_value, confidence)
        
        # Add basic relationships
        relationships = [
            ("physics", "consciousness", "studies", 0.6),
            ("intelligence", "learning", "requires", 0.8),
            ("learning", "reasoning", "enables", 0.7)
        ]
        
        for e1, e2, rel_type, strength in relationships:
            self.current_state.add_relationship(e1, e2, rel_type, strength)
    
    def _initialize_domain_rules(self) -> Dict[str, List[CausalRule]]:
        """Initialize domain-specific causal rules."""
        domain_rules = {
            "physics": [
                CausalRule(
                    rule_id="heat_state_change",
                    condition={"temperature": "increasing", "substance": "water"},
                    effect={"state": "gas"},
                    strength=0.9,
                    confidence=0.95,
                    domain="physics"
                ),
                CausalRule(
                    rule_id="cooling_state_change",
                    condition={"temperature": "decreasing", "substance": "water"},
                    effect={"state": "ice"},
                    strength=0.9,
                    confidence=0.95,
                    domain="physics"
                )
            ],
            "cognitive_science": [
                CausalRule(
                    rule_id="learning_improves_performance",
                    condition={"learning": "active", "practice": "regular"},
                    effect={"performance": "improved"},
                    strength=0.8,
                    confidence=0.85,
                    domain="cognitive_science"
                ),
                CausalRule(
                    rule_id="attention_affects_learning",
                    condition={"attention": "focused", "distraction": "low"},
                    effect={"learning": "enhanced"},
                    strength=0.7,
                    confidence=0.8,
                    domain="cognitive_science"
                )
            ]
        }
        
        # Add rules to main causal rules database
        for domain, rules in domain_rules.items():
            for rule in rules:
                self.causal_rules[rule.rule_id] = rule
        
        return domain_rules
    
    def _generate_simulation_id(self, hypothesis: str) -> str:
        """Generate unique simulation ID."""
        combined = f"{hypothesis[:50]}_{int(time.time())}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _update_simulation_stats(self, result: SimulationResult):
        """Update simulation statistics."""
        self.simulation_stats["total_simulations"] += 1
        
        if result.success:
            self.simulation_stats["successful_simulations"] += 1
        
        # Update average simulation time
        total = self.simulation_stats["total_simulations"]
        current_avg = self.simulation_stats["average_simulation_time"]
        self.simulation_stats["average_simulation_time"] = (
            current_avg * (total - 1) + result.simulation_time
        ) / total
    
    async def _archive_simulation(self, result: SimulationResult, hypothesis: str, context: Dict[str, Any]):
        """Archive simulation for learning and transparency."""
        if self.psi_archive:
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "simulation_id": result.simulation_id,
                "hypothesis": hypothesis,
                "context": context,
                "success": result.success,
                "metrics": {
                    "total_changes": result.total_changes,
                    "cascade_depth": result.cascade_depth,
                    "consistency_score": result.consistency_score,
                    "plausibility_score": result.plausibility_score,
                    "simulation_time": result.simulation_time
                }
            }
            await self.psi_archive.log_world_simulation(archive_data)
    
    async def get_simulation_stats(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        return {
            **self.simulation_stats,
            "world_state": await self.get_world_state_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> bool:
        """Health check for world model."""
        try:
            # Test simulation
            test_result = await self.simulate_effects("test hypothesis")
            
            # Test consistency check
            test_consistency, _, _ = await self.evaluate_model_consistency("test reasoning")
            
            return test_result is not None and isinstance(test_consistency, bool)
        except Exception:
            return False

if __name__ == "__main__":
    # Production test
    async def test_world_model():
        world_model = WorldModel()
        
        # Test simulation
        hypothesis = "If water temperature increases above 100 degrees, then water becomes gas"
        result = await world_model.simulate_effects(hypothesis)
        
        print(f"✅ WorldModel Test Results:")
        print(f"   Simulation ID: {result.simulation_id}")
        print(f"   Success: {result.success}")
        print(f"   Total changes: {result.total_changes}")
        print(f"   Cascade depth: {result.cascade_depth}")
        print(f"   Consistency: {result.consistency_score:.2f}")
        print(f"   Plausibility: {result.plausibility_score:.2f}")
        print(f"   Simulation time: {result.simulation_time:.2f}s")
        
        # Test consistency evaluation
        class MockReasoning:
            def __init__(self):
                self.narrative_explanation = "Water boils at 100 degrees Celsius under normal pressure"
        
        mock_reasoning = MockReasoning()
        is_consistent, score, issues = await world_model.evaluate_model_consistency(mock_reasoning)
        
        print(f"   Consistency check: {is_consistent}")
        print(f"   Consistency score: {score:.2f}")
        print(f"   Issues found: {len(issues)}")
        
        # Test world model update
        update_success = await world_model.update_world_model("learned that AI can reason", 0.8)
        print(f"   Update success: {update_success}")
        
        # Get world state summary
        summary = await world_model.get_world_state_summary()
        print(f"   Total entities: {summary['entities']['total']}")
        print(f"   Total facts: {summary['facts']['total']}")
        print(f"   World consistency: {summary['consistency_score']:.2f}")
    
    import asyncio
    asyncio.run(test_world_model())
