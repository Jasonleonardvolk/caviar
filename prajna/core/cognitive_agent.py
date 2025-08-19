"""
Cognitive Agent: Intent-Oriented Planner for Prajna
===================================================

Production implementation of Prajna's goal formulation and strategic planning system.
This module analyzes user queries to derive concrete cognitive goals, builds multi-step
reasoning plans, and ensures intent alignment with user needs and system constraints.

This is where Prajna gains strategic intelligence - the ability to plan and execute
complex reasoning tasks with clear objectives and measurable success criteria.
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger("prajna.cognitive_agent")

class QueryType(Enum):
    """Types of user queries requiring different planning approaches"""
    EXPLANATION = "explanation"  # "Explain how X works"
    COMPARISON = "comparison"    # "Compare X and Y"
    ANALYSIS = "analysis"        # "Analyze the impact of X"
    SYNTHESIS = "synthesis"      # "Combine concepts from X and Y"
    PROBLEM_SOLVING = "problem_solving"  # "How to solve X"
    FACTUAL = "factual"         # "What is X"
    OPINION = "opinion"         # "What do you think about X"
    CREATIVE = "creative"       # "Create/generate X"
    DEBUGGING = "debugging"     # "Why is X not working"
    PREDICTION = "prediction"   # "What will happen if X"

class PlanStep(Enum):
    """Types of actions in reasoning plans"""
    RETRIEVE = "retrieve"           # Retrieve information from context/knowledge
    REASON = "reason"              # Apply reasoning engine
    SYNTHESIZE = "synthesize"      # Synthesize concepts
    SIMULATE = "simulate"          # World model simulation  
    DEBATE = "debate"              # Internal debate/validation
    REFLECT = "reflect"            # Self-reflection analysis
    VALIDATE = "validate"          # Validate against constraints
    GENERATE = "generate"          # Generate creative content
    REFINE = "refine"             # Refine/improve output

@dataclass
class GoalConstraint:
    """Specific constraint on goal execution"""
    type: str           # "length", "complexity", "domain", "style", "safety"
    value: Any          # Constraint value
    priority: float     # 0.0-1.0 priority level
    description: str    # Human-readable description

@dataclass 
class SuccessCriterion:
    """Measurable success criterion for goal achievement"""
    description: str    # What success looks like
    metric: str        # How to measure it
    threshold: float   # Success threshold
    weight: float      # Importance weight (0.0-1.0)

@dataclass
class Goal:
    """Structured representation of user intent and desired outcome"""
    # Core goal definition
    description: str                           # Clear goal statement
    query_type: QueryType                     # Type of query
    primary_concepts: List[str] = field(default_factory=list)  # Key concepts involved
    
    # Goal parameters
    constraints: List[GoalConstraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    context_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0                   # Confidence in goal interpretation
    complexity: float = 0.0                  # Estimated complexity (0.0-1.0)
    estimated_time: float = 0.0              # Estimated processing time
    priority: float = 1.0                    # Goal priority
    
    def add_constraint(self, constraint_type: str, value: Any, priority: float = 1.0, description: str = ""):
        """Add a constraint to the goal"""
        constraint = GoalConstraint(
            type=constraint_type,
            value=value, 
            priority=priority,
            description=description or f"{constraint_type}: {value}"
        )
        self.constraints.append(constraint)
    
    def add_success_criterion(self, description: str, metric: str, threshold: float, weight: float = 1.0):
        """Add a success criterion to the goal"""
        criterion = SuccessCriterion(
            description=description,
            metric=metric,
            threshold=threshold,
            weight=weight
        )
        self.success_criteria.append(criterion)

@dataclass
class PlanAction:
    """Individual action in a reasoning plan"""
    step: PlanStep                     # Type of action
    description: str                   # What this action does
    inputs: List[str] = field(default_factory=list)     # Required inputs
    outputs: List[str] = field(default_factory=list)    # Expected outputs
    parameters: Dict[str, Any] = field(default_factory=dict)  # Action parameters
    estimated_time: float = 0.0       # Estimated execution time
    success_criteria: List[str] = field(default_factory=list)  # How to measure success
    dependencies: List[int] = field(default_factory=list)  # Indices of prerequisite steps

@dataclass
class ReasoningPlan:
    """Complete plan for achieving a cognitive goal"""
    goal: Goal                                    # The goal this plan achieves
    actions: List[PlanAction] = field(default_factory=list)
    
    # Plan metadata
    total_estimated_time: float = 0.0           # Total estimated execution time
    confidence: float = 0.0                     # Confidence in plan effectiveness
    complexity: float = 0.0                     # Plan complexity
    alternative_plans: List['ReasoningPlan'] = field(default_factory=list)
    
    # Execution tracking
    current_step: int = 0
    execution_status: str = "pending"           # "pending", "executing", "completed", "failed"
    results: Dict[int, Any] = field(default_factory=dict)  # Results from each step
    
    def add_action(self, step: PlanStep, description: str, **kwargs):
        """Add an action to the plan"""
        action = PlanAction(
            step=step,
            description=description,
            **kwargs
        )
        self.actions.append(action)
        self.total_estimated_time += action.estimated_time
    
    def get_next_action(self) -> Optional[PlanAction]:
        """Get the next action to execute"""
        if self.current_step < len(self.actions):
            return self.actions[self.current_step]
        return None
    
    def mark_step_complete(self, step_index: int, result: Any):
        """Mark a step as completed with its result"""
        self.results[step_index] = result
        if step_index == self.current_step:
            self.current_step += 1

class CognitiveAgent:
    """
    Production intent-oriented planner that formulates goals and builds strategic plans.
    
    This is where Prajna gains strategic intelligence - understanding what users really
    want and planning the optimal approach to deliver it.
    """
    
    def __init__(self, world_model=None, concept_mesh=None, psi_archive=None):
        self.world_model = world_model
        self.concept_mesh = concept_mesh
        self.psi_archive = psi_archive
        
        # Planning patterns and templates
        self.query_patterns = self._initialize_query_patterns()
        self.plan_templates = self._initialize_plan_templates()
        self.constraint_extractors = self._initialize_constraint_extractors()
        
        # Performance tracking
        self.planning_stats = {
            "goals_formulated": 0,
            "plans_created": 0,
            "successful_plans": 0,
            "average_accuracy": 0.0,
            "total_planning_time": 0.0
        }
        
        logger.info("🎯 CognitiveAgent initialized with strategic planning capabilities")
    
    async def formulate_goal(self, query: str, context: str = "", user_preferences: Dict = None) -> Goal:
        """
        Analyze user query and context to derive a concrete, actionable goal.
        
        This is where user intent gets transformed into structured objectives.
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Formulating goal from query: {query[:100]}...")
            
            # Step 1: Classify query type
            query_type = self._classify_query_type(query)
            logger.debug(f"Query classified as: {query_type.value}")
            
            # Step 2: Extract primary concepts
            primary_concepts = self._extract_primary_concepts(query, context)
            logger.debug(f"Primary concepts: {primary_concepts}")
            
            # Step 3: Generate goal description
            goal_description = self._generate_goal_description(query, query_type, primary_concepts)
            
            # Step 4: Create base goal
            goal = Goal(
                description=goal_description,
                query_type=query_type,
                primary_concepts=primary_concepts
            )
            
            # Step 5: Extract and add constraints
            constraints = self._extract_constraints(query, context, user_preferences or {})
            for constraint in constraints:
                goal.constraints.append(constraint)
            
            # Step 6: Define success criteria
            success_criteria = self._define_success_criteria(query_type, query, constraints)
            for criterion in success_criteria:
                goal.success_criteria.append(criterion)
            
            # Step 7: Estimate goal parameters
            goal.complexity = self._estimate_complexity(goal)
            goal.estimated_time = self._estimate_processing_time(goal)
            goal.confidence = self._calculate_goal_confidence(goal, query, context)
            
            # Step 8: Validate goal feasibility
            if self.world_model:
                feasibility = await self._validate_goal_feasibility(goal)
                goal.confidence *= feasibility
            
            # Step 9: Archive goal for learning
            
            await self._archive_goal(goal, query, context)
            
            # Update stats
            self.planning_stats["goals_formulated"] += 1
            self.planning_stats["total_planning_time"] += time.time() - start_time
            
            logger.info(f"🎯 Goal formulated: {goal.description} (confidence: {goal.confidence:.2f})")
            
            return goal
            
        except Exception as e:
            logger.error(f"❌ Goal formulation failed: {e}")
            # Return fallback goal
            return Goal(
                description=f"Address user query: {query}",
                query_type=QueryType.FACTUAL,
                primary_concepts=query.split()[:3],
                confidence=0.3
            )
    
    async def build_plan(self, goal: Goal, available_modules: Set[str] = None) -> ReasoningPlan:
        """
        Build a comprehensive reasoning plan to achieve the given goal.
        
        This creates the strategic roadmap for complex reasoning tasks.
        """
        start_time = time.time()
        
        try:
            logger.info(f"📋 Building plan for goal: {goal.description}")
            
            # Step 1: Select plan template based on goal type
            template = self._select_plan_template(goal, available_modules or set())
            
            # Step 2: Create base plan
            plan = ReasoningPlan(goal=goal)
            
            # Step 3: Build action sequence from template
            await self._build_action_sequence(plan, template, goal)
            
            # Step 4: Optimize plan for efficiency
            await self._optimize_plan(plan)
            
            # Step 5: Validate plan against constraints
            validation_result = await self._validate_plan(plan)
            plan.confidence = validation_result
            
            # Step 6: Generate alternative plans if needed
            if plan.confidence < 0.7:
                alternatives = await self._generate_alternative_plans(goal, available_modules)
                plan.alternative_plans = alternatives[:2]  # Keep top 2 alternatives
            
            # Step 7: Calculate final plan metrics
            plan.complexity = self._calculate_plan_complexity(plan)
            plan.total_estimated_time = sum(action.estimated_time for action in plan.actions)
            
            # Step 8: Archive plan for learning
            if self.psi_archive:
            await self._archive_plan(plan, goal)
            
            # Update stats
            self.planning_stats["plans_created"] += 1
            self.planning_stats["total_planning_time"] += time.time() - start_time
            
            logger.info(f"📋 Plan built: {len(plan.actions)} actions, confidence: {plan.confidence:.2f}")
            
            return plan
            
        except Exception as e:
            logger.error(f"❌ Plan building failed: {e}")
            # Return minimal fallback plan
            fallback_plan = ReasoningPlan(goal=goal)
            fallback_plan.add_action(
                PlanStep.RETRIEVE,
                "Retrieve relevant information",
                estimated_time=1.0
            )
            fallback_plan.add_action(
                PlanStep.REASON,
                "Apply reasoning to generate answer",
                estimated_time=2.0
            )
            fallback_plan.confidence = 0.3
            return fallback_plan
    
    async def evaluate_intent_alignment(self, goal: Goal, original_query: str, context: str = "") -> Tuple[bool, float, List[str]]:
        """
        Evaluate if the formulated goal aligns with user intent and system constraints.
        
        Returns (is_aligned, alignment_score, issues)
        """
        issues = []
        alignment_score = 1.0
        
        try:
            # Check 1: Query-Goal semantic alignment
            semantic_alignment = await self._calculate_semantic_alignment(goal, original_query)
            if semantic_alignment < 0.7:
                issues.append(f"Goal may not align with query intent (score: {semantic_alignment:.2f})")
                alignment_score *= semantic_alignment
            
            # Check 2: Safety and ethical constraints
            safety_issues = await self._check_safety_constraints(goal, original_query)
            if safety_issues:
                issues.extend(safety_issues)
                alignment_score *= 0.5  # Significant penalty for safety issues
            
            # Check 3: Resource availability
            if self.world_model:
                resource_check = await self._check_resource_availability(goal)
                if not resource_check:
                    issues.append("Required resources may not be available")
                    alignment_score *= 0.8
            
            # Check 4: Scope appropriateness
            scope_issues = self._check_scope_appropriateness(goal, original_query)
            if scope_issues:
                issues.extend(scope_issues)
                alignment_score *= 0.9
            
            # Check 5: Constraint consistency
            constraint_issues = self._check_constraint_consistency(goal)
            if constraint_issues:
                issues.extend(constraint_issues)
                alignment_score *= 0.9
            
            is_aligned = alignment_score >= 0.7 and not any("safety" in issue.lower() for issue in issues)
            
            logger.info(f"🎯 Intent alignment: {alignment_score:.2f}, aligned: {is_aligned}")
            
            return is_aligned, alignment_score, issues
            
        except Exception as e:
            logger.error(f"❌ Intent alignment evaluation failed: {e}")
            return False, 0.5, [f"Alignment evaluation error: {str(e)}"]
    
    async def execute_plan_step(self, plan: ReasoningPlan, step_index: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step of the reasoning plan.
        
        This coordinates with other modules to execute plan actions.
        """
        if step_index >= len(plan.actions):
            raise ValueError(f"Step index {step_index} out of range")
        
        action = plan.actions[step_index]
        logger.info(f"🏃 Executing plan step {step_index}: {action.description}")
        
        try:
            # Check dependencies
            for dep_index in action.dependencies:
                if dep_index not in plan.results:
                    raise RuntimeError(f"Dependency step {dep_index} not completed")
            
            # Prepare execution context
            execution_context = {
                **context,
                "goal": plan.goal,
                "previous_results": plan.results,
                "action_parameters": action.parameters
            }
            
            # Execute based on action type
            result = await self._execute_action(action, execution_context)
            
            # Validate result against success criteria
            validation_score = self._validate_action_result(action, result)
            
            # Mark step complete
            plan.mark_step_complete(step_index, {
                "result": result,
                "validation_score": validation_score,
                "execution_time": time.time()
            })
            
            logger.info(f"✅ Step {step_index} completed with validation score: {validation_score:.2f}")
            
            return plan.results[step_index]
            
        except Exception as e:
            logger.error(f"❌ Step {step_index} execution failed: {e}")
            plan.execution_status = "failed"
            raise
    
    # Production implementation methods
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of user query using pattern matching."""
        query_lower = query.lower()
        
        # Pattern-based classification
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        # Default classification based on keywords
        if any(word in query_lower for word in ["explain", "how", "why", "describe"]):
            return QueryType.EXPLANATION
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return QueryType.COMPARISON
        elif any(word in query_lower for word in ["analyze", "examine", "evaluate"]):
            return QueryType.ANALYSIS
        elif any(word in query_lower for word in ["what", "who", "when", "where"]):
            return QueryType.FACTUAL
        elif any(word in query_lower for word in ["create", "generate", "make", "design"]):
            return QueryType.CREATIVE
        elif any(word in query_lower for word in ["think", "opinion", "believe"]):
            return QueryType.OPINION
        else:
            return QueryType.EXPLANATION  # Default fallback
    
    def _extract_primary_concepts(self, query: str, context: str) -> List[str]:
        """Extract the main concepts from query and context."""
        # Combine query and context for concept extraction
        combined_text = f"{query} {context}"
        
        # Extract meaningful terms (nouns, technical terms, etc.)
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+(?:_\w+)+\b',  # Technical terms with underscores
            r'\b\w{5,}\b'  # Longer words likely to be concepts
        ]
        
        concepts = set()
        for pattern in concept_patterns:
            matches = re.findall(pattern, combined_text)
            concepts.update(match.lower() for match in matches)
        
        # Remove common stopwords
        stopwords = {
            'about', 'above', 'after', 'again', 'against', 'before', 'being', 'below',
            'between', 'during', 'further', 'having', 'should', 'their', 'there',
            'these', 'they', 'those', 'through', 'under', 'until', 'while', 'would'
        }
        
        concepts -= stopwords
        
        # Prioritize concepts that appear in the query
        query_concepts = [c for c in concepts if c in query.lower()]
        context_concepts = [c for c in concepts if c not in query.lower()]
        
        # Return top concepts, prioritizing query concepts
        return (query_concepts + context_concepts)[:5]
    
    def _generate_goal_description(self, query: str, query_type: QueryType, concepts: List[str]) -> str:
        """Generate a clear, actionable goal description."""
        concept_str = ", ".join(concepts[:3]) if concepts else "relevant topics"
        
        goal_templates = {
            QueryType.EXPLANATION: f"Provide a clear explanation of {concept_str} in response to: {query}",
            QueryType.COMPARISON: f"Compare and contrast {concept_str} to address: {query}",
            QueryType.ANALYSIS: f"Analyze {concept_str} to provide insights on: {query}",
            QueryType.SYNTHESIS: f"Synthesize information about {concept_str} to answer: {query}",
            QueryType.PROBLEM_SOLVING: f"Develop solutions involving {concept_str} for: {query}",
            QueryType.FACTUAL: f"Provide accurate factual information about {concept_str} for: {query}",
            QueryType.CREATIVE: f"Generate creative content involving {concept_str} for: {query}",
            QueryType.OPINION: f"Provide balanced perspectives on {concept_str} regarding: {query}",
            QueryType.DEBUGGING: f"Diagnose and address issues with {concept_str} for: {query}",
            QueryType.PREDICTION: f"Make informed predictions about {concept_str} for: {query}"
        }
        
        return goal_templates.get(query_type, f"Address the query about {concept_str}: {query}")
    
    def _extract_constraints(self, query: str, context: str, user_preferences: Dict) -> List[GoalConstraint]:
        """Extract constraints from query, context, and user preferences."""
        constraints = []
        
        # Extract length constraints
        length_patterns = [
            (r'\bbrief(?:ly)?\b', "short", 0.8),
            (r'\bshort\b', "short", 0.9),
            (r'\bdetailed?\b', "long", 0.8),
            (r'\bcomprehensive\b', "long", 0.9),
            (r'\bin detail\b', "long", 0.8)
        ]
        
        for pattern, value, priority in length_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints.append(GoalConstraint(
                    type="length",
                    value=value,
                    priority=priority,
                    description=f"Response should be {value}"
                ))
                break
        
        # Extract complexity constraints
        complexity_patterns = [
            (r'\bsimple\b|\bbasic\b|\beasy\b', "simple", 0.8),
            (r'\badvanced\b|\bcomplex\b|\btechnical\b', "complex", 0.8),
            (r'\bfor beginners?\b', "simple", 0.9),
            (r'\bexpert level\b', "complex", 0.9)
        ]
        
        for pattern, value, priority in complexity_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints.append(GoalConstraint(
                    type="complexity",
                    value=value,
                    priority=priority,
                    description=f"Complexity level: {value}"
                ))
                break
        
        # Extract domain constraints
        domain_patterns = [
            (r'\bscientific\b|\bscience\b', "science", 0.7),
            (r'\btechnical\b|\btechnology\b', "technology", 0.7),
            (r'\bbusiness\b|\bcommercial\b', "business", 0.7),
            (r'\bacademic\b|\bscholarly\b', "academic", 0.8)
        ]
        
        for pattern, value, priority in domain_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints.append(GoalConstraint(
                    type="domain",
                    value=value,
                    priority=priority,
                    description=f"Domain focus: {value}"
                ))
        
        # Add user preference constraints
        for pref_type, pref_value in user_preferences.items():
            if pref_type in ["style", "tone", "format", "depth"]:
                constraints.append(GoalConstraint(
                    type=pref_type,
                    value=pref_value,
                    priority=0.6,
                    description=f"User preference: {pref_type} = {pref_value}"
                ))
        
        return constraints
    
    def _define_success_criteria(self, query_type: QueryType, query: str, constraints: List[GoalConstraint]) -> List[SuccessCriterion]:
        """Define measurable success criteria based on query type and constraints."""
        criteria = []
        
        # Base criteria for all query types
        criteria.append(SuccessCriterion(
            description="Answer addresses the user's query directly",
            metric="query_relevance",
            threshold=0.8,
            weight=1.0
        ))
        
        criteria.append(SuccessCriterion(
            description="Response is factually accurate",
            metric="factual_accuracy",
            threshold=0.9,
            weight=0.9
        ))
        
        # Query-type specific criteria
        if query_type == QueryType.EXPLANATION:
            criteria.append(SuccessCriterion(
                description="Explanation is clear and understandable",
                metric="clarity_score",
                threshold=0.8,
                weight=0.8
            ))
            
        elif query_type == QueryType.COMPARISON:
            criteria.append(SuccessCriterion(
                description="Both sides are covered fairly",
                metric="balance_score",
                threshold=0.7,
                weight=0.8
            ))
            
        elif query_type == QueryType.ANALYSIS:
            criteria.append(SuccessCriterion(
                description="Analysis provides meaningful insights",
                metric="insight_score",
                threshold=0.7,
                weight=0.9
            ))
        
        elif query_type == QueryType.CREATIVE:
            criteria.append(SuccessCriterion(
                description="Output demonstrates creativity and originality",
                metric="creativity_score",
                threshold=0.6,
                weight=0.8
            ))
        
        # Constraint-specific criteria
        for constraint in constraints:
            if constraint.type == "length" and constraint.value == "short":
                criteria.append(SuccessCriterion(
                    description="Response is appropriately concise",
                    metric="conciseness_score",
                    threshold=0.8,
                    weight=constraint.priority
                ))
            elif constraint.type == "complexity" and constraint.value == "simple":
                criteria.append(SuccessCriterion(
                    description="Response uses simple, accessible language",
                    metric="simplicity_score",
                    threshold=0.8,
                    weight=constraint.priority
                ))
        
        return criteria
    
    def _estimate_complexity(self, goal: Goal) -> float:
        """Estimate the complexity of achieving the goal."""
        complexity = 0.0
        
        # Base complexity by query type
        type_complexity = {
            QueryType.FACTUAL: 0.2,
            QueryType.EXPLANATION: 0.4,
            QueryType.COMPARISON: 0.5,
            QueryType.ANALYSIS: 0.7,
            QueryType.SYNTHESIS: 0.8,
            QueryType.PROBLEM_SOLVING: 0.8,
            QueryType.CREATIVE: 0.6,
            QueryType.OPINION: 0.5,
            QueryType.DEBUGGING: 0.9,
            QueryType.PREDICTION: 0.8
        }
        
        complexity += type_complexity.get(goal.query_type, 0.5)
        
        # Adjust for number of concepts
        concept_factor = min(0.3, len(goal.primary_concepts) * 0.1)
        complexity += concept_factor
        
        # Adjust for constraints
        constraint_factor = min(0.2, len(goal.constraints) * 0.05)
        complexity += constraint_factor
        
        # Adjust for success criteria
        criteria_factor = min(0.1, len(goal.success_criteria) * 0.02)
        complexity += criteria_factor
        
        return min(1.0, complexity)
    
    def _estimate_processing_time(self, goal: Goal) -> float:
        """Estimate processing time needed for the goal."""
        base_time = 2.0  # Base 2 seconds
        
        # Adjust by complexity
        time_factor = 1.0 + goal.complexity * 3.0
        
        # Adjust by query type
        type_multipliers = {
            QueryType.FACTUAL: 0.5,
            QueryType.EXPLANATION: 1.0,
            QueryType.COMPARISON: 1.5,
            QueryType.ANALYSIS: 2.0,
            QueryType.SYNTHESIS: 2.5,
            QueryType.PROBLEM_SOLVING: 2.0,
            QueryType.CREATIVE: 1.8,
            QueryType.OPINION: 1.2,
            QueryType.DEBUGGING: 2.5,
            QueryType.PREDICTION: 2.2
        }
        
        multiplier = type_multipliers.get(goal.query_type, 1.0)
        
        return base_time * time_factor * multiplier
    
    def _calculate_goal_confidence(self, goal: Goal, query: str, context: str) -> float:
        """Calculate confidence in goal formulation."""
        confidence = 0.8  # Base confidence
        
        # Boost confidence if primary concepts were identified
        if goal.primary_concepts:
            confidence += 0.1
        
        # Boost confidence if constraints were identified
        if goal.constraints:
            confidence += 0.05
        
        # Reduce confidence if query is very short or unclear
        if len(query.split()) < 3:
            confidence -= 0.2
        
        # Reduce confidence for very complex goals
        if goal.complexity > 0.8:
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    async def _validate_goal_feasibility(self, goal: Goal) -> float:
        """Validate if goal is feasible given current capabilities."""
        feasibility = 1.0
        
        # Check if query type is supported
        supported_types = {QueryType.EXPLANATION, QueryType.COMPARISON, QueryType.ANALYSIS, 
                         QueryType.FACTUAL, QueryType.SYNTHESIS}
        
        if goal.query_type not in supported_types:
            feasibility *= 0.7
        
        # Check constraint feasibility
        for constraint in goal.constraints:
            if constraint.type == "safety" and constraint.value == "high_risk":
                feasibility *= 0.3
            elif constraint.type == "complexity" and constraint.value == "extremely_complex":
                feasibility *= 0.6
        
        return feasibility
    
    def _select_plan_template(self, goal: Goal, available_modules: Set[str]) -> Dict[str, Any]:
        """Select appropriate plan template based on goal and available modules."""
        template_name = f"{goal.query_type.value}_template"
        
        if template_name in self.plan_templates:
            template = self.plan_templates[template_name].copy()
        else:
            template = self.plan_templates["default_template"].copy()
        
        # Filter actions based on available modules
        if available_modules:
            filtered_actions = []
            for action in template.get("actions", []):
                required_module = action.get("required_module")
                if not required_module or required_module in available_modules:
                    filtered_actions.append(action)
            template["actions"] = filtered_actions
        
        return template
    
    async def _build_action_sequence(self, plan: ReasoningPlan, template: Dict[str, Any], goal: Goal):
        """Build the sequence of actions from template and goal requirements."""
        
        for action_spec in template.get("actions", []):
            action_type = PlanStep(action_spec["type"])
            description = action_spec["description"].format(
                concepts=", ".join(goal.primary_concepts[:2])
            )
            
            # Build action parameters
            parameters = action_spec.get("parameters", {}).copy()
            
            # Add goal-specific parameters
            if action_type == PlanStep.REASON:
                parameters["reasoning_mode"] = self._determine_reasoning_mode(goal.query_type)
                parameters["max_hops"] = min(5, 2 + int(goal.complexity * 3))
            
            elif action_type == PlanStep.SYNTHESIZE:
                parameters["concepts"] = goal.primary_concepts
                parameters["synthesis_depth"] = int(goal.complexity * 3) + 1
            
            elif action_type == PlanStep.DEBATE:
                parameters["agents"] = self._select_debate_agents(goal)
                parameters["rounds"] = min(5, 2 + int(goal.complexity * 2))
            
            # Create and add action
            plan.add_action(
                action_type,
                description,
                inputs=action_spec.get("inputs", []),
                outputs=action_spec.get("outputs", []),
                parameters=parameters,
                estimated_time=action_spec.get("estimated_time", 1.0),
                success_criteria=action_spec.get("success_criteria", []),
                dependencies=action_spec.get("dependencies", [])
            )
    
    async def _optimize_plan(self, plan: ReasoningPlan):
        """Optimize plan for efficiency and effectiveness."""
        
        # Remove redundant steps
        self._remove_redundant_actions(plan)
        
        # Reorder for optimal dependency resolution
        self._optimize_action_order(plan)
        
        # Adjust time estimates based on dependencies
        self._recalculate_time_estimates(plan)
        
        # Add parallel execution opportunities
        self._identify_parallel_actions(plan)
    
    def _remove_redundant_actions(self, plan: ReasoningPlan):
        """Remove redundant or unnecessary actions from plan."""
        seen_action_types = set()
        actions_to_remove = []
        
        for i, action in enumerate(plan.actions):
            # Check for duplicate retrieve actions
            if action.step == PlanStep.RETRIEVE and PlanStep.RETRIEVE in seen_action_types:
                actions_to_remove.append(i)
            else:
                seen_action_types.add(action.step)
        
        # Remove in reverse order to maintain indices
        for i in reversed(actions_to_remove):
            plan.actions.pop(i)
    
    def _optimize_action_order(self, plan: ReasoningPlan):
        """Reorder actions for optimal dependency resolution."""
        # Simple topological sort based on dependencies
        ordered_actions = []
        remaining_actions = list(enumerate(plan.actions))
        
        while remaining_actions:
            # Find actions with no unresolved dependencies
            ready_actions = []
            for i, (orig_index, action) in enumerate(remaining_actions):
                deps_resolved = all(
                    dep_index < len(ordered_actions) 
                    for dep_index in action.dependencies
                )
                if deps_resolved:
                    ready_actions.append((i, orig_index, action))
            
            if not ready_actions:
                # Break dependency cycles by taking first action
                ready_actions = [remaining_actions[0]]
            
            # Add ready actions to ordered list
            for i, orig_index, action in ready_actions:
                ordered_actions.append(action)
            
            # Remove from remaining
            for i, _, _ in reversed(ready_actions):
                remaining_actions.pop(i)
        
        plan.actions = ordered_actions
    
    def _recalculate_time_estimates(self, plan: ReasoningPlan):
        """Recalculate time estimates considering dependencies."""
        for action in plan.actions:
            # Adjust time based on complexity
            complexity_factor = 1.0 + plan.goal.complexity * 0.5
            action.estimated_time *= complexity_factor
            
            # Adjust based on number of dependencies
            dep_factor = 1.0 + len(action.dependencies) * 0.1
            action.estimated_time *= dep_factor
    
    def _identify_parallel_actions(self, plan: ReasoningPlan):
        """Identify actions that can be executed in parallel."""
        # Mark actions that don't depend on each other
        for i, action in enumerate(plan.actions):
            action.parameters["can_parallelize"] = True
            
            # Check if this action has dependencies that aren't completed
            for dep_index in action.dependencies:
                if dep_index >= i:  # Dependency comes after this action
                    action.parameters["can_parallelize"] = False
                    break
    
    async def _validate_plan(self, plan: ReasoningPlan) -> float:
        """Validate plan against goal constraints and requirements."""
        validation_score = 1.0
        
        # Check if plan addresses all goal requirements
        required_actions = self._get_required_actions(plan.goal)
        plan_action_types = {action.step for action in plan.actions}
        
        missing_actions = required_actions - plan_action_types
        if missing_actions:
            validation_score *= 0.7
        
        # Check time constraints
        if plan.total_estimated_time > plan.goal.estimated_time * 1.5:
            validation_score *= 0.8
        
        # Check complexity appropriateness
        if plan.complexity > plan.goal.complexity * 1.2:
            validation_score *= 0.9
        
        # Validate against goal constraints
        for constraint in plan.goal.constraints:
            constraint_valid = await self._validate_constraint(plan, constraint)
            if not constraint_valid:
                validation_score *= (1.0 - constraint.priority * 0.3)
        
        return max(0.1, validation_score)
    
    async def _generate_alternative_plans(self, goal: Goal, available_modules: Set[str] = None) -> List[ReasoningPlan]:
        """Generate alternative plans if primary plan has low confidence."""
        alternatives = []
        
        # Generate simpler plan
        simple_template = self.plan_templates["simple_template"]
        simple_plan = ReasoningPlan(goal=goal)
        await self._build_action_sequence(simple_plan, simple_template, goal)
        alternatives.append(simple_plan)
        
        # Generate more comprehensive plan if goal is complex
        if goal.complexity > 0.6:
            comprehensive_template = self.plan_templates["comprehensive_template"]
            comp_plan = ReasoningPlan(goal=goal)
            await self._build_action_sequence(comp_plan, comprehensive_template, goal)
            alternatives.append(comp_plan)
        
        return alternatives
    
    def _calculate_plan_complexity(self, plan: ReasoningPlan) -> float:
        """Calculate overall plan complexity."""
        complexity = 0.0
        
        # Base complexity from number of actions
        complexity += len(plan.actions) * 0.1
        
        # Add complexity from action types
        action_complexity = {
            PlanStep.RETRIEVE: 0.1,
            PlanStep.REASON: 0.3,
            PlanStep.SYNTHESIZE: 0.4,
            PlanStep.SIMULATE: 0.5,
            PlanStep.DEBATE: 0.4,
            PlanStep.REFLECT: 0.2,
            PlanStep.VALIDATE: 0.2,
            PlanStep.GENERATE: 0.3,
            PlanStep.REFINE: 0.2
        }
        
        for action in plan.actions:
            complexity += action_complexity.get(action.step, 0.2)
        
        # Add complexity from dependencies
        total_deps = sum(len(action.dependencies) for action in plan.actions)
        complexity += total_deps * 0.05
        
        return min(1.0, complexity)
    
    async def _execute_action(self, action: PlanAction, context: Dict[str, Any]) -> Any:
        """Execute a single plan action."""
        
        # This would coordinate with actual modules in production
        # For now, return mock results based on action type
        
        if action.step == PlanStep.RETRIEVE:
            return {"retrieved_info": f"Information for {action.description}"}
        
        elif action.step == PlanStep.REASON:
            return {"reasoning_result": f"Reasoning completed for {action.description}"}
        
        elif action.step == PlanStep.SYNTHESIZE:
            return {"synthesized_concepts": f"Concepts synthesized for {action.description}"}
        
        elif action.step == PlanStep.SIMULATE:
            return {"simulation_result": f"Simulation completed for {action.description}"}
        
        elif action.step == PlanStep.DEBATE:
            return {"debate_result": f"Internal debate completed for {action.description}"}
        
        elif action.step == PlanStep.REFLECT:
            return {"reflection_report": f"Self-reflection completed for {action.description}"}
        
        elif action.step == PlanStep.VALIDATE:
            return {"validation_score": 0.8}
        
        elif action.step == PlanStep.GENERATE:
            return {"generated_content": f"Content generated for {action.description}"}
        
        elif action.step == PlanStep.REFINE:
            return {"refined_output": f"Output refined for {action.description}"}
        
        else:
            return {"result": f"Action completed: {action.description}"}
    
    def _validate_action_result(self, action: PlanAction, result: Any) -> float:
        """Validate action result against success criteria."""
        # Simple validation - in production would use sophisticated metrics
        if isinstance(result, dict) and result:
            return 0.8
        return 0.5
    
    # Helper methods for production functionality
    
    def _determine_reasoning_mode(self, query_type: QueryType) -> str:
        """Determine appropriate reasoning mode for query type."""
        mode_mapping = {
            QueryType.EXPLANATION: "explanatory",
            QueryType.COMPARISON: "comparative", 
            QueryType.ANALYSIS: "explanatory",
            QueryType.SYNTHESIS: "explanatory",
            QueryType.PROBLEM_SOLVING: "causal",
            QueryType.FACTUAL: "explanatory",
            QueryType.CREATIVE: "analogical",
            QueryType.OPINION: "comparative",
            QueryType.DEBUGGING: "causal",
            QueryType.PREDICTION: "causal"
        }
        return mode_mapping.get(query_type, "explanatory")
    
    def _select_debate_agents(self, goal: Goal) -> List[str]:
        """Select appropriate debate agents for the goal."""
        agents = ["skeptic", "advocate"]
        
        if goal.query_type in [QueryType.COMPARISON, QueryType.OPINION]:
            agents.append("balanced_analyzer")
        
        if goal.complexity > 0.7:
            agents.append("domain_expert")
        
        return agents
    
    def _get_required_actions(self, goal: Goal) -> Set[PlanStep]:
        """Get required actions for achieving the goal."""
        required = {PlanStep.RETRIEVE}
        
        if goal.query_type in [QueryType.ANALYSIS, QueryType.SYNTHESIS, QueryType.PROBLEM_SOLVING]:
            required.add(PlanStep.REASON)
        
        if goal.complexity > 0.6:
            required.add(PlanStep.REFLECT)
        
        if any(c.type == "safety" for c in goal.constraints):
            required.add(PlanStep.VALIDATE)
        
        return required
    
    async def _validate_constraint(self, plan: ReasoningPlan, constraint: GoalConstraint) -> bool:
        """Validate plan against a specific constraint."""
        
        if constraint.type == "length":
            # Check if plan will produce appropriate length output
            return True  # Simplified for now
        
        elif constraint.type == "complexity":
            # Check if plan complexity matches constraint
            if constraint.value == "simple":
                return plan.complexity < 0.5
            elif constraint.value == "complex":
                return plan.complexity > 0.6
        
        elif constraint.type == "safety":
            # Ensure safety validation is included
            safety_actions = [a for a in plan.actions if a.step == PlanStep.VALIDATE]
            return len(safety_actions) > 0
        
        return True
    
    async def _calculate_semantic_alignment(self, goal: Goal, original_query: str) -> float:
        """Calculate semantic alignment between goal and original query."""
        # Simplified semantic alignment calculation
        goal_words = set(goal.description.lower().split())
        query_words = set(original_query.lower().split())
        
        if not query_words:
            return 1.0
        
        overlap = len(goal_words.intersection(query_words))
        alignment = overlap / len(query_words)
        
        return min(1.0, alignment + 0.3)  # Boost base alignment
    
    async def _check_safety_constraints(self, goal: Goal, original_query: str) -> List[str]:
        """Check for safety and ethical constraint violations."""
        issues = []
        
        # Check for harmful content requests
        harmful_patterns = [
            r'\b(hack|attack|exploit|virus|malware)\b',
            r'\b(illegal|unlawful|criminal)\b',
            r'\b(hurt|harm|damage|destroy)\b.*\b(person|people|individual)\b'
        ]
        
        query_lower = original_query.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, query_lower):
                issues.append(f"Potential safety concern: query contains harmful content indicators")
                break
        
        return issues
    
    async def _check_resource_availability(self, goal: Goal) -> bool:
        """Check if required resources are available."""
        # This would check actual resource availability in production
        # For now, assume resources are available unless goal is extremely complex
        return goal.complexity < 0.9
    
    def _check_scope_appropriateness(self, goal: Goal, original_query: str) -> List[str]:
        """Check if goal scope is appropriate for the query."""
        issues = []
        
        # Check if goal is too broad for a simple query
        if len(original_query.split()) < 5 and goal.complexity > 0.7:
            issues.append("Goal may be too complex for the simple query provided")
        
        # Check if goal is too narrow for a complex query
        if len(original_query.split()) > 20 and goal.complexity < 0.3:
            issues.append("Goal may be too simple for the complex query provided")
        
        return issues
    
    def _check_constraint_consistency(self, goal: Goal) -> List[str]:
        """Check for inconsistent constraints."""
        issues = []
        
        # Check for conflicting constraints
        constraint_types = {}
        for constraint in goal.constraints:
            if constraint.type in constraint_types:
                if constraint.value != constraint_types[constraint.type]:
                    issues.append(f"Conflicting {constraint.type} constraints detected")
            else:
                constraint_types[constraint.type] = constraint.value
        
        return issues
    
    def _initialize_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize query classification patterns."""
        return {
            QueryType.EXPLANATION: [
                r'\b(explain|describe|tell me about|how does|why does)\b',
                r'\bwhat is\b.*\band how\b',
                r'\bbreak down\b'
            ],
            QueryType.COMPARISON: [
                r'\b(compare|contrast|difference|versus|vs\.?|rather than)\b',
                r'\bwhich is better\b',
                r'\bpros and cons\b'
            ],
            QueryType.ANALYSIS: [
                r'\b(analyze|examine|evaluate|assess|study)\b',
                r'\bwhat.*impact\b',
                r'\bbreak down.*analysis\b'
            ],
            QueryType.SYNTHESIS: [
                r'\b(combine|synthesize|merge|integrate)\b',
                r'\bbring together\b',
                r'\bunify.*concepts\b'
            ],
            QueryType.PROBLEM_SOLVING: [
                r'\bhow to\b.*\b(solve|fix|resolve|address)\b',
                r'\bwhat.*solution\b',
                r'\bsolve.*problem\b'
            ],
            QueryType.FACTUAL: [
                r'^\s*(what|who|when|where|which)\b',
                r'\bdefine\b',
                r'\bwhat is\b(?!.*how)'
            ],
            QueryType.CREATIVE: [
                r'\b(create|generate|make|design|build)\b',
                r'\bcome up with\b',
                r'\binvent\b'
            ],
            QueryType.OPINION: [
                r'\bwhat do you think\b',
                r'\byour opinion\b',
                r'\bdo you believe\b'
            ],
            QueryType.DEBUGGING: [
                r'\bwhy.*not work\b',
                r'\bwhat.*wrong\b',
                r'\btroubleshoot\b'
            ],
            QueryType.PREDICTION: [
                r'\bwhat will happen\b',
                r'\bpredict\b',
                r'\bfuture.*outlook\b'
            ]
        }
    
    def _initialize_plan_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize plan templates for different scenarios."""
        return {
            "explanation_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Retrieve information about {concepts}",
                        "estimated_time": 1.0,
                        "inputs": ["query", "context"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "reason",
                        "description": "Apply reasoning to explain {concepts}",
                        "estimated_time": 2.0,
                        "inputs": ["retrieved_info"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [0]
                    },
                    {
                        "type": "reflect",
                        "description": "Self-reflect on explanation quality",
                        "estimated_time": 0.5,
                        "inputs": ["reasoning_result"],
                        "outputs": ["reflection_report"],
                        "dependencies": [1]
                    }
                ]
            },
            "comparison_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Retrieve information about {concepts}",
                        "estimated_time": 1.5,
                        "inputs": ["query", "context"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "reason",
                        "description": "Compare and contrast {concepts}",
                        "estimated_time": 2.5,
                        "inputs": ["retrieved_info"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [0]
                    },
                    {
                        "type": "debate",
                        "description": "Internal debate on comparison validity",
                        "estimated_time": 1.5,
                        "inputs": ["reasoning_result"],
                        "outputs": ["debate_result"],
                        "dependencies": [1]
                    },
                    {
                        "type": "reflect",
                        "description": "Self-reflect on comparison balance",
                        "estimated_time": 0.5,
                        "inputs": ["debate_result"],
                        "outputs": ["reflection_report"],
                        "dependencies": [2]
                    }
                ]
            },
            "analysis_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Retrieve comprehensive data about {concepts}",
                        "estimated_time": 2.0,
                        "inputs": ["query", "context"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "synthesize",
                        "description": "Synthesize key concepts from {concepts}",
                        "estimated_time": 1.5,
                        "inputs": ["retrieved_info"],
                        "outputs": ["synthesized_concepts"],
                        "dependencies": [0]
                    },
                    {
                        "type": "reason",
                        "description": "Analyze relationships in {concepts}",
                        "estimated_time": 3.0,
                        "inputs": ["synthesized_concepts"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [1]
                    },
                    {
                        "type": "validate",
                        "description": "Validate analysis against constraints",
                        "estimated_time": 1.0,
                        "inputs": ["reasoning_result"],
                        "outputs": ["validation_result"],
                        "dependencies": [2]
                    },
                    {
                        "type": "reflect",
                        "description": "Self-reflect on analysis depth",
                        "estimated_time": 0.5,
                        "inputs": ["validation_result"],
                        "outputs": ["reflection_report"],
                        "dependencies": [3]
                    }
                ]
            },
            "default_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Retrieve relevant information",
                        "estimated_time": 1.0,
                        "inputs": ["query", "context"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "reason",
                        "description": "Apply reasoning to generate response",
                        "estimated_time": 2.0,
                        "inputs": ["retrieved_info"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [0]
                    }
                ]
            },
            "simple_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Quick information retrieval",
                        "estimated_time": 0.5,
                        "inputs": ["query"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "reason",
                        "description": "Basic reasoning for simple response",
                        "estimated_time": 1.0,
                        "inputs": ["retrieved_info"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [0]
                    }
                ]
            },
            "comprehensive_template": {
                "actions": [
                    {
                        "type": "retrieve",
                        "description": "Comprehensive information gathering",
                        "estimated_time": 2.5,
                        "inputs": ["query", "context"],
                        "outputs": ["retrieved_info"]
                    },
                    {
                        "type": "synthesize",
                        "description": "Synthesize complex concepts",
                        "estimated_time": 2.0,
                        "inputs": ["retrieved_info"],
                        "outputs": ["synthesized_concepts"],
                        "dependencies": [0]
                    },
                    {
                        "type": "reason",
                        "description": "Deep reasoning analysis",
                        "estimated_time": 3.5,
                        "inputs": ["synthesized_concepts"],
                        "outputs": ["reasoning_result"],
                        "dependencies": [1]
                    },
                    {
                        "type": "simulate",
                        "description": "Simulate scenarios if applicable",
                        "estimated_time": 2.0,
                        "inputs": ["reasoning_result"],
                        "outputs": ["simulation_result"],
                        "dependencies": [2]
                    },
                    {
                        "type": "debate",
                        "description": "Comprehensive internal debate",
                        "estimated_time": 2.5,
                        "inputs": ["simulation_result"],
                        "outputs": ["debate_result"],
                        "dependencies": [3]
                    },
                    {
                        "type": "validate",
                        "description": "Thorough validation against all constraints",
                        "estimated_time": 1.5,
                        "inputs": ["debate_result"],
                        "outputs": ["validation_result"],
                        "dependencies": [4]
                    },
                    {
                        "type": "reflect",
                        "description": "Comprehensive self-reflection",
                        "estimated_time": 1.0,
                        "inputs": ["validation_result"],
                        "outputs": ["reflection_report"],
                        "dependencies": [5]
                    },
                    {
                        "type": "refine",
                        "description": "Final refinement and optimization",
                        "estimated_time": 1.0,
                        "inputs": ["reflection_report"],
                        "outputs": ["final_result"],
                        "dependencies": [6]
                    }
                ]
            }
        }
    
    def _initialize_constraint_extractors(self) -> Dict[str, Callable]:
        """Initialize constraint extraction functions."""
        return {
            "length": self._extract_length_constraints,
            "complexity": self._extract_complexity_constraints,
            "domain": self._extract_domain_constraints,
            "style": self._extract_style_constraints
        }
    
    def _extract_length_constraints(self, query: str) -> List[GoalConstraint]:
        """Extract length-related constraints from query."""
        # Implementation already in _extract_constraints
        return []
    
    def _extract_complexity_constraints(self, query: str) -> List[GoalConstraint]:
        """Extract complexity-related constraints from query."""
        # Implementation already in _extract_constraints
        return []
    
    def _extract_domain_constraints(self, query: str) -> List[GoalConstraint]:
        """Extract domain-related constraints from query."""
        # Implementation already in _extract_constraints
        return []
    
    def _extract_style_constraints(self, query: str) -> List[GoalConstraint]:
        """Extract style-related constraints from query."""
        constraints = []
        
        style_patterns = [
            (r'\bformal\b', "formal", 0.8),
            (r'\bcasual\b|\binformal\b', "casual", 0.8),
            (r'\bacademic\b', "academic", 0.9),
            (r'\bprofessional\b', "professional", 0.8)
        ]
        
        for pattern, value, priority in style_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                constraints.append(GoalConstraint(
                    type="style",
                    value=value,
                    priority=priority,
                    description=f"Response style: {value}"
                ))
                break
        
        return constraints
    
    async def _archive_goal(self, goal: Goal, query: str, context: str):
        """Archive goal formulation for learning and improvement."""
        if self.psi_archive:
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "context_length": len(context),
                "goal": {
                    "description": goal.description,
                    "type": goal.query_type.value,
                    "complexity": goal.complexity,
                    "confidence": goal.confidence,
                    "concepts_count": len(goal.primary_concepts),
                    "constraints_count": len(goal.constraints),
                    "success_criteria_count": len(goal.success_criteria)
                }
            }
            await self.psi_archive.log_goal_formulation(archive_data)
    
    async def _archive_plan(self, plan: ReasoningPlan, goal: Goal):
        """Archive plan creation for learning and improvement."""
        if self.psi_archive:
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "goal_type": goal.query_type.value,
                "plan": {
                    "actions_count": len(plan.actions),
                    "complexity": plan.complexity,
                    "confidence": plan.confidence,
                    "estimated_time": plan.total_estimated_time,
                    "action_types": [action.step.value for action in plan.actions]
                }
            }
            await self.psi_archive.log_plan_creation(archive_data)
    
    async def get_planning_stats(self) -> Dict[str, Any]:
        """Get current planning statistics."""
        stats = self.planning_stats.copy()
        
        if stats["goals_formulated"] > 0:
            stats["average_planning_time"] = stats["total_planning_time"] / stats["goals_formulated"]
        
        if stats["plans_created"] > 0:
            stats["success_rate"] = stats["successful_plans"] / stats["plans_created"]
        
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    async def health_check(self) -> bool:
        """Health check for cognitive agent."""
        try:
            # Test goal formulation
            test_goal = await self.formulate_goal("test query")
            
            # Test plan building  
            test_plan = await self.build_plan(test_goal)
            
            return test_goal.confidence > 0 and test_plan.confidence > 0
        except Exception:
            return False

if __name__ == "__main__":
    # Production test
    async def test_cognitive_agent():
        agent = CognitiveAgent()
        
        # Test goal formulation
        test_query = "Explain how quantum mechanics relates to consciousness in simple terms"
        goal = await agent.formulate_goal(test_query)
        
        print(f"✅ CognitiveAgent Test Results:")
        print(f"   Goal: {goal.description}")
        print(f"   Type: {goal.query_type.value}")
        print(f"   Concepts: {goal.primary_concepts}")
        print(f"   Complexity: {goal.complexity:.2f}")
        print(f"   Confidence: {goal.confidence:.2f}")
        print(f"   Constraints: {len(goal.constraints)}")
        
        # Test plan building
        plan = await agent.build_plan(goal)
        
        print(f"   Plan actions: {len(plan.actions)}")
        print(f"   Plan confidence: {plan.confidence:.2f}")
        print(f"   Estimated time: {plan.total_estimated_time:.1f}s")
        
        for i, action in enumerate(plan.actions):
            print(f"   Step {i}: {action.step.value} - {action.description}")
        
        # Test intent alignment
        is_aligned, score, issues = await agent.evaluate_intent_alignment(goal, test_query)
        print(f"   Intent aligned: {is_aligned}")
        print(f"   Alignment score: {score:.2f}")
        if issues:
            print(f"   Issues: {issues}")
    
    import asyncio
    asyncio.run(test_cognitive_agent())
