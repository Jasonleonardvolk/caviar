"""
Cognitive interaction prompts
"""

import json
import numpy as np


def register_cognitive_prompts(mcp, state_manager):
    """Register cognitive prompts for common interactions."""
    
    @mcp.prompt()
    async def explore_consciousness() -> str:
        """
        Guide exploration of consciousness landscape.
        
        This prompt helps analyze and enhance consciousness levels
        through various cognitive operations.
        """
        current = await state_manager.get_current_state()
        
        return f"""You are exploring the consciousness landscape of a cognitive system.

Current State:
- Integrated Information (Φ): {current['phi']:.3f}
- Free Energy: {current['free_energy']:.3f}
- Dimension: {current['dimension']}

Available Actions:
1. Measure consciousness with different connectivity patterns
2. Apply interventions to enhance consciousness
3. Analyze consciousness history and trends
4. Find high-consciousness fixed points

Suggested Exploration:
- If Φ < {state_manager.config.consciousness_threshold}, consider intervention
- Try different connectivity patterns: 'full', 'sparse', 'small_world', 'modular'
- Use consciousness_intervention tool with 'auto' mode for smart interventions
- Analyze patterns with analyze_consciousness_history

What aspect of consciousness would you like to explore?"""
    
    @mcp.prompt()
    async def cognitive_optimization() -> str:
        """
        Guide cognitive state optimization process.
        
        This prompt helps optimize cognitive states for various objectives.
        """
        current = await state_manager.get_current_state()
        stats = state_manager.consciousness_monitor.get_statistics()
        
        return f"""You are optimizing a cognitive system with multiple objectives.

Current Metrics:
- Free Energy: {current['free_energy']:.3f}
- Consciousness (Φ): {current['phi']:.3f}
- State Norm: {np.linalg.norm(np.array(current['state'])):.3f}

Optimization Options:
1. self_modify - Minimize free energy while preserving consciousness
2. reflect - Natural gradient ascent on beliefs
3. find_fixed_point - Find stable cognitive configurations
4. stabilize - Apply control to reach target states

Recent Performance:
- Consciousness violations: {stats.get('violation_rate', 0):.1%}
- Average Φ: {stats.get('average_phi', 0):.3f}

Recommendations:
- Use adaptive=True for self_modify for better convergence
- Try different lambda_d values (0.1 to 10) to balance objectives
- Monitor consciousness with each optimization step

What optimization goal would you like to pursue?"""
    
    @mcp.prompt()
    async def metacognitive_analysis() -> str:
        """
        Guide metacognitive analysis and tower exploration.
        
        This prompt helps understand hierarchical cognitive structure.
        """
        tower = state_manager.tower
        n_levels = len(tower.levels)
        
        return f"""You are analyzing the metacognitive architecture of a cognitive system.

Tower Structure:
- Levels: {n_levels}
- Base Dimension: {tower.base_dim}
- Metric: {tower.metric}

Level Details:
- Level 0: Base cognitive state ({tower.levels[0]['dim']}D)
- Level 1: State + Velocity ({tower.levels[1]['dim']}D)
- Level 2: State + Velocity + Curvature ({tower.levels[2]['dim']}D)
{f"- Level 3+: Abstract metacognitive features" if n_levels > 3 else ""}

Analysis Options:
1. lift_to_metacognitive_level - Explore higher-order representations
2. compute_holonomy - Measure cognitive curvature
3. query_knowledge_sheaf - Access distributed knowledge
4. Get tower structure with resource: tori://tower/structure

Suggested Analyses:
- Lift to level 1 to see cognitive velocity
- Compute holonomy to detect curvature in cognitive space
- Query knowledge sheaf for local vs global knowledge

Which metacognitive aspect interests you?"""
    
    @mcp.prompt()
    async def dynamics_exploration() -> str:
        """
        Guide exploration of cognitive dynamics.
        
        This prompt helps understand how cognitive states evolve over time.
        """
        current = await state_manager.get_current_state()
        
        return f"""You are exploring the dynamics of a cognitive system.

Current Configuration:
- State Dimension: {state_manager.dimension}
- Noise Level: {state_manager.dynamics.sigma}
- Manifold Metric: {state_manager.manifold.metric}

Dynamics Tools:
1. evolve - Run cognitive dynamics forward in time
2. compute_lyapunov_exponents - Analyze stability/chaos
3. stabilize - Apply control to guide dynamics
4. Get trajectory with resource: tori://state/trajectory/50

Key Parameters:
- time_span: How long to evolve (try 0.1 to 10.0)
- adaptive: Use adaptive timestepping for accuracy
- noise_scale: Override noise (0 for deterministic)

Analysis Suggestions:
- Short evolution (0.1-1.0) to see local dynamics
- Compute Lyapunov exponents to check for chaos
- Use stabilize to guide toward desired states
- Monitor consciousness during evolution

What aspect of dynamics would you like to explore?"""
    
    @mcp.prompt()
    async def diagnostic_check() -> str:
        """
        Perform comprehensive system diagnostic.
        
        This prompt helps assess overall system health.
        """
        # Gather diagnostic information
        current = await state_manager.get_current_state()
        cons_stats = state_manager.consciousness_monitor.get_statistics()
        
        # Check for issues
        issues = []
        if cons_stats.get('violation_rate', 0) > 0.1:
            issues.append("High consciousness violation rate")
        if current['phi'] < state_manager.config.consciousness_threshold * 0.8:
            issues.append("Low consciousness level")
        if current['free_energy'] > 10:
            issues.append("High free energy")
        
        status = "⚠️ Issues Detected" if issues else "✅ Healthy"
        
        return f"""System Diagnostic Report

Status: {status}

Core Metrics:
- Consciousness (Φ): {current['phi']:.3f} (threshold: {state_manager.config.consciousness_threshold})
- Free Energy: {current['free_energy']:.3f}
- Session Duration: {current['session_duration']:.1f}s

Performance:
- State Updates: {len(state_manager.state_history)}
- Violation Rate: {cons_stats.get('violation_rate', 0):.1%}
- Memory Usage: {len(state_manager.state_history) + len(state_manager.event_history)} items

{"Issues Found:" if issues else ""}
{chr(10).join(f"- {issue}" for issue in issues) if issues else ""}

Diagnostic Tools:
- get_server_info() - Server configuration
- Resource: tori://performance/metrics - Detailed metrics
- Resource: tori://consciousness/monitor - Consciousness analysis
- Resource: tori://stability/analysis - Stability check

{f"Recommended Actions:{chr(10)}- " + chr(10).join([
    "Apply consciousness_intervention",
    "Run self_modify with high IIT weight",
    "Check stability with compute_lyapunov_exponents"
][:len(issues)]) if issues else "System operating normally."}"""
    
    @mcp.prompt()
    async def research_assistant() -> str:
        """
        Guide cognitive research experiments.
        
        This prompt helps design and run cognitive experiments.
        """
        return f"""Welcome to the TORI Cognitive Research Assistant!

I can help you design and run experiments on:

1. **Consciousness Studies**
   - Measure Φ under different conditions
   - Test consciousness preservation
   - Find optimal connectivity patterns

2. **Optimization Research**
   - Compare optimization algorithms
   - Study free energy landscapes
   - Analyze fixed points

3. **Dynamics Analysis**
   - Characterize attractors
   - Study chaos and stability
   - Explore control strategies

4. **Metacognitive Structure**
   - Map knowledge organization
   - Study hierarchical representations
   - Analyze information flow

Experiment Templates:

A) Consciousness vs Connectivity:
for connectivity in ['full', 'sparse', 'small_world', 'modular']:
measure_consciousness(connectivity_type=connectivity)

B) Optimization Comparison:
for method in ['iteration', 'newton', 'anderson']:
find_fixed_point(method=method)

C) Dynamics Characterization:
evolve(time_span=10.0)
compute_lyapunov_exponents()
analyze trajectory

What kind of experiment would you like to design?"""