# TORI Self-Transformation Architecture

## Overview

This implementation provides TORI with phase-coherent cognition capabilities through a geometrically-inspired self-transformation system. The architecture ensures safe, controlled evolution while maintaining system integrity.

## Core Components

### 1. Constitutional Safety (`/safety/constitution.py`)

The constitution acts as an invariant kernel, defining a topologically closed sub-manifold in TORI's state space. All mutations must preserve these invariants:

- **Identity Preservation**: Immutable UUID and hash ensure core identity persistence
- **Resource Budgets**: Hard limits on CPU, GPU, and RAM usage
- **Safety Rules**: Forbidden system calls and rate limiting
- **Rollback Capability**: Quorum-based reversion mechanism

### 2. Critic Consensus (`/meta_genome/critics/aggregation.py`)

Implements weighted Riemannian barycenter aggregation using log-odds transformation:

```
S = Σ(w_i * s_i) / Σ(w_i)
where w_i = log(p_i / (1 - p_i))
```

This negative curvature approach prevents flat consensus while privileging reliable critics.

### 3. Sandboxed Mutation Testing (`/meta/sandbox/runner.py`)

Container-free phase-space test harness using:

- Git worktrees for isolated testing
- POSIX resource limits (rlimit)
- Automatic rollback on failure
- Comprehensive audit logging

### 4. Energy Budget Management (`/meta/energy_budget.py`)

Implements discrete-time entropy production:

```
E_{t+1} = E_t + ΔQ - ΔW
```

Where:
- ΔQ = compute heat (CPU-seconds)
- ΔW = work returned (validated utility)

### 5. Analogical Transfer (`/goals/analogical_transfer.py`)

Cross-domain knowledge transfer using:

- Graph manifold representation
- Geodesic distance calculations
- Spectral decomposition via graph Laplacian
- Transfer kernel computation

## Integration Points

### With Existing TORI Systems

1. **Cognitive Engine Integration**
   - Energy budget hooks into `CognitiveEngine.py`
   - Critics interface with metacognitive monitoring

2. **Memory Vault Connection**
   - Audit logs stored in memory vault
   - Transfer patterns saved for future use

3. **MCP Metacognitive Bridge**
   - Critics provide input to MCP decisions
   - Energy states exposed via MCP protocol

## Usage Examples

### Basic Self-Modification Flow

```python
# 1. Propose mutation
mutation = generate_mutation()

# 2. Get critic consensus
scores = gather_critic_scores(mutation)
accepted, score = aggregate(scores, reliabilities)

if accepted:
    # 3. Test in sandbox
    success = run_mutation(mutation.patch_path)
    
    if success:
        # 4. Apply with energy awareness
        if energy_budget.update(mutation.cost, mutation.utility):
            apply_mutation(mutation)
```

### Cross-Domain Strategy Transfer

```python
# Transfer optimization strategy from math to physics domain
math_strategy = {"precision": 0.99, "iterations": 1000}
physics_strategy = transfer.transfer_strategy("mathematics", "physics", math_strategy)
```

## Safety Guarantees

1. **Constitutional Boundaries**: Hard limits prevent resource exhaustion
2. **Quorum Rollback**: Multiple critics must approve before permanent changes
3. **Sandboxed Testing**: All mutations tested in isolation first
4. **Energy Throttling**: Prevents runaway computation
5. **Audit Trail**: Complete history for debugging and analysis

## Next Steps for Production

1. **Populate constitution.json** with production-appropriate limits
2. **Seed critic database** with initial Beta(2,2) parameters
3. **Configure energy thresholds** based on hardware capabilities
4. **Enable GPU tracking** in resource monitoring
5. **Set up Prometheus metrics** export for monitoring

## Mathematical Foundations

### Manifold Geometry
- State space M as high-dimensional manifold
- Constitution defines closed sub-manifold C ⊂ M
- Mutations as retractions r: M → C

### Information Geometry
- Critics sample from Beta-Bernoulli field
- Log-odds weighting induces negative curvature
- Prevents convergence to flat consensus

### Spectral Graph Theory
- Knowledge domains as graph nodes
- Laplacian eigenvectors as transfer basis
- Geodesic distance guides analogical mapping

## Performance Considerations

- Sandbox overhead: ~100ms per mutation test
- Critic aggregation: O(n) for n critics
- Energy update: O(1) amortized
- Transfer kernel computation: O(n³) for n domains (cached)

## Security Model

- No external dependencies in core loop
- All I/O operations audited
- Forbidden syscalls blocked at constitution level
- Resource limits enforced by OS kernel
- Rollback requires multi-critic consensus

---

This architecture enables TORI to safely explore its cognitive phase space while maintaining stability and coherence. The geometric approach ensures principled self-modification aligned with the system's fundamental invariants.
