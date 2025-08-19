# Hybrid Systems in ELFIN

This file describes the hybrid systems support in ELFIN with a focus on barrier functions and verification.

## Hybrid System Components

ELFIN provides several specialized building blocks for hybrid systems:

- `hybrid_system`: Define systems with both continuous and discrete dynamics
- `hybrid_lyapunov`: Stability certification for hybrid systems
- `hybrid_barrier`: Safety verification via barrier functions
- `hybrid_controller`: Controllers that handle both continuous and discrete actions
- `hybrid_verification`: Formal verification of hybrid system properties
- `hybrid_simulation`: Simulation configuration for hybrid systems

## Key Examples

1. **BouncingBall**: Classic hybrid system with impacts and energy dissipation
2. **Thermostat**: Bang-bang controller with hysteresis
3. **LeggedRobot**: Stance/flight phases with guard conditions
4. **LaneChanging**: Mode-based vehicle navigation with safety barriers
5. **VTOL**: Aircraft with transition between multirotor and fixed-wing modes

## Barrier Function Best Practices

1. **Proper Formulation**:
   - Ensure B(x) > 0 in the safe set
   - Use linear terms (x_max - x) rather than quadratic terms when appropriate
   - For multi-obstacle barriers, use smooth approximations like log-sum-exp

2. **Definition Order**:
   - Define helper variables before they are used (e.g., v_max before B: v_max^2 - v^2)
   - Place alpha_fun outside params block but inside barrier block

3. **Syntax Requirements**:
   - Use ** for exponentiation, not ^ (caret is XOR in the DSL)
   - Use wrap_angle(angle) to normalize angles to [-π, π]
   - Guard against division by zero (e.g., max(V, 1e-6))

## Verification Process

1. Run `elf verify barrier --file=your_file.elfin` to check all barriers
2. Run `elf verify lyapunov --file=your_file.elfin` to check stability
3. Run `elf verify safety --file=your_file.elfin` to verify safety properties
4. Run `elf verify invariant --file=your_file.elfin` to check invariants
5. Run `elf verify all --file=your_file.elfin` for complete verification

## Common Pitfalls

1. **Undefined Symbols**: Variables must be defined before use in barrier functions
2. **Sign Errors**: Ensure barrier convention is consistent (positive = safe)
3. **Smoothness Issues**: Use smooth approximations for min/max operations
4. **Zeno Behavior**: Check for potential infinite jumps in finite time
5. **Numerical Issues**: Add small epsilon values to avoid division by zero
