"""
Stability Code Generation for ELFIN DSL.

This module provides code generation capabilities for the stability-related
constructs in the ELFIN DSL. It generates executable Python code from
parsed Lyapunov function declarations, verification directives, and phase
synchronization monitoring statements.
"""

import logging
import textwrap
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

try:
    from alan_backend.elfin.parser.stability_parser import (
        LyapunovDeclaration, VerificationDirective, StabilityDirective,
        StabilityConstraint, PhaseMonitor, AdaptiveTrigger,
        LyapunovType, VerificationMethod, ComparisonOperator
    )
except ImportError:
    # For standalone testing
    from enum import Enum, auto
    
    class LyapunovType(Enum):
        POLYNOMIAL = auto()
        NEURAL = auto()
        CLF = auto()
        COMPOSITE = auto()
    
    class VerificationMethod(Enum):
        SOS = auto()
        SAMPLING = auto()
        MILP = auto()
        SMT = auto()
    
    class ComparisonOperator(Enum):
        LT = "<"
        GT = ">"
        LE = "<="
        GE = ">="
        EQ = "=="
        APPROX = "≈"
    
    class LyapunovDeclaration:
        def __init__(self, name, lyap_type, domain=None, symbolic_form=None, 
                     parameters=None, verification_hints=None):
            self.name = name
            self.lyap_type = lyap_type
            self.domain = domain or []
            self.symbolic_form = symbolic_form
            self.parameters = parameters or {}
            self.verification_hints = verification_hints or {}
    
    class VerificationDirective:
        def __init__(self, targets, method, options=None):
            self.targets = targets
            self.method = method
            self.options = options or {}
    
    class StabilityDirective:
        def __init__(self, predicate):
            self.predicate = predicate
    
    class StabilityConstraint:
        def __init__(self, predicate):
            self.predicate = predicate
    
    class PhaseMonitor:
        def __init__(self, target, operator, threshold, options=None):
            self.target = target
            self.operator = operator
            self.threshold = threshold
            self.options = options or {}
    
    class AdaptiveTrigger:
        def __init__(self, condition, actions):
            self.condition = condition
            self.actions = actions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StabilityCodeGenerator:
    """Code generator for stability-related constructs."""
    
    def __init__(self):
        """Initialize code generator."""
        self.imports = set()
        self.lyapunov_functions = {}
        self.verifiers = {}
        self.monitors = {}
        self.triggers = {}
    
    def generate_imports(self) -> str:
        """
        Generate import statements.
        
        Returns:
            Import statements as a string
        """
        import_lines = []
        
        # Standard library imports
        if "numpy" in self.imports:
            import_lines.append("import numpy as np")
        
        if "time" in self.imports:
            import_lines.append("import time")
        
        if "logging" in self.imports:
            import_lines.append("import logging")
        
        # ELFIN stability imports
        if "lyapunov" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.lyapunov import *")
        
        if "verifier" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.verifier import *")
        
        if "incremental_verifier" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.incremental_verifier import *")
        
        if "jit_guard" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.jit_guard import *")
        
        if "koopman_bridge" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.koopman_bridge_poc import *")
        
        if "phase_drift_monitor" in self.imports:
            import_lines.append("from alan_backend.elfin.stability.phase_drift_monitor import *")
        
        return "\n".join(import_lines)
    
    def generate_lyapunov_function(self, lyap_decl: LyapunovDeclaration) -> str:
        """
        Generate code for a Lyapunov function declaration.
        
        Args:
            lyap_decl: Lyapunov function declaration
            
        Returns:
            Generated code as a string
        """
        self.imports.add("lyapunov")
        self.imports.add("numpy")
        
        if lyap_decl.lyap_type == LyapunovType.POLYNOMIAL:
            return self._generate_polynomial_lyapunov(lyap_decl)
        elif lyap_decl.lyap_type == LyapunovType.NEURAL:
            return self._generate_neural_lyapunov(lyap_decl)
        elif lyap_decl.lyap_type == LyapunovType.CLF:
            return self._generate_clf_lyapunov(lyap_decl)
        elif lyap_decl.lyap_type == LyapunovType.COMPOSITE:
            return self._generate_composite_lyapunov(lyap_decl)
        else:
            # Default to polynomial if type is unknown
            return self._generate_polynomial_lyapunov(lyap_decl)
    
    def _generate_polynomial_lyapunov(self, lyap_decl: LyapunovDeclaration) -> str:
        """Generate code for a polynomial Lyapunov function."""
        degree = lyap_decl.parameters.get("degree", 2)
        domain_str = ", ".join(f'"{d}"' for d in lyap_decl.domain)
        
        code = f"""
        # Polynomial Lyapunov function: {lyap_decl.name}
        {lyap_decl.name}_Q = np.eye({degree})  # Default to identity matrix
        {lyap_decl.name} = PolynomialLyapunov(
            name="{lyap_decl.name}",
            Q={lyap_decl.name}_Q,
            domain_ids=[{domain_str}]
        )
        """
        
        # Store for later reference
        self.lyapunov_functions[lyap_decl.name] = {
            "type": "polynomial",
            "degree": degree,
            "domain": lyap_decl.domain
        }
        
        return textwrap.dedent(code)
    
    def _generate_neural_lyapunov(self, lyap_decl: LyapunovDeclaration) -> str:
        """Generate code for a neural network Lyapunov function."""
        layers = lyap_decl.parameters.get("layers", [10, 10])
        domain_str = ", ".join(f'"{d}"' for d in lyap_decl.domain)
        
        layers_str = ", ".join(str(l) for l in layers)
        
        code = f"""
        # Neural Lyapunov function: {lyap_decl.name}
        {lyap_decl.name}_layers = [{layers_str}]
        {lyap_decl.name} = NeuralLyapunov(
            name="{lyap_decl.name}",
            layer_dims={lyap_decl.name}_layers,
            domain_ids=[{domain_str}]
        )
        """
        
        # Store for later reference
        self.lyapunov_functions[lyap_decl.name] = {
            "type": "neural",
            "layers": layers,
            "domain": lyap_decl.domain
        }
        
        return textwrap.dedent(code)
    
    def _generate_clf_lyapunov(self, lyap_decl: LyapunovDeclaration) -> str:
        """Generate code for a Control Lyapunov function."""
        variables = lyap_decl.parameters.get("variables", ["u"])
        domain_str = ", ".join(f'"{d}"' for d in lyap_decl.domain)
        variables_str = ", ".join(f'"{v}"' for v in variables)
        
        code = f"""
        # Control Lyapunov function: {lyap_decl.name}
        {lyap_decl.name}_variables = [{variables_str}]
        {lyap_decl.name} = CLFFunction(
            name="{lyap_decl.name}",
            control_variables={lyap_decl.name}_variables,
            domain_ids=[{domain_str}]
        )
        """
        
        # Store for later reference
        self.lyapunov_functions[lyap_decl.name] = {
            "type": "clf",
            "variables": variables,
            "domain": lyap_decl.domain
        }
        
        return textwrap.dedent(code)
    
    def _generate_composite_lyapunov(self, lyap_decl: LyapunovDeclaration) -> str:
        """Generate code for a composite Lyapunov function."""
        components = lyap_decl.parameters.get("components", [])
        domain_str = ", ".join(f'"{d}"' for d in lyap_decl.domain)
        components_str = ", ".join(c for c in components)
        
        code = f"""
        # Composite Lyapunov function: {lyap_decl.name}
        {lyap_decl.name}_components = [{components_str}]
        {lyap_decl.name} = CompositeLyapunov(
            name="{lyap_decl.name}",
            components={lyap_decl.name}_components,
            domain_ids=[{domain_str}]
        )
        """
        
        # Store for later reference
        self.lyapunov_functions[lyap_decl.name] = {
            "type": "composite",
            "components": components,
            "domain": lyap_decl.domain
        }
        
        return textwrap.dedent(code)
    
    def generate_verification_directive(self, directive: VerificationDirective) -> str:
        """
        Generate code for a verification directive.
        
        Args:
            directive: Verification directive
            
        Returns:
            Generated code as a string
        """
        self.imports.add("verifier")
        self.imports.add("incremental_verifier")
        
        targets_str = ", ".join(directive.targets)
        
        if directive.method == VerificationMethod.SOS:
            verifier_class = "SOSVerifier"
        elif directive.method == VerificationMethod.SAMPLING:
            verifier_class = "SamplingVerifier"
        elif directive.method == VerificationMethod.MILP:
            verifier_class = "MILPVerifier"
        elif directive.method == VerificationMethod.SMT:
            verifier_class = "SMTVerifier"
        else:
            verifier_class = "LyapunovVerifier"
        
        # Generate options
        options = []
        for key, value in directive.options.items():
            if isinstance(value, str) and not value.isdigit():
                # Quote string values
                options.append(f"{key}=\"{value}\"")
            else:
                options.append(f"{key}={value}")
        
        options_str = ", ".join(options)
        
        # Generate a unique verifier ID
        verifier_id = f"verifier_{len(self.verifiers) + 1}"
        
        code = f"""
        # Verification directive for: {targets_str}
        {verifier_id} = {verifier_class}({options_str})
        cache = ProofCache(cache_dir="./proof_cache")
        parallel_verifier = ParallelVerifier(verifier={verifier_id}, cache=cache)
        
        # Verify all targets
        verification_tasks = []
        """
        
        # Add each target to verification tasks
        for target in directive.targets:
            code += f"""
            verification_tasks.append(({target}, None, None))
            """
        
        code += """
        print(f"Verifying {len(verification_tasks)} Lyapunov functions...")
        results = parallel_verifier.verify_batch(verification_tasks, show_progress=True)
        
        # Print results
        for proof_hash, result in results.items():
            print(f"  {result.lyapunov_name}: {result.status.name}")
        """
        
        # Store for later reference
        self.verifiers[verifier_id] = {
            "method": directive.method,
            "targets": directive.targets,
            "options": directive.options
        }
        
        return textwrap.dedent(code)
    
    def generate_stability_directive(self, directive: StabilityDirective) -> str:
        """
        Generate code for a stability directive.
        
        Args:
            directive: Stability directive
            
        Returns:
            Generated code as a string
        """
        self.imports.add("jit_guard")
        
        predicate = directive.predicate
        
        # Extract target from predicate
        target = predicate.left.target
        
        # Generate comparison
        op_map = {
            ComparisonOperator.LT: "<",
            ComparisonOperator.GT: ">",
            ComparisonOperator.LE: "<=",
            ComparisonOperator.GE: ">=",
            ComparisonOperator.EQ: "==",
            ComparisonOperator.APPROX: "≈"
        }
        
        op_str = op_map.get(predicate.operator, "<")
        
        # Right side
        if hasattr(predicate.right, "value"):
            right_str = str(predicate.right.value)
        else:
            right_str = f"Lyapunov({predicate.right.target})"
        
        # Generate a unique guard ID
        guard_id = f"stability_guard_{len(self.monitors) + 1}"
        
        code = f"""
        # Stability directive for: {target}
        def stability_violation_callback(x_prev, x, guard):
            print(f"Stability violation detected for {target}")
            print(f"  Previous state: {{x_prev}}")
            print(f"  Current state: {{x}}")
            print(f"  Violation count: {{guard.violations}}")
        
        {guard_id} = StabilityGuard(
            lyap=get_lyapunov_function("{target}"),
            threshold={right_str},
            callback=stability_violation_callback
        )
        
        # Monitor stability during system evolution
        def check_stability_{len(self.monitors) + 1}(x_prev, x):
            return {guard_id}.step(x_prev, x)
        """
        
        # Store for later reference
        self.monitors[guard_id] = {
            "target": target,
            "operator": op_str,
            "threshold": right_str
        }
        
        return textwrap.dedent(code)
    
    def generate_phase_monitor(self, monitor: PhaseMonitor) -> str:
        """
        Generate code for a phase monitor.
        
        Args:
            monitor: Phase monitor
            
        Returns:
            Generated code as a string
        """
        self.imports.add("phase_drift_monitor")
        
        # Parse phase target
        target = monitor.target
        
        # Generate comparison
        op_map = {
            ComparisonOperator.LT: "<",
            ComparisonOperator.GT: ">",
            ComparisonOperator.LE: "<=",
            ComparisonOperator.GE: ">=",
            ComparisonOperator.EQ: "==",
            ComparisonOperator.APPROX: "≈"
        }
        
        op_str = op_map.get(monitor.operator, ">")
        
        # Handle special threshold values
        threshold_str = str(monitor.threshold)
        if threshold_str == "π":
            threshold_str = "np.pi"
        elif threshold_str == "π/2":
            threshold_str = "np.pi/2"
        elif threshold_str == "π/4":
            threshold_str = "np.pi/4"
        elif threshold_str == "π/8":
            threshold_str = "np.pi/8"
        
        # Generate options
        options = []
        for key, value in monitor.options.items():
            if isinstance(value, str) and not value.isdigit():
                # Quote string values
                options.append(f"{key}=\"{value}\"")
            else:
                options.append(f"{key}={value}")
        
        options_str = ", ".join(options)
        
        # Generate a unique monitor ID
        monitor_id = f"phase_monitor_{len(self.monitors) + 1}"
        
        code = f"""
        # Phase drift monitor for: {target}
        def drift_callback(phase_state, drift_amount):
            print(f"Phase drift detected for {target}")
            print(f"  Current phase: {{phase_state.get_phase({target})}}")
            print(f"  Drift amount: {{drift_amount}}")
        
        {monitor_id} = PhaseDriftMonitor(
            target="{target}",
            threshold={threshold_str},
            callback=drift_callback,
            {options_str}
        )
        
        # Register monitor with phase state
        phase_state.register_monitor({monitor_id})
        """
        
        # Store for later reference
        self.monitors[monitor_id] = {
            "target": target,
            "operator": op_str,
            "threshold": threshold_str,
            "options": monitor.options
        }
        
        return textwrap.dedent(code)
    
    def generate_adaptive_trigger(self, trigger: AdaptiveTrigger) -> str:
        """
        Generate code for an adaptive trigger.
        
        Args:
            trigger: Adaptive trigger
            
        Returns:
            Generated code as a string
        """
        self.imports.add("phase_drift_monitor")
        
        # Generate condition
        if hasattr(trigger.condition, "target"):
            # Phase drift condition
            target = trigger.condition.target
            
            op_map = {
                ComparisonOperator.LT: "<",
                ComparisonOperator.GT: ">",
                ComparisonOperator.LE: "<=",
                ComparisonOperator.GE: ">=",
                ComparisonOperator.EQ: "==",
                ComparisonOperator.APPROX: "≈"
            }
            
            op_str = op_map.get(trigger.condition.operator, ">")
            
            # Handle special threshold values
            threshold = trigger.condition.threshold
            if threshold == "π":
                threshold_str = "np.pi"
            elif threshold == "π/2":
                threshold_str = "np.pi/2"
            elif threshold == "π/4":
                threshold_str = "np.pi/4"
            elif threshold == "π/8":
                threshold_str = "np.pi/8"
            else:
                threshold_str = str(threshold)
            
            condition_str = f"get_phase_drift(\"{target}\") {op_str} {threshold_str}"
        else:
            # Lyapunov condition
            left = trigger.condition.left
            right = trigger.condition.right
            
            op_map = {
                ComparisonOperator.LT: "<",
                ComparisonOperator.GT: ">",
                ComparisonOperator.LE: "<=",
                ComparisonOperator.GE: ">=",
                ComparisonOperator.EQ: "==",
                ComparisonOperator.APPROX: "≈"
            }
            
            op_str = op_map.get(trigger.condition.operator, "<")
            
            if hasattr(right, "value"):
                right_str = str(right.value)
            else:
                right_str = f"get_lyapunov_value(\"{right.target}\")"
            
            if left.is_derivative:
                left_str = f"get_lyapunov_derivative(\"{left.target}\")"
            else:
                left_str = f"get_lyapunov_value(\"{left.target}\")"
            
            condition_str = f"{left_str} {op_str} {right_str}"
        
        # Generate actions
        actions = []
        for action in trigger.actions:
            args = []
            for arg in action.arguments:
                if isinstance(arg, str) and not arg.isdigit():
                    args.append(f"\"{arg}\"")
                else:
                    args.append(str(arg))
            
            args_str = ", ".join(args)
            actions.append(f"{action.name}({args_str})")
        
        actions_str = "\n    ".join(actions)
        
        # Generate a unique trigger ID
        trigger_id = f"adaptive_trigger_{len(self.triggers) + 1}"
        
        code = f"""
        # Adaptive trigger for condition: {condition_str}
        def {trigger_id}():
            if {condition_str}:
                # Execute actions
                {actions_str}
        
        # Register trigger with runtime
        register_trigger({trigger_id})
        """
        
        # Store for later reference
        self.triggers[trigger_id] = {
            "condition": condition_str,
            "actions": actions
        }
        
        return textwrap.dedent(code)
    
    def generate_helper_functions(self) -> str:
        """
        Generate helper functions.
        
        Returns:
            Generated code as a string
        """
        code = """
        # Helper functions for stability management
        
        def get_lyapunov_function(name):
            """Get a Lyapunov function by name."""
            # This would connect to a registry in the real implementation
            if name in globals():
                return globals()[name]
            raise ValueError(f"Unknown Lyapunov function: {name}")
        
        def get_lyapunov_value(target):
            """Get the value of a Lyapunov function for the current state."""
            # In a real implementation, this would use the current system state
            lyap = get_lyapunov_function(target)
            return lyap.evaluate(system_state)
        
        def get_lyapunov_derivative(target):
            """Get the derivative of a Lyapunov function for the current state."""
            # In a real implementation, this would compute the actual derivative
            lyap = get_lyapunov_function(target)
            return compute_lyapunov_derivative(lyap, system_state)
        
        def get_phase_drift(target):
            """Get the phase drift for a target."""
            # In a real implementation, this would query the phase monitor
            return phase_state.get_drift(target)
        
        def register_trigger(trigger_fn):
            """Register an adaptive trigger function."""
            # In a real implementation, this would register with the runtime
            global triggers
            if 'triggers' not in globals():
                triggers = []
            triggers.append(trigger_fn)
        
        def check_triggers():
            """Check all registered triggers."""
            for trigger in triggers:
                trigger()
        """
        
        return textwrap.dedent(code)
    
    def generate_code(self, ast_nodes: List[Any]) -> str:
        """
        Generate code for a list of AST nodes.
        
        Args:
            ast_nodes: List of AST nodes
            
        Returns:
            Generated code as a string
        """
        # Reset state
        self.imports = set()
        self.lyapunov_functions = {}
        self.verifiers = {}
        self.monitors = {}
        self.triggers = {}
        
        # Add numpy and time by default
        self.imports.add("numpy")
        self.imports.add("time")
        
        code_sections = []
        
        # Generate code for each node
        for node in ast_nodes:
            if isinstance(node, LyapunovDeclaration):
                code_sections.append(self.generate_lyapunov_function(node))
            elif isinstance(node, VerificationDirective):
                code_sections.append(self.generate_verification_directive(node))
            elif isinstance(node, StabilityDirective):
                code_sections.append(self.generate_stability_directive(node))
            elif isinstance(node, PhaseMonitor):
                code_sections.append(self.generate_phase_monitor(node))
            elif isinstance(node, AdaptiveTrigger):
                code_sections.append(self.generate_adaptive_trigger(node))
        
        # Generate helper functions
        code_sections.append(self.generate_helper_functions())
        
        # Generate main code
        if code_sections:
            code_sections.append("""
            # Main execution
            if __name__ == "__main__":
                # Initialize system state and phase state
                system_state = np.zeros(4)  # Example state
                phase_state = PsiPhaseState()
                
                # Run simulation loop
                print("Running stability-aware simulation...")
                for i in range(100):
                    # Update system state
                    old_state = system_state.copy()
                    system_state = update_system_state(system_state)
                    
                    # Check stability
                    check_triggers()
                    
                    # Print progress
                    if i % 10 == 0:
                        print(f"Step {i}: system_state = {system_state}")
                
                print("Simulation complete!")
            """)
        
        # Combine all code sections
        imports = self.generate_imports()
        
        return imports + "\n\n" + "\n\n".join(code_sections)


def test_stability_codegen():
    """Test the stability code generation."""
    print("Testing ELFIN stability code generation...")
    
    # Create some sample AST nodes
    lyap1 = LyapunovDeclaration(
        name="V_quad",
        lyap_type=LyapunovType.POLYNOMIAL,
        domain=["system1", "system2"],
        symbolic_form="x^T P x",
        parameters={"degree": 2},
        verification_hints={"method": "sos", "verbose": True}
    )
    
    verify1 = VerificationDirective(
        targets=["V_quad"],
        method=VerificationMethod.SOS,
        options={"tolerance": 0.001}
    )
    
    # Create code generator and generate code
    codegen = StabilityCodeGenerator()
    code = codegen.generate_code([lyap1, verify1])
    
    print("\nGenerated code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    print("Code generation test successful!")
    return True


if __name__ == "__main__":
    test_stability_codegen()
