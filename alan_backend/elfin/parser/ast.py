"""
ELFIN AST (Abstract Syntax Tree) Module.

This module defines the abstract syntax tree classes used to represent
parsed ELFIN DSL code before compilation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Union, Set, Tuple


class Node:
    """Base class for all AST nodes."""
    
    def __init__(self):
        """Initialize a node."""
        self.children = []
        
    def add_child(self, child):
        """
        Add a child node.
        
        Args:
            child: Child node to add
        """
        self.children.append(child)


class Expression(Node):
    """Base class for expressions."""
    
    def __init__(self):
        """Initialize an expression."""
        super().__init__()


class ConceptDecl(Node):
    """Concept declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a concept declaration.
        
        Args:
            id: Concept ID
            name: Concept name
            description: Optional concept description
            properties: Optional concept properties
        """
        super().__init__()
        self.id = id
        self.name = name
        self.description = description
        self.properties = properties or {}


class RelationDecl(Node):
    """Relation declaration."""
    
    def __init__(
        self,
        id: str,
        source_id: str,
        target_id: str,
        type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a relation declaration.
        
        Args:
            id: Relation ID
            source_id: Source concept ID
            target_id: Target concept ID
            type: Relation type
            weight: Relation weight
            properties: Optional relation properties
        """
        super().__init__()
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.weight = weight
        self.properties = properties or {}


class FunctionDecl(Node):
    """Function declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        parameters: Optional[List[str]] = None,
        body: Optional[Node] = None
    ):
        """
        Initialize a function declaration.
        
        Args:
            id: Function ID
            name: Function name
            parameters: Optional function parameters
            body: Optional function body
        """
        super().__init__()
        self.id = id
        self.name = name
        self.parameters = parameters or []
        self.body = body


class AgentDirective(Node):
    """Agent directive."""
    
    def __init__(
        self,
        agent_type: str,
        directive: str,
        target_concept_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an agent directive.
        
        Args:
            agent_type: Agent type
            directive: Directive text
            target_concept_ids: Optional target concept IDs
            parameters: Optional directive parameters
        """
        super().__init__()
        self.agent_type = agent_type
        self.directive = directive
        self.target_concept_ids = target_concept_ids or []
        self.parameters = parameters or {}


class GoalDecl(Node):
    """Goal declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        expression: Expression,
        description: Optional[str] = None,
        priority: float = 1.0
    ):
        """
        Initialize a goal declaration.
        
        Args:
            id: Goal ID
            name: Goal name
            expression: Goal expression
            description: Optional goal description
            priority: Goal priority
        """
        super().__init__()
        self.id = id
        self.name = name
        self.expression = expression
        self.description = description
        self.priority = priority


class AssumptionDecl(Node):
    """Assumption declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        expression: Expression,
        confidence: float = 1.0,
        validated: bool = False
    ):
        """
        Initialize an assumption declaration.
        
        Args:
            id: Assumption ID
            name: Assumption name
            expression: Assumption expression
            confidence: Assumption confidence
            validated: Whether the assumption is validated
        """
        super().__init__()
        self.id = id
        self.name = name
        self.expression = expression
        self.confidence = confidence
        self.validated = validated


class StabilityConstraint(Node):
    """Stability constraint."""
    
    def __init__(
        self,
        id: str,
        expression: Expression,
        constraint_type: str = "LYAPUNOV",
        threshold: float = 0.0
    ):
        """
        Initialize a stability constraint.
        
        Args:
            id: Constraint ID
            expression: Constraint expression
            constraint_type: Constraint type
            threshold: Constraint threshold
        """
        super().__init__()
        self.id = id
        self.expression = expression
        self.constraint_type = constraint_type
        self.threshold = threshold


class PsiModeDecl(Node):
    """ψ-mode declaration."""
    
    def __init__(
        self,
        mode_index: int,
        amplitude: float = 1.0,
        phase: float = 0.0,
        is_primary: bool = False
    ):
        """
        Initialize a ψ-mode declaration.
        
        Args:
            mode_index: Mode index
            amplitude: Mode amplitude
            phase: Mode phase
            is_primary: Whether this is a primary mode
        """
        super().__init__()
        self.mode_index = mode_index
        self.amplitude = amplitude
        self.phase = phase
        self.is_primary = is_primary


class LyapunovDecl(Node):
    """Lyapunov function declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        function_type: str,
        expression: Optional[Expression] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Lyapunov function declaration.
        
        Args:
            id: Function ID
            name: Function name
            function_type: Function type (POLYNOMIAL, NEURAL, COMPOSITE)
            expression: Optional function expression
            params: Optional function parameters
        """
        super().__init__()
        self.id = id
        self.name = name
        self.function_type = function_type
        self.expression = expression
        self.params = params or {}


# New Stability Extension AST Nodes

class EnhancedLyapunovDecl(Node):
    """Enhanced Lyapunov function declaration with more details."""
    
    def __init__(
        self,
        id: str,
        name: str,
        lyapunov_type: str,
        type_params: Dict[str, Any],
        domain_concepts: Optional[List[str]] = None,
        symbolic_form: Optional[str] = None,
        learn_directive: Optional['LearnDirective'] = None,
        verification_hint: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an enhanced Lyapunov function declaration.
        
        Args:
            id: Function ID
            name: Function name
            lyapunov_type: Type of Lyapunov function (polynomial, neural, clf, composite)
            type_params: Type-specific parameters (degree, layers, etc.)
            domain_concepts: Concepts in the function's domain
            symbolic_form: Symbolic form of the function
            learn_directive: Optional learning directive
            verification_hint: Verification method and options
        """
        super().__init__()
        self.id = id
        self.name = name
        self.lyapunov_type = lyapunov_type
        self.type_params = type_params
        self.domain_concepts = domain_concepts or []
        self.symbolic_form = symbolic_form
        self.learn_directive = learn_directive
        self.verification_hint = verification_hint or {}


class LearnDirective(Node):
    """Directive to learn a Lyapunov function from data or dynamics."""
    
    def __init__(
        self,
        source_type: str,
        source_name: str,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a learn directive.
        
        Args:
            source_type: Type of source ("data" or "dynamics")
            source_name: Name of data source or dynamics function
            options: Optional learning options
        """
        super().__init__()
        self.source_type = source_type
        self.source_name = source_name
        self.options = options or {}


class VerificationDirective(Node):
    """Directive to verify stability properties."""
    
    def __init__(
        self,
        targets: List[str],
        method: str,
        properties: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a verification directive.
        
        Args:
            targets: Lyapunov functions or concepts to verify
            method: Verification method (sos, milp, smt)
            properties: Stability properties to verify
            options: Optional verification options
        """
        super().__init__()
        self.targets = targets
        self.method = method
        self.properties = properties or ["positive_definite", "decreasing"]
        self.options = options or {}


class KoopmanDecl(Node):
    """Koopman operator declaration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        eigenfunctions: List[str],
        mode_mappings: Optional[Dict[str, float]] = None,
        phase_bindings: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize a Koopman operator declaration.
        
        Args:
            id: Operator ID
            name: Operator name
            eigenfunctions: List of eigenfunction names
            mode_mappings: Mapping from modes to eigenvalues
            phase_bindings: Bindings from concepts to phase indices
        """
        super().__init__()
        self.id = id
        self.name = name
        self.eigenfunctions = eigenfunctions
        self.mode_mappings = mode_mappings or {}
        self.phase_bindings = phase_bindings or []


class VerificationStatus(Enum):
    """Status of a verification result."""
    PROVEN = auto()
    DISPROVEN = auto()
    UNKNOWN = auto()
    IN_PROGRESS = auto()
    ERROR = auto()


class ConstraintIR(Node):
    """
    Constraint Intermediate Representation for verification.
    
    This represents a solver-agnostic constraint that can be passed to
    various verification backends (SOS, SMT, MILP, etc.)
    """
    
    def __init__(
        self,
        id: str,
        variables: List[str],
        expression: str,
        constraint_type: str,
        context: Optional[Dict[str, Any]] = None,
        solver_hint: Optional[str] = None,
        proof_needed: bool = True,
        dependencies: Optional[List[str]] = None
    ):
        """
        Initialize a constraint IR.
        
        Args:
            id: Constraint ID
            variables: List of variables in the constraint
            expression: Constraint expression
            constraint_type: Type of constraint (equality, inequality, etc.)
            context: Additional context information
            solver_hint: Hint for the solver
            proof_needed: Whether a formal proof is needed
            dependencies: List of dependencies
        """
        super().__init__()
        self.id = id
        self.variables = variables
        self.expression = expression
        self.constraint_type = constraint_type
        self.context = context or {}
        self.solver_hint = solver_hint
        self.proof_needed = proof_needed
        self.dependencies = dependencies or []
