"""
AST Node definitions for the ELFIN language.

This module defines the Abstract Syntax Tree (AST) node classes that
represent the structure of an ELFIN program after parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union
from typing import List as TypeList


class Node(ABC):
    """Base class for all AST nodes."""
    
    # Source location information
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __init__(self, line=None, column=None):
        self.line = line
        self.column = column
    
    def __repr__(self):
        attrs = [f"{k}={repr(v)}" for k, v in vars(self).items() 
                 if k not in ('line', 'column') and not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(attrs)})"


@dataclass
class Program(Node):
    """Root node of the AST, representing a complete ELFIN program."""
    sections: TypeList[Node] = field(default_factory=list)
    imports: TypeList['ImportStmt'] = field(default_factory=list)


@dataclass
class ImportStmt(Node):
    """Import statement: import SectionName from "path/to/file.elfin";"""
    section_name: str
    file_path: str


@dataclass
class HelpersSection(Node):
    """Helpers section containing utility functions."""
    name: Optional[str]
    functions: TypeList['HelperFunction'] = field(default_factory=list)


@dataclass
class HelperFunction(Node):
    """Helper function definition."""
    name: str
    body: 'Expression'
    parameters: TypeList[str] = field(default_factory=list)


@dataclass
class ParamDef(Node):
    """Parameter definition with optional unit."""
    name: str
    value: 'Expression'
    unit: Optional[str] = None


@dataclass
class SystemSection(Node):
    """System section defining dynamics."""
    name: str
    continuous_state: TypeList[str] = field(default_factory=list)
    inputs: TypeList[str] = field(default_factory=list)
    params: Dict[str, 'Expression'] = field(default_factory=dict)
    dynamics: Dict[str, 'Expression'] = field(default_factory=dict)


@dataclass
class LyapunovSection(Node):
    """Lyapunov function definition."""
    name: str
    system: str
    v_expression: 'Expression'
    params: Dict[str, 'Expression'] = field(default_factory=dict)


@dataclass
class BarrierSection(Node):
    """Barrier function definition."""
    name: str
    system: str
    b_expression: 'Expression'
    alpha_function: Optional['Expression'] = None
    params: Dict[str, 'Expression'] = field(default_factory=dict)


@dataclass
class ModeSection(Node):
    """Control mode definition."""
    name: str
    system: str
    lyapunov: Optional[str] = None
    barriers: TypeList[str] = field(default_factory=list)
    controller: Dict[str, 'Expression'] = field(default_factory=dict)
    params: Dict[str, 'Expression'] = field(default_factory=dict)


@dataclass
class PlannerSection(Node):
    """Planner section definition."""
    name: str
    system: str
    config: Dict[str, 'Expression'] = field(default_factory=dict)
    obstacles: TypeList['Expression'] = field(default_factory=list)
    params: Dict[str, 'Expression'] = field(default_factory=dict)


@dataclass
class IntegrationSection(Node):
    """Integration section definition."""
    name: str
    planner: str
    controller: str
    config: Dict[str, 'Expression'] = field(default_factory=dict)


# Expression nodes

class Expression(Node):
    """Base class for all expression nodes."""
    pass


@dataclass
class VarRef(Expression):
    """Variable reference."""
    name: str


@dataclass
class Number(Expression):
    """Numeric literal."""
    value: Union[int, float]


@dataclass
class String(Expression):
    """String literal."""
    value: str


@dataclass
class List(Expression):
    """List literal."""
    elements: TypeList[Expression] = field(default_factory=list)


@dataclass
class Object(Expression):
    """Object literal."""
    items: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class MemberAccess(Expression):
    """Member access: obj.member"""
    object: Expression
    member: str


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right"""
    left: Expression
    right: Expression
    op: str


@dataclass
class FunctionCall(Expression):
    """Function call: func(arg1, arg2, ...)"""
    function: str
    arguments: TypeList[Expression] = field(default_factory=list)


@dataclass
class IfExpression(Expression):
    """If expression: if condition then true_expr else false_expr"""
    condition: Expression
    true_expr: Expression
    false_expr: Expression
