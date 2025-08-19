"""
Megatron AST Converter for the ELFIN language.

This module implements the Megatron that converts Lark parse trees
into our own AST nodes, providing a powerful tree transformation capability.
"""

from typing import Dict, List, Any, Optional, Union, cast
from lark import Transformer as LarkTransformer, v_args
import os
from pathlib import Path
import importlib.util

# Import nodes using regular imports which preserves package structure
from alan_backend.elfin.compiler.ast.nodes import (
    Node, Program, ImportStmt, HelpersSection, HelperFunction,
    SystemSection, LyapunovSection, BarrierSection, ModeSection,
    PlannerSection, IntegrationSection, Expression, VarRef, 
    Number, String, List as ListExpr, Object, MemberAccess, 
    BinaryOp, FunctionCall, IfExpression
)


@v_args(inline=True)
class ELFINMegatron(LarkTransformer):
    """
    Transform a Lark parse tree into our own AST nodes.
    
    The Megatron is a powerful engine that converts the parse tree produced by Lark
    into our own AST nodes, which are more suitable for analysis and code generation.
    """
    
    def __init__(self, filename=None):
        super().__init__()
        self.filename = filename
        
    def start(self, *items):
        """Transform the root node of the parse tree."""
        sections = []
        imports = []
        
        for item in items:
            if isinstance(item, ImportStmt):
                imports.append(item)
            elif isinstance(item, Node):
                sections.append(item)
        
        return Program(sections=sections, imports=imports)
    
    def comment(self, text):
        """Transform a comment."""
        # Comments are ignored in the AST
        return None
    
    def import_stmt(self, section_name, file_path):
        """Transform an import statement."""
        # Remove quotes from the file path
        clean_path = file_path.value.strip('"')
        return ImportStmt(section_name=section_name.value, file_path=clean_path)
    
    # Helpers section
    def helpers_section(self, *items):
        """Transform a helpers section."""
        name = None
        functions = []
        
        # First item might be the section name
        if items and hasattr(items[0], 'value'):
            name = items[0].value
            items = items[1:]
        
        # Process helper functions
        for item in items:
            if isinstance(item, HelperFunction):
                functions.append(item)
        
        return HelpersSection(name=name, functions=functions)
    
    def helper_function(self, name, params=None, expr=None):
        """Transform a helper function."""
        parameters = []
        
        # Process parameters if present
        if params:
            parameters = [p.value for p in params]
        
        return HelperFunction(name=name.value, parameters=parameters, body=expr)
    
    def parameter_list(self, *params):
        """Transform a parameter list."""
        return list(params)
    
    # System section
    def system_section(self, name, *elements):
        """Transform a system section."""
        system = SystemSection(name=name.value)
        
        for element in elements:
            if hasattr(element, 'tag'):
                if element.tag == 'continuous_state':
                    system.continuous_state = element.value
                elif element.tag == 'input':
                    system.inputs = element.value
                elif element.tag == 'params':
                    system.params = element.value
                elif element.tag == 'dynamics':
                    system.dynamics = element.value
        
        return system
    
    def continuous_state(self, content):
        """Transform a continuous state definition."""
        if isinstance(content, list):
            return Node(tag='continuous_state', value=content)
        else:
            # For the curly brace syntax with individual variables
            names = [name.value for name in content if hasattr(name, 'value')]
            return Node(tag='continuous_state', value=names)
    
    def input_block(self, content):
        """Transform an input block."""
        if isinstance(content, list):
            return Node(tag='input', value=content)
        else:
            # For the curly brace syntax with individual variables
            names = [name.value for name in content if hasattr(name, 'value')]
            return Node(tag='input', value=names)
    
    def params_block(self, *items):
        """Transform a params block."""
        params = {}
        
        for item in items:
            if hasattr(item, 'name') and hasattr(item, 'value'):
                params[item.name] = item.value
        
        return Node(tag='params', value=params)
    
    def param_def(self, name, expr, unit=None):
        """Transform a parameter definition with optional unit."""
        # Extract unit value if present, stripping brackets
        unit_value = None
        if unit:
            unit_value = unit.value.strip('[]')
        
        # Create a ParamDef node instead of a generic Node
        return ParamDef(name=name.value, value=expr, unit=unit_value)
    
    def flow_dynamics(self, *items):
        """Transform a flow dynamics block."""
        dynamics = {}
        
        for item in items:
            if hasattr(item, 'name') and hasattr(item, 'value'):
                dynamics[item.name] = item.value
        
        return Node(tag='dynamics', value=dynamics)
    
    def equation(self, name, expr):
        """Transform an equation."""
        return Node(name=name.value, value=expr)
    
    # Lyapunov section
    def lyapunov_section(self, name, *elements):
        """Transform a Lyapunov section."""
        lyapunov = LyapunovSection(name=name.value, system="", v_expression=None)
        
        for element in elements:
            if hasattr(element, 'tag'):
                if element.tag == 'system':
                    lyapunov.system = element.value
                elif element.tag == 'v_expression':
                    lyapunov.v_expression = element.value
                elif element.tag == 'params':
                    lyapunov.params = element.value
        
        return lyapunov
    
    # Expression nodes
    def var_ref(self, name):
        """Transform a variable reference."""
        return VarRef(name=name.value)
    
    def number(self, value):
        """Transform a numeric literal."""
        try:
            # Try parsing as int first
            return Number(value=int(value))
        except ValueError:
            # If that fails, parse as float
            return Number(value=float(value))
    
    def string(self, value):
        """Transform a string literal."""
        # Remove quotes
        clean_value = value.value.strip('"')
        return String(value=clean_value)
    
    def list(self, *elements):
        """Transform a list literal."""
        return ListExpr(elements=list(elements))
    
    def object(self, *items):
        """Transform an object literal."""
        obj_items = {}
        
        for item in items:
            if hasattr(item, 'key') and hasattr(item, 'value'):
                obj_items[item.key] = item.value
        
        return Object(items=obj_items)
    
    def object_item(self, key, value):
        """Transform an object item."""
        return Node(key=key.value, value=value)
    
    def member_access(self, obj, member):
        """Transform a member access expression."""
        return MemberAccess(object=obj, member=member.value)
    
    # Binary operations
    def add(self, left, right):
        """Transform an addition operation."""
        return BinaryOp(left=left, right=right, op='+')
    
    def sub(self, left, right):
        """Transform a subtraction operation."""
        return BinaryOp(left=left, right=right, op='-')
    
    def mul(self, left, right):
        """Transform a multiplication operation."""
        return BinaryOp(left=left, right=right, op='*')
    
    def div(self, left, right):
        """Transform a division operation."""
        return BinaryOp(left=left, right=right, op='/')
    
    def power(self, left, right):
        """Transform a power operation."""
        return BinaryOp(left=left, right=right, op='**')
    
    def eq(self, left, right):
        """Transform an equality operation."""
        return BinaryOp(left=left, right=right, op='==')
    
    def neq(self, left, right):
        """Transform an inequality operation."""
        return BinaryOp(left=left, right=right, op='!=')
    
    def lt(self, left, right):
        """Transform a less-than operation."""
        return BinaryOp(left=left, right=right, op='<')
    
    def gt(self, left, right):
        """Transform a greater-than operation."""
        return BinaryOp(left=left, right=right, op='>')
    
    def le(self, left, right):
        """Transform a less-than-or-equal operation."""
        return BinaryOp(left=left, right=right, op='<=')
    
    def ge(self, left, right):
        """Transform a greater-than-or-equal operation."""
        return BinaryOp(left=left, right=right, op='>=')
    
    def if_expr(self, condition, true_expr, false_expr):
        """Transform an if expression."""
        return IfExpression(condition=condition, true_expr=true_expr, false_expr=false_expr)
    
    def function_call(self, function, *arguments):
        """Transform a function call."""
        return FunctionCall(function=function.value, arguments=list(arguments))
    
    def parenthesis(self, expr):
        """Transform a parenthesized expression."""
        # Parentheses don't have their own node in the AST
        return expr
