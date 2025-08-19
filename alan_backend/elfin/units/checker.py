"""
Dimensional analysis and checking for ELFIN expressions.

This module provides a dimensional checker that can analyze expressions in ELFIN
and verify that they are dimensionally consistent. It catches common errors such as
adding quantities with different dimensions or using transcendental functions with
non-dimensionless arguments.
"""

from typing import Dict, Any, Optional, Set, List, Tuple, Union
import ast
import re

from .units import Unit, UnitDimension, UnitTable


class DimensionError(Exception):
    """Error raised when a dimensional inconsistency is detected."""
    pass


class DimensionChecker:
    """
    Checker for dimensional consistency in ELFIN expressions.
    
    This class provides methods for checking that expressions in ELFIN are
    dimensionally consistent. It tracks the dimensions of variables and expressions
    and verifies that operations are valid with respect to dimensions.
    """
    
    def __init__(self, unit_table: Optional[UnitTable] = None):
        """
        Initialize the dimension checker.
        
        Args:
            unit_table: Table of units to use for lookups (optional)
        """
        self.unit_table = unit_table or UnitTable()
        self.symbol_dimensions: Dict[str, UnitDimension] = {}
    
    def register_symbol(self, name: str, unit: Union[str, Unit, UnitDimension]) -> None:
        """
        Register a symbol with its associated unit or dimension.
        
        Args:
            name: Name of the symbol
            unit: Unit or dimension of the symbol
        """
        if isinstance(unit, str):
            # Parse unit string
            try:
                unit_obj = Unit.parse(unit)
                self.symbol_dimensions[name] = unit_obj.dimension
            except ValueError:
                raise ValueError(f"Unknown unit: {unit}")
        elif isinstance(unit, Unit):
            self.symbol_dimensions[name] = unit.dimension
        elif isinstance(unit, UnitDimension):
            self.symbol_dimensions[name] = unit
        else:
            raise ValueError(f"Invalid unit type: {type(unit)}")
    
    def get_symbol_dimension(self, name: str) -> Optional[UnitDimension]:
        """
        Get the dimension of a registered symbol.
        
        Args:
            name: Name of the symbol
            
        Returns:
            Dimension of the symbol, or None if not registered
        """
        return self.symbol_dimensions.get(name)
    
    def check_expression(self, expr: str) -> UnitDimension:
        """
        Check the dimensional consistency of an expression.
        
        This method parses the expression and computes its dimension, raising
        a DimensionError if the expression is dimensionally inconsistent.
        
        Args:
            expr: ELFIN expression to check
            
        Returns:
            Dimension of the expression
            
        Raises:
            DimensionError: If the expression is dimensionally inconsistent
        """
        try:
            # Parse the expression
            expr_ast = ast.parse(expr, mode='eval').body
            
            # Check the dimension of the expression
            return self._check_node(expr_ast)
        except SyntaxError:
            # For ELFIN-specific syntax that doesn't parse as Python
            # We'd need a more sophisticated parser for the full ELFIN language
            raise ValueError(f"Failed to parse expression: {expr}")
    
    def _check_node(self, node: ast.AST) -> UnitDimension:
        """
        Check the dimension of an AST node.
        
        This method recursively checks the dimension of an AST node, raising
        a DimensionError if the node or any of its children are dimensionally
        inconsistent.
        
        Args:
            node: AST node to check
            
        Returns:
            Dimension of the node
            
        Raises:
            DimensionError: If the node is dimensionally inconsistent
        """
        # Handle different node types
        if isinstance(node, ast.Num):
            # Numeric literal (dimensionless)
            return UnitDimension()
        
        elif isinstance(node, ast.Name):
            # Variable reference
            if node.id in self.symbol_dimensions:
                return self.symbol_dimensions[node.id]
            else:
                raise DimensionError(f"Unknown symbol: {node.id}")
        
        elif isinstance(node, ast.BinOp):
            # Binary operation
            left_dim = self._check_node(node.left)
            right_dim = self._check_node(node.right)
            
            if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
                # Addition and subtraction require same dimensions
                if left_dim != right_dim:
                    left_str = str(left_dim)
                    right_str = str(right_dim)
                    raise DimensionError(
                        f"Cannot add or subtract values with different dimensions: "
                        f"{left_str} and {right_str}"
                    )
                return left_dim
            
            elif isinstance(node.op, ast.Mult):
                # Multiplication combines dimensions
                return left_dim * right_dim
            
            elif isinstance(node.op, ast.Div):
                # Division subtracts dimensions
                return left_dim / right_dim
            
            elif isinstance(node.op, ast.Pow):
                # Power
                if not isinstance(node.right, ast.Num):
                    raise DimensionError(
                        "Exponent must be a numeric literal"
                    )
                
                # Check for dimensionless exponentiation
                if not left_dim.is_dimensionless() and node.right.n != int(node.right.n):
                    raise DimensionError(
                        f"Cannot raise dimensional value to non-integer power: "
                        f"{str(left_dim)} ^ {node.right.n}"
                    )
                
                return left_dim ** int(node.right.n)
            
            else:
                # Other binary operations
                raise DimensionError(
                    f"Unsupported binary operation: {type(node.op).__name__}"
                )
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operation (e.g., -x)
            return self._check_node(node.operand)
        
        elif isinstance(node, ast.Call):
            # Function call
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            args_dims = [self._check_node(arg) for arg in node.args]
            
            # Handle transcendental functions
            if func_name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt'):
                # Check that arguments are dimensionless for transcendental functions
                if not args_dims[0].is_dimensionless() and func_name in ('sin', 'cos', 'tan', 'exp', 'log'):
                    raise DimensionError(
                        f"Transcendental function '{func_name}' requires dimensionless argument, "
                        f"got {str(args_dims[0])}"
                    )
                
                # Special case for angle arguments
                if func_name in ('sin', 'cos', 'tan') and args_dims[0].angle == 1 and args_dims[0].is_pure_angle():
                    # Allow angles for trigonometric functions
                    pass
                elif func_name == 'sqrt':
                    # sqrt returns square root of dimension
                    return args_dims[0] ** 0.5
                else:
                    # Other transcendental functions return dimensionless values
                    return UnitDimension()
                
                return UnitDimension()
            
            # Other functions would need case-by-case handling
            raise DimensionError(f"Unknown function: {func_name}")
        
        else:
            # Unsupported node type
            raise DimensionError(f"Unsupported expression type: {type(node).__name__}")


def check_elfin_file(file_path: str) -> List[Tuple[str, DimensionError]]:
    """
    Check dimensional consistency of an ELFIN file.
    
    This function parses an ELFIN file and checks that all expressions are
    dimensionally consistent.
    
    Args:
        file_path: Path to the ELFIN file
        
    Returns:
        List of (expression, error) pairs for dimensionally inconsistent expressions
    """
    # This is a placeholder for a full ELFIN parser
    # In reality, we would use the ELFIN parser to extract symbols and expressions
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract variable declarations with units
    # This is a very basic regex-based approach; a real parser would be more robust
    var_pattern = r'(\w+)\s*:\s*(\w+)\s*\[([^\]]+)\]\s*(?:=\s*([^;]+))?;'
    variables = {}
    for match in re.finditer(var_pattern, content):
        var_name, var_type, unit_str, default_value = match.groups()
        variables[var_name] = (var_type, unit_str, default_value)
    
    # Extract expressions
    # Again, this is a basic approach; a real parser would be more robust
    expr_pattern = r'(\w+)(?:_dot)?\s*=\s*([^;]+);'
    expressions = {}
    for match in re.finditer(expr_pattern, content):
        var_name, expr = match.groups()
        expressions[var_name] = expr
    
    # Create dimension checker
    checker = DimensionChecker()
    
    # Register symbols with their units
    for var_name, (var_type, unit_str, _) in variables.items():
        try:
            checker.register_symbol(var_name, unit_str)
        except ValueError as e:
            print(f"Warning: {e}")
    
    # Check expressions
    errors = []
    for var_name, expr in expressions.items():
        try:
            checker.check_expression(expr)
        except DimensionError as e:
            errors.append((expr, e))
    
    return errors
