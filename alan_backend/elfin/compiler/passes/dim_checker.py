"""
Dimensional consistency checker for ELFIN.

This module provides a compiler pass that checks for dimensional consistency
in ELFIN expressions and emits warnings for inconsistencies.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import time
from pathlib import Path

from alan_backend.elfin.compiler.ast.nodes import (
    Node, Program, Expression, BinaryOp, VarRef, Number, FunctionCall,
    ParamDef, SystemSection
)
from alan_backend.elfin.compiler.symbol import Symbol, SymbolTable
from alan_backend.elfin.units.unit_expr import (
    UnitExpr, BaseUnit, MulUnit, DivUnit, PowUnit, parse_unit_expr
)
from alan_backend.elfin.utils.file_hash import FileHashCache


class Diagnostic:
    """
    Represents a diagnostic message from the compiler.
    
    Diagnostics include warnings and errors with source location information.
    """
    
    # Severity levels
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    
    # Source types
    DIM_CHECKER = "dim"
    CONST_FOLDER = "const_fold"
    
    def __init__(
        self,
        message: str,
        severity: str = WARNING,
        code: str = None,
        line: int = None,
        column: int = None,
        node: Node = None,
        source: str = DIM_CHECKER
    ):
        """
        Initialize a diagnostic.
        
        Args:
            message: Diagnostic message
            severity: Severity level (INFO, WARNING, ERROR)
            code: Diagnostic code (e.g., "DIM_MISMATCH")
            line: Line number in source file
            column: Column number in source file
            node: AST node that triggered the diagnostic
        """
        self.message = message
        self.severity = severity
        self.code = code
        self.source = source
        
        # Set location information
        if node is not None:
            self.line = node.line
            self.column = node.column
        else:
            self.line = line
            self.column = column
    
    def to_dict(self) -> Dict[str, any]:
        """Convert diagnostic to a dictionary for JSON serialization."""
        result = {
            "severity": self.severity,
            "message": self.message
        }
        
        if self.code:
            result["code"] = self.code
        
        if self.line is not None and self.column is not None:
            result["range"] = {
                "start": {"line": self.line, "column": self.column},
                "end": {"line": self.line, "column": self.column + 1}  # Approximate
            }
        
        return result
    
    def __str__(self) -> str:
        """String representation of the diagnostic."""
        location = f"{self.line}:{self.column}" if self.line is not None else ""
        code_str = f"[{self.code}] " if self.code else ""
        return f"{location} {self.severity.upper()}: {code_str}{self.message}"


class DimChecker:
    """
    Dimensional consistency checker for ELFIN.
    
    This class checks for dimensional consistency in ELFIN expressions
    and emits warnings for inconsistencies.
    """
    
    def __init__(self):
        """Initialize the dimension checker."""
        self.symbol_table = SymbolTable()
        self.diagnostics: List[Diagnostic] = []
        self.logger = logging.getLogger(__name__)
        self.hash_cache = FileHashCache()
        
    def run(self, files: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, List[Diagnostic]]:
        """
        Run the dimension checker on multiple files with caching.
        
        This method uses a file hash cache to skip files that haven't changed
        since they were last checked, making repeated runs much faster.
        
        Args:
            files: Path or list of paths to ELFIN files to check
            
        Returns:
            Dictionary mapping file paths to lists of diagnostics
        """
        from alan_backend.elfin.standalone_parser import parse_file
        
        # Convert single file to list
        if isinstance(files, (str, Path)):
            files = [files]
        
        # Convert to Path objects
        file_paths = [Path(f) for f in files]
        
        # Track results
        all_diagnostics: Dict[str, List[Diagnostic]] = {}
        files_checked = 0
        files_skipped = 0
        start_time = time.time()
        
        # Check each file
        for file_path in file_paths:
            # Skip non-existent files
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
                
            # Skip non-ELFIN files
            if file_path.suffix.lower() != '.elfin':
                self.logger.debug(f"Skipping non-ELFIN file: {file_path}")
                continue
            
            # Check if the file has changed since last check
            file_path_str = str(file_path)
            if self.hash_cache.is_unchanged(file_path):
                # File hasn't changed, use empty diagnostics list
                all_diagnostics[file_path_str] = []
                files_skipped += 1
                self.logger.debug(f"Skipping unchanged file: {file_path}")
                continue
            
            try:
                # Parse the file
                ast = parse_file(file_path)
                
                # Run the dimension checker
                diagnostics = self.check_program(ast)
                
                # Store diagnostics
                all_diagnostics[file_path_str] = diagnostics
                
                # Update the hash cache
                self.hash_cache.update_hash(file_path)
                
                files_checked += 1
            except Exception as e:
                self.logger.error(f"Error checking file {file_path}: {e}")
                # Store error as a diagnostic
                all_diagnostics[file_path_str] = [
                    Diagnostic(
                        message=f"Error checking file: {e}",
                        severity=Diagnostic.ERROR,
                        code="CHECK_ERROR",
                        line=1, column=1
                    )
                ]
        
        # Log performance
        elapsed = time.time() - start_time
        self.logger.info(f"Checked {files_checked} files, skipped {files_skipped} files in {elapsed:.3f} seconds")
        
        return all_diagnostics
    
    def check_program(self, program: Program) -> List[Diagnostic]:
        """
        Check a program for dimensional consistency.
        
        Args:
            program: ELFIN program to check
            
        Returns:
            List of diagnostics generated during the check
        """
        self.diagnostics = []
        
        # First pass: Collect all symbols and their dimensions
        self._collect_symbols(program)
        
        # Second pass: Check all expressions for dimensional consistency
        for section in program.sections:
            self._check_section(section)
        
        return self.diagnostics
    
    def _collect_symbols(self, program: Program) -> None:
        """
        Collect symbols and their dimensions from a program.
        
        Args:
            program: ELFIN program to collect symbols from
        """
        # Collect symbols from the program
        for section in program.sections:
            if isinstance(section, SystemSection):
                # Collect state variables
                for var_name in section.continuous_state:
                    symbol = Symbol(name=var_name)
                    self.symbol_table.add_symbol(symbol)
                
                # Collect input variables
                for var_name in section.inputs:
                    symbol = Symbol(name=var_name)
                    self.symbol_table.add_symbol(symbol)
                
                # Collect parameters with units
                for param_name, param_value in section.params.items():
                    if isinstance(param_value, ParamDef) and param_value.unit:
                        try:
                            dim = parse_unit_expr(param_value.unit)
                            symbol = Symbol(name=param_name, dim=dim)
                        except ValueError:
                            symbol = Symbol(name=param_name)
                            self.diagnostics.append(
                                Diagnostic(
                                    message=f"Invalid unit format: {param_value.unit}",
                                    code="INVALID_UNIT",
                                    node=param_value
                                )
                            )
                    else:
                        symbol = Symbol(name=param_name)
                    
                    self.symbol_table.add_symbol(symbol)
    
    def _check_section(self, section: Node) -> None:
        """
        Check a section for dimensional consistency.
        
        Args:
            section: ELFIN section to check
        """
        if isinstance(section, SystemSection):
            # Check dynamics expressions
            for var_name, expr in section.dynamics.items():
                try:
                    dim = self._check_expression(expr)
                    # Store the inferred dimension in the symbol table
                    var_symbol = self.symbol_table.get_symbol(var_name)
                    if var_symbol:
                        self.symbol_table.set_dimension(var_name, dim)
                except Exception as e:
                    self.logger.error(f"Error checking expression for {var_name}: {e}")
    
    def _check_expression(self, expr: Expression) -> Optional[UnitExpr]:
        """
        Check an expression for dimensional consistency.
        
        Args:
            expr: ELFIN expression to check
            
        Returns:
            The dimension of the expression, or None if it cannot be determined
        """
        if isinstance(expr, Number):
            # Numeric literal has no dimension
            return UnitExpr()
        
        elif isinstance(expr, VarRef):
            # Variable reference has the dimension of the variable
            symbol = self.symbol_table.get_symbol(expr.name)
            if symbol and symbol.dim:
                return symbol.dim
            return None
        
        elif isinstance(expr, BinaryOp):
            # Binary operation may combine dimensions
            left_dim = self._check_expression(expr.left)
            right_dim = self._check_expression(expr.right)
            
            # If either dimension is unknown, we can't check
            if left_dim is None or right_dim is None:
                return None
            
            # Check dimensional consistency based on operation
            if expr.op in ["+", "-"]:
                # Addition and subtraction require the same dimension
                if not self._dimensions_compatible(left_dim, right_dim):
                    self.diagnostics.append(
                        Diagnostic(
                            message=f"Cannot add or subtract values with different dimensions: "
                                   f"[{left_dim}] and [{right_dim}]",
                            code="DIM_MISMATCH",
                            node=expr
                        )
                    )
                return left_dim
            
            elif expr.op == "*":
                # Multiplication combines dimensions
                return MulUnit(left_dim, right_dim)
            
            elif expr.op == "/":
                # Division subtracts dimensions
                return DivUnit(left_dim, right_dim)
            
            elif expr.op == "**" or expr.op == "^":
                # Power operation
                if isinstance(expr.right, Number):
                    exponent = expr.right.value
                    if isinstance(exponent, int):
                        return PowUnit(left_dim, exponent)
                    else:
                        self.diagnostics.append(
                            Diagnostic(
                                message=f"Cannot raise dimensional value to non-integer power: "
                                       f"[{left_dim}] ^ {exponent}",
                                code="INVALID_EXPONENT",
                                node=expr
                            )
                        )
                else:
                    self.diagnostics.append(
                        Diagnostic(
                            message="Exponent must be a numeric literal",
                            code="INVALID_EXPONENT",
                            node=expr
                        )
                    )
                return None
            
            # Other operations
            return None
        
        elif isinstance(expr, FunctionCall):
            # Function call may have special dimensional rules
            arg_dims = [self._check_expression(arg) for arg in expr.arguments]
            
            # List of common helper functions
            helper_functions = ["hAbs", "hMin", "hMax", "wrapAngle", "clamp", "lerp"]
            
            # Handle transcendental functions
            if expr.function in ["sin", "cos", "tan", "exp", "log"]:
                # These functions require dimensionless arguments
                if len(arg_dims) > 0 and arg_dims[0] is not None:
                    if not isinstance(arg_dims[0], UnitExpr) or arg_dims[0].__str__() != "dimensionless":
                        self.diagnostics.append(
                            Diagnostic(
                                message=f"Function '{expr.function}' requires dimensionless argument, "
                                       f"got [{arg_dims[0]}]",
                                code="DIM_MISMATCH",
                                node=expr
                            )
                        )
                return UnitExpr()  # Result is dimensionless
            
            elif expr.function == "sqrt":
                # Square root returns half the dimension
                if len(arg_dims) > 0 and arg_dims[0] is not None:
                    return PowUnit(arg_dims[0], 0.5)
                return None
            
            # Check for helper functions that should be imported
            elif expr.function in helper_functions:
                # Check if the function is defined in the current scope
                if not self.symbol_table.get_symbol(expr.function):
                    self.diagnostics.append(
                        Diagnostic(
                            message=f"Function '{expr.function}' is not defined. Consider importing helpers.",
                            code="MISSING_HELPER",
                            severity=Diagnostic.WARNING,
                            node=expr
                        )
                    )
                # Preserve dimensions of arguments (most helper functions do)
                if len(arg_dims) > 0:
                    return arg_dims[0]
                return None
            
            # Other functions would need case-by-case handling
            return None
        
        # Other expression types
        return None
    
    def _dimensions_compatible(self, dim1: UnitExpr, dim2: UnitExpr) -> bool:
        """
        Check if two dimensions are compatible for addition/subtraction.
        
        Args:
            dim1: First dimension
            dim2: Second dimension
            
        Returns:
            True if the dimensions are compatible, False otherwise
        """
        if dim1 is None or dim2 is None:
            return True  # Can't check, assume compatible
        
        # Use the same() method for compatibility check
        return dim1.same(dim2)
