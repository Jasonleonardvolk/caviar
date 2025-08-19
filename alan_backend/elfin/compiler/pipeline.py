"""
Compiler Pipeline for ELFIN.

This module defines the compilation pipeline that applies various passes
to the AST in the correct order.
"""

from typing import List, Dict, Any, Optional, Union
from alan_backend.elfin.compiler.ast.nodes import Node
from alan_backend.elfin.compiler.passes.constant_folder import ConstantFolder
from alan_backend.elfin.compiler.passes.dim_checker import DimChecker


class CompilerPipeline:
    """
    Compiler pipeline that applies passes to the AST.
    
    This class manages the sequence of compiler passes that transform and
    analyze the AST during compilation.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the compiler pipeline.
        
        Args:
            options: Compilation options
        """
        self.options = options or {}
        
        # Initialize passes
        self.constant_folder = ConstantFolder()
        self.dim_checker = DimChecker()
        
        # Track diagnostics from passes
        self.diagnostics = []
    
    def process(self, ast: Node) -> Node:
        """
        Process an AST through the pipeline.
        
        Args:
            ast: The root node of the AST
            
        Returns:
            The processed AST
        """
        # Clear previous diagnostics
        self.diagnostics = []
        
        # Apply constant folding if enabled
        # Default to enabled unless explicitly disabled
        if self.options.get('fold', True):
            ast = self.constant_folder.process(ast)
            # Add any diagnostics from constant folding
            self.diagnostics.extend(self.constant_folder.diagnostics)
        
        # Apply dimensional checking
        # The constant folder should run before dimensional checking so that
        # the checker can see the collapsed dimensions
        dim_diagnostics = self.dim_checker.check_program(ast)
        self.diagnostics.extend(dim_diagnostics)
        
        return ast
    
    def get_diagnostics(self) -> List[Any]:
        """
        Get diagnostics collected during processing.
        
        Returns:
            List of diagnostic objects
        """
        return self.diagnostics
