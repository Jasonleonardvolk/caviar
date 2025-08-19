from dataclasses import dataclass
from typing import List, Optional, Any

from alan_backend.elfin.compiler.ast.nodes import AssignStmt

@dataclass
class Hint:
    """
    A hint to show at a position in the source code.
    
    Attributes:
        line: The line number (0-based)
        col: The column number (0-based)
        label: The hint label to display
    """
    line: int
    col: int
    label: str  # "[m]" etc.

class InlayHintCollector:
    """
    Collect inlay hints for the source code.
    
    This pass collects inlay hints that show the units of expressions
    at the end of assignments.
    """
    
    def run(self, ast: Any, dim_resolver: Any) -> List[Hint]:
        """
        Run the inlay hint collector on the AST.
        
        Args:
            ast: The AST to analyze
            dim_resolver: A dimension resolver for expressions
            
        Returns:
            A list of inlay hints
        """
        hints = []
        
        def _visit(n: Any) -> None:
            """
            Visit a node and collect hints.
            
            Args:
                n: The node to visit
            """
            if isinstance(n, AssignStmt):
                # Try to resolve the dimension of the expression
                unit = dim_resolver.resolve(n.expr) if hasattr(dim_resolver, 'resolve') else None
                
                if unit:
                    # Get the end position of the assignment
                    span = n.span.end if hasattr(n.span, 'end') else None
                    
                    if span:
                        # Create a hint at the end of the line
                        hints.append(Hint(
                            line=span.line,
                            col=span.col,
                            label=f"▸ [{unit}]"
                        ))
            
            # Visit children
            for child in getattr(n, "children", []):
                _visit(child)
        
        # Start the traversal
        _visit(ast)
        
        return hints
