"""
Adapter for working with the ELFIN AST in the language server.

This module provides functions for mapping between AST nodes and document positions,
which is necessary for features like hover and definition.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re

from alan_backend.elfin.standalone_parser import Node, Program


class SourceRange:
    """
    Represents a range in a source file.
    
    This is different from the LSP Range, as it uses 0-based line and column indices.
    """
    
    def __init__(
        self, 
        start_line: int, 
        start_column: int, 
        end_line: Optional[int] = None, 
        end_column: Optional[int] = None
    ):
        """
        Initialize a source range.
        
        Args:
            start_line: Starting line (0-based)
            start_column: Starting column (0-based)
            end_line: Ending line (0-based), defaults to start_line
            end_column: Ending column (0-based), defaults to start_column
        """
        self.start_line = start_line
        self.start_column = start_column
        self.end_line = end_line if end_line is not None else start_line
        self.end_column = end_column if end_column is not None else start_column
    
    def contains_position(self, line: int, column: int) -> bool:
        """
        Check if this range contains the given position.
        
        Args:
            line: Line number (0-based)
            column: Column number (0-based)
            
        Returns:
            True if the position is within this range, False otherwise
        """
        # If the line is before the start line or after the end line, it's outside
        if line < self.start_line or line > self.end_line:
            return False
        
        # If we're on the start line, check the column
        if line == self.start_line and column < self.start_column:
            return False
        
        # If we're on the end line, check the column
        if line == self.end_line and column > self.end_column:
            return False
        
        return True


def calculate_node_range(node: Node, text: str) -> SourceRange:
    """
    Calculate the source range for a node.
    
    This is an approximation based on the node's attributes and the source text.
    
    Args:
        node: AST node
        text: Source text
        
    Returns:
        Source range for the node
    """
    # If the node has line and column information, use that
    if hasattr(node, "line") and hasattr(node, "column"):
        line = getattr(node, "line")
        column = getattr(node, "column")
        
        # Convert from 1-based to 0-based indices
        if line > 0:
            line -= 1
        if column > 0:
            column -= 1
        
        # Find the node's name or text representation
        node_text = ""
        if hasattr(node, "name"):
            node_text = getattr(node, "name")
        elif hasattr(node, "value"):
            node_text = str(getattr(node, "value"))
        
        # If we have node text, try to calculate the end column
        end_column = column + len(node_text) if node_text else column + 1
        
        return SourceRange(line, column, line, end_column)
    
    # If the node has a span attribute, use that
    elif hasattr(node, "span"):
        span = getattr(node, "span")
        if hasattr(span, "start") and hasattr(span, "end"):
            start = getattr(span, "start")
            end = getattr(span, "end")
            
            if hasattr(start, "line") and hasattr(start, "column") and \
               hasattr(end, "line") and hasattr(end, "column"):
                
                start_line = getattr(start, "line")
                start_column = getattr(start, "column")
                end_line = getattr(end, "line")
                end_column = getattr(end, "column")
                
                # Convert from 1-based to 0-based indices
                if start_line > 0:
                    start_line -= 1
                if start_column > 0:
                    start_column -= 1
                if end_line > 0:
                    end_line -= 1
                if end_column > 0:
                    end_column -= 1
                
                return SourceRange(start_line, start_column, end_line, end_column)
    
    # If we couldn't determine the range, return a default range
    return SourceRange(0, 0, 0, 0)


def find_node_at_position(ast: Node, line: int, column: int, text: str) -> Optional[Node]:
    """
    Find the AST node at the given position.
    
    This function traverses the AST and returns the innermost node that contains
    the given position.
    
    Args:
        ast: AST to search
        line: Line number (0-based)
        column: Column number (0-based)
        text: Source text
        
    Returns:
        The node at the given position, or None if not found
    """
    result = None
    
    # Recursively search the AST
    def search(node: Node) -> None:
        nonlocal result
        
        # Calculate the range for this node
        node_range = calculate_node_range(node, text)
        
        # Check if this node contains the position
        if node_range.contains_position(line, column):
            # Update the result if we haven't found a node yet,
            # or if this node is more specific (smaller range) than the current result
            if result is None or \
               (node_range.end_line - node_range.start_line) < \
               (calculate_node_range(result, text).end_line - calculate_node_range(result, text).start_line):
                result = node
        
        # Recursively search child nodes
        for child_name in dir(node):
            # Skip special attributes and methods
            if child_name.startswith("_"):
                continue
            
            child = getattr(node, child_name)
            
            # Handle lists of nodes
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        search(item)
            
            # Handle dictionaries of nodes
            elif isinstance(child, dict):
                for key, value in child.items():
                    if isinstance(value, Node):
                        search(value)
            
            # Handle individual nodes
            elif isinstance(child, Node):
                search(child)
    
    # Start the search from the root
    search(ast)
    
    return result


def extract_symbols_from_ast(ast: Node) -> Dict[str, Any]:
    """
    Extract symbols from the AST with their source ranges.
    
    This function traverses the AST and collects information about
    all symbols defined in the document.
    
    Args:
        ast: AST to extract symbols from
        
    Returns:
        Dictionary mapping symbol names to information about the symbol
    """
    symbols = {}
    
    # TODO: Implement symbol extraction once we have a stable AST structure
    # For now, return an empty dictionary
    
    return symbols
