"""
Hover provider for the ELFIN language server.

This module provides hover information for the ELFIN language server,
showing types, dimensions, and docstrings for symbols.
"""

import logging
import re
from typing import Optional, Dict, List, Any

from pygls.server import LanguageServer
from elfin_lsp.protocol import Position
from pygls.capabilities import types

# Import types directly from the public API
Hover = types.Hover
MarkupContent = types.MarkupContent
MarkupKind = types.MarkupKind
HoverParams = types.HoverParams

logger = logging.getLogger(__name__)

def _get_word_at_position(document, position: Position) -> Optional[str]:
    """
    Get the word at the given position.
    
    Args:
        document: The document
        position: The position
        
    Returns:
        The word at the position, or None if no word is found
    """
    try:
        line = document.lines[position.line]
        start = position.character
        # Go back to the start of the word
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        # Go forward to the end of the word
        end = position.character
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        # Extract the word
        if start <= end:
            return line[start:end]
        return None
    except Exception as e:
        logger.error(f"Error getting word at position: {e}")
        return None

def _get_docstring_for_helper(helper_name: str) -> Optional[str]:
    """
    Get the docstring for a standard library helper function.
    
    Args:
        helper_name: The name of the helper function
        
    Returns:
        The docstring, or None if not found
    """
    # Dictionary of helper functions and their docstrings
    helper_docs = {
        "hAbs": "Absolute value function. Returns the absolute value of a number.",
        "hMin": "Minimum function. Returns the smaller of two values.",
        "hMax": "Maximum function. Returns the larger of two values.",
        "wrapAngle": "Angle wrapping function. Normalizes an angle to the range [-π, π].",
        "clamp": "Clamp function. Restricts a value to be between a minimum and maximum.",
        "lerp": "Linear interpolation function. Interpolates between two values by a factor t.",
    }
    
    return helper_docs.get(helper_name)

def _find_symbol_info(ls: LanguageServer, uri: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Find information about a symbol.
    
    Args:
        ls: The language server
        uri: The document URI
        name: The symbol name
        
    Returns:
        Information about the symbol, or None if not found
    """
    # Try to get the symbol table from the language server
    if hasattr(ls, "showcase") and uri in ls.showcase:
        symbol_table = ls.showcase[uri]
        
        # Check if the symbol exists in the symbol table
        if hasattr(symbol_table, "get_symbol") and callable(symbol_table.get_symbol):
            symbol = symbol_table.get_symbol(name)
            if symbol:
                return {
                    "name": symbol.name,
                    "type": getattr(symbol, "type", None),
                    "dim": getattr(symbol, "dim", None),
                    "value": getattr(symbol, "value", None),
                }
    
    # If it's not in the symbol table, check if it's a standard helper
    docstring = _get_docstring_for_helper(name)
    if docstring:
        return {
            "name": name,
            "docstring": docstring,
            "is_helper": True
        }
    
    return None

def _get_hover_content(symbol_info: Dict[str, Any]) -> str:
    """
    Create hover content from symbol information.
    
    Args:
        symbol_info: Information about the symbol
        
    Returns:
        The hover content as markdown
    """
    content = []
    
    # Add the symbol name
    content.append(f"### {symbol_info['name']}")
    
    # If it's a helper function, add the docstring
    if symbol_info.get("is_helper"):
        content.append(symbol_info.get("docstring", ""))
        content.append("\n*Standard library helper function*")
        return "\n\n".join(content)
    
    # Add the type if available
    if symbol_info.get("type"):
        content.append(f"**Type:** {symbol_info['type']}")
    
    # Add the dimension if available
    if symbol_info.get("dim"):
        dim_str = str(symbol_info["dim"])
        # Format the dimension nicely
        if dim_str.startswith("UnitExpr"):
            # Extract the unit expression from UnitExpr(...)
            match = re.search(r"UnitExpr\((.*)\)", dim_str)
            if match:
                dim_str = match.group(1)
        content.append(f"**Dimension:** [{dim_str}]")
    
    # Add the value if available
    if symbol_info.get("value") is not None:
        content.append(f"**Value:** {symbol_info['value']}")
    
    return "\n\n".join(content)

async def hover(ls: LanguageServer, params: HoverParams) -> Optional[Hover]:
    """
    Handle hover request.
    
    Args:
        ls: The language server
        params: Hover parameters
        
    Returns:
        Hover information, or None if no hover information is available
    """
    try:
        # Get the document
        uri = params.textDocument.uri
        position = params.position
        document = ls.workspace.get_document(uri)
        
        if not document:
            return None
        
        # Get the word at the position
        word = _get_word_at_position(document, position)
        if not word:
            return None
        
        # Get information about the symbol
        symbol_info = _find_symbol_info(ls, uri, word)
        if not symbol_info:
            return None
        
        # Create the hover content
        content = _get_hover_content(symbol_info)
        
        # Create and return the hover
        return Hover(
            contents=MarkupContent(kind=MarkupKind.Markdown, value=content),
            range=None  # No need to highlight a range
        )
    except Exception as e:
        logger.error(f"Error in hover: {e}")
        return None

def register(server: LanguageServer):
    """
    Register the hover provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/hover")
    async def handle_hover(params: HoverParams):
        """Handle hover request."""
        return await hover(server, params)
