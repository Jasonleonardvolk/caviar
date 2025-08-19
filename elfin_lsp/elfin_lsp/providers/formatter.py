"""
Formatter provider for the ELFIN language server.

This module provides formatting capabilities for the ELFIN language server,
ensuring consistent code style across ELFIN files.

Formatting rules:
- 2-space indentation
- Line width ≤ 80 cols
- Vertical alignment of = in param blocks
- Compact unit annotations
"""

import logging
import re
from typing import List, Dict, Optional, Any, Union

from pygls.server import LanguageServer
from elfin_lsp.protocol import Range, Position
from pygls.capabilities import types

# Import types directly from the public API
TextEdit = types.TextEdit
DocumentFormattingParams = types.DocumentFormattingParams
DocumentRangeFormattingParams = types.DocumentRangeFormattingParams

logger = logging.getLogger(__name__)

def _format_elfin_text(text: str) -> str:
    """
    Format ELFIN source code.
    
    Args:
        text: The source code to format
        
    Returns:
        The formatted source code
    """
    lines = text.splitlines()
    result_lines = []
    
    # Keep track of indentation level and param block state
    indent_level = 0
    in_param_block = False
    param_defs = []
    
    # Process line by line
    for line in lines:
        # Skip empty lines
        if not line.strip():
            result_lines.append("")
            continue
        
        # Process line content
        line = line.strip()
        
        # Check for block start
        if "{" in line and "}" not in line:
            # Add the line with current indentation
            result_lines.append("  " * indent_level + line)
            indent_level += 1
            # Check if entering a param block
            if "params" in line:
                in_param_block = True
                param_defs = []
            continue
            
        # Check for block end
        if "}" in line and "{" not in line:
            indent_level = max(0, indent_level - 1)
            # Check if exiting a param block
            if in_param_block:
                # Format and add the collected param definitions with aligned equals
                result_lines.extend(_format_param_block(param_defs, indent_level))
                in_param_block = False
                param_defs = []
                # Add the closing brace
                result_lines.append("  " * indent_level + "}")
                continue
            
            # Add the line with current indentation
            result_lines.append("  " * indent_level + line)
            continue
        
        # If in param block, collect param definitions
        if in_param_block and "=" in line:
            param_defs.append(line)
            continue
            
        # Format compact unit annotations
        if "[" in line and "]" in line:
            line = _format_unit_annotations(line)
            
        # Normal line with current indentation
        result_lines.append("  " * indent_level + line)
    
    return "\n".join(result_lines)

def _format_param_block(param_defs: List[str], indent_level: int) -> List[str]:
    """
    Format a block of parameter definitions with aligned equals signs.
    
    Args:
        param_defs: List of parameter definition lines
        indent_level: Current indentation level
        
    Returns:
        List of formatted parameter definition lines
    """
    if not param_defs:
        return []
    
    # Find the position of the = sign and the maximum name length
    max_name_length = 0
    for param in param_defs:
        name_part = param.split("=")[0].rstrip()
        max_name_length = max(max_name_length, len(name_part))
    
    # Format each parameter definition with aligned equals signs
    formatted_params = []
    for param in param_defs:
        name_part, value_part = param.split("=", 1)
        name_part = name_part.rstrip()
        value_part = value_part.lstrip()
        
        # Add padding to align equals signs
        padding = " " * (max_name_length - len(name_part))
        formatted_param = "  " * indent_level + name_part + padding + " = " + value_part
        
        # Ensure line length is under 80 characters
        if len(formatted_param) > 80:
            # Simplify for long lines: don't add extra padding
            formatted_param = "  " * indent_level + name_part + " = " + value_part
            
        formatted_params.append(formatted_param)
    
    return formatted_params

def _format_unit_annotations(line: str) -> str:
    """
    Format unit annotations in a compact way.
    
    Args:
        line: The line containing unit annotations
        
    Returns:
        The line with compact unit annotations
    """
    # Find all unit annotations (patterns like name: type[unit])
    pattern = r'(\w+):\s*(\w+)\s*\[(.*?)\]\s*'
    if re.search(pattern, line):
        # Replace with compact format: name: type[unit]
        return re.sub(pattern, r'\1: \2[\3] ', line)
    return line

def _create_text_edits(original_text: str, formatted_text: str) -> List[TextEdit]:
    """
    Create text edits for the formatting changes.
    
    Args:
        original_text: The original unformatted text
        formatted_text: The newly formatted text
        
    Returns:
        A list of text edits to transform the original to the formatted text
    """
    # For full document formatting, replace the entire document
    return [
        TextEdit(
            range=Range(
                start=Position(line=0, character=0),
                end=Position(
                    line=len(original_text.splitlines()),
                    character=0
                )
            ),
            new_text=formatted_text
        )
    ]

async def format_document(
    server: LanguageServer, 
    params: Union[DocumentFormattingParams, DocumentRangeFormattingParams]
) -> List[TextEdit]:
    """
    Format an ELFIN document.
    
    Args:
        server: The language server
        params: Formatting parameters
        
    Returns:
        A list of text edits to format the document
    """
    try:
        # Get the document from the workspace
        document = server.workspace.get_document(params.textDocument.uri)
        if not document:
            logger.error(f"Document not found: {params.textDocument.uri}")
            return []
        
        # If range formatting is requested, only format that range
        if hasattr(params, 'range'):
            # Extract the text in the range
            range_start = params.range.start
            range_end = params.range.end
            
            lines = document.source.splitlines()
            
            # Check if the range spans multiple lines
            if range_start.line == range_end.line:
                # Single line range: extract characters in the range
                line = lines[range_start.line]
                original_text = line[range_start.character:range_end.character]
                formatted_text = _format_elfin_text(original_text)
                
                return [
                    TextEdit(
                        range=Range(
                            start=Position(line=range_start.line, character=range_start.character),
                            end=Position(line=range_end.line, character=range_end.character)
                        ),
                        new_text=formatted_text
                    )
                ]
            else:
                # Multi-line range: extract the lines in the range
                range_lines = lines[range_start.line:range_end.line + 1]
                
                # Adjust first and last lines for partial line selection
                if range_start.character > 0:
                    range_lines[0] = range_lines[0][range_start.character:]
                
                if range_end.character < len(lines[range_end.line]):
                    range_lines[-1] = range_lines[-1][:range_end.character]
                
                original_text = "\n".join(range_lines)
                formatted_text = _format_elfin_text(original_text)
                
                return [
                    TextEdit(
                        range=Range(
                            start=Position(line=range_start.line, character=range_start.character),
                            end=Position(line=range_end.line, character=range_end.character)
                        ),
                        new_text=formatted_text
                    )
                ]
        
        # Full document formatting
        original_text = document.source
        formatted_text = _format_elfin_text(original_text)
        
        return _create_text_edits(original_text, formatted_text)
        
    except Exception as e:
        logger.error(f"Error formatting document: {e}")
        return []

def register(server: LanguageServer):
    """
    Register the formatter provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/formatting")
    async def handle_document_formatting(params: DocumentFormattingParams):
        """Handle document formatting request."""
        return await format_document(server, params)
    
    @server.feature("textDocument/rangeFormatting")
    async def handle_document_range_formatting(params: DocumentRangeFormattingParams):
        """Handle document range formatting request."""
        return await format_document(server, params)
