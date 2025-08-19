"""
Rename provider for the ELFIN language server.

This module provides rename symbol capability for the ELFIN language server,
allowing users to rename variables, parameters, and other symbols within a file.
"""

import logging
from typing import Optional, Dict, List

from pygls.server import LanguageServer
from elfin_lsp.protocol import Position, Range
from pygls.capabilities import types

# Import types directly from the public API
WorkspaceEdit = types.WorkspaceEdit
TextEdit = types.TextEdit
RenameParams = types.RenameParams

logger = logging.getLogger(__name__)

def symbol_at(ls: LanguageServer, uri: str, position: Position) -> Optional[str]:
    """
    Find the symbol at the given position in the document.
    
    Args:
        ls: The language server
        uri: Document URI
        position: Position to check
        
    Returns:
        The symbol name at the position, or None if no symbol is found
    """
    try:
        # Get the document
        document = ls.workspace.get_document(uri)
        if not document:
            return None
        
        # Get the line
        line = document.lines[position.line]
        
        # Simple approach: find the word at position
        # This is a simplified approach - in production, use AST to find symbols
        
        # Find the start of the word
        start = position.character
        while start > 0 and line[start-1].isalnum():
            start -= 1
            
        # Find the end of the word
        end = position.character
        while end < len(line) and line[end].isalnum():
            end += 1
            
        # Extract the word
        word = line[start:end]
        
        # Skip empty words or keywords
        if not word or word in ["system", "params", "flow_dynamics", "continuous_state", "inputs"]:
            return None
            
        return word
    except Exception as e:
        logger.error(f"Error finding symbol at position: {e}")
        return None

async def rename_symbol(ls: LanguageServer, params: RenameParams) -> Optional[WorkspaceEdit]:
    """
    Rename a symbol in the document.
    
    Args:
        ls: The language server
        params: Rename parameters
        
    Returns:
        A workspace edit containing the renames, or None if rename is not possible
    """
    try:
        # Access parameters accounting for different pygls versions
        uri = params.text_document.uri if hasattr(params, 'text_document') else params.textDocument.uri
        position = params.position
        new_name = params.new_name if hasattr(params, 'new_name') else params.newName
        
        # Find the symbol at the position
        old_name = symbol_at(ls, uri, position)
        if not old_name:
            logger.warning(f"No symbol found at position {position}")
            return None
        
        logger.info(f"Renaming symbol '{old_name}' to '{new_name}'")
        
        # Get the document
        document = ls.workspace.get_document(uri)
        if not document:
            logger.warning(f"Document not found: {uri}")
            return None
        
        # Find all occurrences of the symbol in the document
        edits = []
        
        # Process each line
        for i, line in enumerate(document.lines):
            # Find all occurrences in the line
            start = 0
            while True:
                # Find the next occurrence
                index = line.find(old_name, start)
                if index == -1:
                    break
                    
                # Check if it's a whole word
                end = index + len(old_name)
                is_whole_word = (
                    (index == 0 or not line[index-1].isalnum()) and
                    (end >= len(line) or not line[end].isalnum())
                )
                
                if is_whole_word:
                    # Create a text edit for this occurrence
                    edits.append(
                        TextEdit(
                            range=Range(
                                start=Position(line=i, character=index),
                                end=Position(line=i, character=end)
                            ),
                            new_text=new_name
                        )
                    )
                
                # Move to the next position
                start = end
        
        # Create a workspace edit with all the renames
        return WorkspaceEdit(changes={uri: edits})
    except Exception as e:
        logger.error(f"Error during rename: {e}")
        return None

def register(server: LanguageServer):
    """
    Register the rename provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/rename")
    async def handle_rename(params: RenameParams):
        """Handle rename request."""
        return await rename_symbol(server, params)
