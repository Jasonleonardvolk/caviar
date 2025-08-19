"""
Definition provider for the ELFIN language server.

This module implements the definition feature, allowing users to navigate
to the definition of symbols by using "Go to Definition" (F12).
"""

from typing import List, Optional, Union

from pygls.server import LanguageServer
from pygls.workspace import Document

from elfin_lsp.protocol import (
    Location, DefinitionParams, Range, Position
)
from elfin_lsp.adapters.ast_adapter import find_node_at_position


def register(server: LanguageServer) -> None:
    """
    Register the definition provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/definition")
    async def definition(ls: LanguageServer, params: DefinitionParams) -> Union[Location, List[Location], None]:
        """
        Handle a definition request.
        
        This is called when the user triggers "Go to Definition" on a symbol.
        
        Args:
            ls: The language server
            params: Parameters for the request
            
        Returns:
            Location(s) of the definition, or None if no definition is found
        """
        # Get document from workspace
        document_uri = params.textDocument.uri
        document = ls.workspace.get_document(document_uri)
        
        if not document:
            return None
        
        # Get the position
        position = params.position
        
        # Check if we have a symbol table for this document
        if not hasattr(ls, "showcase") or document_uri not in ls.showcase:
            return None
        
        # Get the symbol table
        symbol_table = ls.showcase.get(document_uri)
        
        # Look up the symbol at the position
        symbol = symbol_table.lookup(position.line, position.character)
        
        if not symbol:
            return None
        
        # Get the definition location if available
        if hasattr(symbol, "def_location") and symbol.def_location:
            # Extract location information
            def_uri = symbol.def_location.get("uri", document_uri)
            def_range = symbol.def_location.get("range")
            
            if def_range:
                start_line = def_range.get("start_line", 0)
                start_char = def_range.get("start_char", 0)
                end_line = def_range.get("end_line", start_line)
                end_char = def_range.get("end_char", start_char + len(symbol.name))
                
                # Create a location
                return Location(
                    uri=def_uri,
                    range=Range(
                        start=Position(line=start_line, character=start_char),
                        end=Position(line=end_line, character=end_char)
                    )
                )
        
        # If we don't have explicit definition location, attempt to scan the document
        # to find the declaration
        
        # Scan for declarations in the form "name: type = value" or "name = value"
        # This is a simplified heuristic that could be enhanced
        text_lines = document.source.split('\n')
        for i, line in enumerate(text_lines):
            # Simple pattern matching for parameter declarations
            param_pattern = f"{symbol.name}\\s*:"
            assign_pattern = f"{symbol.name}\\s*="
            
            if (param_pattern in line.replace(" ", "") or 
                assign_pattern in line.replace(" ", "")):
                
                # Found a potential declaration
                name_pos = line.find(symbol.name)
                if name_pos >= 0:
                    # Return the location of the declaration
                    return Location(
                        uri=document_uri,
                        range=Range(
                            start=Position(line=i, character=name_pos),
                            end=Position(line=i, character=name_pos + len(symbol.name))
                        )
                    )
        
        # If we can't find a definition, return None
        return None
