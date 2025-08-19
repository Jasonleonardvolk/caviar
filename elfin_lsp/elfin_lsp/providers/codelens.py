"""
CodeLens provider for the ELFIN language server.

This module provides CodeLens capabilities for the ELFIN language server,
showing clickable actions above system declarations.
"""

import logging
from typing import List, Optional

from pygls.server import LanguageServer
from elfin_lsp.protocol import Position, Range
from pygls.capabilities import types

# Import types directly from the public API
CodeLens = types.CodeLens
Command = types.Command
CodeLensParams = types.CodeLensParams

logger = logging.getLogger(__name__)

async def codelens(ls: LanguageServer, params: CodeLensParams) -> List[CodeLens]:
    """
    Handle textDocument/codeLens request.
    
    Args:
        ls: The language server
        params: The code lens parameters
        
    Returns:
        A list of code lenses
    """
    try:
        uri = params.textDocument.uri
        
        # Get the showcase data for this document
        showcase = ls.showcase.get(uri)
        
        # If no showcase data or no systems, return empty list
        if not showcase or not hasattr(showcase, "systems") or not showcase.systems:
            return []
        
        lenses = []
        
        # Create a code lens for each system
        for system in showcase.systems:
            # Get the position of the system declaration
            pos = Position(
                line=system.span.start.line,
                character=system.span.start.col
            )
            
            # Create a documentation lens
            doc_lens = CodeLens(
                range=Range(
                    start=pos,
                    end=pos
                ),
                command=Command(
                    title="📄 Docs",
                    command="elfin.openDocs",
                    arguments=[system.name]
                )
            )
            lenses.append(doc_lens)
            
            # Create a test lens
            test_lens = CodeLens(
                range=Range(
                    start=pos,
                    end=pos
                ),
                command=Command(
                    title="🧪 Run tests",
                    command="elfin.runSystemTests",
                    arguments=[system.name, uri]
                )
            )
            lenses.append(test_lens)
        
        return lenses
    except Exception as e:
        logger.error(f"Error in codelens: {e}")
        return []

def register(server: LanguageServer):
    """
    Register the CodeLens provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/codeLens")
    async def handle_codelens(params: CodeLensParams):
        """Handle codeLens request."""
        return await codelens(server, params)
