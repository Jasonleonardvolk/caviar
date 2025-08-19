"""
Inlay hint provider for the ELFIN language server.

This module provides inlay hint capabilities for the ELFIN language server,
showing dimensional information next to expressions.
"""

import logging
from typing import List, Optional

from pygls.server import LanguageServer
from elfin_lsp.protocol import Position
from pygls.capabilities import types

# Import types directly from the public API
InlayHint = types.InlayHint
InlayHintKind = types.InlayHintKind
InlayHintParams = types.InlayHintParams

logger = logging.getLogger(__name__)

async def inlay_hint(ls: LanguageServer, params: InlayHintParams) -> List[InlayHint]:
    """
    Handle textDocument/inlayHint request.
    
    Args:
        ls: The language server
        params: The inlay hint parameters
        
    Returns:
        A list of inlay hints
    """
    try:
        uri = params.textDocument.uri
        
        # Get the showcase data for this document
        showcase = ls.showcase.get(uri)
        
        # If no showcase data or no hints, return empty list
        if not showcase or not hasattr(showcase, "hints") or not showcase.hints:
            return []
        
        hints = []
        
        # Create an inlay hint for each hint in the showcase
        for hint in showcase.hints:
            # Create the position
            pos = Position(
                line=hint.line,
                character=hint.col
            )
            
            # Create the inlay hint
            inlay = InlayHint(
                position=pos,
                label=hint.label,
                kind=InlayHintKind.Type,
                padding_left=True
            )
            
            hints.append(inlay)
        
        return hints
    except Exception as e:
        logger.error(f"Error in inlay_hint: {e}")
        return []

def register(server: LanguageServer):
    """
    Register the inlay hint provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/inlayHint")
    async def handle_inlay_hint(params: InlayHintParams):
        """Handle inlayHint request."""
        return await inlay_hint(server, params)
