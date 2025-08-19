"""
Code Action provider for the ELFIN language server.

This module provides code action capabilities for the ELFIN language server,
such as quick fixes for missing imports, unit conversions, etc.
"""

from typing import List, Dict, Union, Any, Optional
import logging
import re

from pygls.server import LanguageServer
from elfin_lsp.protocol import Diagnostic, TextDocumentIdentifier
from pygls.capabilities import types

# Import types directly from the public API
CodeAction = types.CodeAction
CodeActionKind = types.CodeActionKind
TextEdit = types.TextEdit
WorkspaceEdit = types.WorkspaceEdit
Range = types.Range
Position = types.Position
CodeActionParams = types.CodeActionParams

# Import other code action providers
from elfin_lsp.providers.unit_conversion import unit_conversion_code_action

logger = logging.getLogger(__name__)

def _create_import_helpers_edit(document_uri: str) -> WorkspaceEdit:
    """
    Create a workspace edit to import helpers at the top of the file.
    
    Args:
        document_uri: The URI of the document to edit
        
    Returns:
        A workspace edit that inserts the import statement
    """
    # Create a text edit at the beginning of the file
    text_edit = TextEdit(
        range=Range(
            start=Position(line=0, character=0),
            end=Position(line=0, character=0)
        ),
        new_text="import Helpers from \"std/helpers.elfin\";\n\n"
    )
    
    # Create a workspace edit with the text edit
    changes = {document_uri: [text_edit]}
    return WorkspaceEdit(changes=changes)

def _get_code_actions(server: LanguageServer, params: CodeActionParams) -> List[CodeAction]:
    """
    Get code actions for the given params.
    
    Args:
        server: The language server
        params: The code action parameters
        
    Returns:
        A list of code actions
    """
    text_document = params.textDocument
    uri = text_document.uri
    
    # Get the document and current diagnostics
    # Check if the document has any diagnostics with code "MISSING_HELPER"
    diagnostics = []
    if hasattr(server.lsp, 'diagnostics') and uri in server.lsp.diagnostics:
        diagnostics = server.lsp.diagnostics[uri]
    
    # Get the document content
    document = server.workspace.get_document(uri)
    
    # Actions to return
    actions = []
    
    # Check if any of the diagnostics have code "MISSING_HELPER"
    missing_helper_diags = [
        d for d in params.context.diagnostics 
        if hasattr(d, 'code') and d.code == "MISSING_HELPER"
    ]
    
    # If there are any missing helper diagnostics, add an import action
    if missing_helper_diags and document:
        # Check if the import is already present to avoid duplicate imports
        if "import Helpers" not in document.source.splitlines()[0][:60]:
            # Create the code action
            action = CodeAction(
                title="Import Helpers",
                kind="quickfix",
                edit=_create_import_helpers_edit(uri),
                diagnostics=missing_helper_diags
            )
            
            actions.append(action)
    
    return actions

async def code_action(server: LanguageServer, params: CodeActionParams):
    """
    Handle textDocument/codeAction request.
    
    Args:
        server: The language server
        params: The code action parameters
        
    Returns:
        A list of code actions
    """
    try:
        # Get import helpers actions
        actions = _get_code_actions(server, params)
        
        # Get unit conversion actions
        unit_actions = await unit_conversion_code_action(server, params)
        actions.extend(unit_actions)
        
        return actions
    except Exception as e:
        logger.error(f"Error getting code actions: {e}")
        return []

def register(server: LanguageServer):
    """
    Register the code action provider with the language server.
    
    Args:
        server: The language server to register with
    """
    @server.feature("textDocument/codeAction")
    async def handle_code_action(params: CodeActionParams):
        return await code_action(server, params)
