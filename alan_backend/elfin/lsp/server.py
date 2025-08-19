"""
ELFIN Language Server Protocol (LSP) implementation.

This module provides a Language Server Protocol implementation for the ELFIN
language, enabling rich IDE features like diagnostics, hover information,
code completion, and more.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from pygls.server import LanguageServer
from pygls.lsp.types import (
    Diagnostic, 
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    CompletionParams,
    CompletionList,
    CompletionItem,
    CompletionItemKind,
    Hover,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
    TextDocumentPositionParams,
)
from pygls.lsp.methods import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_SAVE,
    COMPLETION,
    HOVER,
)

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from alan_backend.elfin.standalone_parser import parse_string
from alan_backend.elfin.compiler.passes.dim_checker import DimChecker


# Create language server
elfin_server = LanguageServer('elfin-language-server', 'v0.1')

# Configure logging
logging.basicConfig(filename='elfin-lsp.log', level=logging.DEBUG)
log = logging.getLogger(__name__)


@elfin_server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams) -> None:
    """
    Process a document open notification from the client.
    
    This will analyze the document and send diagnostics to the client.
    
    Args:
        ls: Language server instance
        params: Document open parameters
    """
    log.info(f"Document opened: {params.text_document.uri}")
    document = ls.workspace.get_document(params.text_document.uri)
    validate_document(ls, document)


@elfin_server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams) -> None:
    """
    Process a document change notification from the client.
    
    This will re-analyze the document and update diagnostics for the client.
    
    Args:
        ls: Language server instance
        params: Document change parameters
    """
    log.info(f"Document changed: {params.text_document.uri}")
    document = ls.workspace.get_document(params.text_document.uri)
    validate_document(ls, document)


@elfin_server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams) -> None:
    """
    Process a document save notification from the client.
    
    This will perform a thorough analysis of the document and send
    comprehensive diagnostics to the client.
    
    Args:
        ls: Language server instance
        params: Document save parameters
    """
    log.info(f"Document saved: {params.text_document.uri}")
    document = ls.workspace.get_document(params.text_document.uri)
    validate_document(ls, document)


@elfin_server.feature(COMPLETION)
def completion(ls: LanguageServer, params: CompletionParams) -> CompletionList:
    """
    Process a completion request from the client.
    
    This will provide intelligent code completion based on the context
    at the given document position.
    
    Args:
        ls: Language server instance
        params: Completion parameters
        
    Returns:
        A list of completion items
    """
    document = ls.workspace.get_document(params.text_document.uri)
    position = params.position
    
    # Get the document line at the current position
    line = document.lines[position.line]
    
    # Helper function auto-completion
    if 'import' in line:
        # Suggest standard library imports
        return CompletionList(
            is_incomplete=False,
            items=[
                CompletionItem(
                    label="StdHelpers",
                    kind=CompletionItemKind.Module,
                    detail="ELFIN Standard Helpers",
                    documentation=MarkupContent(
                        kind=MarkupKind.Markdown,
                        value="```elfin\nimport StdHelpers from \"std/helpers.elfin\";\n```\n\n"
                              "Standard helper functions like wrapAngle, hAbs, etc."
                    ),
                    insert_text="StdHelpers from \"std/helpers.elfin\";"
                )
            ]
        )
    
    # Standard helper function completion
    if 'StdHelpers' in line or 'h' in line or 'wrap' in line:
        # Suggest helper functions
        return CompletionList(
            is_incomplete=False,
            items=[
                CompletionItem(
                    label="wrapAngle",
                    kind=CompletionItemKind.Function,
                    detail="wrapAngle(theta)",
                    documentation="Wraps an angle to [-π, π]",
                    insert_text="wrapAngle(${1:theta})"
                ),
                CompletionItem(
                    label="hAbs",
                    kind=CompletionItemKind.Function,
                    detail="hAbs(x)",
                    documentation="Absolute value function",
                    insert_text="hAbs(${1:x})"
                ),
                CompletionItem(
                    label="hMin",
                    kind=CompletionItemKind.Function,
                    detail="hMin(a, b)",
                    documentation="Minimum of two values",
                    insert_text="hMin(${1:a}, ${2:b})"
                ),
                CompletionItem(
                    label="hMax",
                    kind=CompletionItemKind.Function,
                    detail="hMax(a, b)",
                    documentation="Maximum of two values",
                    insert_text="hMax(${1:a}, ${2:b})"
                ),
            ]
        )
    
    # Block/keyword completion
    if position.character == 0 or line.strip() == "":
        # Suggest top-level blocks
        return CompletionList(
            is_incomplete=False,
            items=[
                CompletionItem(
                    label="system",
                    kind=CompletionItemKind.Class,
                    detail="system <name> { ... }",
                    documentation="Define a system with state variables and dynamics",
                    insert_text="system ${1:SystemName} {\n  continuous_state: [${2:x, y, z}];\n  input: [${3:u}];\n  \n  params {\n    ${4}\n  }\n  \n  flow_dynamics {\n    ${5}\n  }\n}"
                ),
                CompletionItem(
                    label="lyapunov",
                    kind=CompletionItemKind.Class,
                    detail="lyapunov <name> { ... }",
                    documentation="Define a Lyapunov function for stability analysis",
                    insert_text="lyapunov ${1:LyapName} {\n  system ${2:SystemName};\n  \n  V = ${3:0.5*x^2};\n}"
                ),
                CompletionItem(
                    label="barrier",
                    kind=CompletionItemKind.Class,
                    detail="barrier <name> { ... }",
                    documentation="Define a barrier function for safety verification",
                    insert_text="barrier ${1:BarrierName} {\n  system ${2:SystemName};\n  \n  B = ${3:x_max - x};\n  alpha_fun = ${4:alpha * B};\n  \n  params {\n    ${5:alpha: 1.0;}\n  }\n}"
                ),
                CompletionItem(
                    label="mode",
                    kind=CompletionItemKind.Class,
                    detail="mode <name> { ... }",
                    documentation="Define a control mode with controller logic",
                    insert_text="mode ${1:ModeName} {\n  system ${2:SystemName};\n  \n  controller {\n    ${3}\n  }\n}"
                ),
                CompletionItem(
                    label="import",
                    kind=CompletionItemKind.Keyword,
                    detail="import <module> from \"<path>\";",
                    documentation="Import a module from a file",
                    insert_text="import ${1:StdHelpers} from \"${2:std/helpers.elfin}\";"
                ),
            ]
        )
    
    # Default to empty completion list
    return CompletionList(is_incomplete=False, items=[])


@elfin_server.feature(HOVER)
def hover(ls: LanguageServer, params: TextDocumentPositionParams) -> Optional[Hover]:
    """
    Process a hover request from the client.
    
    This will provide information about the symbol under the cursor,
    including type information and documentation.
    
    Args:
        ls: Language server instance
        params: Hover parameters
        
    Returns:
        Hover information or None if not available
    """
    document = ls.workspace.get_document(params.text_document.uri)
    position = params.position
    
    # Get the current line and find the word under cursor
    line = document.lines[position.line]
    word_start = position.character
    while word_start > 0 and line[word_start-1].isalnum():
        word_start -= 1
    
    word_end = position.character
    while word_end < len(line) and line[word_end].isalnum():
        word_end += 1
    
    word = line[word_start:word_end]
    
    # Standard helper functions
    if word == "wrapAngle":
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value="```elfin\nwrapAngle(theta)\n```\n\n"
                      "Wraps an angle to [-π, π], normalizing it to the principal range.\n\n"
                      "**Parameters:**\n"
                      "- `theta`: Angle to wrap (rad)\n\n"
                      "**Returns:** Wrapped angle in [-π, π]"
            ),
            range=Range(
                start=Position(line=position.line, character=word_start),
                end=Position(line=position.line, character=word_end)
            )
        )
    elif word == "hAbs":
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value="```elfin\nhAbs(x)\n```\n\n"
                      "Absolute value function.\n\n"
                      "**Parameters:**\n"
                      "- `x`: Input value\n\n"
                      "**Returns:** Absolute value of x"
            ),
            range=Range(
                start=Position(line=position.line, character=word_start),
                end=Position(line=position.line, character=word_end)
            )
        )
    elif word == "hMin":
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value="```elfin\nhMin(a, b)\n```\n\n"
                      "Returns the minimum of two values.\n\n"
                      "**Parameters:**\n"
                      "- `a`: First value\n"
                      "- `b`: Second value\n\n"
                      "**Returns:** Minimum value"
            ),
            range=Range(
                start=Position(line=position.line, character=word_start),
                end=Position(line=position.line, character=word_end)
            )
        )
    elif word == "hMax":
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value="```elfin\nhMax(a, b)\n```\n\n"
                      "Returns the maximum of two values.\n\n"
                      "**Parameters:**\n"
                      "- `a`: First value\n"
                      "- `b`: Second value\n\n"
                      "**Returns:** Maximum value"
            ),
            range=Range(
                start=Position(line=position.line, character=word_start),
                end=Position(line=position.line, character=word_end)
            )
        )
    
    # Keywords
    elif word in ("system", "lyapunov", "barrier", "mode", "params", "flow_dynamics", "controller"):
        keyword_info = {
            "system": "Define a system with state variables, inputs, parameters, and dynamics",
            "lyapunov": "Define a Lyapunov function for stability analysis",
            "barrier": "Define a barrier function for safety verification",
            "mode": "Define a control mode with controller logic",
            "params": "Define parameters with optional units and default values",
            "flow_dynamics": "Define continuous dynamics (differential equations)",
            "controller": "Define control logic within a mode"
        }
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=f"**{word}**\n\n{keyword_info.get(word, '')}"
            ),
            range=Range(
                start=Position(line=position.line, character=word_start),
                end=Position(line=position.line, character=word_end)
            )
        )
    
    # Default to no hover information
    return None


def validate_document(ls: LanguageServer, document) -> None:
    """
    Validate an ELFIN document and send diagnostics to the client.
    
    Args:
        ls: Language server instance
        document: Document to validate
    """
    diagnostics = []
    
    try:
        # Parse the document
        ast = parse_string(document.source)
        
        # Run dimensional checking pass
        checker = DimChecker()
        dim_diagnostics = checker.check_program(ast)
        
        # Convert diagnostics to LSP format
        for diag in dim_diagnostics:
            severity = DiagnosticSeverity.Warning
            
            # Create range from location info
            line = diag.line - 1 if diag.line else 0
            col = diag.column - 1 if diag.column else 0
            range = Range(
                start=Position(line=line, character=col),
                end=Position(line=line, character=col + 10)  # Approximate range
            )
            
            # Add diagnostic
            diagnostics.append(
                Diagnostic(
                    range=range,
                    message=diag.message,
                    severity=severity,
                    source="elfin-dim-checker",
                    code=diag.code
                )
            )
        
    except Exception as e:
        # If parsing fails, add a generic diagnostic
        log.error(f"Error validating document: {str(e)}")
        diagnostics.append(
            Diagnostic(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=0, character=1)
                ),
                message=f"Error parsing ELFIN file: {str(e)}",
                severity=DiagnosticSeverity.Error,
                source="elfin-lsp"
            )
        )
    
    # Send diagnostics to client
    ls.publish_diagnostics(document.uri, diagnostics)


def start_server() -> None:
    """Start the ELFIN language server."""
    elfin_server.start_io()


if __name__ == "__main__":
    start_server()
