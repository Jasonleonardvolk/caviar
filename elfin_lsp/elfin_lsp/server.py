"""
ELFIN language server implementation.

This module implements a Language Server Protocol (LSP) server for the ELFIN language,
using pygls for the JSON-RPC communication.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pygls.server import LanguageServer
from pygls.workspace import Document

from elfin_lsp.protocol import (
    Diagnostic, Position, Range,
    InitializeParams, DidOpenTextDocumentParams, DidChangeTextDocumentParams,
)
from pygls.capabilities import types
# Use pygls's PublishDiagnosticsParams to avoid conflicts
PublishDiagnosticsParams = types.PublishDiagnosticsParams

# Import providers
from elfin_lsp.providers import hover, definition, code_action, formatter, rename, codelens, inlay

# Import ELFIN compiler components
from alan_backend.elfin.standalone_parser import parse
from alan_backend.elfin.compiler.pipeline import CompilerPipeline
from alan_backend.elfin.compiler.passes.dim_checker import DimChecker

# Import file watcher
from elfin_lsp.adapters.fs_watch import ElfinFileWatcher

# Set up logging
logging.basicConfig(
    filename="elfin_lsp.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("elfin_lsp")

# Create the language server
ELFIN_LS = LanguageServer("elfin-lsp", "1.0.0")

# Initialize the showcase attribute to store symbol tables
ELFIN_LS.showcase = {}

# Initialize file watcher
ELFIN_LS.file_watcher = None

# Register providers
hover.register(ELFIN_LS)
definition.register(ELFIN_LS)
code_action.register(ELFIN_LS)
formatter.register(ELFIN_LS)
rename.register(ELFIN_LS)
codelens.register(ELFIN_LS)
inlay.register(ELFIN_LS)

# Initialize server capabilities for code actions
def handle_file_change(path: str):
    """
    Handle a file change event from the file watcher.
    
    Args:
        path: Path to the file that changed
    """
    # Convert the file path to a URI
    file_uri = f"file://{path}"
    
    try:
        # Read the file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process the document
        process_document(ELFIN_LS, file_uri, content)
        
        logger.info(f"Processed external file change: {path}")
    except Exception as e:
        logger.error(f"Error processing external file change {path}: {e}")


def start_file_watcher(root_dirs: List[str]):
    """
    Start the file watcher.
    
    Args:
        root_dirs: List of root directories to watch
    """
    if not ELFIN_LS.file_watcher:
        # Create the file watcher
        ELFIN_LS.file_watcher = ElfinFileWatcher(
            on_change=handle_file_change,
            root_dirs=root_dirs,
            extensions=[".elfin"]
        )
        
        # Start watching
        ELFIN_LS.file_watcher.start_watching()


def convert_diagnostic(elfin_diag: Any) -> Diagnostic:
    """
    Convert an ELFIN diagnostic to an LSP diagnostic.

    Args:
        elfin_diag: An ELFIN diagnostic from the compiler

    Returns:
        An LSP diagnostic
    """
    # Get line and column, defaulting to 0 if not available
    line = getattr(elfin_diag, "line", 0) or 0
    column = getattr(elfin_diag, "column", 0) or 0
    
    # Convert to 0-based indexing if necessary (LSP uses 0-based, ELFIN might use 1-based)
    line = max(0, line - 1) if line > 0 else 0
    column = max(0, column - 1) if column > 0 else 0
    
    # Create position and range
    start_pos = Position(line, column)
    # For now, we make the range just one character wide
    end_pos = Position(line, column + 1)
    rng = Range(start_pos, end_pos)

    # Determine severity (1=error, 2=warning, 3=info, 4=hint)
    severity = 2  # Default to warning
    if hasattr(elfin_diag, "severity"):
        severity_str = getattr(elfin_diag, "severity", "").lower()
        if severity_str == "error":
            severity = 1
        elif severity_str == "info":
            severity = 3
        elif severity_str == "hint":
            severity = 4

    # Get the diagnostic code
    code = getattr(elfin_diag, "code", None)
    
    # Get the source
    source = getattr(elfin_diag, "source", "elfin")
    
    # Create and return the LSP diagnostic
    return Diagnostic(
        range=rng,
        severity=severity,
        message=elfin_diag.message,
        code=code,
        source=source,
    )


@ELFIN_LS.feature("initialize")
async def initialize(ls: LanguageServer, params: InitializeParams):
    """
    Handle initialize request.
    
    Args:
        ls: The language server
        params: Initialize parameters
    """
    logger.info("Initializing ELFIN language server")
    
    # Store initialization parameters
    ls.initialization_params = params
    
    # Return server capabilities
    return {
        "capabilities": {
            "textDocumentSync": {
                "openClose": True,
                "change": 1,  # Full text sync
                "willSave": False,
                "willSaveWaitUntil": False,
                "save": {"includeText": False},
            },
            "hoverProvider": True,  # Enable hover support
            "definitionProvider": True,  # Enable go-to-definition support
            "diagnosticProvider": {
                "interFileDependencies": False,
                "workspaceDiagnostics": False,
            },
            "codeActionProvider": True,  # Enable code actions
            "documentFormattingProvider": True,  # Enable formatting
            "documentRangeFormattingProvider": True,  # Enable range formatting
            "renameProvider": True,  # Enable rename support
            "codeLensProvider": {  # Enable code lens
                "resolveProvider": False
            },
            "inlayHintProvider": {  # Enable inlay hints
                "resolveProvider": False
            }
        }
    }


@ELFIN_LS.feature("initialized")
async def initialized(ls: LanguageServer, params):
    """
    Handle initialized notification.
    
    This is called after the client has received the initialize result
    and before the client starts sending requests or notifications.
    
    Args:
        ls: The language server
        params: Parameters for the notification
    """
    logger.info("ELFIN language server initialized")
    
    # Determine workspace folders
    workspace_folders = determine_workspace_folders(ls)
    logger.info(f"Workspace folders: {workspace_folders}")
    
    # Start file watcher for workspace folders
    if workspace_folders:
        start_file_watcher(workspace_folders)
        logger.info("File watcher started")


@ELFIN_LS.feature("textDocument/didOpen")
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    """
    Handle textDocument/didOpen notification.
    
    This is called when a document is opened in the editor.
    
    Args:
        ls: The language server
        params: Parameters for the notification
    """
    logger.info(f"Document opened: {params.textDocument.uri}")
    
    # Get the document text
    uri = params.textDocument.uri
    text = params.textDocument.text
    
    # Process the document and publish diagnostics
    process_document(ls, uri, text)


@ELFIN_LS.feature("textDocument/didChange")
async def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams):
    """
    Handle textDocument/didChange notification.
    
    This is called when a document is changed in the editor.
    
    Args:
        ls: The language server
        params: Parameters for the notification
    """
    logger.info(f"Document changed: {params.textDocument.uri}")
    
    # Get the document from the workspace
    document = ls.workspace.get_document(params.textDocument.uri)
    if document:
        # Process the document and publish diagnostics
        process_document(ls, document.uri, document.source)


def process_document(ls: LanguageServer, uri: str, text: str):
    """
    Process a document and publish diagnostics.
    
    Args:
        ls: The language server
        uri: The document URI
        text: The document text
    """
    try:
        # Parse the document
        ast = parse(text)
        
        # Run the compiler pipeline
        pipeline = CompilerPipeline()
        pipeline.process(ast)
        
        # Store the symbol table for hover capabilities
        if hasattr(pipeline, "constant_folder") and hasattr(pipeline.constant_folder, "symbol_table"):
            # Store the symbol table keyed by URI
            ls.showcase[uri] = pipeline.constant_folder.symbol_table
            logger.info(f"Stored symbol table for {uri} with {len(pipeline.constant_folder.symbol_table.symbols)} symbols")
        
        # Get diagnostics from the pipeline
        elfin_diagnostics = pipeline.get_diagnostics()
        
        # Convert to LSP diagnostics
        lsp_diagnostics = [convert_diagnostic(diag) for diag in elfin_diagnostics]
        
        # Publish diagnostics
        ls.publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics))
        
        logger.info(f"Published {len(lsp_diagnostics)} diagnostics for {uri}")
        
    except Exception as e:
        # Log the error
        logger.error(f"Error processing document {uri}: {e}")
        
        # Create an error diagnostic
        error_diag = Diagnostic(
            range=Range(Position(0, 0), Position(0, 1)),
            severity=1,  # Error
            message=f"Error processing document: {e}",
            source="elfin-lsp",
        )
        
        # Publish the error diagnostic
        ls.publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=[error_diag]))


def determine_workspace_folders(ls: LanguageServer) -> List[str]:
    """
    Determine the workspace folders to watch.
    
    Args:
        ls: The language server
    
    Returns:
        List of workspace folder paths
    """
    workspace_folders = []
    
    # Check if we have initialization parameters with workspace folders
    if hasattr(ls, "initialization_params") and ls.initialization_params:
        # Extract workspace folders from initialization parameters
        params = ls.initialization_params
        
        # Check for workspaceFolders
        if hasattr(params, "workspaceFolders") and params.workspaceFolders:
            for folder in params.workspaceFolders:
                if hasattr(folder, "uri") and folder.uri.startswith("file://"):
                    # Convert URI to path
                    folder_path = folder.uri[7:]  # Remove "file://"
                    workspace_folders.append(folder_path)
        
        # If no workspace folders, check for rootUri
        elif hasattr(params, "rootUri") and params.rootUri and params.rootUri.startswith("file://"):
            # Convert URI to path
            root_path = params.rootUri[7:]  # Remove "file://"
            workspace_folders.append(root_path)
        
        # If no rootUri, check for rootPath (deprecated)
        elif hasattr(params, "rootPath") and params.rootPath:
            workspace_folders.append(params.rootPath)
    
    # If no workspace folders found, use current directory
    if not workspace_folders:
        workspace_folders.append(str(Path.cwd()))
    
    return workspace_folders


def run():
    """Run the language server."""
    logger.info("Starting ELFIN language server")
    
    # Start the server
    ELFIN_LS.start_io()


if __name__ == "__main__":
    run()
