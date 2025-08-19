"""
LSP protocol definitions for ELFIN.

This module contains dataclass definitions for the subset of the Language Server Protocol
that is used by the ELFIN language server.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---  LSP → Server -------------------------------------------------
@dataclass
class InitializeParams:
    """
    Parameters passed to initialize request.
    """
    rootUri: Optional[str]
    capabilities: Optional[Dict[str, Any]] = None
    workspaceFolders: Optional[List[Any]] = None
    processId: Optional[int] = None


@dataclass
class DidOpenTextDocumentParams:
    """
    Parameters for textDocument/didOpen notification.
    """
    textDocument: "TextDocumentItem"


@dataclass
class DidChangeTextDocumentParams:
    """
    Parameters for textDocument/didChange notification.
    """
    textDocument: "VersionedTextDocumentIdentifier"
    contentChanges: List["TextDocumentContentChangeEvent"]


@dataclass
class VersionedTextDocumentIdentifier:
    """
    Text document identifier with a version.
    """
    uri: str
    version: int


@dataclass
class TextDocumentContentChangeEvent:
    """
    Event describing a change to a text document.
    """
    text: str
    range: Optional["Range"] = None
    rangeLength: Optional[int] = None


@dataclass
class TextDocumentItem:
    """
    Item to transfer a text document from the client to the server.
    """
    uri: str
    languageId: str
    version: int
    text: str


@dataclass
class TextDocumentPositionParams:
    """
    Parameters for requests that require a text document and a position.
    """
    textDocument: "TextDocumentIdentifier"
    position: "Position"


@dataclass
class TextDocumentIdentifier:
    """
    Text document identifier.
    """
    uri: str


@dataclass
class HoverParams(TextDocumentPositionParams):
    """
    Parameters for hover request.
    """
    pass


@dataclass
class DefinitionParams(TextDocumentPositionParams):
    """
    Parameters for definition request.
    """
    pass


# ---  Server → LSP -------------------------------------------------
@dataclass
class Diagnostic:
    """
    Represents a diagnostic, such as a compiler error or warning.
    """
    range: "Range"
    severity: int         # 1=Error, 2=Warning, 3=Info, 4=Hint
    message: str
    code: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[int]] = None


@dataclass
class Range:
    """
    A range in a text document.
    """
    start: "Position"
    end: "Position"


@dataclass
class Position:
    """
    A position in a text document.
    """
    line: int
    character: int


@dataclass
class Location:
    """
    Represents a location inside a resource, such as a line inside a text file.
    """
    uri: str
    range: Range


@dataclass
class PublishDiagnosticsParams:
    """
    Parameters for textDocument/publishDiagnostics notification.
    """
    uri: str
    diagnostics: List[Diagnostic]
    version: Optional[int] = None


@dataclass
class MarkupContent:
    """
    A markdown string or plaintext string.
    """
    kind: str  # "plaintext" | "markdown"
    value: str


@dataclass
class Hover:
    """
    The result of a hover request.
    """
    contents: Union[MarkupContent, str, List[str]]
    range: Optional[Range] = None


@dataclass
class CompletionItem:
    """
    A completion item.
    """
    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None
    documentation: Optional[Union[str, MarkupContent]] = None
    insertText: Optional[str] = None
    data: Optional[Any] = None


@dataclass
class CompletionList:
    """
    A list of completion items.
    """
    isIncomplete: bool
    items: List[CompletionItem]


@dataclass
class CompletionParams(TextDocumentPositionParams):
    """
    Parameters for completion request.
    """
    context: Optional[Any] = None
