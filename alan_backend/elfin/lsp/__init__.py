"""
ELFIN Language Server Protocol (LSP) package.

This package provides LSP support for the ELFIN language, enabling
rich IDE features in editors that support the Language Server Protocol.
"""

from alan_backend.elfin.lsp.server import start_server

__all__ = ['start_server']
