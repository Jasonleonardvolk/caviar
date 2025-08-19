"""
Pytest configuration for ELFIN LSP tests.

This file is automatically loaded by pytest and contains shared fixtures and setup
for tests.
"""

import asyncio
from pygls.protocol import LanguageServerProtocol
from pygls.capabilities import types

def _bf_initialize(self, params):
    """Blocking initialise helper for **tests only**, compatible with pygls ≥ 1.0."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        # The request method is on the endpoint, not directly on the server
        self._server._endpoint.request("initialize", types.InitializeParams(**params))
    )

# Monkey-patch once
LanguageServerProtocol.bf_initialize = _bf_initialize
