import pytest
import textwrap
from elfin_lsp.server import ELFIN_LS
from pygls.capabilities import types

@pytest.mark.asyncio
async def test_inlay_hints():
    """Test that inlay hints are generated for assignments with dimensions."""
    # Initialize the server
    ls = ELFIN_LS
    await ls.lsp.bf_initialize({"processId": 1, "rootUri": None})
    
    # Create a sample document with assignments that have dimensions
    txt = textwrap.dedent("""
    system Test {
      continuous_state: [x, v];
      inputs: [u];
      
      params {
        m: mass[kg] = 1.0;
        k: spring_const[N/m] = 10.0;
      }
      
      flow_dynamics {
        # Position derivative
        x_dot = v;
        
        # Velocity derivative
        v_dot = (-k * x - m * v + u) / m;
      }
    }
    """)
    uri = "mem://inlay_test.elfin"
    
    # Notify the server that the document was opened
    ls.lsp.notify(
        "textDocument/didOpen",
        types.DidOpenTextDocumentParams(
            textDocument=types.TextDocumentItem(
                uri=uri,
                languageId="elfin",
                version=1,
                text=txt
            )
        )
    )
    
    # Request inlay hints for the document
    hints = await ls.lsp.inlay_hint(
        types.InlayHintParams(
            textDocument=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=100, character=100)
            )
        )
    )
    
    # Check that we have hints
    assert len(hints) > 0
    
    # Check that we have hints for position and velocity derivatives
    position_hint = False
    velocity_hint = False
    
    for hint in hints:
        if "m" in hint.label:  # Position derivative should have meters
            position_hint = True
        if "m/s" in hint.label:  # Velocity derivative might have meters/second
            velocity_hint = True
    
    # Verify at least one hint was found
    assert position_hint or velocity_hint, "No dimension hints found for derivatives"
