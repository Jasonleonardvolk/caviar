import pytest
import textwrap
from elfin_lsp.server import ELFIN_LS
from pygls.capabilities import types

@pytest.mark.asyncio
async def test_system_codelens():
    """Test that code lenses are generated for system declarations."""
    # Initialize the server
    ls = ELFIN_LS
    await ls.lsp.bf_initialize({"processId": 1, "rootUri": None})
    
    # Create a sample document with a system declaration
    txt = textwrap.dedent("""
        system MyPlant {
          params { m: [kg] = 1 }
        }
    """)
    uri = "mem://lens.elfin"
    
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
    
    # Request code lenses for the document
    lenses = await ls.lsp.code_lens(
        types.CodeLensParams(
            textDocument=types.TextDocumentIdentifier(uri=uri)
        )
    )
    
    # Check that we have at least two lenses (docs and tests)
    assert len(lenses) == 2
    
    # Check that the first lens is for documentation
    assert lenses[0].command.title == "📄 Docs"
    assert lenses[0].command.command == "elfin.openDocs"
    assert lenses[0].command.arguments[0] == "MyPlant"
    
    # Check that the second lens is for tests
    assert lenses[1].command.title == "🧪 Run tests"
    assert lenses[1].command.command == "elfin.runSystemTests"
    assert lenses[1].command.arguments[0] == "MyPlant"
    assert lenses[1].command.arguments[1] == uri
