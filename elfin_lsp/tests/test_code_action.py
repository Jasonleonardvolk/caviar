import pytest
from elfin_lsp.server import ELFIN_LS
from pygls.capabilities import types
from elfin_lsp.protocol import Position, Range

InitializeParams = types.InitializeParams
DidOpenTextDocumentParams = types.DidOpenTextDocumentParams
TextDocumentItem = types.TextDocumentItem
CodeActionParams = types.CodeActionParams

@pytest.mark.asyncio
async def test_import_helpers_quickfix():
    ls = ELFIN_LS
    # Use bf_initialize instead of initialize (for backward compatibility)
    ls.lsp.bf_initialize({"processId": 1, "rootUri": None})

    text = "angle = wrapAngle(theta);"
    uri = "mem://helpers.elfin"
    ls.lsp.notify(
        "textDocument/didOpen",
        DidOpenTextDocumentParams(
            textDocument=TextDocumentItem(uri=uri, languageId="elfin", text=text, version=1)
        )
    )

    # Wait for the server to process the document and generate diagnostics
    # In a real test, we'd use asyncio.sleep or a better synchronization mechanism
    import asyncio
    await asyncio.sleep(0.1)
    
    # Create a mock diagnostic for MISSING_HELPER
    mock_diagnostic = types.Diagnostic(
        range=Range(Position(line=0, character=10), Position(line=0, character=19)),
        message="Function 'wrapAngle' is not defined. Consider importing helpers.",
        severity=2,  # Warning
        code="MISSING_HELPER"
    )

    # Create diagnostics list in the server if it doesn't exist
    if not hasattr(ls.lsp, 'diagnostics'):
        ls.lsp.diagnostics = {}
    
    # Add mock diagnostics to the server
    ls.lsp.diagnostics[uri] = [mock_diagnostic]

    # Fake a code-action request on the offending line/column
    params = CodeActionParams(
        textDocument=types.TextDocumentIdentifier(uri=uri),
        range=Range(Position(line=0, character=10), Position(line=0, character=19)),
        context=types.CodeActionContext(diagnostics=[mock_diagnostic])
    )
    
    # Call the code_action endpoint
    actions = await ls.lsp._endpoint.request("textDocument/codeAction", params)

    # Verify that we get at least one code action with title "Import Helpers"
    assert any(a.title == "Import Helpers" for a in actions), "No 'Import Helpers' code action found"
    
    # Get the Import Helpers action
    import_action = next(a for a in actions if a.title == "Import Helpers")
    
    # Verify it has an edit
    assert hasattr(import_action, 'edit'), "Code action has no edit"
    assert import_action.edit is not None, "Code action edit is None"
    
    # Verify the edit contains changes for our file
    assert uri in import_action.edit.changes, f"No changes for {uri} in the edit"
    
    # Verify there's at least one text edit
    text_edits = import_action.edit.changes[uri]
    assert len(text_edits) > 0, "No text edits in the changes"
    
    # Verify the text edit will add the import statement
    first_edit = text_edits[0]
    assert "import Helpers" in first_edit.new_text, "Import statement not found in the edit"
    
    print("✅ 'Import Helpers' code action test passed!")
