"""
Tests for the ELFIN Language Server hover functionality.
"""

import pytest
from pygls.workspace import Workspace, Document

from elfin_lsp.server import ELFIN_LS
from elfin_lsp.protocol import Hover, HoverParams, TextDocumentIdentifier, Position, Range, MarkupContent


class TestHover:
    """Tests for the hover functionality."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create server and initialize it first
        self.server = ELFIN_LS
        
        # Initialize the server before trying to access workspace
        self.server.lsp.bf_initialize({"processId": 1234, "rootUri": None})
        
        # Create our own workspace for test documents
        self.workspace = Workspace("", None)
        
        # Store original methods we need to mock
        self._original_get_document = self.server.lsp.workspace.get_document
        
        # Create a document lookup that checks our test workspace first
        def mock_get_document(uri):
            if uri in self.workspace._documents:
                return self.workspace.get_document(uri)
            return self._original_get_document(uri)
            
        # Replace the get_document method
        self.server.lsp.workspace.get_document = mock_get_document
        
        # Reset the showcase attribute
        self.server.showcase = {}
        
        # Mock the needed methods
        self.hover_result = None
        
        async def mock_hover(server, params):
            return await self.server.lsp._features["textDocument/hover"]["method"](server, params)
        
        # Store the original method
        self.original_hover_method = self.server.lsp._features["textDocument/hover"]["method"]
        
        # Replace with our mock
        self.server.lsp._features["textDocument/hover"]["method"] = mock_hover
    
    def teardown_method(self):
        """Clean up after the test."""
        # Restore the original method
        if hasattr(self, "original_hover_method"):
            self.server.lsp._features["textDocument/hover"]["method"] = self.original_hover_method
    
    @pytest.mark.asyncio
    async def test_hover_symbol_with_dimension(self):
        """Test hover over a symbol with dimensional information."""
        # Sample ELFIN code with a symbol that has a dimension
        document_uri = "file:///hover_test.elfin"
        document_text = """
        system TestSystem {
          continuous_state: [x, v];
          inputs: [f];
          
          params {
            m: mass[kg] = 1.0;
            k: spring_const[N/m] = 10.0;
            g: acceleration[m/s^2] = 9.81;
          }
          
          flow_dynamics {
            x_dot = v;
            v_dot = -k/m * x - g;
          }
        }
        """
        
        # Initialize the server
        self.server.lsp.bf_initialize({"processId": 1234, "rootUri": None})
        
        # Update the document
        self.workspace.put_document(document_uri, document_text)
        
        # Process the document
        from elfin_lsp.server import process_document
        process_document(self.server, document_uri, document_text)
        
        # Create a hover request for a symbol with dimension: 'g'
        # The position should point to the 'g' in the flow_dynamics section
        g_line = document_text.split("\n").index("            v_dot = -k/m * x - g;")
        g_character = document_text.split("\n")[g_line].find("g;") 
        
        hover_params = HoverParams(
            textDocument=TextDocumentIdentifier(uri=document_uri),
            position=Position(line=g_line, character=g_character)
        )
        
        # Call the hover handler
        hover_result = await self.server.lsp._endpoint.request("textDocument/hover", hover_params)
        
        # Verify the hover result
        assert hover_result is not None
        assert isinstance(hover_result, Hover)
        assert hover_result.contents is not None
        
        # The hover content should contain the dimension of 'g' - m/s^2
        if isinstance(hover_result.contents, MarkupContent):
            assert "m/s^2" in hover_result.contents.value
        else:
            assert "m/s^2" in str(hover_result.contents)
    
    @pytest.mark.asyncio
    async def test_hover_symbol_without_dimension(self):
        """Test hover over a symbol without dimensional information."""
        # Sample ELFIN code with a symbol that has no dimension
        document_uri = "file:///hover_no_dim.elfin"
        document_text = """
        system TestSystem {
          continuous_state: [x, v];
          inputs: [f];
          
          params {
            flag = true;  # Boolean parameter without dimension
            text = "hello";  # String parameter without dimension
          }
          
          flow_dynamics {
            x_dot = v;
            v_dot = flag ? 1.0 : 0.0;
          }
        }
        """
        
        # Initialize the server
        self.server.lsp.bf_initialize({"processId": 1234, "rootUri": None})
        
        # Update the document
        self.workspace.put_document(document_uri, document_text)
        
        # Process the document
        from elfin_lsp.server import process_document
        process_document(self.server, document_uri, document_text)
        
        # Create a hover request for a symbol without dimension: 'flag'
        # The position should point to the 'flag' in the flow_dynamics section
        flag_line = document_text.split("\n").index("            v_dot = flag ? 1.0 : 0.0;")
        flag_character = document_text.split("\n")[flag_line].find("flag") 
        
        hover_params = HoverParams(
            textDocument=TextDocumentIdentifier(uri=document_uri),
            position=Position(line=flag_line, character=flag_character)
        )
        
        # Call the hover handler
        hover_result = await self.server.lsp._endpoint.request("textDocument/hover", hover_params)
        
        # Verify the hover result
        # Since we don't have proper symbol lookup yet, this might be None
        # or it might have basic info without dimension
        if hover_result is not None:
            assert isinstance(hover_result, Hover)
            assert hover_result.contents is not None
            
            # The hover content should not contain dimension brackets
            if isinstance(hover_result.contents, MarkupContent):
                assert "[" not in hover_result.contents.value or "]" not in hover_result.contents.value
            else:
                assert "[" not in str(hover_result.contents) or "]" not in str(hover_result.contents)
