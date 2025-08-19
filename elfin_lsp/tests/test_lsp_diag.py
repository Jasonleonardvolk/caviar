"""
Tests for the ELFIN Language Server diagnostics functionality.
"""

import pytest
from pygls.workspace import Workspace, Document

from elfin_lsp.server import ELFIN_LS
from elfin_lsp.protocol import PublishDiagnosticsParams, Diagnostic


class TestDiagnostics:
    """Tests for the diagnostics functionality."""
    
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
        
        # Mock the publish_diagnostics method
        self.published_diagnostics = {}
        
        def mock_publish_diagnostics(params: PublishDiagnosticsParams):
            self.published_diagnostics[params.uri] = params.diagnostics
        
        self.original_publish_diagnostics = self.server.publish_diagnostics
        self.server.publish_diagnostics = mock_publish_diagnostics
    
    def teardown_method(self):
        """Clean up after the test."""
        # Restore the original publish_diagnostics method
        self.server.publish_diagnostics = self.original_publish_diagnostics
    
    def test_dimensional_error_detection(self):
        """Test that the server correctly identifies dimensional errors."""
        # Sample ELFIN code with a dimensional error (adding meters and seconds)
        document_uri = "file:///test.elfin"
        document_text = """
        system TestSystem {
          continuous_state: [x, v];
          inputs: [f];
          
          params {
            m: mass[kg] = 1.0;
            k: spring_const[N/m] = 10.0;
            b: damping[N*s/m] = 0.5;
            g: acceleration[m/s^2] = 9.81;
            t: time[s] = 0.1;
          }
          
          flow_dynamics {
            # Position derivative
            x_dot = v;
            
            # Error: adding position (x) and time (t)
            v_dot = x + t;  # Dimensional error: [m] + [s]
          }
        }
        """
        
        # Initialize the server
        self.server.lsp.bf_initialize({"processId": 1234, "rootUri": None})
        
        # Update the document
        self.workspace.put_document(document_uri, document_text)
        
        # Notify the server about the document
        notification_params = {
            "textDocument": {
                "uri": document_uri,
                "languageId": "elfin",
                "version": 1,
                "text": document_text
            }
        }
        
        # Process the document
        self.server.lsp._endpoint.notify("textDocument/didOpen", notification_params)
        
        # Verify diagnostics were published
        assert document_uri in self.published_diagnostics
        
        # There should be at least one diagnostic for the dimensional error
        diags = self.published_diagnostics[document_uri]
        assert len(diags) > 0
        
        # Find the diagnostic related to dimensional mismatch
        dim_mismatch_diags = [d for d in diags if "dimension" in d.message.lower()]
        assert len(dim_mismatch_diags) > 0
        
        # The diagnostic should be related to the line with v_dot = x + t
        for diag in dim_mismatch_diags:
            # Using 0-based line indexing, looking for the line with the error
            error_line = document_text.split("\n").index("            v_dot = x + t;  # Dimensional error: [m] + [s]")
            assert diag.range.start.line == error_line
            assert "dimension" in diag.message.lower()
    
    def test_valid_document(self):
        """Test that the server correctly processes a valid document with no errors."""
        # Sample valid ELFIN code
        document_uri = "file:///valid.elfin"
        document_text = """
        system ValidSystem {
          continuous_state: [x, v];
          inputs: [f];
          
          params {
            m: mass[kg] = 1.0;
            k: spring_const[N/m] = 10.0;
            b: damping[N*s/m] = 0.5;
          }
          
          flow_dynamics {
            # Position derivative
            x_dot = v;
            
            # Valid force equation: F = ma
            v_dot = f / m;
          }
        }
        """
        
        # Initialize the server
        self.server.lsp.bf_initialize({"processId": 1234, "rootUri": None})
        
        # Update the document
        self.workspace.put_document(document_uri, document_text)
        
        # Notify the server about the document
        notification_params = {
            "textDocument": {
                "uri": document_uri,
                "languageId": "elfin",
                "version": 1,
                "text": document_text
            }
        }
        
        # Process the document
        self.server.lsp._endpoint.notify("textDocument/didOpen", notification_params)
        
        # Verify no dimensional error diagnostics were published
        assert document_uri in self.published_diagnostics
        
        # There should be no diagnostics for dimensional errors
        diags = self.published_diagnostics[document_uri]
        dim_mismatch_diags = [d for d in diags if "dimension" in d.message.lower()]
        assert len(dim_mismatch_diags) == 0
