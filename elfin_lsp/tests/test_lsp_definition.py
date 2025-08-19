"""
Tests for the ELFIN Language Server definition (go-to) functionality.
"""

import pytest
from pygls.workspace import Workspace, Document

from elfin_lsp.server import ELFIN_LS
from elfin_lsp.protocol import (
    Location, DefinitionParams, TextDocumentIdentifier, Position, Range
)


class TestDefinition:
    """Tests for the definition functionality."""
    
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
        self.definition_result = None
        
        async def mock_definition(server, params):
            return await self.server.lsp._features["textDocument/definition"]["method"](server, params)
        
        # Store the original method
        self.original_definition_method = self.server.lsp._features["textDocument/definition"]["method"]
        
        # Replace with our mock
        self.server.lsp._features["textDocument/definition"]["method"] = mock_definition
    
    def teardown_method(self):
        """Clean up after the test."""
        # Restore the original method
        if hasattr(self, "original_definition_method"):
            self.server.lsp._features["textDocument/definition"]["method"] = self.original_definition_method
    
    @pytest.mark.asyncio
    async def test_definition_in_same_file(self):
        """Test finding definition in the same file."""
        # Sample ELFIN code with symbol definitions and references
        document_uri = "file:///definition_test.elfin"
        document_text = """
        system TestSystem {
          continuous_state: [position, velocity];
          inputs: [force];
          
          params {
            m: mass[kg] = 1.0;                # Mass of the object
            k: spring_const[N/m] = 10.0;      # Spring constant
            b: damping[N*s/m] = 0.5;          # Damping coefficient
          }
          
          flow_dynamics {
            # Position derivative is velocity
            position_dot = velocity;
            
            # Spring-mass-damper equation
            velocity_dot = (-k * position - b * velocity + force) / m;
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
        
        # Create a definition request for 'm' in the flow_dynamics section
        # First find line with "m" reference in velocity_dot equation
        lines = document_text.split("\n")
        for i, line in enumerate(lines):
            if "velocity_dot" in line and "/m" in line:
                ref_line = i
                ref_char = line.rfind("m")  # Last 'm' in line (should be the divisor)
                break
        
        definition_params = DefinitionParams(
            textDocument=TextDocumentIdentifier(uri=document_uri),
            position=Position(line=ref_line, character=ref_char)
        )
        
        # Call the definition handler
        definition_result = await self.server.lsp._endpoint.request("textDocument/definition", definition_params)
        
        # Verify the definition result
        assert definition_result is not None
        
        # The result can be a single Location or a list of Locations
        if isinstance(definition_result, list):
            locations = definition_result
        else:
            locations = [definition_result]
        
        assert len(locations) > 0
        
        # Find the line with m: mass[kg] definition in the params section
        def_line = 0
        for i, line in enumerate(lines):
            if "m: mass[kg]" in line:
                def_line = i
                break
        
        # At least one of the locations should point to the 'm' definition
        found = False
        for location in locations:
            if (location.uri == document_uri and 
                location.range.start.line == def_line and
                "m" in lines[def_line][location.range.start.character:location.range.end.character]):
                found = True
                break
        
        assert found, "Definition location for 'm' not found"
    
    @pytest.mark.asyncio
    async def test_no_definition_for_unknown_symbol(self):
        """Test that no definition is returned for unknown symbols."""
        # Sample ELFIN code without a definition for 'unknown_var'
        document_uri = "file:///no_def_test.elfin"
        document_text = """
        system TestSystem {
          continuous_state: [x, v];
          inputs: [f];
          
          params {
            alpha = 1.0;
          }
          
          flow_dynamics {
            x_dot = v;
            v_dot = f;
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
        
        # Create a definition request for a non-existent symbol 'unknown_var'
        # Just point to an empty part of the file
        definition_params = DefinitionParams(
            textDocument=TextDocumentIdentifier(uri=document_uri),
            position=Position(line=10, character=20)  # Empty area in the file
        )
        
        # Call the definition handler
        definition_result = await self.server.lsp._endpoint.request("textDocument/definition", definition_params)
        
        # Verify the definition result is None or empty list
        assert definition_result is None or (isinstance(definition_result, list) and len(definition_result) == 0)
