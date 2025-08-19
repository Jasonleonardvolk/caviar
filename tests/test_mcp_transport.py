"""
Test for MCP transport fix
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestMCPTransport(unittest.TestCase):
    """Test MCP server transport configuration"""
    
    @patch('mcp_metacognitive.server.FastMCP')
    @patch('mcp_metacognitive.server.config')
    def test_sse_transport_fix(self, mock_config, mock_fastmcp_class):
        """Test that SSE transport passes host and port separately"""
        # Setup mocks
        mock_config.transport_type = "sse"
        mock_config.server_host = "0.0.0.0"
        mock_config.server_port = 8100
        mock_config.server_name = "TORI"
        mock_config.server_version = "1.0.0"
        
        mock_mcp_instance = Mock()
        mock_fastmcp_class.return_value = mock_mcp_instance
        
        # Import and run the server setup
        from mcp_metacognitive.server import main, MCP_AVAILABLE
        
        # Override MCP_AVAILABLE for testing
        import mcp_metacognitive.server as server_module
        server_module.MCP_AVAILABLE = True
        server_module.mcp = mock_mcp_instance
        
        # Call main
        main()
        
        # Verify FastMCP was instantiated correctly
        mock_fastmcp_class.assert_called_once_with(
            name="TORI",
            version="1.0.0"
        )
        
        # Verify run was called with correct parameters
        mock_mcp_instance.run.assert_called_once_with(
            transport="sse",
            host="0.0.0.0",
            port=8100
        )
    
    @patch('mcp_metacognitive.server.FastMCP')
    @patch('mcp_metacognitive.server.config')
    def test_stdio_transport(self, mock_config, mock_fastmcp_class):
        """Test that stdio transport works correctly"""
        # Setup mocks
        mock_config.transport_type = "stdio"
        mock_config.server_name = "TORI"
        mock_config.server_version = "1.0.0"
        
        mock_mcp_instance = Mock()
        mock_fastmcp_class.return_value = mock_mcp_instance
        
        # Import and run the server setup
        from mcp_metacognitive.server import main, MCP_AVAILABLE
        
        # Override MCP_AVAILABLE for testing
        import mcp_metacognitive.server as server_module
        server_module.MCP_AVAILABLE = True
        server_module.mcp = mock_mcp_instance
        
        # Call main
        main()
        
        # Verify run was called without parameters for stdio
        mock_mcp_instance.run.assert_called_once_with()

if __name__ == '__main__':
    unittest.main()
