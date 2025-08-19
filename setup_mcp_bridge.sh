#!/bin/bash
# Quick setup script for MCP-TORI bridge

echo "Setting up MCP-TORI Bridge..."

# Check if MCP is built
if [ ! -d "mcp-server-architecture/dist" ]; then
    echo "Building MCP server..."
    cd mcp-server-architecture
    npm install
    npm run build
    cd ..
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install httpx pytest pytest-asyncio pytest-benchmark

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
MCP_GATEWAY_URL=http://localhost:8080
MCP_AUTH_TOKEN=your-secure-token-here
PYTHON_BRIDGE_TOKEN=your-secure-token-here
EOF
fi

# Start services
echo "Starting services..."
echo "1. Start MCP: cd mcp-server-architecture && npm start"
echo "2. Start Python: python run_stable_server.py"
echo "3. Run tests: pytest tests/test_mcp_tori_integration.py -v"

echo "Setup complete!"