# Helper script to register MCP component readiness
import requests
import sys
import time

def register_mcp_ready(port=8002):
    """Register MCP metacognitive server as ready"""
    try:
        # Mark MCP as ready
        response = requests.post(
            f'http://localhost:{port}/api/system/components/mcp_metacognitive/ready',
            json={"port": 8100, "transport": "sse"}
        )
        if response.status_code == 200:
            print("✅ MCP metacognitive server registered as ready")
            return True
    except Exception as e:
        print(f"❌ Failed to register MCP: {e}")
    return False

if __name__ == "__main__":
    # Wait a bit for MCP to fully initialize
    time.sleep(2)
    
    # Try to register
    api_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    register_mcp_ready(api_port)
