#!/usr/bin/env python3
"""
TORI API Integration Module
Integrates the enhanced API endpoints into the existing Prajna API
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "api"))

def integrate_enhanced_api():
    """
    Integrate enhanced API endpoints into the existing Prajna API app
    """
    try:
        # Import the existing Prajna API app
        from prajna.api.prajna_api import app as prajna_app
        
        # Import enhanced API endpoints
        from api.enhanced_api import (
            multiply_matrices,
            process_intent,
            extended_health_check,
            websocket_endpoint,
            stability_websocket,
            chaos_websocket,
            websocket_test_page,
            graphql_status,
            manager,
            startup_enhanced_api
        )
        
        # The enhanced_api module already adds routes to the app
        # This function just ensures everything is loaded
        
        print("✅ Enhanced API endpoints integrated successfully!")
        print("   - POST /multiply - Hyperbolic matrix multiplication")
        print("   - POST /intent - Intent-driven reasoning")
        print("   - GET /health/extended - Extended health check")
        print("   - WS /ws/stability - Stability monitoring WebSocket")
        print("   - WS /ws/chaos - Chaos events WebSocket")
        print("   - GET /ws/test - WebSocket test page")
        print("   - GET /graphql/status - GraphQL availability")
        
        return prajna_app
        
    except ImportError as e:
        print(f"⚠️ Failed to integrate enhanced API: {e}")
        print("   Make sure prajna_api.py exists and is properly configured")
        return None

# Auto-integrate when imported
app = integrate_enhanced_api()

if __name__ == "__main__":
    print("""
    TORI API Integration
    ===================
    
    This module integrates the enhanced API with the existing Prajna API.
    
    The integration adds:
    - /multiply endpoint for hyperbolic matrix operations
    - /intent endpoint for intent-driven reasoning
    - WebSocket support for real-time updates
    - Extended health monitoring
    
    The integrated API maintains all existing Prajna endpoints while
    adding the new functionality.
    """)
