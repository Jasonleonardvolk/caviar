"""
FastMCP - Proper FastAPI subclass for TORI MCP
===============================================

This makes the MCP server properly ASGI-callable and includes all routers.
"""

from fastapi import FastAPI, APIRouter
from typing import Optional

class FastMCP(FastAPI):
    """FastMCP class that properly extends FastAPI with MCP functionality"""
    
    def __init__(self, name: str = "TORI MCP", version: str = "0.1.0", **kwargs):
        # Initialize FastAPI with our defaults
        super().__init__(
            title=name,
            version=version,
            **kwargs
        )
        
        # Store MCP-specific attributes
        self.name = name
        self.version = version
        
        # Create routers for different components
        self.tool_router = APIRouter(prefix="/tools", tags=["tools"])
        self.resource_router = APIRouter(prefix="/resources", tags=["resources"])
        self.prompt_router = APIRouter(prefix="/prompts", tags=["prompts"])
        
        # Include all routers
        self.include_router(self.tool_router)
        self.include_router(self.resource_router)
        self.include_router(self.prompt_router)
        
        # Add system status endpoint
        @self.get("/api/system/status")
        async def system_status():
            return {
                "status": "operational",
                "server": self.name,
                "version": self.version,
                "mcp_available": True,
                "routers": ["tools", "resources", "prompts"]
            }
        
        # Add health check
        @self.get("/health")
        async def health():
            return {"status": "healthy", "server": self.name}
    
    def __call__(self, scope, receive, send):
        """Make the instance ASGI-callable"""
        return super().__call__(scope, receive, send)
    
    def tool(self):
        """Decorator for registering tools"""
        def decorator(func):
            # Check if tool already registered
            route_path = f"/{func.__name__}"
            
            # Check if route already exists
            for route in self.tool_router.routes:
                if hasattr(route, 'path') and route.path == route_path:
                    # Already registered, skip
                    return func
            
            # Register the tool endpoint
            self.tool_router.add_api_route(
                route_path,
                func,
                methods=["POST"],
                name=func.__name__
            )
            return func
        return decorator
    
    def resource(self):
        """Decorator for registering resources"""
        def decorator(func):
            # Check if resource already registered
            route_path = f"/{func.__name__}"
            
            # Check if route already exists
            for route in self.resource_router.routes:
                if hasattr(route, 'path') and route.path == route_path:
                    # Already registered, skip
                    return func
            
            # Register the resource endpoint
            self.resource_router.add_api_route(
                route_path,
                func,
                methods=["GET"],
                name=func.__name__
            )
            return func
        return decorator
    
    def prompt(self):
        """Decorator for registering prompts"""
        def decorator(func):
            # Check if prompt already registered
            route_path = f"/{func.__name__}"
            
            # Check if route already exists
            for route in self.prompt_router.routes:
                if hasattr(route, 'path') and route.path == route_path:
                    # Already registered, skip
                    return func
            
            # Register the prompt endpoint
            self.prompt_router.add_api_route(
                route_path,
                func,
                methods=["POST"],
                name=func.__name__
            )
            return func
        return decorator
