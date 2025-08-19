"""
Example: Creating Your Own MCP Server
=====================================

This template shows how to create a custom MCP server using the framework.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Import the base components
from kha.mcp_metacognitive.core.agent_registry import Agent
from kha.mcp_metacognitive.core.psi_archive import psi_archive
from kha.mcp_metacognitive.agents.kaizen_metrics import metrics_exporter

class MyCustomMCPServer(Agent):
    """
    Template for creating your own MCP server.
    
    This example creates a simple task management MCP server.
    """
    
    # Metadata for discovery
    _metadata = {
        "name": "task_manager",
        "description": "Custom MCP server for task management",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/tasks", "method": "GET", "description": "List all tasks"},
            {"path": "/tasks", "method": "POST", "description": "Create new task"},
            {"path": "/tasks/{id}", "method": "PUT", "description": "Update task"},
            {"path": "/tasks/{id}", "method": "DELETE", "description": "Delete task"}
        ],
        "version": "1.0.0"
    }
    
    def __init__(self, name: str = "task_manager"):
        super().__init__(name)
        self.tasks = {}  # In-memory task storage
        self.task_counter = 0
        
        # Log initialization
        psi_archive.log_event("mcp_server_initialized", {
            "server_name": name,
            "capabilities": list(self._metadata["endpoints"])
        })
    
    async def execute(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main execution handler for MCP commands.
        
        Commands:
        - list_tasks: Get all tasks
        - create_task: Create a new task
        - update_task: Update existing task
        - delete_task: Remove a task
        - get_status: Get server status
        """
        
        if command == "list_tasks":
            return self._list_tasks()
            
        elif command == "create_task":
            return await self._create_task(params or {})
            
        elif command == "update_task":
            return await self._update_task(params or {})
            
        elif command == "delete_task":
            return self._delete_task(params or {})
            
        elif command == "get_status":
            return self._get_status()
            
        else:
            return {
                "error": f"Unknown command: {command}",
                "available_commands": [
                    "list_tasks", "create_task", "update_task", 
                    "delete_task", "get_status"
                ]
            }
    
    def _list_tasks(self) -> Dict[str, Any]:
        """List all tasks"""
        return {
            "tasks": list(self.tasks.values()),
            "total": len(self.tasks)
        }
    
    async def _create_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task"""
        # Validate required fields
        if "title" not in params:
            return {"error": "Missing required field: title"}
        
        # Generate ID and create task
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        task = {
            "id": task_id,
            "title": params["title"],
            "description": params.get("description", ""),
            "status": params.get("status", "pending"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": params.get("metadata", {})
        }
        
        self.tasks[task_id] = task
        
        # Log event for monitoring
        psi_archive.log_event("task_created", {
            "task_id": task_id,
            "title": task["title"]
        })
        
        # Record metric if available
        if metrics_exporter:
            metrics_exporter.record_custom_event("tasks_created")
        
        return {
            "success": True,
            "task": task
        }
    
    async def _update_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing task"""
        task_id = params.get("id")
        if not task_id or task_id not in self.tasks:
            return {"error": "Task not found"}
        
        # Update fields
        task = self.tasks[task_id]
        if "title" in params:
            task["title"] = params["title"]
        if "description" in params:
            task["description"] = params["description"]
        if "status" in params:
            task["status"] = params["status"]
        if "metadata" in params:
            task["metadata"].update(params["metadata"])
        
        task["updated_at"] = datetime.utcnow().isoformat()
        
        return {
            "success": True,
            "task": task
        }
    
    def _delete_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a task"""
        task_id = params.get("id")
        if not task_id or task_id not in self.tasks:
            return {"error": "Task not found"}
        
        deleted_task = self.tasks.pop(task_id)
        
        return {
            "success": True,
            "deleted": deleted_task
        }
    
    def _get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "server": self.name,
            "version": self._metadata["version"],
            "tasks_count": len(self.tasks),
            "uptime": "TODO: Calculate uptime",
            "memory_usage": "TODO: Calculate memory",
            "capabilities": self._metadata["endpoints"]
        }


# Example: Creating a more complex MCP server with tools
class ToolCallingMCPServer(Agent):
    """
    MCP Server that can call external tools and services.
    """
    
    _metadata = {
        "name": "tool_caller",
        "description": "MCP server that orchestrates tool calls",
        "enabled": True,
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"query": "string"}
            },
            {
                "name": "calculator",
                "description": "Perform calculations",
                "parameters": {"expression": "string"}
            },
            {
                "name": "file_reader",
                "description": "Read file contents",
                "parameters": {"path": "string"}
            }
        ]
    }
    
    async def execute(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tool-based commands"""
        
        if command == "call_tool":
            tool_name = params.get("tool")
            tool_params = params.get("parameters", {})
            
            if tool_name == "web_search":
                return await self._web_search(tool_params)
            elif tool_name == "calculator":
                return self._calculate(tool_params)
            elif tool_name == "file_reader":
                return await self._read_file(tool_params)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        elif command == "list_tools":
            return {"tools": self._metadata["tools"]}
            
        else:
            return {"error": "Unknown command"}
    
    async def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web search"""
        query = params.get("query", "")
        
        # In a real implementation, you'd call an actual search API
        results = [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2"}
        ]
        
        return {
            "query": query,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform calculation"""
        expression = params.get("expression", "")
        
        try:
            # SAFETY: In production, use a proper expression parser
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "error": f"Calculation failed: {str(e)}"
            }
    
    async def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents"""
        file_path = params.get("path", "")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return {
                "path": file_path,
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {
                "error": f"Failed to read file: {str(e)}"
            }


# FastAPI integration for HTTP access
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class TaskRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    status: Optional[str] = "pending"
    metadata: Optional[Dict[str, Any]] = {}

def create_mcp_api(mcp_server: Agent) -> FastAPI:
    """Create a FastAPI app for your MCP server"""
    
    app = FastAPI(title=f"{mcp_server.name} MCP API")
    
    @app.get("/")
    async def root():
        return {"message": f"Welcome to {mcp_server.name} MCP Server"}
    
    @app.get("/status")
    async def get_status():
        result = await mcp_server.execute("get_status")
        return result
    
    @app.get("/tasks")
    async def list_tasks():
        result = await mcp_server.execute("list_tasks")
        return result
    
    @app.post("/tasks")
    async def create_task(task: TaskRequest):
        result = await mcp_server.execute("create_task", task.dict())
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    @app.put("/tasks/{task_id}")
    async def update_task(task_id: str, task: TaskRequest):
        params = task.dict()
        params["id"] = task_id
        result = await mcp_server.execute("update_task", params)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    
    @app.delete("/tasks/{task_id}")
    async def delete_task(task_id: str):
        result = await mcp_server.execute("delete_task", {"id": task_id})
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Create your MCP server
    task_server = MyCustomMCPServer()
    
    # Create API
    app = create_mcp_api(task_server)
    
    # Add to agent registry for discovery
    from kha.mcp_metacognitive.core.agent_registry import agent_registry
    agent_registry.register(task_server)
    
    # Run the server
    print("🚀 Starting Custom MCP Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
