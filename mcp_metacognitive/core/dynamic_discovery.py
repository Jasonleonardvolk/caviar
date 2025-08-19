"""
Dynamic Server Discovery and Registration System
==============================================

This module provides automatic discovery and registration of all MCP servers/agents.
"""

import os
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
import json
import asyncio

from ..core.agent_registry import Agent, agent_registry
from ..core.psi_archive import psi_archive

logger = logging.getLogger(__name__)

class ServerManifest:
    """Represents metadata about a server/agent"""
    def __init__(self, name: str, module_path: str, class_name: str, 
                 config: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.module_path = module_path
        self.class_name = class_name
        self.config = config or {}
        self.metadata = metadata or {}
        self.enabled = self.metadata.get("enabled", True)
        self.auto_start = self.metadata.get("auto_start", True)
        self.description = self.metadata.get("description", "")
        self.endpoints = self.metadata.get("endpoints", [])
        self.dependencies = self.metadata.get("dependencies", [])

class DynamicServerDiscovery:
    """Automatically discovers and manages all available servers/agents"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.discovered_servers: Dict[str, ServerManifest] = {}
        self.loaded_servers: Dict[str, Agent] = {}
        self.server_tasks: Dict[str, asyncio.Task] = {}
        
    def discover_servers(self) -> Dict[str, ServerManifest]:
        """Discover all available servers/agents"""
        logger.info("🔍 Starting dynamic server discovery...")
        
        # Paths to search for servers
        search_paths = [
            self.base_path / "agents",
            self.base_path / "servers",
            self.base_path / "mcp_servers",
            self.base_path / "extensions"
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                self._scan_directory(search_path)
        
        # Also check for manifest files
        self._load_server_manifests()
        
        logger.info(f"✅ Discovered {len(self.discovered_servers)} servers")
        return self.discovered_servers
    
    def _scan_directory(self, directory: Path):
        """Scan a directory for server/agent modules"""
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            module_name = file_path.stem
            relative_path = file_path.relative_to(self.base_path)
            module_path = str(relative_path).replace(os.sep, ".")[:-3]  # Remove .py
            
            try:
                # Try to load the module
                spec = importlib.util.spec_from_file_location(module_path, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for Agent subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Agent) and 
                            obj != Agent and
                            not name.startswith("_")):
                            
                            # Check for metadata
                            metadata = getattr(obj, "_metadata", {})
                            config = getattr(obj, "_default_config", {})
                            
                            # Create manifest
                            server_name = metadata.get("name", module_name)
                            manifest = ServerManifest(
                                name=server_name,
                                module_path=module_path,
                                class_name=name,
                                config=config,
                                metadata=metadata
                            )
                            
                            self.discovered_servers[server_name] = manifest
                            logger.info(f"  📦 Found server: {server_name} ({name})")
                            
            except Exception as e:
                logger.warning(f"  ⚠️ Error scanning {file_path}: {e}")
    
    def _load_server_manifests(self):
        """Load server manifests from JSON files"""
        manifest_files = [
            self.base_path / "servers.json",
            self.base_path / "mcp_servers.json",
            self.base_path / "agent_manifest.json"
        ]
        
        for manifest_file in manifest_files:
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r') as f:
                        manifests = json.load(f)
                        
                    for server_data in manifests.get("servers", []):
                        manifest = ServerManifest(
                            name=server_data["name"],
                            module_path=server_data["module"],
                            class_name=server_data["class"],
                            config=server_data.get("config", {}),
                            metadata=server_data.get("metadata", {})
                        )
                        
                        if manifest.enabled:
                            self.discovered_servers[manifest.name] = manifest
                            logger.info(f"  📋 Loaded from manifest: {manifest.name}")
                            
                except Exception as e:
                    logger.error(f"Error loading manifest {manifest_file}: {e}")
    
    async def load_server(self, server_name: str) -> Optional[Agent]:
        """Load a specific server/agent"""
        if server_name in self.loaded_servers:
            return self.loaded_servers[server_name]
        
        manifest = self.discovered_servers.get(server_name)
        if not manifest:
            logger.error(f"Server {server_name} not found")
            return None
        
        try:
            # Import the module
            module = importlib.import_module(manifest.module_path)
            
            # Get the class
            server_class = getattr(module, manifest.class_name)
            
            # Get configuration from environment and manifest
            config = self._build_server_config(manifest)
            
            # Create instance
            instance = server_class(name=server_name, config=config)
            
            # Register with agent registry
            agent_registry.register(server_name, instance)
            
            # Store reference
            self.loaded_servers[server_name] = instance
            
            logger.info(f"✅ Loaded server: {server_name}")
            
            # Log to PsiArchive
            psi_archive.log_event("server_loaded", {
                "server": server_name,
                "class": manifest.class_name,
                "config": config
            })
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load server {server_name}: {e}")
            psi_archive.log_event("server_load_failed", {
                "server": server_name,
                "error": str(e)
            })
            return None
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific server if it has a start method"""
        server = await self.load_server(server_name)
        if not server:
            return False
        
        manifest = self.discovered_servers[server_name]
        
        try:
            # Check for start method
            if hasattr(server, 'start'):
                logger.info(f"🚀 Starting server: {server_name}")
                
                # Create task for async start
                if inspect.iscoroutinefunction(server.start):
                    task = asyncio.create_task(server.start())
                    self.server_tasks[server_name] = task
                else:
                    server.start()
                
                psi_archive.log_event("server_started", {
                    "server": server_name
                })
                
            # Special handling for Kaizen - start continuous improvement
            if hasattr(server, 'start_continuous_improvement'):
                if manifest.auto_start:
                    logger.info(f"🔄 Starting continuous improvement for {server_name}")
                    await server.start_continuous_improvement()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            return False
    
    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all discovered servers that are enabled and auto-start"""
        results = {}
        
        # Sort by dependencies if specified
        sorted_servers = self._sort_by_dependencies()
        
        for server_name in sorted_servers:
            manifest = self.discovered_servers[server_name]
            
            if manifest.enabled and manifest.auto_start:
                logger.info(f"Starting {server_name}...")
                success = await self.start_server(server_name)
                results[server_name] = success
                
                # Small delay between starts
                await asyncio.sleep(0.5)
        
        return results
    
    def _sort_by_dependencies(self) -> List[str]:
        """Sort servers by dependencies"""
        # Simple topological sort
        sorted_list = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            manifest = self.discovered_servers.get(name)
            if manifest:
                for dep in manifest.dependencies:
                    if dep in self.discovered_servers:
                        visit(dep)
            
            sorted_list.append(name)
        
        for server_name in self.discovered_servers:
            visit(server_name)
        
        return sorted_list
    
    def _build_server_config(self, manifest: ServerManifest) -> Dict[str, Any]:
        """Build configuration from manifest and environment"""
        config = manifest.config.copy()
        
        # Override with environment variables
        env_prefix = f"{manifest.name.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                
                # Try to parse JSON values
                try:
                    config[config_key] = json.loads(value)
                except:
                    config[config_key] = value
        
        return config
    
    async def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers"""
        status = {}
        
        for server_name, manifest in self.discovered_servers.items():
            server_status = {
                "discovered": True,
                "enabled": manifest.enabled,
                "auto_start": manifest.auto_start,
                "loaded": server_name in self.loaded_servers,
                "running": False,
                "description": manifest.description,
                "endpoints": manifest.endpoints
            }
            
            # Check if server is running
            if server_name in self.loaded_servers:
                server = self.loaded_servers[server_name]
                
                # Check for is_running attribute
                if hasattr(server, 'is_running'):
                    server_status["running"] = server.is_running
                elif server_name in self.server_tasks:
                    task = self.server_tasks[server_name]
                    server_status["running"] = not task.done()
                else:
                    server_status["running"] = True  # Assume running if loaded
                
                # Get server-specific status if available
                if hasattr(server, 'get_status'):
                    try:
                        if inspect.iscoroutinefunction(server.get_status):
                            server_status["details"] = await server.get_status()
                        else:
                            server_status["details"] = server.get_status()
                    except:
                        pass
            
            status[server_name] = server_status
        
        return status
    
    async def stop_all_servers(self):
        """Stop all running servers"""
        for server_name, server in self.loaded_servers.items():
            try:
                logger.info(f"🛑 Stopping server: {server_name}")
                
                # Call shutdown if available
                if hasattr(server, 'shutdown'):
                    if inspect.iscoroutinefunction(server.shutdown):
                        await server.shutdown()
                    else:
                        server.shutdown()
                
                # Cancel any running tasks
                if server_name in self.server_tasks:
                    task = self.server_tasks[server_name]
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                psi_archive.log_event("server_stopped", {
                    "server": server_name
                })
                
            except Exception as e:
                logger.error(f"Error stopping server {server_name}: {e}")

# Global discovery instance
server_discovery = DynamicServerDiscovery()

# Export
__all__ = ['DynamicServerDiscovery', 'server_discovery', 'ServerManifest']
