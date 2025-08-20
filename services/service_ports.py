"""
Service Ports Management Module
Central source of truth for all service ports
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
import logging

LOG = logging.getLogger(__name__)


class ServicePorts:
    """Centralized management of service ports with persistence"""
    
    DEFAULT_PORTS = {
        "api": 8002,
        "ui": 3000,
        "mcp": 6660,
        "bridge_audio": 8501,
        "bridge_concept_mesh": 8502,
        "concept_mesh_visualization": 8503,
        "semantic_core": 8504,
        "knowledge_graph": 8505,
        "inference_engine": 8506
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize service ports manager"""
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path("D:/Dev/kha/bridge_config.json")
            
        self.ports = self.DEFAULT_PORTS.copy()
        self.load_config()
        
    def load_config(self) -> None:
        """Load port configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    stored_config = json.load(f)
                    # Update with stored values
                    for key in self.DEFAULT_PORTS:
                        if f"{key}_port" in stored_config:
                            self.ports[key] = stored_config[f"{key}_port"]
                        elif key in stored_config:
                            self.ports[key] = stored_config[key]
                            
                LOG.info("Loaded port configuration from %s", self.config_path)
            except Exception as e:
                LOG.warning("Failed to load port config: %s, using defaults", e)
                self.save_config()
        else:
            LOG.info("No config file found, creating with defaults")
            self.save_config()
            
    def save_config(self) -> None:
        """Save current port configuration"""
        config = {}
        for key, port in self.ports.items():
            config[f"{key}_port"] = port
            
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            LOG.info("Saved port configuration to %s", self.config_path)
        except Exception as e:
            LOG.error("Failed to save port config: %s", e)
            
    def get_port(self, service: str) -> int:
        """Get port for a specific service"""
        return self.ports.get(service, self.DEFAULT_PORTS.get(service, 8000))
        
    def set_port(self, service: str, port: int) -> None:
        """Set port for a specific service"""
        self.ports[service] = port
        self.save_config()
        
    def get_all_ports(self) -> Dict[str, int]:
        """Get all service ports"""
        return self.ports.copy()
        
    def get_api_url(self, service: str) -> str:
        """Get full API URL for a service"""
        port = self.get_port(service)
        return f"http://localhost:{port}"
        
    def get_health_url(self, service: str) -> str:
        """Get health check URL for a service"""
        return f"{self.get_api_url(service)}/health"
        
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return False
            except OSError:
                return True
                
    def find_free_port(self, start_port: int = 8000, max_tries: int = 100) -> int:
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_tries):
            if not self.is_port_in_use(port):
                return port
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_tries}")
        
    def allocate_dynamic_port(self, service: str) -> int:
        """Allocate a dynamic port for a service"""
        # Start searching from the default port
        default_port = self.DEFAULT_PORTS.get(service, 8000)
        port = self.find_free_port(default_port)
        self.set_port(service, port)
        return port
        
    def __str__(self) -> str:
        """String representation of service ports"""
        lines = ["Service Ports Configuration:"]
        for service, port in sorted(self.ports.items()):
            status = "IN USE" if self.is_port_in_use(port) else "FREE"
            lines.append(f"  {service:.<30} {port:5d} [{status}]")
        return "\n".join(lines)


# Global singleton instance
_service_ports_instance = None


def get_service_ports(config_path: Optional[str] = None) -> ServicePorts:
    """Get the global ServicePorts instance"""
    global _service_ports_instance
    if _service_ports_instance is None:
        _service_ports_instance = ServicePorts(config_path)
    return _service_ports_instance


# Convenience functions
def get_port(service: str) -> int:
    """Get port for a service"""
    return get_service_ports().get_port(service)


def get_api_url(service: str) -> str:
    """Get API URL for a service"""
    return get_service_ports().get_api_url(service)


def get_health_url(service: str) -> str:
    """Get health URL for a service"""
    return get_service_ports().get_health_url(service)


if __name__ == "__main__":
    # Test the service ports manager
    ports = ServicePorts()
    print(ports)
    
    # Test port allocation
    print(f"\nAPI Port: {ports.get_port('api')}")
    print(f"API URL: {ports.get_api_url('api')}")
    print(f"Health URL: {ports.get_health_url('api')}")
    
    # Test dynamic allocation
    if not ports.is_port_in_use(8888):
        ports.set_port("test_service", 8888)
        print(f"\nAllocated port 8888 to test_service")
    else:
        dynamic_port = ports.allocate_dynamic_port("test_service")
        print(f"\nDynamically allocated port {dynamic_port} to test_service")
        
    print(f"\n{ports}")
