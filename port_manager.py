"""
Port Manager Module
Dynamic port allocation and management for TORI services
"""

import socket
import json
import os
from pathlib import Path
from typing import Set, Dict, Optional
from contextlib import closing
import logging
import atexit

LOG = logging.getLogger(__name__)


class PortManager:
    """Manages dynamic port allocation and cleanup"""
    
    def __init__(self, state_file: Optional[str] = None):
        """Initialize the port manager"""
        self.allocated_ports: Set[int] = set()
        self.service_ports: Dict[str, int] = {}
        
        if state_file:
            self.state_file = Path(state_file)
        else:
            self.state_file = Path("D:/Dev/kha/.port_manager_state.json")
            
        self.load_state()
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all_ports)
        
    def load_state(self):
        """Load persisted port state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.allocated_ports = set(state.get("allocated_ports", []))
                    self.service_ports = state.get("service_ports", {})
                    LOG.debug("Loaded port state: %d allocated ports", len(self.allocated_ports))
            except Exception as e:
                LOG.warning("Failed to load port state: %s", e)
                
    def save_state(self):
        """Save port state to disk"""
        try:
            state = {
                "allocated_ports": list(self.allocated_ports),
                "service_ports": self.service_ports
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            LOG.error("Failed to save port state: %s", e)
            
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding"""
        if port in self.allocated_ports:
            return False
            
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return True
            except OSError:
                return False
                
    def find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> Optional[int]:
        """Find a free port in the given range"""
        for port in range(start_port, end_port):
            if port not in self.allocated_ports and self.is_port_available(port):
                return port
        return None
        
    def allocate_port(self, service: Optional[str] = None, preferred_port: Optional[int] = None) -> int:
        """Allocate a port for a service"""
        # Try preferred port first
        if preferred_port and self.is_port_available(preferred_port):
            port = preferred_port
        else:
            # Find a free port
            port = self.find_free_port()
            if port is None:
                raise RuntimeError("No free ports available")
                
        self.allocated_ports.add(port)
        if service:
            self.service_ports[service] = port
            
        self.save_state()
        LOG.info("Allocated port %d%s", port, f" for {service}" if service else "")
        return port
        
    def release_port(self, port: int):
        """Release an allocated port"""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            
            # Remove from service ports
            services_to_remove = [s for s, p in self.service_ports.items() if p == port]
            for service in services_to_remove:
                del self.service_ports[service]
                
            self.save_state()
            LOG.info("Released port %d", port)
            
    def release_service_port(self, service: str):
        """Release port allocated to a specific service"""
        if service in self.service_ports:
            port = self.service_ports[service]
            self.release_port(port)
            
    def get_service_port(self, service: str, default_port: Optional[int] = None) -> int:
        """Get or allocate a port for a service"""
        if service in self.service_ports:
            port = self.service_ports[service]
            # Verify it's still available
            if self.is_port_available(port):
                return port
            else:
                LOG.warning("Previously allocated port %d for %s is now in use", port, service)
                
        # Allocate a new port
        return self.allocate_port(service, default_port)
        
    def cleanup_all_ports(self):
        """Clean up all allocated ports"""
        LOG.info("Cleaning up %d allocated ports", len(self.allocated_ports))
        
        # Kill processes using our ports (Windows specific)
        if os.name == 'nt':
            for port in self.allocated_ports:
                self._kill_port_process_windows(port)
                
        self.allocated_ports.clear()
        self.service_ports.clear()
        self.save_state()
        
    def _kill_port_process_windows(self, port: int):
        """Kill process using a port on Windows"""
        try:
            import subprocess
            # Find process using the port
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                check=False
            )
            
            for line in result.stdout.splitlines():
                if f':{port} ' in line and 'LISTENING' in line:
                    # Extract PID
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid], check=False)
                            LOG.debug("Killed process %s using port %d", pid, port)
                        except Exception:
                            pass
        except Exception as e:
            LOG.debug("Failed to kill port process: %s", e)
            
    def get_allocated_ports(self) -> Set[int]:
        """Get all currently allocated ports"""
        return self.allocated_ports.copy()
        
    def get_service_ports_map(self) -> Dict[str, int]:
        """Get mapping of services to ports"""
        return self.service_ports.copy()
        
    def __str__(self) -> str:
        """String representation"""
        lines = ["Port Manager Status:"]
        lines.append(f"  Allocated ports: {sorted(self.allocated_ports)}")
        if self.service_ports:
            lines.append("  Service ports:")
            for service, port in sorted(self.service_ports.items()):
                lines.append(f"    {service}: {port}")
        return "\n".join(lines)


# Global singleton instance
port_manager = PortManager()


if __name__ == "__main__":
    # Test the port manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Port Manager")
    print("-" * 40)
    
    # Allocate some ports
    api_port = port_manager.allocate_port("api", 8002)
    print(f"API port: {api_port}")
    
    ui_port = port_manager.allocate_port("ui", 3000)
    print(f"UI port: {ui_port}")
    
    mcp_port = port_manager.allocate_port("mcp", 6660)
    print(f"MCP port: {mcp_port}")
    
    print("\n" + str(port_manager))
    
    # Test service port retrieval
    print(f"\nRetrieving API port: {port_manager.get_service_port('api')}")
    
    # Test cleanup
    print("\nCleaning up...")
    port_manager.cleanup_all_ports()
    print(str(port_manager))
