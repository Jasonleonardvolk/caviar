# port_manager.py
import socket
import time
import json
import os
from typing import List, Dict
import psutil

class PortManager:
    def __init__(self, config_file: str = "port_config.json"):
        self.config_file = config_file
        self.allocated_ports = self._load_config()

    def _load_config(self) -> Dict[str, int]:
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_config(self) -> None:
        with open(self.config_file, 'w') as f:
            json.dump(self.allocated_ports, f, indent=2)

    def is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return True
            except OSError:
                return False

    def get_pids_using_port(self, port: int) -> List[int]:
        pids = []
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.pid:
                pids.append(conn.pid)
        return list(set(pids))

    def force_free_port(self, port: int, timeout: float = 5.0) -> bool:
        pids = self.get_pids_using_port(port)
        if not pids:
            return True
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                name = proc.name().lower()
                # Only kill recognized dev processes
                if "python" in name or "node" in name or "bridge" in name:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.5)
        return self.is_port_free(port)

    def find_free_port(self,
                       preferred: int,
                       service_name: str,
                       max_attempts: int = 50) -> int:
        # Try preferred
        if self.is_port_free(preferred) or self.force_free_port(preferred):
            self.allocated_ports[service_name] = preferred
            self._save_config()
            return preferred
        # Try next ones
        for p in range(preferred + 1, preferred + max_attempts):
            if self.is_port_free(p):
                self.allocated_ports[service_name] = p
                self._save_config()
                return p
        raise RuntimeError(f"No free port found in {preferred}–{preferred + max_attempts}")

    def cleanup_all_ports(self) -> None:
        for svc, port in self.allocated_ports.items():
            print(f"🧹 Cleaning up {svc} on port {port}")
            self.force_free_port(port)
        self.allocated_ports.clear()
        self._save_config()

# Global instance
port_manager = PortManager()
