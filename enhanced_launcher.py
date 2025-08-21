# ENHANCED TORI LAUNCHER - BULLETPROOF EDITION v3.0
# D:\Dev\kha\enhanced_launcher.py
import os, sys, logging, argparse, subprocess, signal, time, atexit
from pathlib import Path
import threading, json, urllib.request

try:
    from port_manager import port_manager
except Exception as e:
    # Fallback if port_manager import fails (should not in production)
    class _DummyPM:
        def get_service_port(self, service, default_port=None):
            return int(os.environ.get(f"{service.upper()}_PORT", default_port or 8002))
        def cleanup_all_ports(self): pass
    port_manager = _DummyPM()

# Set production memory vault intervals
os.environ["TORI_VAULT_SNAPSHOT_INTERVAL_SEC"] = "300"   # 5 min snapshots
os.environ["TORI_VAULT_SESSION_ROTATE_SEC"] = "3600"      # 1 hour rotation

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("launcher")

# Warning suppression (keep noise down)
import warnings
for _pat in (".*already exists.*", ".*shadows an attribute.*", ".*0 concepts.*"):
    warnings.filterwarnings("ignore", message=_pat)

def wait_http(url: str, timeout: float = 45.0, interval: float = 0.5):
    """Wait until a simple HTTP GET returns 2xx."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            try:
                import requests
                r = requests.get(url, timeout=2)
                if r.ok:
                    return
            except Exception:
                # urllib fallback (no requests)
                from urllib.request import urlopen
                with urlopen(url, timeout=2) as resp:
                    if 200 <= getattr(resp, "status", 200) < 300:
                        return
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"Timeout waiting for {url}")

def _poll_memory_health(host: str, api_port: int, user_id: str = "me", interval_s: float = 5.0):
    """Poll memory health metrics and log status"""
    url = f"http://{host}:{api_port}/api/memory/state/{user_id}"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                obj = json.loads(r.read().decode("utf-8"))
                status = obj.get("status", "unknown")
                reasons = ",".join(obj.get("status_reasons", []))
                C = obj.get("coherence")
                E = obj.get("energy")
                Lver = obj.get("laplacian_version")
                if C is not None and E is not None:
                    print(f"[MEM] status={status} reasons={reasons} | C={C:.3f} E={E:.6f} Lver={Lver}")
                else:
                    print(f"[MEM] status={status} reasons={reasons} | Lver={Lver}")
        except Exception:
            # keep quiet but persistent in face of API restarts
            pass
        time.sleep(interval_s)

class Launcher:
    def __init__(self, api_port=None, ui_port=None, mcp_port=None, api_only=False, debug=False):
        self.api_port = api_port or port_manager.get_service_port("api", 8002)
        self.ui_port  = ui_port  or port_manager.get_service_port("ui", 3000)
        self.mcp_port = mcp_port or port_manager.get_service_port("mcp", 6660)
        self.api_only = api_only
        self.debug = debug
        self.processes = []
        atexit.register(self.shutdown)

    def _creationflags(self):
        if os.name == "nt":
            return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return 0

    def start(self):
        LOG.info("LAUNCH v3.0 - ports: api=%s ui=%s mcp=%s", self.api_port, self.ui_port, self.mcp_port)
        self.start_api()
        if not self.api_only:
            self.start_ui()
        LOG.info("All services launched.")
        
        # Start memory health polling after API is ready
        try:
            threading.Thread(
                target=_poll_memory_health, 
                args=("127.0.0.1", self.api_port, "me", 5.0), 
                daemon=True
            ).start()
            LOG.info("Memory health monitoring started")
        except Exception as e:
            LOG.warning(f"Could not start memory health monitoring: {e}")
        
        # Keep the launcher running
        self.wait_for_processes()

    def start_api(self):
        env = os.environ.copy()
        env["API_PORT"] = str(self.api_port)
        cmd = [sys.executable, "-m", "uvicorn", "api.main:app",
               "--host", "0.0.0.0", "--port", str(self.api_port)]
        LOG.info("Starting API: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env, creationflags=self._creationflags())
        self.processes.append(proc)
        wait_http(f"http://127.0.0.1:{self.api_port}/health", timeout=45.0)
        LOG.info("API healthy at http://127.0.0.1:%s/health", self.api_port)

    def start_ui(self):
        repo = Path(__file__).resolve().parent
        # Prefer built server if present, else dev server
        build_index = repo / "tori_ui_svelte" / "build" / "index.js"
        if build_index.exists():
            cmd = ["node", str(build_index)]
            env = os.environ.copy()
            env["PORT"] = str(self.ui_port)
            LOG.info("Starting UI (build): node %s (PORT=%s)", build_index, self.ui_port)
            proc = subprocess.Popen(cmd, env=env, creationflags=self._creationflags())
        else:
            # SvelteKit dev: pnpm -C tori_ui_svelte dev -- --port <ui_port>
            cmd = ["pnpm", "-C", "tori_ui_svelte", "dev", "--", "--port", str(self.ui_port)]
            LOG.info("Starting UI (dev): %s", " ".join(cmd))
            proc = subprocess.Popen(cmd, creationflags=self._creationflags())
        self.processes.append(proc)

    def wait_for_processes(self):
        """Wait for all processes to complete or until interrupted"""
        try:
            while True:
                # Check if any process has terminated
                for proc in self.processes:
                    if proc.poll() is not None:
                        LOG.warning(f"Process {proc.pid} terminated with code {proc.returncode}")
                        # Remove terminated process
                        self.processes.remove(proc)
                        
                # If all processes are dead, exit
                if not self.processes:
                    LOG.warning("All processes have terminated")
                    break
                    
                # Sleep a bit to avoid busy waiting
                time.sleep(1)
        except KeyboardInterrupt:
            LOG.info("Interrupt received in wait loop")
            raise
    
    def shutdown(self):
        LOG.info("Shutting down launcher...")
        for p in self.processes:
            try:
                if os.name == "nt":
                    try:
                        p.send_signal(signal.CTRL_BREAK_EVENT)  # requires CREATE_NEW_PROCESS_GROUP
                    except Exception:
                        p.terminate()
                else:
                    p.terminate()
            except Exception:
                pass
        for p in self.processes:
            try:
                p.wait(timeout=8)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        try:
            port_manager.cleanup_all_ports()
        except Exception:
            pass
        LOG.info("Shutdown complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Enhanced TORI Launcher v3.0")
    ap.add_argument("--port", "--api-port", dest="api_port", type=int, default=None, help="API port (default 8002)")
    ap.add_argument("--ui-port", dest="ui_port", type=int, default=None, help="UI port (default 3000)")
    ap.add_argument("--mcp-port", dest="mcp_port", type=int, default=None, help="MCP port (default 6660)")
    ap.add_argument("--api-only", action="store_true", help="Start API only (no UI)")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = ap.parse_args()
    
    launcher = Launcher(api_port=args.api_port, ui_port=args.ui_port, mcp_port=args.mcp_port,
                       api_only=args.api_only, debug=args.debug)
    try:
        launcher.start()
    except KeyboardInterrupt:
        LOG.info("Received interrupt signal")
    finally:
        launcher.shutdown()
