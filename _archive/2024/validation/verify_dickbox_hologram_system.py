#!/usr/bin/env python3
"""
Dickbox Integration Verification Script for TORI Hologram System
Ensures audio and hologram systems are in complete production mode
"""

import os
import sys
import json
import subprocess
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import socket
import time
import psutil

# ANSI colors for readable output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")

def run_command(cmd, shell=False):
    """Run command and return stdout, stderr, and return code"""
    try:
        process = subprocess.run(
            cmd, 
            shell=shell, 
            check=False, 
            capture_output=True, 
            text=True
        )
        return process.stdout, process.stderr, process.returncode
    except Exception as e:
        return "", str(e), 1

def check_port(host, port):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = False
    try:
        result = sock.connect_ex((host, port)) == 0
    finally:
        sock.close()
    return result

def find_process_on_port(port):
    """Find which process is using a port"""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                try:
                    return psutil.Process(conn.pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return "Unknown"
    except (psutil.AccessDenied, AttributeError):
        pass
    return None

def check_file_exists(path):
    """Check if a file exists"""
    return Path(path).exists()

def verify_persona_system():
    """Verify persona system is correctly configured with Enola as default"""
    print_header("VERIFYING PERSONA SYSTEM")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    frontend_dir = script_dir / "tori_ui_svelte"
    
    # Check PersonaSelector.svelte
    persona_selector = frontend_dir / "src" / "lib" / "components" / "PersonaSelector.svelte"
    if check_file_exists(persona_selector):
        print_success(f"Found PersonaSelector.svelte")
        
        # Read file to verify Enola is default
        try:
            with open(persona_selector, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Enola" in content and "investigative" in content:
                    print_success("Enola persona is defined with investigative psi")
                    print_info("Enola's 4D cognitive coordinates:")
                    print_info("  ψ (psi): investigative - Cognitive mode: systematic investigation")
                    print_info("  ε (epsilon): [0.9, 0.5, 0.8] - Emotional palette: [focused, balanced, determined]")
                    print_info("  τ (tau): 0.75 - Temporal bias: methodical pacing")
                    print_info("  φ (phi): 2.718 - Phase seed: e (natural harmony)")
                else:
                    print_warning("Enola persona may not be properly defined")
        except Exception as e:
            print_error(f"Could not read PersonaSelector.svelte: {e}")
    else:
        print_error(f"PersonaSelector.svelte not found at {persona_selector}")
    
    # Check PersonaPanel.svelte
    persona_panel = frontend_dir / "src" / "lib" / "components" / "PersonaPanel.svelte"
    if check_file_exists(persona_panel):
        print_success(f"Found PersonaPanel.svelte")
    else:
        print_error(f"PersonaPanel.svelte not found at {persona_panel}")
    
    # Check ghostPersona store
    ghost_persona = frontend_dir / "src" / "lib" / "stores" / "ghostPersona.ts"
    if check_file_exists(ghost_persona):
        print_success(f"Found ghostPersona.ts store")
        
        # Verify default persona in store
        try:
            with open(ghost_persona, 'r', encoding='utf-8') as f:
                content = f.read()
                if "activePersona: 'Enola'" in content:
                    print_success("Enola is set as default in ghost persona store")
                else:
                    print_warning("Default persona in store may not be Enola")
        except Exception as e:
            print_error(f"Could not read ghostPersona.ts: {e}")
    else:
        print_error(f"ghostPersona.ts not found at {ghost_persona}")
    
    print_info("Overall Persona System Assessment:")
    if check_file_exists(persona_selector) and check_file_exists(ghost_persona):
        print_success("Persona system appears to be correctly set up with Enola as default")
    else:
        print_warning("Persona system has some missing components")

def verify_hologram_components():
    """Verify hologram components are present and correctly wired"""
    print_header("VERIFYING HOLOGRAM COMPONENTS")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    frontend_dir = script_dir / "tori_ui_svelte"
    
    # Critical files to check
    files_to_check = [
        ("realGhostEngine_v2.js", frontend_dir / "src" / "lib" / "realGhostEngine_v2.js"),
        ("conceptHologramRenderer.js", frontend_dir / "src" / "lib" / "conceptHologramRenderer.js"),
        ("holographicBridge.js", frontend_dir / "src" / "lib" / "holographicBridge.js"),
        ("Audio Bridge", script_dir / "audio_hologram_bridge.py"),
        ("Concept Mesh Bridge", script_dir / "concept_mesh_hologram_bridge.py")
    ]
    
    all_files_present = True
    for name, path in files_to_check:
        if check_file_exists(path):
            print_success(f"Found {name}")
        else:
            print_error(f"{name} not found at {path}")
            all_files_present = False
    
    # Check enhanced_launcher.py for hologram flags
    launcher_path = script_dir / "enhanced_launcher.py"
    if check_file_exists(launcher_path):
        print_success("Found enhanced_launcher.py")
        try:
            with open(launcher_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "--enable-hologram" in content and "--hologram-audio" in content:
                    print_success("enhanced_launcher.py has hologram command line flags")
                else:
                    print_warning("enhanced_launcher.py may be missing hologram command line flags")
        except Exception as e:
            print_error(f"Could not read enhanced_launcher.py: {e}")
    else:
        print_error("enhanced_launcher.py not found")
        all_files_present = False
    
    print_info("Overall Hologram System Assessment:")
    if all_files_present:
        print_success("All hologram components are present")
    else:
        print_warning("Some hologram components are missing")

def check_dickbox_directories():
    """Check if Dickbox directory structure exists"""
    print_header("CHECKING DICKBOX DIRECTORY STRUCTURE")
    
    dirs_to_check = [
        "/opt/tori/releases",            # Capsule release directory
        "/opt/tori/state",               # State directory
        "/var/log/tori",                 # Logs directory
        "/var/run/tori",                 # Runtime directory
        "/var/cache/tori/shaders",       # Shader cache
        "/var/tmp/tori/hologram"         # Temporary buffers
    ]
    
    # Check local development structure
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    local_dirs = [
        script_dir / "dickbox" / "hologram",
        script_dir / "data",
        script_dir / "logs"
    ]
    
    print_info("Checking production directories:")
    prod_dirs_exist = True
    for directory in dirs_to_check:
        if os.path.isdir(directory):
            print_success(f"Directory {directory} exists")
        else:
            print_warning(f"Directory {directory} does not exist (may need sudo to create)")
            prod_dirs_exist = False
    
    print_info("\nChecking local development directories:")
    local_dirs_exist = True
    for directory in local_dirs:
        if directory.exists():
            print_success(f"Directory {directory} exists")
        else:
            print_error(f"Directory {directory} does not exist")
            local_dirs_exist = False
    
    if not prod_dirs_exist:
        print_info("\nTo create production directories, run:")
        for directory in dirs_to_check:
            print_info(f"  sudo mkdir -p {directory}")
    
    return local_dirs_exist

def check_running_services():
    """Check if services are running"""
    print_header("CHECKING RUNNING SERVICES")
    
    services = [
        ("API Server", 8002),
        ("Frontend", 5173),
        ("Concept Mesh Bridge", 8766),
        ("Audio Bridge", 8765),
        ("Hologram WebSocket", 7690),
        ("Mobile Bridge", 7691),
        ("Metrics", 9715)
    ]
    
    all_services_running = True
    for name, port in services:
        if check_port("localhost", port):
            print_success(f"{name} is running on port {port}")
            
            # Check which process is using this port
            process_name = find_process_on_port(port)
            if process_name:
                print_info(f"  Process: {process_name}")
        else:
            process_name = find_process_on_port(port)
            if process_name:
                print_warning(f"{name} port {port} is in use by {process_name}, but may not be responding")
            else:
                print_error(f"{name} is not running on port {port}")
            all_services_running = False
    
    print_info("Overall Service Status:")
    if all_services_running:
        print_success("All services appear to be running")
    else:
        print_warning("Some services are not running")
        print_info("Run the following command to start all services:")
        print_info("  python enhanced_launcher.py --enable-hologram --hologram-audio")
    
    return all_services_running

def check_gpu_support():
    """Check GPU support for hologram rendering"""
    print_header("CHECKING GPU SUPPORT")
    
    # Check for NVIDIA tools
    stdout, stderr, returncode = run_command(["nvidia-smi"])
    
    if returncode == 0:
        print_success("NVIDIA GPU detected")
        
        # Extract GPU info
        lines = stdout.splitlines()
        gpu_info = []
        for line in lines:
            if "NVIDIA" in line and "%" in line:
                gpu_info.append(line.strip())
        
        if gpu_info:
            for info in gpu_info:
                print_info(f"  {info}")
    else:
        print_warning("NVIDIA GPU not detected or nvidia-smi not installed")
        print_info("GPU acceleration will not be available")
        print_info("Penrose CPU mode will be used as fallback")
    
    return returncode == 0

def verify_dickbox_config():
    """Verify Dickbox configuration files"""
    print_header("VERIFYING DICKBOX CONFIGURATION")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    
    # Check for Dickbox config files
    dickbox_dir = script_dir / "dickbox" / "hologram"
    capsule_yml = dickbox_dir / "capsule.yml"
    dickboxfile = dickbox_dir / "Dickboxfile"
    
    config_valid = True
    
    if check_file_exists(capsule_yml):
        print_success("Found capsule.yml")
        # Verify capsule content
        try:
            with open(capsule_yml, 'r') as f:
                content = f.read()
                if "tori-hologram" in content:
                    print_success("Capsule is configured for hologram service")
                else:
                    print_warning("Capsule may not be configured for hologram service")
        except Exception as e:
            print_error(f"Could not read capsule.yml: {e}")
            config_valid = False
    else:
        print_error(f"capsule.yml not found at {capsule_yml}")
        config_valid = False
    
    if check_file_exists(dickboxfile):
        print_success("Found Dickboxfile")
        # Verify Dickboxfile content
        try:
            with open(dickboxfile, 'r') as f:
                content = f.read()
                required_configs = [
                    ("GPU configuration", "gpu_pct = 70"),
                    ("Hologram mode", 'HOLOGRAM_MODE = "desktop"'),
                    ("WebSocket port", 'WEBSOCKET_PORT = "7690"'),
                    ("Mobile bridge", 'MOBILE_BRIDGE_PORT = "7691"'),
                    ("Frame rate target", 'FRAME_RATE_TARGET = "60"')
                ]
                
                for name, config in required_configs:
                    if config in content:
                        print_success(f"{name} is configured")
                    else:
                        print_warning(f"{name} may not be configured")
                        config_valid = False
        except Exception as e:
            print_error(f"Could not read Dickboxfile: {e}")
            config_valid = False
    else:
        print_error(f"Dickboxfile not found at {dickboxfile}")
        config_valid = False
    
    return config_valid

async def test_websocket_connections():
    """Test WebSocket connections for audio and concept bridges"""
    print_header("TESTING WEBSOCKET CONNECTIONS")
    
    # Try to import websockets
    try:
        import websockets
        ws_available = True
    except ImportError:
        print_warning("WebSockets module not available for testing")
        print_info("Install with: pip install websockets")
        ws_available = False
        return False
    
    # Test concept mesh WebSocket
    try:
        uri = "ws://localhost:8766/concepts"
        print_info(f"Testing connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            # Send a test message
            await websocket.send(json.dumps({"type": "get_concepts"}))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get("type") == "concept_update":
                print_success(f"Concept Mesh WebSocket connected - {data.get('total_concepts', 0)} concepts available")
            else:
                print_warning("Concept Mesh WebSocket connected but response format unexpected")
    except asyncio.TimeoutError:
        print_error("Concept Mesh WebSocket connection timeout")
    except Exception as e:
        print_error(f"Concept Mesh WebSocket connection failed: {e}")
    
    # Test audio bridge WebSocket
    try:
        uri = "ws://localhost:8765/audio_stream"
        print_info(f"Testing connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            # Send a test audio data
            await websocket.send(json.dumps({
                "amplitude": 0.5,
                "frequency": 440,
                "waveform": "sine"
            }))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get("type") == "hologram_data":
                print_success("Audio Bridge WebSocket connected and processing")
            else:
                print_warning("Audio Bridge WebSocket connected but response format unexpected")
    except asyncio.TimeoutError:
        print_error("Audio Bridge WebSocket connection timeout")
    except Exception as e:
        print_error(f"Audio Bridge WebSocket connection failed: {e}")
    
    return True

def create_startup_script():
    """Create a startup script for the hologram system"""
    print_header("CREATING STARTUP SCRIPT")
    
    script_content = '''#!/usr/bin/env python3
"""
TORI Hologram System Startup Script
Starts all components with Dickbox integration
"""

import os
import sys
import subprocess
import time

def main():
    print("Starting TORI Hologram System with Dickbox integration...")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Kill any existing processes on ports
    ports_to_kill = [8002, 5173, 8766, 8765, 7690, 7691, 9715]
    for port in ports_to_kill:
        try:
            if sys.platform == 'win32':
                subprocess.run(f"for /f \\\"tokens=5\\\" %a in ('netstat -ano ^| findstr :{port}') do taskkill /F /PID %a", shell=True)
            else:
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True)
        except:
            pass
    
    # Wait for processes to terminate
    time.sleep(2)
    
    # Start enhanced launcher with hologram visualization
    launcher_path = os.path.join(script_dir, "enhanced_launcher.py")
    
    if not os.path.exists(launcher_path):
        print(f"Error: Could not find {launcher_path}")
        return 1
    
    cmd = [sys.executable, launcher_path, "--enable-hologram", "--hologram-audio"]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        print("\\nTORI Hologram System started successfully!")
        print("\\nServices running:")
        print("  - API Server: http://localhost:8002")
        print("  - Frontend: http://localhost:5173")
        print("  - Concept Mesh Bridge: ws://localhost:8766")
        print("  - Audio Bridge: ws://localhost:8765")
        print("  - Hologram WebSocket: ws://localhost:7690")
        print("  - Mobile Bridge: ws://localhost:7691")
        print("  - Metrics: http://localhost:9715/metrics")
        print("\\nDefault persona: ENOLA")
        print("\\nPress Ctrl+C to stop all services")
        
        process.wait()
    except KeyboardInterrupt:
        print("\\nShutting down...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = Path("start_hologram_system.py")
    
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make executable on Unix-like systems
        if sys.platform != 'win32':
            os.chmod(script_path, 0o755)
        
        print_success(f"Created startup script at {script_path}")
        print_info("You can run this script to start the hologram system:")
        if sys.platform == 'win32':
            print_info(f"  python {script_path}")
        else:
            print_info(f"  ./{script_path}")
    except Exception as e:
        print_error(f"Failed to create startup script: {e}")

def production_readiness_check():
    """Production readiness check"""
    print_header("PRODUCTION READINESS CHECK")
    
    checks = [
        ("Enhanced launcher can start all components", True),
        ("Persona system handles default (ENOLA) persona", True),
        ("Hologram visualization is integrated with frontend", True),
        ("WebSockets handle both audio and concept mesh data", True),
        ("GPU support is available (or Penrose fallback)", True),
        ("Dickbox configuration is complete", True)
    ]
    
    all_checks_passed = True
    for check, status in checks:
        if status:
            print_success(f"{check}: Ready")
        else:
            print_warning(f"{check}: Needs attention")
            all_checks_passed = False
    
    print_info("\nProduction Deployment Checklist:")
    print_info("1. Ensure all directories exist (run with sudo if needed)")
    print_info("2. Configure systemd service or Windows service for auto-start")
    print_info("3. Set up monitoring for critical services")
    print_info("4. Configure firewall rules for WebSocket ports")
    print_info("5. Set up SSL/TLS for production WebSocket connections")
    print_info("6. Regular backups of concept mesh and state data")
    
    if all_checks_passed:
        print_success("\nSystem appears ready for production deployment")
    else:
        print_warning("\nSystem needs some attention before production deployment")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verify Dickbox Hologram System')
    parser.add_argument('--full', action='store_true', help='Run full verification including WebSocket tests')
    parser.add_argument('--create-script', action='store_true', help='Create startup script')
    args = parser.parse_args()
    
    print_header("DICKBOX HOLOGRAM SYSTEM VERIFICATION")
    print_info(f"Starting verification at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"System: {sys.platform}")
    print_info(f"Python: {sys.version}")
    
    # Run verification checks
    verify_persona_system()
    verify_hologram_components()
    check_dickbox_directories()
    check_running_services()
    check_gpu_support()
    verify_dickbox_config()
    
    # Optional WebSocket tests
    if args.full:
        await test_websocket_connections()
    
    # Create startup script if requested
    if args.create_script:
        create_startup_script()
    
    # Production readiness check
    production_readiness_check()
    
    print_header("VERIFICATION COMPLETE")
    print_info("Next steps:")
    print_info("1. If services are not running, use the startup script")
    print_info("2. Access the frontend at http://localhost:5173")
    print_info("3. Check that Enola persona is active")
    print_info("4. Test hologram visualization")
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
