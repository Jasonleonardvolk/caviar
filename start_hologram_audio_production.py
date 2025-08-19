#!/usr/bin/env python3
"""
TORI Hologram Audio Production Startup
Quick script to get audio and hologram systems into complete production mode
"""

import os
import sys
import subprocess
import time
import asyncio
import json
from pathlib import Path

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║           TORI HOLOGRAPHIC AUDIO SYSTEM STARTUP               ║")
    print("║                  Production Mode Launcher                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

def print_status(message, status="info"):
    if status == "success":
        print(f"{Colors.GREEN}[✓] {message}{Colors.RESET}")
    elif status == "error":
        print(f"{Colors.RED}[✗] {message}{Colors.RESET}")
    elif status == "warning":
        print(f"{Colors.YELLOW}[!] {message}{Colors.RESET}")
    else:
        print(f"{Colors.CYAN}[→] {message}{Colors.RESET}")

def kill_existing_processes():
    """Kill any existing processes on our ports"""
    print_status("Cleaning up existing processes...")
    
    ports = [8002, 5173, 8765, 8766, 7690, 7691, 9715]
    
    for port in ports:
        try:
            if sys.platform == 'win32':
                # Windows command to kill process by port
                cmd = f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}\') do taskkill /F /PID %a'
                subprocess.run(cmd, shell=True, capture_output=True)
            else:
                # Unix command to kill process by port
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, capture_output=True)
        except:
            pass
    
    time.sleep(2)
    print_status("Process cleanup complete", "success")

def start_audio_bridge():
    """Start the audio hologram bridge"""
    print_status("Starting Audio Bridge on port 8765...")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    audio_bridge_path = script_dir / "audio_hologram_bridge.py"
    
    if not audio_bridge_path.exists():
        print_status(f"Audio bridge not found at {audio_bridge_path}", "error")
        return None
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(audio_bridge_path), "--host", "127.0.0.1", "--port", "8765"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        
        if process.poll() is None:
            print_status("Audio Bridge started successfully", "success")
            return process
        else:
            print_status("Audio Bridge failed to start", "error")
            return None
    except Exception as e:
        print_status(f"Error starting Audio Bridge: {e}", "error")
        return None

def start_concept_mesh_bridge():
    """Start the concept mesh hologram bridge"""
    print_status("Starting Concept Mesh Bridge on port 8766...")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    concept_bridge_path = script_dir / "concept_mesh_hologram_bridge.py"
    
    if not concept_bridge_path.exists():
        print_status(f"Concept bridge not found at {concept_bridge_path}", "error")
        return None
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(concept_bridge_path), "--host", "127.0.0.1", "--port", "8766"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        
        if process.poll() is None:
            print_status("Concept Mesh Bridge started successfully", "success")
            return process
        else:
            print_status("Concept Mesh Bridge failed to start", "error")
            return None
    except Exception as e:
        print_status(f"Error starting Concept Mesh Bridge: {e}", "error")
        return None

def start_enhanced_launcher():
    """Start the main enhanced launcher with hologram flags"""
    print_status("Starting Enhanced Launcher with hologram support...")
    
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    launcher_path = script_dir / "enhanced_launcher.py"
    
    if not launcher_path.exists():
        print_status(f"Enhanced launcher not found at {launcher_path}", "error")
        return None
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(launcher_path), "--enable-hologram", "--hologram-audio"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it more time to start up
        print_status("Waiting for services to initialize...")
        time.sleep(10)
        
        if process.poll() is None:
            print_status("Enhanced Launcher started successfully", "success")
            return process
        else:
            print_status("Enhanced Launcher failed to start", "error")
            stderr = process.stderr.read() if process.stderr else ""
            if stderr:
                print_status(f"Error output: {stderr}", "error")
            return None
    except Exception as e:
        print_status(f"Error starting Enhanced Launcher: {e}", "error")
        return None

def verify_services():
    """Verify all services are running"""
    print_status("\nVerifying service status...")
    
    import socket
    
    services = [
        ("API Server", "localhost", 8002),
        ("Frontend", "localhost", 5173),
        ("Audio Bridge", "localhost", 8765),
        ("Concept Mesh Bridge", "localhost", 8766),
        ("Hologram WebSocket", "localhost", 7690),
        ("Mobile Bridge", "localhost", 7691),
        ("Metrics", "localhost", 9715)
    ]
    
    all_running = True
    
    for name, host, port in services:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print_status(f"{name} is running on {host}:{port}", "success")
        else:
            print_status(f"{name} is NOT running on {host}:{port}", "error")
            all_running = False
    
    return all_running

def create_persona_verification_script():
    """Create a script to verify Enola persona is active"""
    script_content = '''
// Run this in the browser console to verify Enola is active
(function verifyEnola() {
    // Check if persona selector exists
    const personaSelector = document.querySelector('[class*="PersonaSelector"]');
    if (!personaSelector) {
        console.error("PersonaSelector not found!");
        return false;
    }
    
    // Check localStorage for saved persona
    const savedPersona = localStorage.getItem('tori-last-persona');
    console.log("Saved persona:", savedPersona);
    
    // Check current active persona from store
    if (window.__svelte__ && window.__svelte__.stores) {
        const stores = window.__svelte__.stores;
        const ghostPersona = stores.get('ghostPersona');
        if (ghostPersona) {
            const state = ghostPersona.get();
            console.log("Current persona state:", state);
            
            if (state.activePersona === 'Enola') {
                console.log("✓ Enola is active!");
                console.log("4D Coordinates:");
                console.log("  ψ: investigative");
                console.log("  ε: [0.9, 0.5, 0.8]");
                console.log("  τ: 0.75");
                console.log("  φ: 2.718");
                return true;
            } else {
                console.warn("! Active persona is", state.activePersona, "not Enola");
                // Try to set Enola
                ghostPersona.update(s => ({...s, activePersona: 'Enola', persona: 'Enola'}));
                console.log("→ Switched to Enola");
            }
        }
    }
    
    // Visual check
    const enolaButton = Array.from(document.querySelectorAll('button')).find(
        btn => btn.textContent.includes('Enola')
    );
    if (enolaButton) {
        const isActive = enolaButton.classList.toString().includes('ring-2') || 
                        enolaButton.classList.toString().includes('border-current');
        console.log("Enola button found, active:", isActive);
        if (!isActive) {
            enolaButton.click();
            console.log("→ Clicked Enola button");
        }
    }
    
    return true;
})();
'''
    
    print_status("\nBrowser verification script created!")
    print_status("After the frontend loads, open browser console and run this script to verify Enola persona")
    
    # Save to file
    script_path = Path("verify_enola_persona.js")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print_status(f"Script saved to: {script_path}", "success")

def main():
    """Main startup sequence"""
    print_banner()
    
    # Step 1: Kill existing processes
    kill_existing_processes()
    
    # Step 2: Start bridges
    audio_process = start_audio_bridge()
    concept_process = start_concept_mesh_bridge()
    
    if not audio_process or not concept_process:
        print_status("Failed to start bridge services", "error")
        return 1
    
    # Step 3: Start main launcher
    launcher_process = start_enhanced_launcher()
    
    if not launcher_process:
        print_status("Failed to start enhanced launcher", "error")
        # Clean up bridges
        if audio_process:
            audio_process.terminate()
        if concept_process:
            concept_process.terminate()
        return 1
    
    # Step 4: Verify services
    time.sleep(5)
    all_running = verify_services()
    
    if all_running:
        print_status("\n✨ All services started successfully! ✨", "success")
    else:
        print_status("\n⚠️  Some services failed to start", "warning")
    
    # Step 5: Create persona verification script
    create_persona_verification_script()
    
    # Print access information
    print(f"\n{Colors.BOLD}{Colors.BLUE}Access Points:{Colors.RESET}")
    print(f"{Colors.CYAN}  Frontend:          http://localhost:5173{Colors.RESET}")
    print(f"{Colors.CYAN}  API:               http://localhost:8002{Colors.RESET}")
    print(f"{Colors.CYAN}  Audio Bridge:      ws://localhost:8765{Colors.RESET}")
    print(f"{Colors.CYAN}  Concept Bridge:    ws://localhost:8766{Colors.RESET}")
    print(f"{Colors.CYAN}  Hologram Socket:   ws://localhost:7690{Colors.RESET}")
    print(f"{Colors.CYAN}  Metrics:           http://localhost:9715/metrics{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Default Persona: ENOLA (Investigative){Colors.RESET}")
    print(f"{Colors.CYAN}  ψ: investigative - Systematic investigation{Colors.RESET}")
    print(f"{Colors.CYAN}  ε: [0.9, 0.5, 0.8] - Focused, balanced, determined{Colors.RESET}")
    print(f"{Colors.CYAN}  τ: 0.75 - Methodical pacing{Colors.RESET}")
    print(f"{Colors.CYAN}  φ: 2.718 - Natural harmony (e){Colors.RESET}")
    
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.RESET}")
    
    try:
        # Keep running
        launcher_process.wait()
    except KeyboardInterrupt:
        print_status("\n\nShutting down services...", "warning")
        
        # Terminate all processes
        for process in [launcher_process, audio_process, concept_process]:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        print_status("All services stopped", "success")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
