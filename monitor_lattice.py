"""
Oscillator Lattice Monitor

This script continuously monitors the oscillator lattice status
to help diagnose and confirm the fix is working.
"""

import requests
import time
import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8002"
CHECK_INTERVAL = 5  # seconds
MAX_CHECKS = 20
DIAGNOSTICS_FILE = Path(__file__).parent / "lattice_diagnostics.log"

def check_lattice_status():
    """Check lattice status and return a dictionary of relevant info."""
    try:
        response = requests.get(f"{API_BASE}/api/lattice/snapshot")
        if response.status_code == 200:
            data = response.json()
            summary = data.get("summary", {})
            return {
                "oscillators": summary.get("oscillators", 0),
                "concept_oscillators": summary.get("concept_oscillators", 0),
                "R": summary.get("R", 0.0),
                "H": summary.get("H", 0.0),
                "status": "success"
            }
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": f"Failed to get lattice status: {response.status_code}"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_concept_mesh_stats():
    """Check concept mesh statistics."""
    try:
        response = requests.get(f"{API_BASE}/api/concept_mesh/stats")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def rebuild_lattice():
    """Trigger lattice rebuild."""
    try:
        response = requests.post(f"{API_BASE}/api/lattice/rebuild")
        return response.status_code == 200
    except:
        return False

def check_environment_variables():
    """Check relevant environment variables."""
    env_vars = {
        "TORI_ENABLE_ENTROPY_PRUNING": os.environ.get("TORI_ENABLE_ENTROPY_PRUNING", "Not set"),
        "ENABLE_ENTROPY_PRUNING": os.environ.get("ENABLE_ENTROPY_PRUNING", "Not set"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "Not set"),
        "ENABLE_EMOJI_LOGS": os.environ.get("ENABLE_EMOJI_LOGS", "Not set"),
        "MAX_CONCEPT_OSCILLATORS": os.environ.get("MAX_CONCEPT_OSCILLATORS", "Not set")
    }
    return env_vars

def save_diagnostics(data):
    """Save diagnostic information to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DIAGNOSTICS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {json.dumps(data)}\n")

def main():
    """Main monitoring function."""
    print(f"=== OSCILLATOR LATTICE MONITOR ===")
    print(f"Monitoring {API_BASE}/api/lattice/snapshot")
    print(f"Log file: {DIAGNOSTICS_FILE}")
    print("\nChecking environment variables:")
    env_vars = check_environment_variables()
    for var, value in env_vars.items():
        print(f"  {var}: {value}")
    
    print("\nChecking concept mesh stats:")
    mesh_stats = check_concept_mesh_stats()
    print(json.dumps(mesh_stats, indent=2))
    
    print("\nMonitoring lattice status:")
    print(f"{'Time':<10} | {'Oscillators':<12} | {'Concept Osc':<12} | {'R Value':<10} | {'H Value':<10}")
    print("-" * 65)
    
    for i in range(MAX_CHECKS):
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = check_lattice_status()
        
        if status.get("status") == "success":
            print(f"{timestamp} | {status.get('oscillators', 0):<12} | {status.get('concept_oscillators', 0):<12} | {status.get('R', 0.0):<10.3f} | {status.get('H', 0.0):<10.3f}")
            
            # Save diagnostics to file
            diagnostic_data = {
                "timestamp": timestamp,
                "lattice": status,
                "env_vars": env_vars,
                "check_number": i + 1
            }
            save_diagnostics(diagnostic_data)
            
            # If zero oscillators after a few checks, try to rebuild
            if i == 3 and status.get("oscillators", 0) == 0:
                print("\nZero oscillators detected, attempting to rebuild lattice...")
                success = rebuild_lattice()
                print(f"Rebuild request {'successful' if success else 'failed'}")
                print("-" * 65)
        else:
            print(f"{timestamp} | ERROR: {status.get('message', 'Unknown error')}")
        
        # Last check - don't sleep
        if i < MAX_CHECKS - 1:
            time.sleep(CHECK_INTERVAL)
    
    # Final stats
    print("\nFinal concept mesh stats:")
    mesh_stats = check_concept_mesh_stats()
    print(json.dumps(mesh_stats, indent=2))
    
    print("\nMonitoring complete!")
    print(f"Diagnostics saved to: {DIAGNOSTICS_FILE}")

if __name__ == "__main__":
    main()
