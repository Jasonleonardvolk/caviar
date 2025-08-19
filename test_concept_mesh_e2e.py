#!/usr/bin/env python3
"""
Enhanced End-to-End Test for Concept Mesh System
With intelligent pre-flight checks, diagnostics, and actionable hints
"""

import asyncio
import json
import requests
import websockets
import time
import sys
import socket
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_AUDIO_URL = "ws://localhost:8765"
WS_CONCEPT_URL = "ws://localhost:8766"
TIMEOUT = 10

# Service status tracking
service_status = {
    "API": False,
    "Audio WebSocket": False,
    "Concept Mesh WebSocket": False,
    "Data Files": False,
    "Concept Mesh Loaded": False
}

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "skipped": []
}

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored(text: str, color: str) -> str:
    """Add color to text for terminal output"""
    if sys.platform == "win32":
        # Windows might not support ANSI colors without colorama
        return text
    return f"{color}{text}{Colors.ENDC}"

def print_test_header(test_name: str):
    """Print a test header"""
    print(f"\n{'='*60}")
    print(colored(f"TEST: {test_name}", Colors.HEADER))
    print(f"{'='*60}")

def record_result(test_name: str, success: bool, message: str = "", hint: str = ""):
    """Record test result with optional hint"""
    if success:
        test_results["passed"].append(test_name)
        print(colored(f"✅ PASS: {test_name}", Colors.OKGREEN))
    else:
        test_results["failed"].append((test_name, message))
        print(colored(f"❌ FAIL: {test_name}", Colors.FAIL))
        if message:
            print(f"   Error: {message}")
        
        # Add intelligent hints based on error
        if hint:
            print(colored(f"   HINT: {hint}", Colors.WARNING))
        elif "Connection refused" in message or "actively refused" in message:
            print(colored("   HINT: Service not running. Start with 'python enhanced_launcher.py'", Colors.WARNING))
        elif "not found" in message or "404" in message:
            print(colored("   HINT: Endpoint missing. Check API routes in enhanced_launcher.py", Colors.WARNING))
        elif "timeout" in message.lower():
            print(colored("   HINT: Service may be overloaded or hanging. Check logs/tori.log", Colors.WARNING))
        elif "parse" in message.lower() or "json" in message.lower():
            print(colored("   HINT: Data corruption. Check file format and encoding", Colors.WARNING))

def record_warning(test_name: str, message: str):
    """Record a warning"""
    test_results["warnings"].append((test_name, message))
    print(colored(f"⚠️  WARNING: {message}", Colors.WARNING))

def record_skip(test_name: str, reason: str):
    """Record a skipped test"""
    test_results["skipped"].append((test_name, reason))
    print(colored(f"⏭️  SKIP: {test_name} - {reason}", Colors.OKCYAN))

# Pre-flight checks
def check_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        return result == 0

def check_process_running(process_name: str) -> bool:
    """Check if a process is running (Windows/Unix compatible)"""
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            if process_name.lower() in proc.info['name'].lower():
                return True
    except ImportError:
        # Fallback if psutil not available
        pass
    return False

def check_data_directories() -> Dict[str, bool]:
    """Check for required data directories and files"""
    required_paths = {
        "data/": "Main data directory",
        "data/concepts.json": "Concept database",
        "logs/": "Log directory",
        ".venv/": "Virtual environment"
    }
    
    status = {}
    missing = []
    
    print("\nChecking data directories and files...")
    for path, description in required_paths.items():
        exists = Path(path).exists()
        status[path] = exists
        if exists:
            print(f"  ✅ {path} - {description}")
        else:
            print(colored(f"  ❌ {path} - {description} MISSING", Colors.FAIL))
            missing.append(path)
    
    if missing:
        print(colored("\nMissing required files/directories:", Colors.FAIL))
        for path in missing:
            if path.endswith('/'):
                print(f"  Create with: mkdir -p {path}")
            else:
                print(f"  Missing file: {path}")
                if "concepts.json" in path:
                    print("  Initialize with: python init_concept_mesh_data.py")
    
    return status

def preflight_check(force: bool = False) -> Dict[str, bool]:
    """Comprehensive pre-flight system check"""
    print("\n" + "="*60)
    print(colored("TORI SYSTEM PRE-FLIGHT CHECK", Colors.BOLD))
    print("="*60)
    
    # Check ports
    ports = {
        "API": (8000, "enhanced_launcher.py"),
        "Audio WebSocket": (8765, "audio_hologram_bridge.py"),
        "Concept Mesh WebSocket": (8766, "concept_mesh_hologram_bridge.py"),
    }
    
    status = {}
    all_good = True
    
    print("\nPort Status:")
    print("-" * 40)
    for name, (port, script) in ports.items():
        is_open = check_port('localhost', port)
        status[name] = is_open
        service_status[name] = is_open
        
        if is_open:
            print(f"{name:<24} Port {port}: " + colored("OPEN", Colors.OKGREEN))
        else:
            print(f"{name:<24} Port {port}: " + colored("CLOSED", Colors.FAIL))
            print(colored(f"  → Start with: python {script}", Colors.WARNING))
            all_good = False
    
    # Check critical services
    print("\nCritical Service Check:")
    print("-" * 40)
    
    if not status["API"]:
        print(colored("❌ API is NOT running on port 8000", Colors.FAIL))
        print(colored("   This is required for all tests.", Colors.WARNING))
        print(colored("   Start with: python enhanced_launcher.py", Colors.WARNING))
        
        if not force:
            print(colored("\nAborting tests. Use --force to run anyway.", Colors.FAIL))
            return None
    else:
        print(colored("✅ API is running", Colors.OKGREEN))
    
    # Check data directories
    print("\nData Directory Check:")
    print("-" * 40)
    dir_status = check_data_directories()
    service_status["Data Files"] = any(dir_status.values())
    
    # Alternative ports check
    print("\nAlternative Port Check:")
    print("-" * 40)
    alt_ports = [8002, 3000, 5173]
    for port in alt_ports:
        if check_port('localhost', port):
            print(colored(f"  ℹ️  Port {port} is also open - might be frontend or alt API", Colors.OKCYAN))
    
    print("\n" + "-"*60)
    
    if not all_good and not force:
        print(colored("\n⚠️  Pre-flight checks failed. Fix issues above and retry.", Colors.FAIL))
        print("Use --force to run tests anyway (not recommended)")
        return None
    
    return status

def detect_duplicate_files():
    """Detect duplicate concept data files"""
    concept_patterns = [
        "concepts.json",
        "concept_db.json",
        "concept_mesh_data.json",
        "soliton_concept_memory.json"
    ]
    
    found_files = defaultdict(list)
    
    # Search in common locations
    search_paths = [".", "data", "config", "python/core/data"]
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            for pattern in concept_patterns:
                for file in path.glob(f"**/{pattern}"):
                    found_files[pattern].append(file)
    
    # Report duplicates
    duplicates = {k: v for k, v in found_files.items() if len(v) > 1}
    if duplicates:
        print(colored("\n⚠️  Duplicate concept files detected:", Colors.WARNING))
        for pattern, files in duplicates.items():
            print(f"  {pattern} found in:")
            for file in files:
                print(f"    - {file}")
        print(colored("  This may cause inconsistent behavior!", Colors.WARNING))
    
    return found_files

# Enhanced API tests with better diagnostics
def test_api_health() -> bool:
    """Test API health endpoints with detailed diagnostics"""
    print_test_header("API Health Check")
    
    if not service_status.get("API"):
        record_skip("API Health", "API service not running")
        return False
    
    endpoints = [
        ("/health", "Basic health"),
        ("/api/health", "API health"),
        ("/api/v1/health", "Versioned health")
    ]
    
    success = False
    working_endpoints = []
    
    for endpoint, description in endpoints:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            response = requests.get(url, timeout=TIMEOUT)
            
            if response.status_code == 200:
                working_endpoints.append(endpoint)
                record_result(f"API Health - {endpoint}", True)
                
                # Try to parse response
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        status = data.get("status", "unknown")
                        print(f"   Status: {status}")
                        if "services" in data:
                            print(f"   Services: {', '.join(data['services'])}")
                except:
                    print(f"   Response: {response.text[:100]}...")
                
                success = True
            else:
                record_result(f"API Health - {endpoint}", False, 
                            f"Status {response.status_code}",
                            "Check API route configuration")
        except requests.exceptions.ConnectionError:
            record_result(f"API Health - {endpoint}", False,
                        "Connection refused",
                        "API not running on expected port")
        except requests.exceptions.Timeout:
            record_result(f"API Health - {endpoint}", False,
                        "Request timeout",
                        "API may be hanging - check logs")
        except Exception as e:
            record_result(f"API Health - {endpoint}", False, str(e))
    
    if working_endpoints:
        print(colored(f"\n✅ Working endpoints: {', '.join(working_endpoints)}", Colors.OKGREEN))
    else:
        print(colored("\n❌ No health endpoints responding", Colors.FAIL))
        print(colored("   Check if API is bound to correct port", Colors.WARNING))
    
    return success

# Enhanced concept mesh tests
def test_concept_mesh_api():
    """Test concept mesh API endpoints with consistency checks"""
    print_test_header("Concept Mesh API")
    
    if not service_status.get("API"):
        record_skip("Concept Mesh API", "API service not running")
        return
    
    # Test endpoints with expected response types
    endpoints = [
        ("/api/v1/concepts", "GET", None, "list"),
        ("/api/v1/concept-mesh/status", "GET", None, "dict"),
        ("/api/v1/concept-mesh/concepts", "GET", None, "list_or_dict"),
        ("/api/v1/soliton/concepts", "GET", None, "list"),
    ]
    
    concept_counts = {}
    responses = {}
    
    for endpoint, method, data, expected_type in endpoints:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=TIMEOUT)
            
            if response.status_code in [200, 201]:
                record_result(f"Concept API - {endpoint}", True)
                
                try:
                    data = response.json()
                    responses[endpoint] = data
                    
                    # Count concepts
                    if isinstance(data, list):
                        concept_counts[endpoint] = len(data)
                        print(f"   Found {len(data)} concepts")
                    elif isinstance(data, dict):
                        if "concepts" in data:
                            concept_counts[endpoint] = len(data['concepts'])
                            print(f"   Found {len(data['concepts'])} concepts")
                        if "status" in data:
                            print(f"   Status: {data['status']}")
                        if "mesh_available" in data:
                            service_status["Concept Mesh Loaded"] = data["mesh_available"]
                            print(f"   Mesh available: {data['mesh_available']}")
                except Exception as e:
                    record_warning(endpoint, f"Could not parse response: {e}")
            
            elif response.status_code == 404:
                record_result(f"Concept API - {endpoint}", False, 
                            "Endpoint not found",
                            "Check route registration in API")
            else:
                record_result(f"Concept API - {endpoint}", False, 
                            f"Status {response.status_code}")
                            
        except requests.exceptions.ConnectionError:
            record_result(f"Concept API - {endpoint}", False, "Connection error")
        except Exception as e:
            record_result(f"Concept API - {endpoint}", False, str(e))
    
    # Check consistency
    if len(concept_counts) > 1:
        counts = list(concept_counts.values())
        if len(set(counts)) > 1:
            record_warning("Concept Count Consistency",
                         f"Different endpoints report different counts: {concept_counts}")
            print(colored("   This may indicate multiple data sources!", Colors.WARNING))
        else:
            print(colored(f"\n✅ All endpoints report {counts[0]} concepts consistently", Colors.OKGREEN))

# Enhanced WebSocket tests
async def test_websocket_audio():
    """Test WebSocket audio bridge with detailed diagnostics"""
    print_test_header("WebSocket Audio Bridge")
    
    if not service_status.get("Audio WebSocket"):
        record_skip("WebSocket Audio", "Audio bridge not running")
        return
    
    try:
        async with websockets.connect(WS_AUDIO_URL) as websocket:
            # Send test audio data
            test_data = {
                "amplitude": 0.7,
                "frequency": 440,
                "waveform": "sine"
            }
            
            await websocket.send(json.dumps(test_data))
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                # Verify response structure
                if data.get("type") == "hologram_data":
                    record_result("WebSocket Audio Bridge", True)
                    print(f"   Brightness: {data.get('hologram_params', {}).get('brightness')}")
                    print(f"   Particle count: {data.get('hologram_params', {}).get('particle_count')}")
                else:
                    record_result("WebSocket Audio Bridge", False, 
                                "Invalid response format",
                                "Check bridge message handling")
            except asyncio.TimeoutError:
                record_result("WebSocket Audio Bridge", False,
                            "No response received",
                            "Bridge may not be processing messages")
                
    except ConnectionRefusedError:
        record_result("WebSocket Audio Bridge", False,
                    "Connection refused",
                    f"Start bridge with: python audio_hologram_bridge.py --port {WS_AUDIO_URL.split(':')[-1]}")
    except websockets.exceptions.InvalidHandshake as e:
        record_result("WebSocket Audio Bridge", False,
                    f"Invalid handshake: {e}",
                    "Port is open but not a WebSocket server")
    except Exception as e:
        record_result("WebSocket Audio Bridge", False, str(e))

async def test_websocket_concepts():
    """Test WebSocket concept mesh bridge with mock detection"""
    print_test_header("WebSocket Concept Mesh Bridge")
    
    if not service_status.get("Concept Mesh WebSocket"):
        record_skip("WebSocket Concepts", "Concept bridge not running")
        return
    
    try:
        async with websockets.connect(WS_CONCEPT_URL) as websocket:
            # Wait for initial concept data
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                if data.get("type") == "concept_update":
                    concepts = data.get("concepts", [])
                    mesh_available = data.get("mesh_available", False)
                    
                    record_result("WebSocket Concept Bridge - Initial Data", True)
                    print(f"   Received {len(concepts)} concepts")
                    print(f"   Mesh available: {mesh_available}")
                    
                    # Check if using mock data
                    if not mesh_available:
                        record_warning("Concept Bridge", 
                                     "Using MOCK concepts - real mesh not available")
                    
                    # Detect mock concepts
                    if concepts and len(concepts) == 5:
                        concept_names = [c.get("name", "").lower() for c in concepts]
                        mock_names = ["consciousness", "cognition", "awareness", "intelligence", "learning"]
                        if all(name in concept_names for name in mock_names):
                            record_warning("Concept Bridge",
                                         "Detected default mock concepts - no real data loaded")
                    
                    # Test additional requests...
                    # [rest of the test implementation]
                    
                else:
                    record_result("WebSocket Concept Bridge", False,
                                "No initial data received")
                    
            except asyncio.TimeoutError:
                record_result("WebSocket Concept Bridge", False,
                            "Initial data timeout",
                            "Bridge may not be sending concept updates")
                
    except ConnectionRefusedError:
        record_result("WebSocket Concept Bridge", False,
                    "Connection refused",
                    f"Start bridge with: python concept_mesh_hologram_bridge.py --port {WS_CONCEPT_URL.split(':')[-1]}")
    except Exception as e:
        record_result("WebSocket Concept Bridge", False, str(e))

# Service status summary
def print_service_status_table():
    """Print a comprehensive service status table"""
    print("\n" + "="*60)
    print(colored("TORI SERVICE STATUS SUMMARY", Colors.BOLD))
    print("="*60)
    
    max_len = max(len(name) for name in service_status.keys())
    
    for name, is_up in service_status.items():
        status = colored("✅ UP", Colors.OKGREEN) if is_up else colored("❌ DOWN", Colors.FAIL)
        print(f"{name:<{max_len + 2}}: {status}")
    
    print("-"*60)
    
    # Overall health
    up_count = sum(1 for v in service_status.values() if v)
    total_count = len(service_status)
    health_percent = (up_count / total_count) * 100 if total_count > 0 else 0
    
    if health_percent == 100:
        health_color = Colors.OKGREEN
        health_status = "FULLY OPERATIONAL"
    elif health_percent >= 80:
        health_color = Colors.WARNING
        health_status = "PARTIALLY OPERATIONAL"
    else:
        health_color = Colors.FAIL
        health_status = "DEGRADED"
    
    print(f"Overall System Health: {colored(f'{health_status} ({up_count}/{total_count})', health_color)}")

def print_fix_suggestions():
    """Print specific fix suggestions based on failures"""
    if not test_results["failed"]:
        return
    
    print("\n" + "="*60)
    print(colored("RECOMMENDED FIXES", Colors.BOLD))
    print("="*60)
    
    # Analyze failures and suggest fixes
    fixes = set()
    
    for test_name, error in test_results["failed"]:
        if "API" in test_name and "refused" in error:
            fixes.add("1. Start API: python enhanced_launcher.py")
        elif "WebSocket Audio" in test_name:
            fixes.add("2. Start Audio Bridge: python audio_hologram_bridge.py")
        elif "WebSocket Concept" in test_name:
            fixes.add("3. Start Concept Bridge: python concept_mesh_hologram_bridge.py")
        elif "File" in test_name or "data" in error.lower():
            fixes.add("4. Initialize data: python init_concept_mesh_data.py")
    
    for fix in sorted(fixes):
        print(f"  {fix}")
    
    print("\nFor detailed logs, check:")
    print("  - logs/tori.log")
    print("  - logs/enhanced_launcher.log")

# Main test runner
async def run_all_tests(force: bool = False):
    """Run all tests with intelligent ordering"""
    print("\n" + "="*60)
    print(colored("TORI CONCEPT MESH END-TO-END TEST SUITE", Colors.BOLD))
    print("="*60)
    print(f"API URL: {API_BASE_URL}")
    print(f"Audio WebSocket: {WS_AUDIO_URL}")
    print(f"Concept WebSocket: {WS_CONCEPT_URL}")
    
    # Pre-flight checks
    preflight_status = preflight_check(force)
    if preflight_status is None and not force:
        return 1
    
    # Detect duplicates
    detect_duplicate_files()
    
    # Run tests in logical order
    if test_api_health():
        test_concept_mesh_api()
    else:
        print(colored("\n⚠️  Skipping remaining API tests due to health check failure", Colors.WARNING))
    
    # Run async tests
    await test_websocket_audio()
    await test_websocket_concepts()
    
    # Print summaries
    print("\n" + "="*60)
    print(colored("TEST SUMMARY", Colors.BOLD))
    print("="*60)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"⏭️  Skipped: {len(test_results['skipped'])}")
    
    if test_results['failed']:
        print(colored("\nFailed Tests:", Colors.FAIL))
        for test_name, error in test_results['failed']:
            print(f"  - {test_name}: {error}")
    
    if test_results['warnings']:
        print(colored("\nWarnings:", Colors.WARNING))
        for test_name, warning in test_results['warnings']:
            print(f"  - {warning}")
    
    # Service status table
    print_service_status_table()
    
    # Fix suggestions
    print_fix_suggestions()
    
    # Return appropriate exit code
    if not test_results['failed']:
        print(colored("\n✅ All tests passed!", Colors.OKGREEN))
        return 0
    else:
        print(colored(f"\n❌ {len(test_results['failed'])} tests failed", Colors.FAIL))
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TORI Concept Mesh E2E Test Suite')
    parser.add_argument('--force', action='store_true', 
                       help='Run all tests even if pre-flight checks fail')
    parser.add_argument('--api-url', default=API_BASE_URL,
                       help=f'API base URL (default: {API_BASE_URL})')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Update globals if custom URL provided
    if args.api_url != API_BASE_URL:
        API_BASE_URL = args.api_url
        print(f"Using custom API URL: {API_BASE_URL}")
    
    # Disable colors if requested
    if args.no_color or sys.platform == "win32":
        # Reset all color codes to empty strings
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Run tests
    exit_code = asyncio.run(run_all_tests(args.force))
    sys.exit(exit_code)
