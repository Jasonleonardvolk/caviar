#!/usr/bin/env python3
"""
TORI Master Launcher - Orchestrates the complete startup sequence
"""

import subprocess
import sys
import time
import os
from datetime import datetime

class Colors:
    """Console color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}🚀 {text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def run_command(cmd, description, critical=True):
    """Run a command and handle the result"""
    print(f"\n{Colors.YELLOW}⏳ {description}...{Colors.RESET}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully")
            if result.stdout and len(result.stdout.strip()) < 200:
                print(f"   {result.stdout.strip()}")
            return True
        else:
            print_error(f"{description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if critical:
                return False
            else:
                print_warning("Continuing despite failure (non-critical component)")
                return True
                
    except Exception as e:
        print_error(f"{description} failed with exception: {e}")
        return False

def check_python_version():
    """Ensure Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ required. You have {sys.version}")
        return False
    print_success(f"Python version {sys.version.split()[0]} is compatible")
    return True

def main():
    """Main launcher sequence"""
    os.chdir("C:\\Users\\jason\\Desktop\\tori\\kha")
    
    # Print welcome banner
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║             TORI MASTER LAUNCHER v2.0                     ║")
    print("║           Unified System Startup Orchestrator             ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 0: Environment checks
    print_header("Phase 0: Environment Validation")
    
    if not check_python_version():
        sys.exit(1)
    
    # Phase 1: Pre-flight validation
    print_header("Phase 1: Pre-Flight Validation")
    
    if not run_command("python pre_flight_check.py", "Pre-flight checks"):
        print_warning("Pre-flight validation failed. Attempting automatic fixes...")
        
        if run_command("python tori_emergency_fix.py", "Emergency fixes"):
            print_info("Re-running pre-flight checks...")
            
            if not run_command("python pre_flight_check.py", "Pre-flight checks (retry)"):
                print_error("System still not ready after emergency fixes.")
                print_info("Please check the error messages above and fix manually.")
                input("\nPress Enter to exit...")
                sys.exit(1)
        else:
            print_error("Emergency fix failed. Manual intervention required.")
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Phase 2: Component startup
    print_header("Phase 2: Isolated Component Startup")
    
    print_info("Starting components in isolated mode...")
    
    # Start the isolated launcher in a new window
    if os.name == 'nt':
        subprocess.Popen('start "TORI Components" cmd /k python isolated_startup.py', shell=True)
    else:
        subprocess.Popen('gnome-terminal -- python isolated_startup.py', shell=True)
    
    # Wait for services to initialize
    print_info("Waiting for services to initialize...")
    wait_time = 30
    for i in range(wait_time, 0, -1):
        print(f"\r   {i} seconds remaining...", end='', flush=True)
        time.sleep(1)
    print("\r   Services should be initialized!    ")
    
    # Phase 3: Verification
    print_header("Phase 3: System Verification")
    
    # Quick health check
    try:
        import requests
        
        # Check API
        try:
            response = requests.get("http://localhost:8002/api/health", timeout=5)
            if response.status_code == 200:
                print_success("API server is healthy")
            else:
                print_warning(f"API server returned status {response.status_code}")
        except:
            print_error("API server is not responding")
        
        # Check Frontend
        try:
            response = requests.get("http://localhost:5173", timeout=5)
            if response.status_code == 200:
                print_success("Frontend is accessible")
            else:
                print_warning(f"Frontend returned status {response.status_code}")
        except:
            print_error("Frontend is not responding")
            
    except ImportError:
        print_warning("requests module not installed, skipping health checks")
    
    # Phase 4: Launch monitoring (optional)
    print_header("Phase 4: Optional Services")
    
    response = input("\nStart health monitor? (y/n): ")
    if response.lower() == 'y':
        if os.name == 'nt':
            subprocess.Popen('start "Health Monitor" cmd /k python health_monitor.py', shell=True)
        else:
            subprocess.Popen('gnome-terminal -- python health_monitor.py', shell=True)
        print_success("Health monitor started")
    
    response = input("Start auto-recovery service? (y/n): ")
    if response.lower() == 'y':
        if os.name == 'nt':
            subprocess.Popen('start "Auto Recovery" cmd /k python auto_recovery.py', shell=True)
        else:
            subprocess.Popen('gnome-terminal -- python auto_recovery.py', shell=True)
        print_success("Auto-recovery service started")
    
    # Final summary
    print_header("TORI System Launch Complete!")
    
    print(f"\n{Colors.BOLD}📍 Access Points:{Colors.RESET}")
    print(f"   Frontend:     {Colors.BLUE}http://localhost:5173{Colors.RESET}")
    print(f"   API Docs:     {Colors.BLUE}http://localhost:8002/docs{Colors.RESET}")
    print(f"   Health Check: {Colors.BLUE}http://localhost:8002/api/health{Colors.RESET}")
    print(f"   Dashboard:    {Colors.BLUE}Open dashboard.html in your browser{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}📝 Quick Commands:{Colors.RESET}")
    print(f"   Stop TORI:      {Colors.YELLOW}python shutdown_tori.py{Colors.RESET}")
    print(f"   Check Health:   {Colors.YELLOW}python health_monitor.py{Colors.RESET}")
    print(f"   Run Tests:      {Colors.YELLOW}python test_basic.py{Colors.RESET}")
    print(f"   Emergency Fix:  {Colors.YELLOW}python tori_emergency_fix.py{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}💡 Tips:{Colors.RESET}")
    print("   • Check the component window for detailed status")
    print("   • Monitor the health check window for real-time updates")
    print("   • Review logs/tori_errors.log if issues occur")
    print("   • Use dashboard.html for visual monitoring")
    
    # Open browser
    response = input("\nOpen frontend in browser? (y/n): ")
    if response.lower() == 'y':
        import webbrowser
        webbrowser.open("http://localhost:5173")
        print_success("Browser opened")
    
    print(f"\n{Colors.GREEN}✅ All done! TORI should now be running.{Colors.RESET}")
    print(f"{Colors.YELLOW}This window can be closed safely.{Colors.RESET}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Launch cancelled by user{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        input("\nPress Enter to exit...")
        sys.exit(1)
