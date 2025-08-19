#!/usr/bin/env python3
"""
Dickbox Management Script for TORI/Saigon
==========================================
Manages dickbox containers for the TORI system.
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_command(cmd):
    """Run a dickbox command."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{RED}Error: {result.stderr}{RESET}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"{RED}Failed to run command: {e}{RESET}")
        return False

def build_images():
    """Build dickbox images."""
    print(f"{BLUE}Building TORI API image...{RESET}")
    if not run_command("dickbox build -f Dickboxfile -t tori/saigon:latest ."):
        return False
    
    print(f"{BLUE}Building frontend image...{RESET}")
    if not run_command("dickbox build -f frontend/hybrid/Dickboxfile.frontend -t tori/frontend:latest frontend/hybrid"):
        return False
    
    print(f"{GREEN}✓ Images built successfully{RESET}")
    return True

def start_containers():
    """Start all containers."""
    print(f"{BLUE}Starting TORI containers...{RESET}")
    
    if not run_command("dickbox up -c dickbox.toml"):
        return False
    
    print(f"{GREEN}✓ Containers started{RESET}")
    print(f"\n{CYAN}Services available at:{RESET}")
    print(f"  API:        http://localhost:8001")
    print(f"  Frontend:   http://localhost:3000")
    print(f"  Prometheus: http://localhost:9090")
    print(f"  Grafana:    http://localhost:3001")
    return True

def stop_containers():
    """Stop all containers."""
    print(f"{BLUE}Stopping TORI containers...{RESET}")
    
    if not run_command("dickbox down -c dickbox.toml"):
        return False
    
    print(f"{GREEN}✓ Containers stopped{RESET}")
    return True

def restart_containers():
    """Restart all containers."""
    print(f"{BLUE}Restarting TORI containers...{RESET}")
    
    if not run_command("dickbox restart -c dickbox.toml"):
        return False
    
    print(f"{GREEN}✓ Containers restarted{RESET}")
    return True

def show_status():
    """Show container status."""
    print(f"{BLUE}TORI Container Status:{RESET}")
    run_command("dickbox ps -c dickbox.toml")

def show_logs(container=None, follow=False):
    """Show container logs."""
    if container:
        cmd = f"dickbox logs {container}"
    else:
        cmd = "dickbox logs -c dickbox.toml"
    
    if follow:
        cmd += " -f"
    
    run_command(cmd)

def exec_shell(container="tori-api"):
    """Execute shell in container."""
    print(f"{BLUE}Opening shell in {container}...{RESET}")
    subprocess.run(f"dickbox exec {container} /bin/bash", shell=True)

def clean_volumes():
    """Clean up volumes."""
    print(f"{YELLOW}⚠ This will delete all persistent data!{RESET}")
    response = input("Are you sure? (y/n): ")
    
    if response.lower() == 'y':
        print(f"{BLUE}Cleaning volumes...{RESET}")
        run_command("dickbox volume rm redis-data postgres-data prometheus-data grafana-data")
        print(f"{GREEN}✓ Volumes cleaned{RESET}")
    else:
        print("Cancelled")

def backup_data():
    """Backup important data."""
    print(f"{BLUE}Creating backup...{RESET}")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup adapters
    run_command(f"cp -r models/adapters {backup_dir}/")
    
    # Backup mesh contexts
    run_command(f"cp -r data/mesh_contexts {backup_dir}/")
    
    # Backup training data
    run_command(f"cp -r data/training {backup_dir}/")
    
    print(f"{GREEN}✓ Backup created at {backup_dir}{RESET}")

def gpu_info():
    """Show GPU information."""
    print(f"{BLUE}GPU Information:{RESET}")
    run_command("dickbox exec tori-api nvidia-smi")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dickbox management for TORI/Saigon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./dickbox.py build        # Build images
  ./dickbox.py up           # Start containers
  ./dickbox.py down         # Stop containers
  ./dickbox.py status       # Show status
  ./dickbox.py logs         # Show logs
  ./dickbox.py shell        # Open shell
        """
    )
    
    parser.add_argument('command', 
                       choices=['build', 'up', 'down', 'restart', 'status', 
                               'logs', 'shell', 'clean', 'backup', 'gpu'],
                       help='Command to execute')
    parser.add_argument('--container', '-c', help='Container name (for logs/shell)')
    parser.add_argument('--follow', '-f', action='store_true', help='Follow logs')
    
    args = parser.parse_args()
    
    # Print banner
    print(f"{CYAN}{'='*50}")
    print(f"{BOLD}TORI/Saigon Dickbox Manager{RESET}")
    print(f"{CYAN}{'='*50}{RESET}\n")
    
    # Execute command
    if args.command == 'build':
        build_images()
    elif args.command == 'up':
        start_containers()
    elif args.command == 'down':
        stop_containers()
    elif args.command == 'restart':
        restart_containers()
    elif args.command == 'status':
        show_status()
    elif args.command == 'logs':
        show_logs(args.container, args.follow)
    elif args.command == 'shell':
        exec_shell(args.container or 'tori-api')
    elif args.command == 'clean':
        clean_volumes()
    elif args.command == 'backup':
        backup_data()
    elif args.command == 'gpu':
        gpu_info()

if __name__ == "__main__":
    main()
