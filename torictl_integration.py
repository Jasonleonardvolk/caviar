#!/usr/bin/env python3
"""
torictl_integration.py - Integrate Beyond Metacognition with existing torictl

Add Beyond commands to your existing torictl CLI tool.
This can be imported into your main torictl or used standalone.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Detect TORI installation
TORI_BASE = Path("/opt/tori")
VENV_PYTHON = TORI_BASE / "venv" / "bin" / "python"
CURRENT_DIR = TORI_BASE / "current"

class BeyondCommands:
    """Beyond Metacognition commands for torictl"""
    
    @staticmethod
    def patch(args):
        """Apply Beyond Metacognition patches"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "apply_beyond_patches.py")]
        
        if args.dry:
            cmd.append("--dry")
        if args.verify:
            cmd.append("--verify")
        if args.single:
            cmd.extend(["--single", args.single])
            
        return subprocess.call(cmd)
    
    @staticmethod
    def verify(args):
        """Verify Beyond Metacognition installation"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "verify_beyond_integration.py")]
        return subprocess.call(cmd)
    
    @staticmethod
    def diagnose(args):
        """Run Beyond diagnostics"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "beyond_diagnostics.py")]
        return subprocess.call(cmd)
    
    @staticmethod
    def demo(args):
        """Run Beyond demonstration"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "torictl.py"), "demo", args.scenario]
        
        if args.plot:
            cmd.append("--plot")
            
        return subprocess.call(cmd)
    
    @staticmethod
    def monitor(args):
        """Monitor Beyond metrics live"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "torictl.py"), "monitor"]
        return subprocess.call(cmd)
    
    @staticmethod
    def rollback(args):
        """Rollback Beyond Metacognition"""
        script = CURRENT_DIR / "scripts" / "beyond_rollback.sh"
        
        if not script.exists():
            print(f"❌ Rollback script not found: {script}")
            return 1
            
        cmd = ["bash", str(script), str(args.versions)]
        return subprocess.call(cmd)
    
    @staticmethod
    def status(args):
        """Show Beyond Metacognition status"""
        cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "kha" / "torictl.py"), "status"]
        return subprocess.call(cmd)

def integrate_with_torictl():
    """
    Integration function for existing torictl.
    
    In your main torictl, add:
    
    from kha.torictl_integration import add_beyond_commands
    add_beyond_commands(subparsers)
    """
    pass

def add_beyond_commands(subparsers):
    """Add Beyond commands to existing torictl argument parser"""
    
    # Beyond parent parser for common options
    beyond_parent = argparse.ArgumentParser(add_help=False)
    beyond_parser = subparsers.add_parser(
        'beyond',
        help='Beyond Metacognition commands',
        description='Manage Beyond Metacognition features'
    )
    
    beyond_subparsers = beyond_parser.add_subparsers(
        dest='beyond_command',
        help='Beyond subcommands'
    )
    
    # Patch command
    patch_parser = beyond_subparsers.add_parser(
        'patch',
        help='Apply Beyond patches to TORI'
    )
    patch_parser.add_argument('--dry', action='store_true', help='Preview changes')
    patch_parser.add_argument('--verify', action='store_true', help='Verify after patching')
    patch_parser.add_argument('--single', type=str, help='Patch single file only')
    patch_parser.set_defaults(func=BeyondCommands.patch)
    
    # Verify command
    verify_parser = beyond_subparsers.add_parser(
        'verify',
        help='Verify Beyond installation'
    )
    verify_parser.set_defaults(func=BeyondCommands.verify)
    
    # Diagnose command
    diagnose_parser = beyond_subparsers.add_parser(
        'diagnose',
        help='Run Beyond diagnostics'
    )
    diagnose_parser.set_defaults(func=BeyondCommands.diagnose)
    
    # Demo command
    demo_parser = beyond_subparsers.add_parser(
        'demo',
        help='Run Beyond demonstrations'
    )
    demo_parser.add_argument(
        'scenario',
        choices=['emergence', 'creative', 'reflexive', 'temporal'],
        help='Demo scenario to run'
    )
    demo_parser.add_argument('--plot', action='store_true', help='Generate plot')
    demo_parser.set_defaults(func=BeyondCommands.demo)
    
    # Monitor command
    monitor_parser = beyond_subparsers.add_parser(
        'monitor',
        help='Monitor Beyond metrics live'
    )
    monitor_parser.set_defaults(func=BeyondCommands.monitor)
    
    # Rollback command
    rollback_parser = beyond_subparsers.add_parser(
        'rollback',
        help='Rollback Beyond changes'
    )
    rollback_parser.add_argument(
        'versions',
        type=int,
        default=1,
        nargs='?',
        help='Number of versions to rollback (default: 1)'
    )
    rollback_parser.set_defaults(func=BeyondCommands.rollback)
    
    # Status command
    status_parser = beyond_subparsers.add_parser(
        'status',
        help='Show Beyond status'
    )
    status_parser.set_defaults(func=BeyondCommands.status)

def standalone_main():
    """Run as standalone torictl extension"""
    parser = argparse.ArgumentParser(
        prog='torictl-beyond',
        description='Beyond Metacognition extension for torictl'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Add all Beyond commands
    add_beyond_commands(subparsers)
    
    # Parse and execute
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)

# Example: How to integrate into existing torictl
INTEGRATION_EXAMPLE = '''
# In your existing torictl script, add:

from kha.torictl_integration import add_beyond_commands

# In your main argument parser setup:
parser = argparse.ArgumentParser(prog='torictl')
subparsers = parser.add_subparsers(dest='command')

# Add your existing commands...
# ...

# Add Beyond Metacognition commands
add_beyond_commands(subparsers)

# Now you can use:
# torictl beyond patch --verify
# torictl beyond demo emergence
# torictl beyond monitor
# etc.
'''

if __name__ == "__main__":
    # Run standalone
    standalone_main()
