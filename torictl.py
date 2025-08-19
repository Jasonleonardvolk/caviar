#!/usr/bin/env python3
"""
torictl.py – Command-line interface for Beyond Metacognition tooling

Usage examples:
  torictl list                  # see available demo scenarios
  torictl demo emergence        # run a specific demo scenario
  torictl verify                # execute verify_beyond_integration.py
  torictl demo creative --plot  # run demo with matplotlib enabled
  torictl patch                 # apply Beyond Metacognition patches
  torictl patch --dry           # preview patches without applying
  torictl status                # show integration status
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def run_demo(scenario: str, plot: bool = False):
    """Run a specific Beyond Metacognition demo scenario"""
    try:
        # Import demo module
        sys.path.insert(0, str(ROOT))
        from beyond_demo import BeyondMetacognitionDemo
        
        # Create and run demo
        demo = BeyondMetacognitionDemo()
        
        # Run the selected scenario
        asyncio.run(demo.run_scenario(scenario))
        
        # Optionally plot results
        if plot and demo.history['novelty']:
            try:
                demo.plot_history()
                print("\n📈 Plot saved to: beyond_metacognition_demo.png")
            except ImportError:
                print("\n⚠️  matplotlib not available - install with: pip install matplotlib")
            except Exception as e:
                print(f"\n⚠️  Could not create plot: {e}")
                
    except Exception as e:
        sys.exit(f"❌ Demo error: {e}")

def list_scenarios():
    """List available demo scenarios"""
    scenarios = {
        "emergence": "Demonstrate dimensional emergence detection",
        "creative": "Creative exploration with entropy injection", 
        "reflexive": "Self-measurement and metacognition",
        "temporal": "Multi-scale temporal braiding"
    }
    
    print("\n🌌 Available Beyond Metacognition scenarios:")
    print("=" * 60)
    for name, desc in scenarios.items():
        print(f"  {name:<12} - {desc}")
    print()

def run_verify():
    """Run the Beyond Metacognition verification script"""
    verify_script = ROOT / "verify_beyond_integration.py"
    if verify_script.exists():
        subprocess.call([sys.executable, str(verify_script)])
    else:
        sys.exit(f"❌ Verification script not found: {verify_script}")

def run_patch(dry: bool = False, verify: bool = False):
    """Apply Beyond Metacognition patches"""
    patch_script = ROOT / "apply_beyond_patches.py"
    if patch_script.exists():
        cmd = [sys.executable, str(patch_script)]
        if dry:
            cmd.append("--dry")
        if verify:
            cmd.append("--verify")
        subprocess.call(cmd)
    else:
        sys.exit(f"❌ Patch script not found: {patch_script}")

def show_status():
    """Show Beyond Metacognition integration status"""
    status_file = ROOT / "BEYOND_METACOGNITION_STATUS.json"
    
    if not status_file.exists():
        print("⚠️  No status file found. Run 'torictl patch' first.")
        return
    
    try:
        with open(status_file) as f:
            status = json.load(f)
        
        print("\n🌌 Beyond Metacognition Integration Status")
        print("=" * 60)
        print(f"Last updated: {status.get('timestamp', 'Unknown')}")
        print(f"Version: {status.get('patch_version', 'Unknown')}")
        print(f"\nComponents:")
        
        components = status.get('components', {})
        for comp, desc in components.items():
            print(f"  ✅ {comp}: {desc}")
        
        print(f"\nPatched files: {len(status.get('patched_files', []))}")
        for file in status.get('patched_files', []):
            print(f"  ✓ {file}")
            
        if skipped := status.get('skipped_files', []):
            print(f"\nSkipped files: {len(skipped)}")
            for file in skipped:
                print(f"  - {file}")
                
    except Exception as e:
        print(f"❌ Error reading status: {e}")

def run_monitor():
    """Live monitoring dashboard (simple version)"""
    print("\n🔍 Beyond Metacognition Live Monitor")
    print("=" * 60)
    
    try:
        sys.path.insert(0, str(ROOT))
        from alan_backend.origin_sentry import OriginSentry
        from python.core.braid_buffers import get_braiding_engine, TimeScale
        from python.core.creative_feedback import get_creative_feedback
        
        origin = OriginSentry()
        braiding = get_braiding_engine()
        creative = get_creative_feedback()
        
        print("Press Ctrl+C to stop monitoring\n")
        
        while True:
            # Get current metrics
            origin_metrics = origin.metrics
            creative_metrics = creative.get_creative_metrics()
            braid_status = braiding.get_status()
            
            # Clear screen (platform-specific)
            print("\033[2J\033[H" if sys.platform != "win32" else "\n" * 50)
            
            # Display metrics
            print(f"🌌 Beyond Metacognition Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            print("\n📊 Origin Sentry:")
            print(f"  Dimension: {origin_metrics['current_dimension']}")
            print(f"  Dimension births: {origin_metrics['dimension_expansions']}")
            print(f"  Coherence: {origin_metrics['coherence_state']}")
            print(f"  Novelty: {origin_metrics['novelty_score']:.3f}")
            
            print("\n🎨 Creative Feedback:")
            print(f"  Mode: {creative_metrics['current_mode']}")
            print(f"  Injections: {creative_metrics['total_injections']}")
            print(f"  Success rate: {creative_metrics.get('success_rate', 0.0):.1%}")
            
            print("\n🕰️ Temporal Braiding:")
            for scale, buffer_data in braid_status['buffers'].items():
                print(f"  {scale}: {buffer_data['count']} events ({buffer_data['fill_ratio']:.1%} full)")
            
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
    except ImportError as e:
        print(f"❌ Required components not found. Run 'torictl verify' first.")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Monitor error: {e}")

def main():
    parser = argparse.ArgumentParser(
        prog="torictl",
        description="Beyond Metacognition control interface"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    subparsers.add_parser("list", help="List available demo scenarios")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a Beyond Metacognition demo")
    demo_parser.add_argument("scenario", 
                            choices=["emergence", "creative", "reflexive", "temporal"],
                            help="Demo scenario to run")
    demo_parser.add_argument("--plot", action="store_true",
                            help="Generate matplotlib plot after demo")
    
    # Verify command
    subparsers.add_parser("verify", help="Run integration verification")
    
    # Patch command
    patch_parser = subparsers.add_parser("patch", help="Apply Beyond Metacognition patches")
    patch_parser.add_argument("--dry", action="store_true",
                             help="Preview changes without applying")
    patch_parser.add_argument("--verify", action="store_true",
                             help="Run verification after patching")
    
    # Status command
    subparsers.add_parser("status", help="Show integration status")
    
    # Monitor command
    subparsers.add_parser("monitor", help="Live monitoring dashboard")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "list":
        list_scenarios()
    elif args.command == "demo":
        run_demo(args.scenario, args.plot)
    elif args.command == "verify":
        run_verify()
    elif args.command == "patch":
        run_patch(args.dry, args.verify)
    elif args.command == "status":
        show_status()
    elif args.command == "monitor":
        run_monitor()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
