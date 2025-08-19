#!/usr/bin/env python3
"""
TORI Manual Mesh Compaction Runner

PURPOSE:
    Interactive utility for testing and maintenance of mesh compaction.
    Provides manual control over compaction operations with real-time feedback.

WHAT IT DOES:
    - Checks which meshes need compaction
    - Runs selective or full compaction
    - Tests snapshot restore functionality
    - Provides interactive prompts and progress feedback
    - Validates compaction status and metrics

USAGE:
    python run_compaction.py [OPTIONS]
    
    Options:
        --check-only     Only check what needs compaction
        --force          Force compaction even if not needed
        --specific MESH  Compact specific mesh (e.g. user:alice)
        --restore MESH   Restore mesh from snapshot
        --snapshot PATH  Snapshot file to restore from
        --config FILE    Config file path
        -y, --yes        Skip confirmation prompts

EXAMPLES:
    python run_compaction.py --check-only
    python run_compaction.py --specific user:alice
    python run_compaction.py --restore user:alice --snapshot backup.json.gz

AUTHOR: TORI System Maintenance
LAST UPDATED: 2025-01-26
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import with fallbacks for robustness
try:
    from scripts.compact_all_meshes import MeshCompactor
except ImportError:
    print("Warning: MeshCompactor not available - limited functionality")
    MeshCompactor = None

try:
    from core.metrics import MetricsCollector
except ImportError:
    print("Warning: MetricsCollector not available - using mock")
    
    class MetricsCollector:
        def get_compaction_report(self):
            return {
                'total_meshes': 0,
                'needs_compaction': 0,
                'total_mesh_size_mb': 0.0,
                'total_wal_size_mb': 0.0,
                'details': []
            }

import json


def print_header(text):
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"{text:^60}")
    print(f"{'=' * 60}\n")


async def run_compaction(args):
    """Run compaction with given arguments"""
    if not MeshCompactor:
        print("❌ MeshCompactor not available - cannot run compaction")
        return 1
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    compactor = MeshCompactor(config)
    
    if args.check_only:
        # Just check what needs compaction
        print_header("Compaction Status Check")
        
        collector = MetricsCollector()
        report = collector.get_compaction_report()
        
        print(f"Total meshes: {report['total_meshes']}")
        print(f"Need compaction: {report['needs_compaction']}")
        print(f"Total mesh size: {report['total_mesh_size_mb']:.2f} MB")
        print(f"Total WAL size: {report['total_wal_size_mb']:.2f} MB")
        
        if report['details']:
            print("\nMeshes needing compaction:")
            for detail in report['details']:
                print(f"  - {detail['scope']}:{detail['scope_id']}: {detail['reason']}")
        else:
            print("\nNo meshes need compaction at this time.")
        
        return
    
    if args.specific:
        # Compact specific mesh
        scope, scope_id = args.specific.split(':')
        print_header(f"Compacting {scope}:{scope_id}")
        
        try:
            success = await compactor._compact_single_mesh(scope, scope_id)
            if success:
                print(f"✓ Successfully compacted {scope}:{scope_id}")
            else:
                print(f"✗ Failed to compact {scope}:{scope_id}")
                return 1
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
    
    else:
        # Compact all meshes
        print_header("Running Full Compaction")
        
        if not args.force:
            # Show what will be done
            collector = MetricsCollector()
            report = collector.get_compaction_report()
            
            if report['needs_compaction'] == 0:
                print("No meshes need compaction.")
                if not args.yes:
                    response = input("\nRun anyway? (y/n): ")
                    if response.lower() != 'y':
                        print("Aborted.")
                        return 0
            else:
                print(f"Will compact {report['needs_compaction']} meshes:")
                for detail in report['details']:
                    print(f"  - {detail['scope']}:{detail['scope_id']}")
                
                if not args.yes:
                    response = input("\nProceed? (y/n): ")
                    if response.lower() != 'y':
                        print("Aborted.")
                        return 0
        
        # Run compaction
        print("\nStarting compaction...")
        results = await compactor.compact_all_meshes(force=args.force)
        
        # Print results
        print(f"\n✓ Compaction complete!")
        print(f"  Meshes processed: {results['meshes_processed']}")
        print(f"  Meshes compacted: {results['meshes_compacted']}")
        print(f"  Snapshots created: {results['snapshots_created']}")
        print(f"  Duration: {results['duration_seconds']:.1f}s")
        
        if results.get('daily_checkpoint'):
            print(f"  Daily checkpoint: {results['daily_checkpoint']}")
        
        if results['errors']:
            print(f"\n✗ Errors occurred:")
            for error in results['errors']:
                print(f"  - {error}")
            return 1
    
    return 0


async def test_restore(args):
    """Test snapshot restore"""
    print_header("Testing Snapshot Restore")
    
    scope, scope_id = args.restore.split(':')
    
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    compactor = MeshCompactor(config)
    
    snapshot_path = None
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        if not snapshot_path.exists():
            print(f"✗ Snapshot not found: {snapshot_path}")
            return 1
    
    print(f"Restoring {scope}:{scope_id} from snapshot...")
    success = await compactor.restore_from_snapshot(scope, scope_id, snapshot_path)
    
    if success:
        print(f"✓ Successfully restored {scope}:{scope_id}")
        return 0
    else:
        print(f"✗ Failed to restore {scope}:{scope_id}")
        return 1


async def main():
    parser = argparse.ArgumentParser(
        description='TORI Mesh Compaction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check compaction status
  %(prog)s --check-only
  
  # Run compaction (only what's needed)
  %(prog)s
  
  # Force compact all meshes
  %(prog)s --force
  
  # Compact specific mesh
  %(prog)s --specific user:alice
  
  # Restore from snapshot
  %(prog)s --restore user:alice --snapshot /path/to/snapshot.json.gz
  
  # Use custom config
  %(prog)s --config custom_config.json
        """
    )
    
    parser.add_argument('--check-only', action='store_true',
                        help='Only check what needs compaction')
    parser.add_argument('--force', action='store_true',
                        help='Force compaction even if not needed')
    parser.add_argument('--specific', metavar='SCOPE:ID',
                        help='Compact specific mesh (e.g. user:alice)')
    parser.add_argument('--restore', metavar='SCOPE:ID',
                        help='Restore mesh from snapshot')
    parser.add_argument('--snapshot', metavar='PATH',
                        help='Snapshot file to restore from')
    parser.add_argument('--config', metavar='FILE',
                        help='Config file path')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    try:
        if args.restore:
            return await test_restore(args)
        else:
            return await run_compaction(args)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
