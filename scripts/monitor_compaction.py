#!/usr/bin/env python3
"""
TORI Mesh Compaction Monitor

PURPOSE:
    Real-time monitoring of mesh compaction status and health.
    Provides live dashboard and status reporting for TORI concept meshes.

WHAT IT DOES:
    - Shows real-time mesh size, WAL status, and compaction needs
    - Displays recent compaction runs and results
    - Shows scheduled tasks and automation status
    - Provides both live dashboard and JSON output modes
    - Monitors mesh health and performance metrics

USAGE:
    python monitor_compaction.py [OPTIONS]
    
    Options:
        --live           Live monitoring mode (requires rich library)
        --interval N     Refresh interval in seconds (default: 30)
        --json           Output status as JSON
        --data-dir DIR   Custom data directory path

EXAMPLES:
    python monitor_compaction.py --live
    python monitor_compaction.py --json | jq .
    python monitor_compaction.py --interval 10

DEPENDENCIES:
    - rich (optional, for enhanced display)
    - Core TORI mesh components

AUTHOR: TORI System Maintenance
LAST UPDATED: 2025-01-26
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components with fallbacks
try:
    from core.metrics import MetricsCollector
except ImportError:
    print("Warning: MetricsCollector not available - using mock")
    
    class MetricsCollector:
        def __init__(self, data_dir=None):
            self.data_dir = data_dir
        
        def check_all_meshes(self):
            return []
        
        def get_compaction_report(self):
            return {
                'total_meshes': 0,
                'needs_compaction': 0,
                'total_mesh_size_mb': 0.0,
                'total_wal_size_mb': 0.0,
                'details': []
            }

try:
    from scripts.compact_all_meshes import MeshCompactor
except ImportError:
    print("Warning: MeshCompactor not available")
    MeshCompactor = None

# Try to import rich for better display
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print
    print("Note: Install 'rich' library for enhanced display: pip install rich")


class CompactionMonitor:
    """Monitor compaction status and health"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path('data')
        self.collector = MetricsCollector(self.data_dir)
        self.console = Console() if RICH_AVAILABLE else None
        
    def get_status_table(self) -> Table:
        """Create status table"""
        table = Table(title="Mesh Compaction Status", show_header=True)
        
        table.add_column("Mesh", style="cyan", no_wrap=True)
        table.add_column("Size (MB)", justify="right", style="yellow")
        table.add_column("WAL (MB)", justify="right", style="yellow")
        table.add_column("Last Modified", style="green")
        table.add_column("Last Compact", style="blue")
        table.add_column("Status", style="bold")
        
        # Get all mesh metrics
        all_metrics = self.collector.check_all_meshes()
        
        for metrics in sorted(all_metrics, key=lambda m: (m.scope, m.scope_id)):
            # Format times
            last_mod = f"{metrics.last_modified_hours:.1f}h ago" if metrics.last_modified_hours else "Unknown"
            last_compact = f"{metrics.last_compact_hours:.1f}h ago" if metrics.last_compact_hours else "Never"
            
            # Status
            if metrics.needs_compaction:
                status = f"[red]NEEDS COMPACT[/red]"
            else:
                status = "[green]OK[/green]"
            
            table.add_row(
                f"{metrics.scope}:{metrics.scope_id}",
                f"{metrics.mesh_size_mb:.2f}",
                f"{metrics.wal_size_mb:.2f}",
                last_mod,
                last_compact,
                status
            )
        
        return table
    
    def get_summary_panel(self) -> Panel:
        """Create summary panel"""
        report = self.collector.get_compaction_report()
        
        content = f"""
[bold]Total Meshes:[/bold] {report['total_meshes']}
[bold]Need Compaction:[/bold] {report['needs_compaction']}
[bold]Total Size:[/bold] {report['total_mesh_size_mb']:.2f} MB
[bold]Total WAL:[/bold] {report['total_wal_size_mb']:.2f} MB

[bold]Last Check:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return Panel(content.strip(), title="Summary", border_style="blue")
    
    def get_schedule_info(self) -> Panel:
        """Get scheduled task information"""
        content = []
        
        # Check for cron jobs (Linux)
        try:
            import subprocess
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                cron_jobs = [line for line in result.stdout.split('\n') if 'compact_all_meshes.py' in line]
                if cron_jobs:
                    content.append("[bold]Cron Jobs:[/bold]")
                    for job in cron_jobs:
                        content.append(f"  {job}")
        except:
            pass
        
        # Check for Windows scheduled tasks
        try:
            import subprocess
            result = subprocess.run(['schtasks', '/query', '/tn', 'TORI Compaction*', '/fo', 'csv'], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                content.append("\n[bold]Windows Tasks:[/bold]")
                # Parse CSV output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) > 2:
                            task_name = parts[0].strip('"')
                            status = parts[2].strip('"')
                            content.append(f"  {task_name}: {status}")
        except:
            pass
        
        # Check for systemd timers
        try:
            import subprocess
            result = subprocess.run(['systemctl', 'list-timers', 'tori-compact*'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and 'tori-compact' in result.stdout:
                content.append("\n[bold]Systemd Timers:[/bold]")
                content.append("  Active (check: systemctl list-timers)")
        except:
            pass
        
        if not content:
            content.append("[yellow]No scheduled tasks found[/yellow]")
        
        return Panel('\n'.join(content), title="Scheduled Tasks", border_style="green")
    
    def get_recent_runs(self) -> Panel:
        """Get recent compaction runs"""
        log_dir = self.data_dir.parent / 'logs'
        content = []
        
        if log_dir.exists():
            # Find recent compaction result files
            result_files = sorted(log_dir.glob('compaction_*.json'), 
                                key=lambda p: p.stat().st_mtime, 
                                reverse=True)[:5]
            
            if result_files:
                content.append("[bold]Recent Runs:[/bold]")
                for result_file in result_files:
                    try:
                        with open(result_file) as f:
                            result = json.load(f)
                        
                        timestamp = datetime.fromisoformat(result['start_time'])
                        duration = result.get('duration_seconds', 0)
                        compacted = result.get('meshes_compacted', 0)
                        
                        content.append(f"  {timestamp.strftime('%Y-%m-%d %H:%M')} - "
                                     f"{compacted} compacted in {duration:.1f}s")
                    except:
                        pass
            else:
                content.append("[yellow]No recent runs found[/yellow]")
        else:
            content.append("[yellow]Log directory not found[/yellow]")
        
        return Panel('\n'.join(content), title="Recent Runs", border_style="magenta")
    
    def monitor_live(self, refresh_interval: int = 30):
        """Live monitoring with auto-refresh"""
        if not RICH_AVAILABLE:
            print("Live monitoring requires 'rich' library. Install with: pip install rich")
            return
        
        layout = Layout()
        layout.split_column(
            Layout(name="summary", size=8),
            Layout(name="table", ratio=2),
            Layout(name="info")
        )
        
        layout["info"].split_row(
            Layout(name="schedule"),
            Layout(name="recent")
        )
        
        with Live(layout, refresh_per_second=0.5, screen=True) as live:
            while True:
                try:
                    # Update displays
                    layout["summary"].update(self.get_summary_panel())
                    layout["table"].update(self.get_status_table())
                    layout["schedule"].update(self.get_schedule_info())
                    layout["recent"].update(self.get_recent_runs())
                    
                    # Wait for refresh interval
                    time.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(5)
    
    def print_simple_status(self):
        """Print simple status (no rich library)"""
        report = self.collector.get_compaction_report()
        
        print(f"\nCompaction Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Total meshes: {report['total_meshes']}")
        print(f"Need compaction: {report['needs_compaction']}")
        print(f"Total size: {report['total_mesh_size_mb']:.2f} MB")
        print(f"Total WAL: {report['total_wal_size_mb']:.2f} MB")
        
        if report['details']:
            print("\nMeshes needing compaction:")
            for detail in report['details']:
                print(f"  - {detail['scope']}:{detail['scope_id']}: {detail['reason']}")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Monitor TORI mesh compaction status')
    parser.add_argument('--live', action='store_true', 
                        help='Live monitoring mode (requires rich)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--data-dir', type=Path,
                        help='Data directory path')
    
    args = parser.parse_args()
    
    monitor = CompactionMonitor(args.data_dir)
    
    if args.json:
        # JSON output
        report = monitor.collector.get_compaction_report()
        print(json.dumps(report, indent=2))
    
    elif args.live:
        # Live monitoring
        print("Starting live monitoring... Press Ctrl+C to exit")
        monitor.monitor_live(args.interval)
    
    else:
        # Simple status
        if RICH_AVAILABLE:
            console = Console()
            console.print(monitor.get_summary_panel())
            console.print(monitor.get_status_table())
            console.print(monitor.get_schedule_info())
            console.print(monitor.get_recent_runs())
        else:
            monitor.print_simple_status()


if __name__ == '__main__':
    main()
