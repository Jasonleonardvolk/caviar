#!/usr/bin/env python3
"""
TORI Doctor - One-command health check
Checks: clock skew, seals, index drift, snapshot age
"""

import click
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.psi_archive_extended import PSI_ARCHIVER


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--fix', is_flag=True, help='Attempt to fix issues')
def doctor(verbose: bool, fix: bool):
    """
    TORI health check - ensures system integrity
    
    Checks:
    - Archive seals are up to date
    - Snapshots are recent
    - Mini-indices are fresh
    - Disk space is adequate
    - Clock is synchronized
    """
    print("🩺 TORI Doctor - System Health Check")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # 1. Check archive seals
    print("\n📋 Checking archive seals...")
    try:
        yesterday = datetime.now().date() - timedelta(days=1)
        yesterday_file = PSI_ARCHIVER.archive_dir / yesterday.strftime("%Y-%m") / f"day-{yesterday.strftime('%d')}.ndjson"
        yesterday_gz = yesterday_file.with_suffix('.ndjson.gz')
        
        if yesterday_file.exists():
            issues.append(f"Yesterday's archive not sealed: {yesterday_file}")
            if fix:
                print("  🔧 Sealing yesterday's archive...")
                PSI_ARCHIVER.seal_yesterday()
        elif yesterday_gz.exists():
            print("  ✅ Yesterday's archive is sealed")
        else:
            warnings.append("No archive file for yesterday (OK if system was idle)")
            
    except Exception as e:
        issues.append(f"Failed to check seals: {e}")
    
    # 2. Check snapshot age
    print("\n📸 Checking snapshots...")
    try:
        snapshots_dir = Path("data/snapshots")
        if snapshots_dir.exists():
            snapshots = sorted(snapshots_dir.glob("????-??-??"))
            if snapshots:
                latest_snapshot = snapshots[-1]
                snapshot_age = datetime.now() - datetime.fromisoformat(latest_snapshot.name)
                
                if snapshot_age > timedelta(days=7):
                    issues.append(f"Latest snapshot is {snapshot_age.days} days old (should be <7)")
                else:
                    print(f"  ✅ Latest snapshot: {latest_snapshot.name} ({snapshot_age.days} days old)")
                    
                # Check for full snapshot
                full_snapshots = list(snapshots_dir.glob("*-full"))
                if full_snapshots:
                    print(f"  ✅ Full snapshot exists: {full_snapshots[-1].name}")
                else:
                    warnings.append("No full snapshot found")
            else:
                issues.append("No snapshots found")
        else:
            issues.append("Snapshots directory not found")
            
    except Exception as e:
        issues.append(f"Failed to check snapshots: {e}")
    
    # 3. Check mini-index freshness
    print("\n🗂️  Checking mini-indices...")
    try:
        today = datetime.now().date()
        today_index = PSI_ARCHIVER.archive_dir / today.strftime("%Y-%m") / f"index-{today.isoformat()}.jsonl"
        
        if today_index.exists():
            print(f"  ✅ Today's mini-index exists")
        else:
            warnings.append("Today's mini-index not yet created (will be created on first event)")
            
        # Check index consistency
        if PSI_ARCHIVER.day_indices:
            print(f"  ✅ {len(PSI_ARCHIVER.day_indices)} day indices loaded")
        else:
            warnings.append("No day indices found in cache")
            
    except Exception as e:
        issues.append(f"Failed to check indices: {e}")
    
    # 4. Check disk space
    print("\n💾 Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(PSI_ARCHIVER.archive_dir)
        free_gb = free / (1024**3)
        usage_pct = (used / total) * 100
        
        if free_gb < 5:
            issues.append(f"Low disk space: {free_gb:.1f}GB free ({usage_pct:.0f}% used)")
        elif free_gb < 10:
            warnings.append(f"Disk space getting low: {free_gb:.1f}GB free ({usage_pct:.0f}% used)")
        else:
            print(f"  ✅ Disk space OK: {free_gb:.1f}GB free ({usage_pct:.0f}% used)")
            
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")
    
    # 5. Check clock sync (basic check)
    print("\n🕐 Checking system clock...")
    try:
        # Try to get NTP offset (platform-specific)
        if sys.platform == "win32":
            # Windows: w32tm /query /status
            result = subprocess.run(["w32tm", "/query", "/status"], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  ✅ Windows time service is running")
            else:
                warnings.append("Could not query Windows time service")
        else:
            # Unix: check if NTP is installed
            for ntp_cmd in ["ntpstat", "timedatectl"]:
                try:
                    result = subprocess.run([ntp_cmd], capture_output=True, 
                                            text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"  ✅ Time sync OK ({ntp_cmd})")
                        break
                except FileNotFoundError:
                    continue
            else:
                warnings.append("NTP not found - cannot verify time sync")
                
    except Exception as e:
        warnings.append(f"Could not check time sync: {e}")
    
    # 6. Check API health
    print("\n🌐 Checking API endpoint...")
    try:
        import requests
        response = requests.get("http://localhost:8000/api/archive/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ API is healthy")
            if verbose:
                print(f"     Events: ~{health.get('estimated_total_events', 0):,}")
                print(f"     Archive size: {health.get('current_file_size_bytes', 0) / 1024:.1f}KB")
        else:
            warnings.append(f"API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        warnings.append("API not running (start with: uvicorn api.enhanced_api_with_archive:app)")
    except Exception as e:
        warnings.append(f"Could not check API: {e}")
    
    # 7. Check Penrose cache
    print("\n🎯 Checking Penrose cache...")
    try:
        penrose_cache = Path(os.environ.get('PENROSE_CACHE', 'data/.penrose_cache'))
        if penrose_cache.exists():
            cache_files = list(penrose_cache.glob("*.npy"))
            print(f"  ✅ Penrose cache exists: {len(cache_files)} projectors cached")
            if verbose and cache_files:
                for cf in cache_files[:3]:  # Show first 3
                    print(f"     - {cf.name}")
        else:
            warnings.append("Penrose cache directory not found (will be created on first use)")
            
    except Exception as e:
        warnings.append(f"Could not check Penrose cache: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    if not issues and not warnings:
        print("  ✅ All systems healthy!")
        return 0
    
    if warnings:
        print(f"\n  ⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     - {w}")
    
    if issues:
        print(f"\n  ❌ {len(issues)} issue(s) found:")
        for i in issues:
            print(f"     - {i}")
        
        if fix:
            print("\n  🔧 Some issues were auto-fixed. Run again to verify.")
        else:
            print("\n  💡 Run with --fix to attempt automatic fixes")
    
    print("\n📝 Quick commands:")
    print("  - Seal archives:  python tools/cron_daily_seal.py")
    print("  - Create snapshot: python tools/psi_replay.py --until now")
    print("  - Start API:      uvicorn api.enhanced_api_with_archive:app")
    print("  - View events:    curl -N http://localhost:8000/api/archive/events | jq")
    
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(doctor())
