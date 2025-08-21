"""
TORI Memory Vault Production Verification
Quick sanity checks after production configuration
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, r"D:\Dev\kha")

def check_env_vars():
    """Verify production environment variables"""
    print("="*60)
    print("ENVIRONMENT CONFIGURATION")
    print("="*60)
    
    snapshot_interval = os.environ.get("TORI_VAULT_SNAPSHOT_INTERVAL_SEC", "not set")
    session_rotate = os.environ.get("TORI_VAULT_SESSION_ROTATE_SEC", "not set")
    vault_dir = os.environ.get("TORI_VAULT_DIR", r"D:\Dev\kha\data\memory_vault")
    
    print(f"Snapshot Interval: {snapshot_interval} seconds")
    print(f"Session Rotation:  {session_rotate} seconds")
    print(f"Vault Directory:   {vault_dir}")
    
    if snapshot_interval == "300" and session_rotate == "3600":
        print("✅ Production intervals configured correctly")
    else:
        print("⚠️ Non-production intervals detected")
    
    return vault_dir

def check_vault_health(vault_dir):
    """Check vault file health"""
    print("\n" + "="*60)
    print("VAULT HEALTH CHECK")
    print("="*60)
    
    vault_path = Path(vault_dir)
    logs_dir = vault_path / "logs"
    
    # Check directories
    if not vault_path.exists():
        print(f"❌ Vault directory missing: {vault_path}")
        return False
    
    print(f"✅ Vault directory exists: {vault_path}")
    
    # Check files
    live_log = logs_dir / "vault_live.jsonl"
    snapshot = logs_dir / "vault_snapshot.json"
    
    if live_log.exists():
        size_mb = live_log.stat().st_size / (1024 * 1024)
        print(f"✅ Live log: {size_mb:.2f} MB")
    else:
        print("⚠️ Live log missing (will be created on first write)")
    
    if snapshot.exists():
        size_mb = snapshot.stat().st_size / (1024 * 1024)
        print(f"✅ Snapshot: {size_mb:.2f} MB")
        
        # Check snapshot validity
        try:
            with open(snapshot, 'r') as f:
                data = json.load(f)
                entry_count = data.get('entry_count', 0)
                timestamp = data.get('timestamp', 'unknown')
                print(f"   - Entries: {entry_count}")
                print(f"   - Created: {timestamp}")
        except Exception as e:
            print(f"⚠️ Snapshot may be corrupted: {e}")
    else:
        print("⚠️ Snapshot missing (will be created after first interval)")
    
    # Check sessions
    sessions = list(logs_dir.glob("session_*.jsonl"))
    print(f"✅ Active sessions: {len(sessions)}")
    if len(sessions) > 50:
        print("⚠️ Consider running vault rotation (>50 session files)")
    
    # Check archive
    archive_dir = vault_path / "archive"
    if archive_dir.exists():
        archives = list(archive_dir.glob("*.zip"))
        print(f"✅ Archived sessions: {len(archives)}")
    
    return True

def test_api_endpoint():
    """Test memory API endpoint if running"""
    print("\n" + "="*60)
    print("API ENDPOINT TEST")
    print("="*60)
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:8002/api/memory/state/admin", timeout=2)
        if response.ok:
            data = response.json()
            print("✅ API endpoint responding")
            print(f"   - Live log: {data.get('live_log_bytes', 0)} bytes")
            print(f"   - Snapshot: {data.get('snapshot_bytes', 0)} bytes")
            print(f"   - Sessions: {data.get('sessions_count', 0)}")
            if 'fsm_coherence' in data:
                print(f"   - FSM Coherence: {data['fsm_coherence']:.3f}")
        else:
            print(f"⚠️ API returned status {response.status_code}")
    except Exception as e:
        print("ℹ️ API not running or not accessible (this is OK if launcher isn't running)")

def run_vault_inspector(vault_dir):
    """Run the vault inspector tool"""
    print("\n" + "="*60)
    print("VAULT INSPECTOR")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, r"D:\Dev\kha\tools\vault_inspector.py", "--dir", vault_dir],
            capture_output=True, text=True, timeout=5
        )
        print(result.stdout)
    except Exception as e:
        print(f"Could not run vault inspector: {e}")

def main():
    print("="*60)
    print("TORI MEMORY VAULT - PRODUCTION VERIFICATION")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    # Run checks
    vault_dir = check_env_vars()
    vault_health = check_vault_health(vault_dir)
    test_api_endpoint()
    run_vault_inspector(vault_dir)
    
    # Summary
    print("\n" + "="*60)
    print("PRODUCTION READINESS SUMMARY")
    print("="*60)
    
    if vault_health:
        print("✅ Vault structure healthy")
    else:
        print("❌ Vault structure needs attention")
    
    print("\nNext steps:")
    print("1. Run launcher: powershell -ExecutionPolicy Bypass -File D:\\Dev\\kha\\START_TORI_BULLETPROOF_NOW.ps1")
    print("2. Schedule rotation: D:\\Dev\\kha\\tools\\schedule_vault_rotation.bat (run as Admin)")
    print("3. Monitor at: http://localhost:3000/admin/memory")

if __name__ == "__main__":
    main()
