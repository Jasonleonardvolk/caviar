from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\Dev\kha\data\memory_vault", help="Vault directory")
    args = ap.parse_args()

    root = Path(args.dir)
    logs = root / "logs"
    live = logs / "vault_live.jsonl"
    snap = logs / "vault_snapshot.json"

    print(f"[vault] {root}")
    print(f"  live_log:    {live}  {'OK' if live.exists() else 'MISSING'}")
    print(f"  snapshot:    {snap}  {'OK' if snap.exists() else 'MISSING'}")
    if live.exists():
        print(f"  live_bytes:  {live.stat().st_size}")
    if snap.exists():
        print(f"  snap_bytes:  {snap.stat().st_size}")

    sessions = sorted([p.name for p in logs.glob("session_*.jsonl")]) if logs.exists() else []
    print(f"  sessions:    {len(sessions)}")
    for s in sessions[-5:]:
        print(f"    - {s}")

if __name__ == "__main__":
    main()
