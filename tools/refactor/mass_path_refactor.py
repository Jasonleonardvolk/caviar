# Create robust, resumable mass refactor tooling for Jason.
from pathlib import Path
from datetime import datetime

base = Path("/mnt/data")
base.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

mass_py = r'''#!/usr/bin/env python3
"""
mass_path_refactor_v2.py
Robust, resumable refactor that replaces absolute path prefixes across a large repo without hanging.

Features:
- Excludes heavy dirs by default (.git, .venv, node_modules, dist, build, tools/dawn, .cache, .pytest_cache, target, .idea, .vscode)
- Size cap per file (default 2 MB) to avoid huge/binary blobs
- Detects candidate files by scanning raw bytes for OLD prefix before decoding (fast)
- Python files: injects Path header once and replaces OLD with "{PROJECT_ROOT}"
- Other text files: replaces OLD with "${IRIS_ROOT}"
- Multiprocessing with --workers (safe for I/O bound with chunking)
- Resume: skips files already listed in state file when --resume is used
- Dry run: produces CSV plan with counts, no writes
- Logs: writes changes/errors under tools/refactor/

Usage (examples):
  python mass_path_refactor_v2.py --root D:\Dev\kha --old "{PROJECT_ROOT}"
  python mass_path_refactor_v2.py --root D:\Dev\kha --old "{PROJECT_ROOT}" --workers 8 --backup-dir D:\Backups\IrisRefactor
  python mass_path_refactor_v2.py --root D:\Dev\kha --old "{PROJECT_ROOT}" --dry-run --plan plan.csv

Notes:
- ${IRIS_ROOT}: for TS/JS/Svelte/WGSL/TXT/JSON you can resolve this at runtime from env/config.
- If you prefer a different token, use --text-token "IRIS_ROOT_DIR" (it will output ${IRIS_ROOT_DIR}).
"""
import argparse, os, sys, csv, shutil, json, time, ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "dist", "build", "out", ".cache", ".pytest_cache", "target",
    ".idea", ".vscode", "tools/dawn", "tools\\dawn"
]
DEFAULT_EXTS = [".py", ".ts", ".tsx", ".js", ".svelte", ".wgsl", ".txt", ".json", ".md", ".yaml", ".yml"]

HEADER_PY = "from pathlib import Path\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\n"

def is_probably_text(chunk: bytes) -> bool:
    # Basic heuristic: reject if NUL bytes or >30% high-bit noise
    if b"\x00" in chunk:
        return False
    if not chunk:
        return True
    high = sum(1 for b in chunk if b > 127)
    return (high / len(chunk)) < 0.30

def find_candidates(root: Path, old_bytes: bytes, include_exts, excludes, max_bytes):
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter dirs
        pruned = []
        for d in list(dirnames):
            rel = os.path.relpath(os.path.join(dirpath, d), root)
            # exclude if top-level dir matches or any component matches excludes
            if any(part in excludes for part in Path(rel).parts):
                continue
            pruned.append(d)
        dirnames[:] = pruned
        for fn in filenames:
            p = Path(dirpath) / fn
            if include_exts and p.suffix.lower() not in include_exts:
                continue
            try:
                sz = p.stat().st_size
                if sz > max_bytes:
                    continue
                with open(p, "rb") as f:
                    chunk = f.read(min(65536, sz))
                if not is_probably_text(chunk):
                    continue
                if old_bytes in chunk or (sz > 65536 and old_bytes in p.read_bytes()):
                    yield p
            except Exception:
                continue

def patch_python_text(raw: str, old: str) -> str:
    if old not in raw:
        return raw
    new = raw
    if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in new:
        if "from pathlib import Path" in new:
            new = new.replace("from pathlib import Path\n", HEADER_PY, 1)
        else:
            new = HEADER_PY + new
    new = new.replace(old, "{PROJECT_ROOT}")
    return new

def patch_textual(raw: str, old: str, token: str) -> str:
    if old not in raw:
        return raw
    return raw.replace(old, "${" + token + "}")

def worker(p: str, old: str, token: str, backup_dir: str|None) -> tuple[str, str, int]:
    path = Path(p)
    try:
        rawb = path.read_bytes()
    except Exception as e:
        return (p, "read-bytes-fail", 0)
    oldb = old.encode("utf-8", errors="ignore")
    if oldb not in rawb:
        return (p, "nochange", 0)
    # decode once
    try:
        raw = rawb.decode("utf-8", errors="ignore")
    except Exception as e:
        return (p, "decode-fail", 0)
    if path.suffix.lower() == ".py":
        new = patch_python_text(raw, old)
    else:
        new = patch_textual(raw, old, token)
    if new == raw:
        return (p, "nochange", 0)
    try:
        if backup_dir:
            backup_path = Path(backup_dir) / path.relative_to(Path(args.root))
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                backup_path.write_bytes(rawb)
        path.write_text(new, encoding="utf-8")
        count = new.count("{PROJECT_ROOT}") + new.count("${" + token + "}")
        return (p, "patched", count)
    except Exception as e:
        return (p, "write-fail", 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root (e.g., D:\\Dev\\kha)")
    ap.add_argument("--old", required=True, help="Absolute path prefix to replace")
    ap.add_argument("--include-exts", default=",".join(DEFAULT_EXTS), help="Comma-separated extensions")
    ap.add_argument("--exclude-dirs", default="", help="Comma-separated extra dirs to exclude")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max file size to process")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", default="tools/refactor/refactor_plan.csv")
    ap.add_argument("--resume", action="store_true", help="Skip files already in state file")
    ap.add_argument("--state", default="tools/refactor/refactor_state.json")
    ap.add_argument("--backup-dir", default="", help="If set, saves originals of changed files under this dir")
    ap.add_argument("--text-token", default="IRIS_ROOT", help="Token name for non-Python replacements (${TOKEN})")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print("Root not found:", root)
        sys.exit(2)

    ref_dir = root / "tools" / "refactor"
    ref_dir.mkdir(parents=True, exist_ok=True)
    plan_csv = root / args.plan
    state_json = root / args.state

    include_exts = [e.strip().lower() for e in args.include_exts.split(",") if e.strip()]
    excludes = set(DEFAULT_EXCLUDES + [e.strip() for e in args.exclude_dirs.split(",") if e.strip()])

    # Resume state
    done: set[str] = set()
    if args.resume and state_json.exists():
        try:
            done = set(json.loads(state_json.read_text()))
        except Exception:
            done = set()

    old = args.old
    old_bytes = old.encode("utf-8", errors="ignore")

    # Find candidates
    candidates = []
    for p in find_candidates(root, old_bytes, include_exts, excludes, args.max_bytes):
        rp = str(p)
        if args.resume and rp in done:
            continue
        candidates.append(p)

    # DRY RUN: produce a plan CSV with counts only (no writes)
    if args.dry_run:
        with open(plan_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "occurrences"])
            for p in candidates:
                try:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    cnt = raw.count(old)
                    if cnt > 0:
                        w.writerow([str(p), cnt])
                except Exception:
                    continue
        print(f"Dry run complete. Plan at {plan_csv} (candidates: {len(candidates)})")
        sys.exit(0)

    # Parallel patching
    changes_log = (ref_dir / "refactor_changes.log").open("a", encoding="utf-8")
    errors_log = (ref_dir / "refactor_errors.log").open("a", encoding="utf-8")
    total = len(candidates)
    patched = 0
    start = time.time()

    backup_dir = args.backup_dir or ""
    if backup_dir:
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, str(p), old, args.text_token, backup_dir): str(p) for p in candidates}
        i = 0
        for fut in as_completed(futs):
            i += 1
            p = futs[fut]
            try:
                path, status, cnt = fut.result()
            except Exception as e:
                path, status, cnt = p, "worker-exc", 0
            if status == "patched":
                patched += 1
                changes_log.write(f"{path} :: {status} :: occurrences={cnt}\n")
            elif status not in ("nochange",):
                errors_log.write(f"{path} :: {status}\n")

            # progress heartbeat every 200 files
            if i % 200 == 0 or i == total:
                elapsed = time.time() - start
                print(f"[{i}/{total}] patched={patched} elapsed={elapsed:.1f}s")

            # update resume state every 500 files
            if args.resume and i % 500 == 0:
                try:
                    done.update([p for p in list(done)])
                    state_json.write_text(json.dumps(list(done | set([p]))))
                except Exception:
                    pass

    # final state save
    if args.resume:
        try:
            done.update([str(p) for p in candidates])
            state_json.write_text(json.dumps(list(done)))
        except Exception:
            pass

    changes_log.close()
    errors_log.close()
    print(f"Done. candidates={total}, patched={patched}. Logs under {ref_dir}")
'''
