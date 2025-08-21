from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List
import os, json, pathlib, subprocess, sys, asyncio

ROOT = pathlib.Path(r"D:\Dev\kha").resolve()
ART = ROOT / "artifacts"
DEMO = ART / "demo"
LOG = ART / "ricci_burn_log.jsonl"
MANIFEST = DEMO / "manifest.json"
BUNDLE = DEMO / "bundle.zip"

router = APIRouter()


def _safe(relpath: str) -> pathlib.Path:
    """Resolve a relative path under ROOT, forbid traversal outside ROOT."""
    p = (ROOT / relpath).resolve()
    if not str(p).startswith(str(ROOT)):
        raise HTTPException(status_code=400, detail="Invalid path (outside ROOT)")
    return p


@router.get("/log")
async def get_log(as_list: bool = Query(False)):
    """Return the ricci_burn_log.jsonl (full). If as_list true, returns a JSON array."""
    if not LOG.exists():
        raise HTTPException(status_code=404, detail="ricci_burn_log.jsonl not found")
    lines = LOG.read_text(encoding="utf-8").splitlines()
    if as_list:
        items = []
        for ln in lines:
            try:
                items.append(json.loads(ln))
            except Exception:
                continue
        return JSONResponse(items)
    # default: return raw JSONL (text/plain)
    async def _gen():
        for ln in lines:
            yield (ln + "\n").encode("utf-8")
    return StreamingResponse(_gen(), media_type="text/plain")


@router.get("/manifest")
async def get_manifest():
    if not MANIFEST.exists():
        raise HTTPException(status_code=404, detail="manifest.json not found; run packaging first")
    try:
        return JSONResponse(json.loads(MANIFEST.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="manifest.json is not valid JSON")


@router.get("/files")
async def list_files() -> List[dict]:
    """List notable artifact files under artifacts/ and artifacts/demo/."""
    if not ART.exists():
        return []
    rows: List[dict] = []
    for root, _, fnames in os.walk(ART):
        for fn in fnames:
            full = pathlib.Path(root) / fn
            rel = full.relative_to(ROOT)
            if ("\\demo\\" in str(rel).lower()) or fn.lower().endswith((".npz",".json",".jsonl",".zip")):
                rows.append({
                    "path": str(rel),
                    "bytes": full.stat().st_size,
                    "mtime": int(full.stat().st_mtime)
                })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


@router.get("/download")
async def download(path: str = Query(..., description="Path relative to ROOT, e.g. artifacts/demo/bundle.zip")):
    p = _safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(p))


@router.post("/package")
async def package_demo(glob: str = Query('artifacts\\demo\\**\\*'), out: str = Query('artifacts\\demo\\bundle.zip')):
    """Run scripts/package_demo_artifacts.py to produce bundle.zip and manifest.json."""
    script = ROOT / "scripts" / "package_demo_artifacts.py"
    if not script.exists():
        raise HTTPException(status_code=404, detail="package_demo_artifacts.py not found")
    # Use the current Python interpreter
    cmd = [sys.executable, str(script), "--glob", glob, "--out", out]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"spawn failed: {e}")
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"packager failed: {stderr.decode('utf-8',errors='ignore')}")
    try:
        return JSONResponse(json.loads(stdout.decode("utf-8")))
    except Exception:
        return JSONResponse({"stdout": stdout.decode("utf-8"), "stderr": stderr.decode("utf-8")})
