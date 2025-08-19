from pathlib import Path
from datetime import datetime

base = Path("/mnt/data")
base.mkdir(exist_ok=True, parents=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ps_patch = r'''<#
Apply-Iris-Fixes.ps1
Purpose:
  - Fix Quilt generator path references
  - Add hard guard against tools\dawn
  - Create missing TS stubs so the build can progress
  - Report all changes

Usage:
  powershell -ExecutionPolicy Bypass -File .\Apply-Iris-Fixes.ps1 -RepoRoot "D:\Dev\kha"
#>

param(
  [string]$RepoRoot = "D:\Dev\kha"
)

function Ensure-Dir($p) { if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }
function Write-Change($msg) { Write-Host ("[CHANGE] " + $msg) }
function Write-Info($msg) { Write-Host ("[INFO] " + $msg) }
function Replace-InFile($file, $pattern, $replacement) {
  if (!(Test-Path $file)) { return $false }
  $text = Get-Content -Raw -Path $file
  $new = $text -replace $pattern, $replacement
  if ($new -ne $text) {
    Set-Content -Path $file -Value $new -Encoding UTF8
    Write-Change "$file : replaced '$pattern' -> '$replacement'"
    return $true
  }
  return $false
}

# 0) Guards / constants
$quiltGenRel = "tools\quilt\WebGPU\QuiltGenerator.ts"
$quiltGenFull = Join-Path $RepoRoot $quiltGenRel

# 1) Patch IrisOneButton.ps1 and Run-IrisReleaseGate.ps1
$IrisOneButton = Join-Path $RepoRoot "tools\release\IrisOneButton.ps1"
$ReleaseGate   = Join-Path $RepoRoot "Run-IrisReleaseGate.ps1"

$patterns = @(
  "tools\\quilt\\WebGPUQuiltGenerator\.ts",          # wrong old name
  "tools\\quilt\\WebGPU\\WebGPUQuiltGenerator\.ts",  # wrong folder + name
  "tools\\quilt\\QuiltGenerator\.ts"                 # missing WebGPU subfolder
)

foreach ($pat in $patterns) {
  Replace-InFile $IrisOneButton $pat ([regex]::Escape($quiltGenRel)) | Out-Null
  Replace-InFile $ReleaseGate   $pat ([regex]::Escape($quiltGenRel)) | Out-Null
}

# Ensure the Quilt generator exists (create skeleton if missing)
if (!(Test-Path $quiltGenFull)) {
  Ensure-Dir (Split-Path -Parent $quiltGenFull)
  @"
/**
 * tools/quilt/WebGPU/QuiltGenerator.ts
 * Skeleton generator: copies canonical WGSL from
 *   frontend/lib/webgpu/shaders/** -> frontend/public/hybrid/wgsl/**
 * Replace with your real logic when ready.
 */
import { cpSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

const repo = process.cwd();
const src = join(repo, 'frontend', 'lib', 'webgpu', 'shaders');
const dst = join(repo, 'frontend', 'public', 'hybrid', 'wgsl');

if (!existsSync(src)) {
  console.error('Canonical shader dir missing:', src);
  process.exit(2);
}
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log('Quilt generator copied', src, '->', dst);
"@ | Set-Content -Path $quiltGenFull -Encoding UTF8
  Write-Change "Created skeleton $quiltGenRel"
}

# 2) Add HARD guard to Run-IrisReleaseGate.ps1 (if not present)
if (Test-Path $ReleaseGate) {
  $guard = '$DawnPath = Join-Path $RepoRoot "tools\dawn"'
  $gateText = Get-Content -Raw -Path $ReleaseGate
  if ($gateText -notmatch [regex]::Escape($guard)) {
    $inject = @"
# HARD GUARD: forbid vendoring Dawn/ANGLE/Tests into repo
$guard
if (Test-Path $DawnPath) {
  Write-Host "ERROR: Remove $DawnPath (3rd-party repo/test suites). Use tools\shaders\bin\tint.exe instead."
  exit 2
}
"@
    # Put guard after param() block
    $gateText = $gateText -replace "(?s)(param\([^\)]*\)\s*)", "`$1`r`n$inject`r`n"
    Set-Content -Path $ReleaseGate -Value $gateText -Encoding UTF8
    Write-Change "Injected Dawn guard into Run-IrisReleaseGate.ps1"
  } else {
    Write-Info "Dawn guard already present in Run-IrisReleaseGate.ps1"
  }
}

# 3) Create missing TS stubs (only if they don't exist)
$stubs = @{
  "frontend\lib\webgpu\validateDeviceLimits.ts" = @"
export type LimitCheck = { name: string; required: number; actual: number; ok: boolean };
export async function validateDeviceLimits(device: GPUDevice, required: Partial<GPUDeviceLimits> = {}): Promise<{ ok: boolean; checks: LimitCheck[]; limits: GPUDeviceLimits }>
{
  const limits = device.limits;
  const req: any = required ?? {};
  const checks: LimitCheck[] = Object.keys(limits).map((k: any) => {
    const actual = (limits as any)[k] as number;
    const need = (req as any)[k] as number | undefined;
    const ok = need == null ? true : actual >= need;
    return { name: String(k), required: need ?? -1, actual, ok };
  });
  const ok = checks.every(c => c.ok);
  return { ok, checks, limits };
}
export default validateDeviceLimits;
"@;

  "frontend\lib\hybrid\cpu_projector.ts" = @"
export interface CPUProjector {
  project(input: Float32Array, width: number, height: number): Float32Array;
}
export class SimpleCPUProjector implements CPUProjector {
  project(input: Float32Array, _w: number, _h: number): Float32Array {
    // Identity projector placeholder
    return input;
  }
}
export default SimpleCPUProjector;
"@;

  "frontend\lib\webgpu\pipelines\ios26_compositor.ts" = @"
export interface IOS26CompositorConfig { numViews: number; }
export class IOS26Compositor {
  constructor(public device: GPUDevice, public config: IOS26CompositorConfig) {}
  compose(): void {
    // TODO: implement compositor
  }
}
export default IOS26Compositor;
"@
}

foreach ($rel in $stubs.Keys) {
  $full = Join-Path $RepoRoot $rel
  if (!(Test-Path $full)) {
    Ensure-Dir (Split-Path -Parent $full)
    Set-Content -Path $full -Value $stubs[$rel] -Encoding UTF8
    Write-Change "Created stub: $rel"
  } else {
    Write-Info "Exists: $rel"
  }
}

# 4) Ensure engine.ts exports minimal symbols if present
$engine = Join-Path $RepoRoot "frontend\lib\webgpu\engine.ts"
if (Test-Path $engine) {
  $content = Get-Content -Raw -Path $engine
  if ($content -notmatch "export\s+class\s+WebGPUEngine" -and $content -notmatch "export\s+default") {
    Add-Content -Path $engine -Value @"

// Minimal export to satisfy imports during refactor
export class WebGPUEngine {
  device!: GPUDevice;
  constructor(public canvas?: HTMLCanvasElement) {}
}
export default WebGPUEngine;
"@ -Encoding UTF8
    Write-Change "Patched engine.ts with minimal exports"
  } else {
    Write-Info "engine.ts already exports symbols"
  }
}

Write-Host ""
Write-Host "Complete. Review the [CHANGE] lines above. Re-run your gate when ready."
'''

refactor_py = r'''"""
mass_path_refactor.py
- Replaces absolute "C:\Users\jason\Desktop\tori\kha" prefixes with dynamic project root usage.
- For Python files: injects `from pathlib import Path` and `PROJECT_ROOT = ...` if needed.
- For TS/JS/Svelte: replaces the absolute prefix string with an ENV token ${IRIS_ROOT}.
  (This is conservative textual replacement; adjust callers to read process.env.IRIS_ROOT or a config if needed.)

Usage:
  python mass_path_refactor.py D:\Dev\kha
"""

import sys, os
from pathlib import Path

ROOT_ABS = r"C:\Users\jason\Desktop\tori\kha"

if len(sys.argv) != 2:
    print("Usage: python mass_path_refactor.py <repo_root>")
    sys.exit(2)

repo = Path(sys.argv[1]).resolve()
if not repo.exists():
    print("Repo not found:", repo)
    sys.exit(2)

GLOBS = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.svelte", "**/*.txt", "**/*.wgsl", "**/*.json"]

def patch_python(text: str) -> str:
    if ROOT_ABS not in text:
        return text
    header = "from pathlib import Path\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\n"
    if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in text:
        if "from pathlib import Path" in text:
            text = text.replace("from pathlib import Path\n", header, 1)
        else:
            text = header + text
    return text.replace(ROOT_ABS, "{PROJECT_ROOT}")

def patch_textual(text: str) -> str:
    if ROOT_ABS not in text:
        return text
    return text.replace(ROOT_ABS, "${IRIS_ROOT}")

changed = 0
for pattern in GLOBS:
    for p in repo.glob(pattern):
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        new = raw
        if p.suffix == ".py":
            new = patch_python(raw)
        else:
            new = patch_textual(raw)
        if new != raw:
            p.write_text(new, encoding="utf-8")
            print(f"[patched] {p}")
            changed += 1

print(f"Done. Files changed: {changed}")
'''

(base / "Apply-Iris-Fixes.ps1").write_text(ps_patch, encoding="utf-8")
(base / "mass_path_refactor.py").write_text(refactor_py, encoding="utf-8")
(base / "IRIS_FIXES_README.txt").write_text(f"Artifacts created at {timestamp}", encoding="utf-8")

print("Files created:")
for f in ["Apply-Iris-Fixes.ps1", "mass_path_refactor.py", "IRIS_FIXES_README.txt"]:
    print(" -", base / f)
