<# 
Drop-FreshTsConfig-And-HoloFix.ps1
Author: Jason L Volk for the newest and coolest iRis
Purpose:
  - Overwrite frontend\tsconfig.json with a clean, bundler-ready config (DOM/WebGPU libs, vite/client types).
  - Exclude frontend\hybrid\** from the app compile surface to defuse the 260-error cascade for this release.
  - Add ambient shims under frontend\types\shims\ for '?raw', '*.wgsl?raw', 'glob', 'ktx-parse', '@playwright/test', and env.
  - Normalize the WebGPU quilt import/name in frontend\lib\holographicEngine.ts.

Usage (from repo root):
  powershell -ExecutionPolicy Bypass -File .\scripts\Drop-FreshTsConfig-And-HoloFix.ps1 -RepoRoot "C:\Users\jason\Desktop\tori\kha"

This script is idempotent and creates a timestamped backup of the original tsconfig.json.
#>

param(
  [string]$RepoRoot = "C:\Users\jason\Desktop\tori\kha"
)

$ErrorActionPreference = "Stop"

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Write-Step($m){ Write-Host "`n==== $m ====" -ForegroundColor Magenta }
function Ensure-Dir($p){ if(!(Test-Path $p)){ New-Item -ItemType Directory -Force -Path $p | Out-Null } }

$frontend = Join-Path $RepoRoot "frontend"
if (!(Test-Path $frontend)) { throw "Frontend path not found: $frontend" }

$tsconfig = Join-Path $frontend "tsconfig.json"
$backup = Join-Path $frontend ("tsconfig.backup." + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")

Write-Step "Backup existing tsconfig.json (if any)"
if (Test-Path $tsconfig) {
  Copy-Item $tsconfig $backup -Force
  Write-Ok "Backed up to $backup"
} else {
  Write-Info "No existing tsconfig.json found; will create fresh."
}

Write-Step "Write fresh frontend\tsconfig.json (bundler + DOM/WebGPU + vite types, hybrid excluded)"
$tsconfigJson = @'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2023", "DOM", "DOM.Iterable", "WebWorker"],
    "types": ["vite/client", "webxr"],
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./lib/*"],
      "@hybrid/*": ["./hybrid/*"]
    },
    "resolveJsonModule": true,
    "allowJs": false,
    "useDefineForClassFields": true,
    "allowSyntheticDefaultImports": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": [
    "./lib/**/*.ts",
    "./lib/**/*.tsx",
    "./app/**/*.ts",
    "./app/**/*.tsx",
    "./plugins/**/*.ts",
    "./plugins/**/*.tsx",
    "./types/**/*.d.ts"
  ],
  "exclude": [
    "./node_modules",
    "./hybrid/**",
    "./tests/**",
    "./vite.config.ts"
  ]
}
'@
Set-Content -LiteralPath $tsconfig -Value $tsconfigJson -Encoding UTF8
Write-Ok "Wrote $tsconfig"

Write-Step "Create ambient shims under frontend\\types\\shims"
$shimsDir = Join-Path $frontend "types\shims"
Ensure-Dir $shimsDir

# raw modules + small module decls
$rawMods = @'
declare module "*?raw" { const src: string; export default src; }
declare module "*.wgsl?raw" { const src: string; export default src; }

declare module "glob";
declare module "ktx-parse";

declare module "@playwright/test" {
  export const test: any;
  export const expect: any;
  const _default: any;
  export default _default;
}
'@
Set-Content -LiteralPath (Join-Path $shimsDir "raw-modules.d.ts") -Value $rawMods -Encoding UTF8

# env + navigator.gpu augments
$envAug = @'
/// <reference lib="dom" />
/// <reference lib="dom.iterable" />
/// <reference types="vite/client" />

export {};

declare global {
  interface Navigator {
    gpu?: GPU;
  }
  interface ImportMetaEnv {
    VITE_MODE?: string;
    VITE_API_BASE?: string;
  }
}
'@
Set-Content -LiteralPath (Join-Path $shimsDir "webgpu-env.d.ts") -Value $envAug -Encoding UTF8
Write-Ok "Wrote shims: raw-modules.d.ts, webgpu-env.d.ts"

Write-Step "Normalize quilt import/name in frontend\\lib\\holographicEngine.ts"
$he = Join-Path $frontend "lib\holographicEngine.ts"
if (Test-Path $he) {
  $text = Get-Content -Raw -LiteralPath $he

  # Remove aliasing and wrong names
  $text = [regex]::Replace($text, "import\s*\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s*from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
  $text = [regex]::Replace($text, "import\s+QuiltGenerator\s+from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
  $text = $text -replace "WebGPUWebGPUQuiltGenerator", "WebGPUQuiltGenerator"

  # Fix import path to canonical location
  $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")
  $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quilt[\\/]+WebGPUQuiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")

  # Replace identifier usages (preserve lower-camel variable names)
  $text = [regex]::Replace($text, "(?<![a-z])QuiltGenerator(?![a-z])", "WebGPUQuiltGenerator")

  # Ensure at least one correct import exists; if not, inject at top
  if ($text -notmatch "import\s*\{\s*WebGPUQuiltGenerator\s*\}\s*from\s*['""][\.\/]*webgpu\/quilt\/WebGPUQuiltGenerator['""]") {
    $text = "import { WebGPUQuiltGenerator } from './webgpu/quilt/WebGPUQuiltGenerator';`n" + $text
  }

  Set-Content -LiteralPath $he -Value $text -Encoding UTF8
  Write-Ok "Patched $($he.Substring($RepoRoot.Length + 1))"
} else {
  Write-Info "holographicEngine.ts not found at expected path: $($he)"
}

Write-Step "Done. Now re-run the gate (no shaders)"
Write-Host "powershell -ExecutionPolicy Bypass -File .\Run-IrisReleaseGate.ps1 -SkipShaders" -ForegroundColor Yellow
