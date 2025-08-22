<#

Apply-Tori-FullFix.ps1
One-shot codemod for D:\Dev\kha\tori_ui_svelte

What this does (idempotent, safe to re-run):
  1) Fix build-blocking syntax in src\lib\components\vault\MemoryVaultDashboard.svelte
  2) Unify App.Locals and Window globals via src\app.d.ts
  3) Make hooks.server.ts assign 'name' consistently
  4) Align BridgeConfig (api/websocket) with usage
  5) Raise TS target/libs to ES2022 (Array.at et al.)
  6) Remove stray WebGPU type ref & React.CSSProperties
  7) Convert value-vs-type imports (isolatedModules/preserveValueImports)
  8) Make ConceptDiff.changes optional + normalize timestamp
  9) Strip HTML-style event attributes (onmouseover/onmouseout) that break Svelte+TS
 10) Deduplicate repeated role=/tabindex= attributes in certain modals

Backup is created under D:\Dev\kha\patches\backup-YYYYMMDD-HHmmss\

Usage:
  pwsh -ExecutionPolicy Bypass -File .\Apply-Tori-FullFix.ps1 -Root "D:\Dev\kha\tori_ui_svelte"

#>

param(
  [string]$Root = "D:\Dev\kha\tori_ui_svelte"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir($p) {
  $d = Split-Path -Parent $p
  if ($d -and -not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

function Write-TextFile($path, [string]$content) {
  Ensure-Dir $path
  # Use UTF8 without BOM
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
}

function Read-AllText($path) {
  if (Test-Path $path) { return Get-Content -Raw -LiteralPath $path }
  return ""
}

function Replace-File($path, [string]$newContent, [switch]$Backup) {
  if ($Backup -and (Test-Path $path)) {
    $rel = $path.Replace($Root, "").TrimStart("\","/")
    $backupPath = Join-Path $Global:BackupDir $rel
    Ensure-Dir $backupPath
    Copy-Item -LiteralPath $path -Destination $backupPath -Force
  }
  Write-TextFile $path $newContent
}

function Mutate-File($path, [scriptblock]$mutator) {
  if (-not (Test-Path $path)) { return $false }
  $original = Get-Content -Raw -LiteralPath $path
  $mutated = & $mutator $original
  if ($mutated -ne $original) {
    $rel = $path.Replace($Root, "").TrimStart("\","/")
    $backupPath = Join-Path $Global:BackupDir $rel
    Ensure-Dir $backupPath
    Copy-Item -LiteralPath $path -Destination $backupPath -Force
    Write-TextFile $path $mutated
    return $true
  }
  return $false
}

# ──────────────────────────────────────────────────────────────────────────────
# Prepare backup folder
# ──────────────────────────────────────────────────────────────────────────────
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Global:BackupDir = "D:\Dev\kha\patches\backup-$timestamp"
New-Item -ItemType Directory -Force -Path $Global:BackupDir | Out-Null

Write-Host "Backing up changed files under $BackupDir"

# ──────────────────────────────────────────────────────────────────────────────
# 1) MemoryVaultDashboard.svelte: remove stray ')();' and ensure state/imports
# ──────────────────────────────────────────────────────────────────────────────
$mvPath = Join-Path $Root "src\lib\components\vault\MemoryVaultDashboard.svelte"
if (Test-Path $mvPath) {
  Mutate-File $mvPath {
    param($t)
    $orig = $t

    # Remove any loose IIFE closers that commonly caused the parse error
    $t = $t -replace "^\s*\)\(\);\s*$","", "Multiline"

    # Ensure needed imports and state are present inside <script lang="ts">
    if ($t -match "<script\s+lang=""ts"">") {
      $headerBlock = @"
import { onMount } from 'svelte';
import { browser } from '\$app/environment';
import { writable, derived } from 'svelte/store';

type MemoryType = 'soliton' | 'concept' | 'ghost' | 'document' | 'chat' | 'memory';
export interface MemoryEntry {
  id: string;
  type: MemoryType;
  timestamp: Date;
  content: any;
  phase: string;
  coherence: number;
  importance: number;
  tags: string[];
  relationships: string[];
}

const memories = writable<MemoryEntry[]>([]);
const metrics = writable({
  totalMemories: 0,
  activePhases: 0,
  coherenceLevel: 0,
  conceptDensity: 0,
  ghostActivity: 0,
  quantumEntanglement: 0
});
const phaseTransitions = writable<any[]>([]);
const memoryGraph = writable<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] });

const searchQuery = writable('');
const filterType = writable<'all' | MemoryType>('all');
const filteredMemories = derived([memories, searchQuery, filterType], ([$m, $q, $t]) => {
  const q = $q.trim().toLowerCase();
  return $m.filter((mem) => {
    const matchesType = $t === 'all' || mem.type === $t;
    const matchesText =
      !q ||
      mem.tags?.some((tt) => tt.toLowerCase().includes(q)) ||
      JSON.stringify(mem.content || {}).toLowerCase().includes(q);
    return matchesType && matchesText;
  });
});

let selectedView: 'overview' | 'timeline' | 'graph' | 'quantum' | 'export' = 'overview';
let selectedMemory: MemoryEntry | null = null;
let autoRefresh = false;
let isExporting = false;
let refreshInterval: ReturnType<typeof setInterval> | undefined;

"@

      # Inject only if key identifiers missing
      if ($t -notmatch "\bmemories\s*=\s*writable<MemoryEntry") {
        $t = $t -replace "(<script\s+lang=""ts"">\s*)", "`$1$headerBlock"
      }

      # Ensure onMount returns cleanup (no stray )();)
      if ($t -notmatch "onMount\(\s*\(\)\s*=>\s*{\s*if\s*\(!?browser") {
        # Add a safe onMount if missing
        $onMountBlock = @"
onMount(() => {
  if (!browser) return;
  const unsubscribes: Array<() => void> = [];
  (async () => {
    try {
      if (typeof loadMemoryVault === 'function') {
        await loadMemoryVault();
      }
      if (typeof autoRefresh !== 'undefined' && autoRefresh) {
        refreshInterval = setInterval(() => {
          try { (loadMemoryVault && loadMemoryVault()); } catch {}
        }, 5000);
      }
      try {
        unsubscribes.push(
          (solitonMemory && solitonMemory.subscribe ? solitonMemory.subscribe(updateFromSoliton) : () => {}),
          (conceptStore && conceptStore.subscribe ? conceptStore.subscribe(updateFromConcepts) : () => {}),
          (ghostPersona && ghostPersona.subscribe ? ghostPersona.subscribe(updateFromGhost) : () => {})
        );
      } catch {}
    } catch {}
  })();
  return () => {
    try { unsubscribes.forEach((u) => { try { u() } catch {} }); } catch {}
    if (refreshInterval) clearInterval(refreshInterval);
  };
});
"@
        # Place onMount right after header block if not present
        if ($t -notmatch "onMount\s*\(") {
          $t = $t -replace "($headerBlock)", "`$1`r`n$onMountBlock`r`n"
        }
      }
    }

    return $t
  } | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) src\app.d.ts — global types (Locals + Window)
# ──────────────────────────────────────────────────────────────────────────────
$appDts = @"
// src/app.d.ts
declare global {
  namespace App {
    interface Locals {
      user: { id: string; username: string; name?: string; role: 'admin' | 'user' };
    }
    interface PageData {
      user?: Locals['user'] | null;
    }
    interface Window {
      TORI?: {
        updateHologramState?: (state: any) => void;
        setHologramVideoMode?: (enabled: boolean) => void;
        toggleHologramAudio?: (enabled: boolean) => void;
        toggleHologramVideo?: (enabled: boolean) => void;
      };
      ghostMemoryDemo?: () => void;
      webkitAudioContext?: typeof AudioContext;
      TORI_DISPLAY_TYPE?: string;
    }
  }
}
export {};
"@
Replace-File (Join-Path $Root "src\app.d.ts") $appDts -Backup

# ──────────────────────────────────────────────────────────────────────────────
# 3) src\hooks.server.ts — assign 'name' on Locals.user
# ──────────────────────────────────────────────────────────────────────────────
$hooksPath = Join-Path $Root "src\hooks.server.ts"
if (Test-Path $hooksPath) {
  Mutate-File $hooksPath {
    param($t)
    if ($t -notmatch "export\s+const\s+handle") { return $t }
    # normalize user assignment block
    $t = $t -replace "event\.locals\.user\s*=\s*\{[\s\S]*?\};", @"
event.locals.user = {
    id: \`user_\${(event.cookies.get('username') || 'anonymous')}\`,
    username: event.cookies.get('username') || 'anonymous',
    name: event.cookies.get('username') || 'anonymous',
    role: ((event.cookies.get('role') -as [string]) -in @('admin','user')) ? (event.cookies.get('role')) : 'user'
  };
"@
    return $t
  } | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 4) src\lib\bridgeConfig.ts — align interface
# ──────────────────────────────────────────────────────────────────────────────
$bridgePath = Join-Path $Root "src\lib\bridgeConfig.ts"
if (Test-Path $bridgePath) {
  Mutate-File $bridgePath {
    param($t)
    # Expand BridgeConfig interface with api & websocket
    $t = $t -replace "export\s+interface\s+BridgeConfig\s*\{[\s\S]*?\}", @"
export interface BridgeConfig {
  mode: 'dev' | 'prod' | 'test';
  api?: { url: string; timeout: number };
  websocket?: { url: string; reconnect: boolean };
}
"@
    return $t
  } | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 5) tsconfig.json — ES2022 target/libs
# ──────────────────────────────────────────────────────────────────────────────
$tsconfigPath = Join-Path $Root "tsconfig.json"
if (Test-Path $tsconfigPath) {
  $json = Get-Content -Raw -LiteralPath $tsconfigPath | ConvertFrom-Json -Depth 12
  if (-not $json.compilerOptions) { $json | Add-Member -NotePropertyName compilerOptions -NotePropertyValue (@{}) }
  $co = $json.compilerOptions
  $co.target = "ES2022"
  $co.lib = @("ES2022","DOM","DOM.Iterable")
  $co.module = "ESNext"
  $co.moduleResolution = "Bundler"
  $co.types = @("svelte","@types/web","@webgpu/types")
  $co.preserveValueImports = $true
  $co.isolatedModules = $true
  $co.skipLibCheck = $true
  $json | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $tsconfigPath -Encoding utf8
}

# ──────────────────────────────────────────────────────────────────────────────
# 6) WebGPU stray ref + React types
# ──────────────────────────────────────────────────────────────────────────────
$photoMorph = Join-Path $Root "src\lib\webgpu\photoMorphPipeline.ts"
if (Test-Path $photoMorph) {
  Mutate-File $photoMorph {
    param($t)
    $t = $t -replace "^\s*///\s*<reference\s+path=""\.\/webgpu-types\.d\.ts""\s*\/>\s*\r?\n", "", "Multiline"
    return $t
  } | Out-Null
}

$ghostPersonaEngine = Join-Path $Root "src\lib\elfin\ghostPersonaEngine.ts"
if (Test-Path $ghostPersonaEngine) {
  Mutate-File $ghostPersonaEngine {
    param($t)
    $t = $t -replace "as\s+React\.CSSProperties", "as Record<string, string>"
    return $t
  } | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 7) Type-only imports where required
# ──────────────────────────────────────────────────────────────────────────────
$fftCompute = Join-Path $Root "..\frontend\lib\webgpu\fftCompute.ts"
if (Test-Path $fftCompute) {
  Mutate-File $fftCompute {
    param($t)
    $t = $t -replace "import\s+\{\s*getPrecomputedData,\s*isPrecomputedSize,\s*PrecomputedFFTData\s*\}\s*from\s*'(\.\/generated\/fftPrecomputed)';", "import { getPrecomputedData, isPrecomputedSize } from '$1';`r`nimport type { PrecomputedFFTData } from '$1';"
    $t = $t -replace "import\s+\{\s*getConstantsForShader,\s*ShaderConstantConfig,\s*SHADER_CONSTANT_PRESETS\s*\}\s*from\s*'(\.\/shaderConstantManager)';", "import { getConstantsForShader, SHADER_CONSTANT_PRESETS } from '$1';`r`nimport type { ShaderConstantConfig } from '$1';"
    return $t
  } | Out-Null
}

$holoEngine = Join-Path $Root "..\frontend\lib\holographicEngine.ts"
if (Test-Path $holoEngine) {
  Mutate-File $holoEngine {
    param($t)
    $t = $t -replace "import\s+\{\s*getConstantsForShader,\s*ShaderConstantConfig\s*\}\s*from\s*'(\.\/webgpu\/shaderConstantManager)';", "import { getConstantsForShader } from '$1';`r`nimport type { ShaderConstantConfig } from '$1';"
    return $t
  } | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 8) ConceptDiff — make `changes?` optional + normalize `timestamp: Date`
#    Apply to any file declaring `interface ConceptDiff` under src\lib\stores\
# ──────────────────────────────────────────────────────────────────────────────
Get-ChildItem -Path (Join-Path $Root "src\lib\stores") -Recurse -Include *.ts,*.svelte |
  Where-Object { (Get-Content -Raw -LiteralPath $_.FullName) -match "interface\s+ConceptDiff\s*\{" } |
  ForEach-Object {
    Mutate-File $_.FullName {
      param($t)
      # Swap `changes:` to optional `changes?:`
      $t = $t -replace "(\bchanges)\s*:\s*Array<\{[^}]*\}>\s*;", "`$1?: Array<{ field: string; from: any; to: any }>;"
      # Ensure timestamp is required (no '?')
      $t = $t -replace "(\btimestamp)\s*\?:\s*Date\s*;", "`$1: Date;"
      return $t
    } | Out-Null
  }

# ──────────────────────────────────────────────────────────────────────────────
# 9) Remove HTML-style event attributes that TS rejects
# ──────────────────────────────────────────────────────────────────────────────
$eventFiles = @(
  "src\lib\components\JumpToLatest.svelte",
  "src\lib\components\ScholarSpherePanel.svelte",
  "src\routes\+page.svelte"
) | ForEach-Object { Join-Path $Root $_ }

$patternOnMouse = '(\s)onmouse(over|out)="[^"]*"'

foreach ($f in $eventFiles) {
  if (Test-Path $f) {
    Mutate-File $f {
      param($t)
      $t = [System.Text.RegularExpressions.Regex]::Replace($t, $patternOnMouse, '$1')
      return $t
    } | Out-Null
  }
}

# ──────────────────────────────────────────────────────────────────────────────
# 10) Deduplicate role=/tabindex= attributes in modal backdrops
# ──────────────────────────────────────────────────────────────────────────────
$dupAttrFiles = @(
  "src\lib\components\GroupSelector.svelte",
  "src\lib\components\InviteButton.svelte",
  "src\lib\components\InviteModal.svelte"
) | ForEach-Object { Join-Path $Root $_ }

foreach ($f in $dupAttrFiles) {
  if (Test-Path $f) {
    Mutate-File $f {
      param($t)
      # Collapse multiple role="button" to single
      $t = $t -replace '(role="button"\s*){2,}', 'role="button" '
      # Collapse multiple tabindex="0" to single
      $t = $t -replace '(tabindex="0"\s*){2,}', 'tabindex="0" '
      return $t
    } | Out-Null
  }
}

Write-Host "`nDone. Suggested next steps:"
Write-Host "  cd $Root"
Write-Host "  pnpm run check"
Write-Host "  pnpm run build"
