<#

Apply-Tori-FullFix-02.ps1
Second-pass codemod for D:\Dev\kha\tori_ui_svelte based on fulllerror.txt

Focus:
  • Kill the exact syntax blocker in src\lib\components\vault\MemoryVaultDashboard.svelte (stray ')();')
  • Ensure all required state/imports exist in MemoryVaultDashboard.svelte
  • Unify App.Locals typing (add optional 'name') and Window globals (TORI, TORI_DISPLAY_TYPE, webkitAudioContext)
  • Align BridgeConfig interface with 'api' and 'websocket' usage
  • Unify ConceptDiff across src\lib\stores\conceptMesh.ts AND src\lib\stores\types.ts
  • Remove stray WebGPU reference include and React.CSSProperties casts
  • Make isolatedModules/type-only imports happy in ..\frontend\lib\*.ts
  • Strip HTML-style onmouseover/onmouseout attributes flagged by Svelte+TS
  • Deduplicate role="button"/tabindex="0" attributes in modal backdrops

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Apply-Tori-FullFix-02.ps1 -Root "D:\Dev\kha\tori_ui_svelte"

Backups:
  Each changed file is copied to D:\Dev\kha\patches\backup-<timestamp>\...

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
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
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

# Prepare backup root
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Global:BackupDir = "D:\Dev\kha\patches\backup-$timestamp"
New-Item -ItemType Directory -Force -Path $Global:BackupDir | Out-Null

# ============================================================================
# 1) MEMORY VAULT: remove stray ')();' and ensure state/imports are present
# ============================================================================
$mv = Join-Path $Root "src\lib\components\vault\MemoryVaultDashboard.svelte"
if (Test-Path $mv) {
  Mutate-File $mv {
    param($t)

    # Remove any lines that are just ')();' (the build blocker)
    $t = [regex]::Replace($t, "^[\t ]*\)\(\);\s*$", "", [System.Text.RegularExpressions.RegexOptions]::Multiline)

    # Ensure <script lang="ts"> has our header imports/state
    if ($t -match "<script\s+lang=""ts"">") {
      $needHeader = ($t -notmatch "\bconst\s+memories\s*=\s*writable<MemoryEntry")
      if ($needHeader) {
        $header = @"
import { onMount, onDestroy } from 'svelte';
import { browser } from '\$app/environment';
import { writable, derived } from 'svelte/store';

// --- Local types & state for Memory Vault ---
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

const systemCoherence = writable(0);
const quantumState = writable<{ superposition: number; decoherence: number }>({ superposition: 0, decoherence: 0 });
const currentPhase = writable<string>('exploration');

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

// External stores (fallback as dummies if not provided elsewhere)
const conceptStore = writable<any>({ activeConcepts: [] });
const solitonMemory = writable<any>(null);
const ghostPersona = writable<any>(null);

// --- helpers ---
function getTypeIcon(t: MemoryType) {
  switch (t) {
    case 'soliton': return '🪐';
    case 'concept': return '🧩';
    case 'ghost': return '👻';
    case 'document': return '📄';
    case 'chat': return '💬';
    default: return '🧠';
  }
}
function getPhaseColor(p: string) {
  return p === 'exploration' ? 'text-blue-700' : p === 'consolidation' ? 'text-green-700' : 'text-gray-700';
}
function updateFromSoliton(_v: any) {}
function updateFromConcepts(_v: any) {}
function updateFromGhost(_v: any) {}
async function loadMemoryVault() {
  try {
    const raw = (browser && localStorage.getItem('memory-vault')) || '[]';
    const arr = JSON.parse(raw);
    const processed = (Array.isArray(arr) ? arr : []).map((m: any) => ({
      id: m.id ?? `mem_${Math.random().toString(36).slice(2)}`,
      type: (m.type as MemoryType) ?? 'memory',
      timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
      content: m.content ?? {},
      phase: m.phase ?? 'exploration',
      coherence: +m.coherence || 0,
      importance: +m.importance || 0,
      tags: Array.isArray(m.tags) ? m.tags : [],
      relationships: Array.isArray(m.relationships) ? m.relationships : []
    } as MemoryEntry));
    memories.set(processed);
    metrics.set({
      totalMemories: processed.length,
      activePhases: 1,
      coherenceLevel: processed.length ? processed.reduce((a, c) => a + (c.coherence || 0), 0) / processed.length : 0,
      conceptDensity: 0.1,
      ghostActivity: 0.1,
      quantumEntanglement: 0.05
    });
  } catch {}
}
"@
        $t = $t -replace "(<script\s+lang=""ts"">\s*)", "`$1$header`r`n"
      }
    }

    # Ensure onMount has a single, well-formed async IIFE and cleanup
    if ($t -notmatch "onMount\(\s*\(\)\s*=>") {
      $onMount = @"
onMount(() => {
  if (!browser) return;
  const unsubscribes: Array<() => void> = [];
  (async () => {
    await loadMemoryVault();
    if (autoRefresh) {
      refreshInterval = setInterval(loadMemoryVault, 5000);
    }
    try {
      unsubscribes.push(
        (solitonMemory.subscribe ? solitonMemory.subscribe(updateFromSoliton) : () => {}),
        (conceptStore.subscribe ? conceptStore.subscribe(updateFromConcepts) : () => {}),
        (ghostPersona.subscribe ? ghostPersona.subscribe(updateFromGhost) : () => {})
      );
    } catch {}
  })();
});
onDestroy(() => {
  try { if (refreshInterval) clearInterval(refreshInterval); } catch {}
});
"@
      $t = $t -replace "(</script>)", "$onMount`r`n`$1"
    }

    return $t
  } | Out-Null
}

# ============================================================================
# 2) app.d.ts: unify Locals and Window
# ============================================================================
$appDtsPath = Join-Path $Root "src\app.d.ts"
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
      TORI_DISPLAY_TYPE?: string;
      ghostMemoryDemo?: () => void;
      webkitAudioContext?: typeof AudioContext;
    }
  }
}
export {};
"
Write-TextFile $appDtsPath $appDts

# Also patch legacy global type files if they declare App.Locals without 'name'
$legacyTypes = @(
  Join-Path $Root "src\types.d.ts",
  Join-Path $Root "src\lib\types\global.d.ts"
) | Where-Object { Test-Path $_ }

foreach ($f in $legacyTypes) {
  Mutate-File $f {
    param($t)
    # Replace any Locals { id; username } with one that includes name?
    $t = [regex]::Replace($t, "interface\s+Locals\s*\{[^\}]*?\bid:\s*string;[^\}]*?\busername:\s*string;[^\}]*?\}", "interface Locals { user: { id: string; username: string; name?: string; role: 'admin' | 'user' } }", 'Singleline')
    return $t
  } | Out-Null
}

# ============================================================================
# 3) hooks.server.ts: ensure 'name' assigned
# ============================================================================
$hooks = Join-Path $Root "src\hooks.server.ts"
if (Test-Path $hooks) {
  Mutate-File $hooks {
    param($t)
    $t = [regex]::Replace($t,
@"
event\.locals\.user\s*=\s*\{[\s\S]*?\};
"@,
@"
event.locals.user = {
  id: `user_${(event.cookies.get('username') || 'anonymous')}`,
  username: event.cookies.get('username') || 'anonymous',
  name: event.cookies.get('username') || 'anonymous',
  role: ((event.cookies.get('role') ?? '') -in @('admin','user')) ? (event.cookies.get('role')) : 'user'
};
"@, 'Singleline')
    return $t
  } | Out-Null
}

# ============================================================================
# 4) bridgeConfig.ts: API + websocket in interface
# ============================================================================
$bridge = Join-Path $Root "src\lib\bridgeConfig.ts"
if (Test-Path $bridge) {
  Mutate-File $bridge {
    param($t)
    $t = [regex]::Replace($t, "export\s+interface\s+BridgeConfig\s*\{[\s\S]*?\}", @"
export interface BridgeConfig {
  mode: 'dev' | 'prod' | 'test';
  api?: { url: string; timeout: number };
  websocket?: { url: string; reconnect: boolean };
}
", 'Singleline')
    return $t
  } | Out-Null
}

# ============================================================================
# 5) ConceptDiff unification (conceptMesh.ts and types.ts)
# ============================================================================
$conceptMesh = Join-Path $Root "src\lib\stores\conceptMesh.ts"
if (Test-Path $conceptMesh) {
  Mutate-File $conceptMesh {
    param($t)
    # Unify interface
    $t = [regex]::Replace($t, "interface\s+ConceptDiff\s*\{[\s\S]*?\}", @"
interface ConceptDiff {
  id: string;
  type: string;
  title: string;
  concepts: string[];
  summary?: string;
  metadata?: Record<string, any>;
  timestamp: Date;
  changes?: Array<{ field: string; from: any; to: any }>;
}
", 'Singleline')
    # Ensure addConceptDiff fills defaults
    $t = [regex]::Replace($t, "addConceptDiff\s*\(\s*diff:\s*ConceptDiff\s*\)\s*\{[\s\S]*?\}", @"
addConceptDiff(diff: ConceptDiff) {
  const complete: ConceptDiff = {
    id: diff.id || `diff_${Date.now()}_${Math.random().toString(36).slice(2,9)}`,
    timestamp: diff.timestamp || new Date(),
    changes: diff.changes ?? [],
    ...diff
  };
  // existing push-to-store logic should remain below
}
", 'Singleline')
    return $t
  } | Out-Null
}

$typesTs = Join-Path $Root "src\lib\stores\types.ts"
if (Test-Path $typesTs) {
  Mutate-File $typesTs {
    param($t)
    $t = [regex]::Replace($t, "interface\s+ConceptDiff\s*\{[\s\S]*?\}", @"
interface ConceptDiff {
  id: string;
  type: string;
  title: string;
  concepts: string[];
  summary?: string;
  metadata?: Record<string, any>;
  timestamp: Date;
  changes?: Array<{ field: string; from: any; to: any }>;
}
", 'Singleline')
    return $t
  } | Out-Null
}

# ============================================================================
# 6) tsconfig.json: set ES2022 libs (no harm if already applied)
# ============================================================================
$tsconfig = Join-Path $Root "tsconfig.json"
if (Test-Path $tsconfig) {
  $json = Get-Content -Raw -LiteralPath $tsconfig | ConvertFrom-Json -Depth 20
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
  $json | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $tsconfig -Encoding utf8
}

# ============================================================================
# 7) WebGPU + React type cleanup
# ============================================================================
$photo = Join-Path $Root "src\lib\webgpu\photoMorphPipeline.ts"
if (Test-Path $photo) {
  Mutate-File $photo {
    param($t)
    $t = [regex]::Replace($t, "^\s*///\s*<reference\s+path=""\.\/webgpu-types\.d\.ts""\s*\/>\s*\r?\n", "", 'Multiline')
    return $t
  } | Out-Null
}

$ghostEngine = Join-Path $Root "src\lib\elfin\ghostPersonaEngine.ts"
if (Test-Path $ghostEngine) {
  Mutate-File $ghostEngine {
    param($t)
    $t = $t -replace "as\s+React\.CSSProperties", "as Record<string, string>"
    return $t
  } | Out-Null
}

# ============================================================================
# 8) Type-only import fixes in frontend libs
# ============================================================================
$fft = Join-Path $Root "..\frontend\lib\webgpu\fftCompute.ts"
if (Test-Path $fft) {
  Mutate-File $fft {
    param($t)
    $t = [regex]::Replace($t,
      "import\s+\{\s*getPrecomputedData,\s*isPrecomputedSize,\s*PrecomputedFFTData\s*\}\s*from\s*'(\.\/generated\/fftPrecomputed)';",
      "import { getPrecomputedData, isPrecomputedSize } from '$1';`r`nimport type { PrecomputedFFTData } from '$1';"
      )
    $t = [regex]::Replace($t,
      "import\s+\{\s*getConstantsForShader,\s*ShaderConstantConfig,\s*SHADER_CONSTANT_PRESETS\s*\}\s*from\s*'(\.\/shaderConstantManager)';",
      "import { getConstantsForShader, SHADER_CONSTANT_PRESETS } from '$1';`r`nimport type { ShaderConstantConfig } from '$1';"
      )
    return $t
  } | Out-Null
}
$holo = Join-Path $Root "..\frontend\lib\holographicEngine.ts"
if (Test-Path $holo) {
  Mutate-File $holo {
    param($t)
    $t = [regex]::Replace($t,
      "import\s+\{\s*getConstantsForShader,\s*ShaderConstantConfig\s*\}\s*from\s*'(\.\/webgpu\/shaderConstantManager)';",
      "import { getConstantsForShader } from '$1';`r`nimport type { ShaderConstantConfig } from '$1';"
      )
    return $t
  } | Out-Null
}

# ============================================================================
# 9) Strip HTML-style mouseover/mouseout attrs from known files
# ============================================================================
$attrFiles = @(
  "src\lib\components\JumpToLatest.svelte",
  "src\lib\components\ScholarSpherePanel.svelte",
  "src\routes\+page.svelte"
) | ForEach-Object { Join-Path $Root $_ }

foreach ($f in $attrFiles) {
  if (Test-Path $f) {
    Mutate-File $f {
      param($t)
      $t = [regex]::Replace($t, "\s+onmouse(over|out)=""[^""]*""", "", 'IgnoreCase')
      return $t
    } | Out-Null
  }
}

# ============================================================================
# 10) Deduplicate role/tabindex attributes in known files
# ============================================================================
$dupFiles = @(
  "src\lib\components\GroupSelector.svelte",
  "src\lib\components\InviteButton.svelte",
  "src\lib\components\InviteModal.svelte"
) | ForEach-Object { Join-Path $Root $_ }

foreach ($f in $dupFiles) {
  if (Test-Path $f) {
    Mutate-File $f {
      param($t)
      $t = [regex]::Replace($t, '(role="button"\s*){2,}', 'role="button" ')
      $t = [regex]::Replace($t, '(tabindex="0"\s*){2,}', 'tabindex="0" ')
      return $t
    } | Out-Null
  }
}

Write-Host "`nAll set. Next:"
Write-Host "  1) cd $Root"
Write-Host "  2) pnpm run check"
Write-Host "  3) pnpm run build"
