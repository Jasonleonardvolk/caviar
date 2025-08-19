<#

Apply-PackAB-Codemod.ps1
One-shot codemod for PACK A+B surgical edits that are hard to express as a static patch.

Edits:
  A1) routes\+page.svelte: remove stray 'await' line before a 'for (...)' loop
  A2) lib\stores\conceptMesh.ts: normalize ConceptDiff types (remove duplicates / junk tokens) and inject canonical block once
  B3) routes\api\soliton\[...path]\+server.ts: make params.path safe (string|string[]) across handlers
  B4) lib\components\vault\MemoryVaultDashboard.svelte: move types to module and relax optional fields; ensure selectedView union alignment

Backups to D:\Dev\kha\patches\backup-AB-<timestamp>\...

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Apply-PackAB-Codemod.ps1 -Root "D:\Dev\kha\tori_ui_svelte"

#>

param(
  [string]$Root = "D:\Dev\kha\tori_ui_svelte"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir($p) {
  $d = Split-Path -Parent $p
  if ($d -and -not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

function Backup-And-Write($path, [string]$content) {
  $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $rel = $path.Replace($Root, "").TrimStart("\","/")
  $backup = "D:\Dev\kha\patches\backup-AB-$timestamp\$rel"
  Ensure-Dir $backup
  if (Test-Path $path) { Copy-Item -LiteralPath $path -Destination $backup -Force }
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
}

function Mutate($path, [scriptblock]$fn) {
  if (-not (Test-Path $path)) { return $false }
  $original = Get-Content -Raw -LiteralPath $path
  $mutated = & $fn $original
  if ($mutated -ne $original) {
    Backup-And-Write $path $mutated
    return $true
  }
  return $false
}

# Timestamped backup root
$global:TS = Get-Date -Format "yyyyMMdd-HHmmss"

# ─────────────────────────────────────────────────────────────────────────────
# A1) +page.svelte: remove stray 'await' before 'for (...)'
# ─────────────────────────────────────────────────────────────────────────────
$page = Join-Path $Root "src\routes\+page.svelte"
if (Test-Path $page) {
  Mutate $page {
    param($t)
    # Transform: 'docs.push(result.document);\s*await\s*\r?\n\s*for (' -> 'docs.push(result.document);\nfor ('
    $t = [regex]::Replace($t, "docs\.push\(result\.document\);\s*await\s*\r?\n\s*for\s*\(", "docs.push(result.document);`r`nfor (")
    return $t
  } | Out-Null
}

# ─────────────────────────────────────────────────────────────────────────────
# A2) conceptMesh.ts: canonicalize ConceptDiff types and remove junk
# ─────────────────────────────────────────────────────────────────────────────
$cm = Join-Path $Root "src\lib\stores\conceptMesh.ts"
if (Test-Path $cm) {
  Mutate $cm {
    param($t)
    $orig = $t

    # Remove stray junk lines like '>;' from previous broken merges
    $t = [regex]::Replace($t, "^[ \t]*>;\s*$", "", 'Multiline')

    # Strip all existing ConceptDiffType/ConceptDiff interface blocks
    $t = [regex]::Replace($t, "export\s+type\s+ConceptDiffType\s*=\s*[\s\S]*?;", "", 'Singleline')
    $t = [regex]::Replace($t, "(export\s+)?interface\s+ConceptDiff\s*\{[\s\S]*?\}", "", 'Singleline')
    $t = [regex]::Replace($t, "export\s+const\s+buildConceptDiff\s*=\s*\([\s\S]*?\)\s*=>\s*\{[\s\S]*?\};", "", 'Singleline')

    # Inject canonical types just after first import block
    $canonical = @"
// ---- Canonical ConceptDiff types (single source of truth) ----
export type ConceptDiffType =
  | 'document' | 'manual' | 'chat' | 'system'
  | 'add' | 'remove' | 'modify' | 'relate' | 'unrelate'
  | 'extract' | 'link' | 'memory';

export interface ConceptDiff {
  id: string;
  type: ConceptDiffType;
  title: string;
  concepts: string[];
  summary?: string;
  metadata?: Record<string, any>;
  timestamp: Date;
  changes?: Array<{ field: string; from: any; to: any }>;
}

export const buildConceptDiff = (
  diff: Omit<ConceptDiff, 'id' | 'timestamp'> & Partial<Pick<ConceptDiff, 'id' | 'timestamp' | 'changes'>>
): ConceptDiff => ({
  id: diff.id ?? ``diff_`${Date.now()}_`${Math.random().toString(36).slice(2, 9)}``,
  timestamp: diff.timestamp ?? new Date(),
  changes: diff.changes ?? [],
  ...diff
});
// ---- End canonical types ----
"@

    if ($t -match "import\s") {
      $t = [regex]::Replace($t, "(\A(?:.|\r|\n)*?\bimport[\s\S]*?;[\r\n]+)", "``$1$canonical``r``n", 'Singleline')
    } else {
      $t = $canonical + "``r``n" + $t
    }

    return $t
  } | Out-Null
}

# ─────────────────────────────────────────────────────────────────────────────
# B3) soliton route: make params.path robust (string|string[]) across handlers
# ─────────────────────────────────────────────────────────────────────────────
$soliton = Join-Path $Root "src\routes\api\soliton\[...path]\+server.ts"
if (Test-Path $soliton) {
  Mutate $soliton {
    param($t)
    # Replace common patterns of params.path.join('/') with robust expression
    $t = $t -replace "params\.path\.join\('/'\)", "(Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''))"
    $t = $t -replace "\(params\.path\)\.join\('/'\)", "(Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''))"
    return $t
  } | Out-Null
}

# ─────────────────────────────────────────────────────────────────────────────
# B4) MemoryVaultDashboard.svelte: move types to module & relax optionals
# ─────────────────────────────────────────────────────────────────────────────
$mvd = Join-Path $Root "src\lib\components\vault\MemoryVaultDashboard.svelte"
if (Test-Path $mvd) {
  Mutate $mvd {
    param($t)
    $orig = $t

    # Ensure a <script context="module" lang="ts"> block exists with types
    $moduleBlock = @"
<script context="module" lang="ts">
  export type MemoryType = 'soliton' | 'concept' | 'ghost' | 'document' | 'chat' | 'memory';
  export interface MemoryEntry {
    id: string;
    type: MemoryType;
    timestamp: Date;
    content: any;
    phase?: string;
    coherence?: number;
    importance?: number;
    tags?: string[];
    relationships?: string[];
  }
</script>
"

    if ($t -notmatch "<script\s+context=""module""\s+lang=""ts"">") {
      # insert module block at top
      $t = $moduleBlock + "``r``n" + $t
    }

    # If instance script mistakenly has 'export interface MemoryEntry', strip 'export '
    $t = $t -replace "export\s+interface\s+MemoryEntry", "interface MemoryEntry"

    # selectedView union alignment (if a const array exists, keep it)
    $t = $t -replace "let\s+selectedView\s*:\s*'overview'\s*\|\s*'timeline'\s*\|\s*'graph'\s*\|\s*'quantum'\s*\|\s*'export'\s*=\s*'overview';",
                     "let selectedView: 'overview' | 'timeline' | 'graph' | 'quantum' | 'export' = 'overview';"

    return $t
  } | Out-Null
}

Write-Host "``nPACK A+B codemod complete."
Write-Host "Next:"
Write-Host "  1) git switch -c fix/packAB"
Write-Host "  2) git apply --3way --whitespace=fix D:\Dev\kha\patches\Tori-PackAB.patch"
Write-Host "  3) powershell -NoProfile -ExecutionPolicy Bypass -File D:\Dev\kha\patches\Apply-PackAB-Codemod.ps1 -Root ""D:\Dev\kha\tori_ui_svelte"""
Write-Host "  4) pnpm run check"
Write-Host "  5) pnpm run build"