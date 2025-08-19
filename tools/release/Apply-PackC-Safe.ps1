Param(
  [switch]$DryRun,
  [switch]$NoBackup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[PackC:Safe] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[PackC:Safe] OK: $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[PackC:Safe] WARN: $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[PackC:Safe] ERROR: $m" -ForegroundColor Red }

# --- Locate project root ---
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Candidates = @(
  (Join-Path $RepoRoot "tori_ui_svelte"),
  (Join-Path $RepoRoot "frontend"),
  $RepoRoot
)
$ProjectRoot = $null
foreach($c in $Candidates){ if(Test-Path (Join-Path $c "src")){ $ProjectRoot = $c; break } }
if(-not $ProjectRoot){ Err "Could not find a project containing 'src' under: $($Candidates -join '; ')"; exit 2 }
Info "ProjectRoot: $ProjectRoot"

# Helpers
function Ensure-Dir([string]$Path){
  if(-not (Test-Path $Path)){ if(-not $DryRun){ New-Item -ItemType Directory -Path $Path -Force | Out-Null }; Ok "mkdir $Path" }
}
function ReadText($p){ if(Test-Path $p){ return [IO.File]::ReadAllText($p) } else { return $null } }
function WriteText($p, $t){
  Ensure-Dir ([IO.Path]::GetDirectoryName($p))
  if(-not $DryRun){ [IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false))) }
  Ok "write $p"
}
function Backup($p){ if($NoBackup){ return }; if(Test-Path $p){ $b="$p.bak"; if(-not (Test-Path $b)){ if(-not $DryRun){ Copy-Item $p $b -Force }; Info "backup $b" } } }
function Replace-InFile($p, $pattern, $replacement, $label){
  if(-not (Test-Path $p)){ Warn "skip $label (missing $p)"; return }
  $t = ReadText $p
  $new = [Regex]::Replace($t, $pattern, $replacement, [Text.RegularExpressions.RegexOptions]::Singleline -bor [Text.RegularExpressions.RegexOptions]::Multiline)
  if($t -ne $new){
    Backup $p
    WriteText $p $new
    Ok $label
  } else { Info "nochange $label" }
}

# --------- PHASE A: unblock preprocess ---------
# A1) src\routes\+page.svelte — remove stray 'await' line before for (...)
$page = Join-Path $ProjectRoot "src\routes\+page.svelte"
# remove the single line starting with 'await // ...' after docs.push(result.document);
Replace-InFile $page '(?ms)(docs\.push\(result\.document\);\s*)await\s*//[^\r\n]*\r?\n' '$1' "+page.svelte: removed bare 'await' before for()"

# A2) global cleanup across .svelte
$allSvelte = Get-ChildItem -Path (Join-Path $ProjectRoot "src") -Recurse -Filter *.svelte | % { $_.FullName }
foreach($f in $allSvelte){
  $t = ReadText $f
  if($null -eq $t){ continue }
  $orig = $t
  # on:on:* → on:*
  $t = $t -replace 'on:on:', 'on:'
  # on:(e)=>fn(e) → on:{fn}
  $t = [Regex]::Replace($t, 'on:(\w+)\s*=\s*\{\s*\(\s*\w+\s*\)\s*=>\s*([A-Za-z0-9_]+)\s*\(\s*\w*\s*\)\s*\}', 'on:$1={$2}', 'Singleline, Multiline')
  if($t -ne $orig){ Backup $f; WriteText $f $t; Ok "svelte handlers: $([IO.Path]::GetFileName($f))" }
}

# A3) remove known malformed hover attributes
Replace-InFile $page 'on:mouseover=\{\{[^}]+\}\}' '' "+page.svelte: removed malformed on:mouseover {{...}}"
Replace-InFile $page 'on:mouseout=\{[^}]+\}' '' "+page.svelte: removed malformed on:mouseout"
$scholar = Join-Path $ProjectRoot "src\lib\components\ScholarSpherePanel.svelte"
Replace-InFile $scholar 'on:mouseover=\{[^}]+\}\s*\r?\n\s*on:mouseout=\{[^}]+\}' '' "ScholarSpherePanel.svelte: removed inline hover style handlers"

# A4) patch broken modal backdrops/tags in three files
$modalBackdrop_Group = @'
<div class="modal-backdrop" role="button" tabindex="0" on:click={() => (showCreateModal = false)} on:keydown={(e) => e.key === 'Escape' && (showCreateModal = false)}>
'@
$modalBackdrop_InviteBtn = @'
<div class="modal-backdrop" role="button" tabindex="0" on:click={toggleInviteModal} on:keydown={(e) => e.key === 'Escape' && toggleInviteModal()}>
'@
$modalBackdrop_Invite = @'
<div class="modal-backdrop" role="button" tabindex="0" on:click={close} on:keydown={(e) => e.key === 'Escape' && close()}>
'@
$modalOpen = @'
<div class="modal" on:click|stopPropagation>
'@

$group = Join-Path $ProjectRoot "src\lib\components\GroupSelector.svelte"
$inviteBtn = Join-Path $ProjectRoot "src\lib\components\InviteButton.svelte"
$invite = Join-Path $ProjectRoot "src\lib\components\InviteModal.svelte"

Replace-InFile $group '<div class="modal-backdrop"[^>]*>' ([Regex]::Escape($modalBackdrop_Group)) "GroupSelector.svelte: fixed modal-backdrop"
Replace-InFile $group '<div class="modal"[^>]*>' ([Regex]::Escape($modalOpen)) "GroupSelector.svelte: fixed modal open"

Replace-InFile $inviteBtn '<div class="modal-backdrop"[^>]*>' ([Regex]::Escape($modalBackdrop_InviteBtn)) "InviteButton.svelte: fixed modal-backdrop"
Replace-InFile $inviteBtn '<div class="modal"[^>]*>' ([Regex]::Escape($modalOpen)) "InviteButton.svelte: fixed modal open"

Replace-InFile $invite '<div class="modal-backdrop"[^>]*>' ([Regex]::Escape($modalBackdrop_Invite)) "InviteModal.svelte: fixed modal-backdrop"
Replace-InFile $invite '<div class="modal"[^>]*>' ([Regex]::Escape($modalOpen)) "InviteModal.svelte: fixed modal open"

# A5) unwrap InviteModalSecure wrappers
$inviteSecure = Join-Path $ProjectRoot "src\lib\components\InviteModalSecure.svelte"
Replace-InFile $inviteSecure 'on:click=\{\s*\(\s*\w+\s*\)\s*=>\s*close\(\s*\w*\s*\)\s*\}' 'on:click={close}' "InviteModalSecure: close wrapper → direct"
Replace-InFile $inviteSecure 'on:click=\{\s*\(\s*\w+\s*\)\s*=>\s*joinGroup\(\s*\w*\s*\)\s*\}' 'on:click={joinGroup}' "InviteModalSecure: joinGroup wrapper → direct"
Replace-InFile $inviteSecure 'on:beforeunload=\{\s*\(\s*\w+\s*\)\s*=>\s*cleanup\(\s*\w*\s*\)\s*\}' 'on:beforeunload={cleanup}' "InviteModalSecure: beforeunload wrapper → direct"

# --------- PHASE B: minimal TS calm-down (won't affect runtime) ---------
# B1) interpreter.ts — unknown -> cast at callsite
$interp = Join-Path $ProjectRoot "src\lib\elfin\interpreter.ts"
Replace-InFile $interp 'this\.scripts\[''onUpload''\]\s*=\s*\(ctx:\s*UploadContext\)\s*=>\s*onUpload\(ctx\);' 'this.scripts[''onUpload''] = (ctx) => onUpload(ctx as UploadContext);' "interpreter: onUpload typed→cast"
Replace-InFile $interp 'this\.scripts\[''onConceptChange''\]\s*=\s*\(ctx:\s*ConceptChangeContext\)\s*=>\s*onConceptChange\(ctx\);' 'this.scripts[''onConceptChange''] = (ctx) => onConceptChange(ctx as ConceptChangeContext);' "interpreter: onConceptChange typed→cast"
Replace-InFile $interp 'this\.scripts\[''onGhostStateChange''\]\s*=\s*\(ctx:\s*GhostStateChangeContext\)\s*=>\s*onGhostStateChange\(ctx\);' 'this.scripts[''onGhostStateChange''] = (ctx) => onGhostStateChange(ctx as GhostStateChangeContext);' "interpreter: onGhostStateChange typed→cast"

# B2) toriInit.ts — guard getExecutionStats with any-cast
$tori = Join-Path $ProjectRoot "src\lib\toriInit.ts"
Replace-InFile $tori 'checkStats:\s*\(\)\s*=>\s*[^,]+' 'checkStats: () => { const eng = activeEngine as any; return (eng && typeof eng.getExecutionStats === "function") ? eng.getExecutionStats() : "Engine not available"; }' "toriInit: guarded checkStats"

# B3) dynamicApi.ts — safe error.message
$dyn = Join-Path $ProjectRoot "src\lib\dynamicApi.ts"
Replace-InFile $dyn 'error\.message' '(error instanceof Error ? error.message : String(error))' "dynamicApi: safe error.message"

# B4) conceptScoring_enhanced.ts — add Bench & casts
$concept = Join-Path $ProjectRoot "src\lib\cognitive\conceptScoring_enhanced.ts"
if(Test-Path $concept){
  $txt = ReadText $concept
  if($txt -notmatch 'type\s+Bench\s*=\s*\{'){
    $bench = "type Bench = { labels: string[]; method_used?: string; quality_metrics?: Record<string, number>; performance_stats?: Record<string, number>; status?: string; };`r`n"
    Backup $concept; WriteText $concept ($bench + $txt); Ok "conceptScoring: inserted Bench type"
  }
  Replace-InFile $concept 'benchmarkResult\.' '(benchmarkResult as Bench).' "conceptScoring: cast benchmarkResult.*"
  Replace-InFile $concept 'healthResult\.status' '(healthResult as any).status' "conceptScoring: cast healthResult.status"
}

# B5) types.ts — import type, remove stray lines, add aliases
$types = Join-Path $ProjectRoot "src\lib\cognitive\types.ts"
if(Test-Path $types){
  Replace-InFile $types 'import\s+\{\s*ConceptDiff\s*,\s*ConceptDiffType\s*\}\s+from\s+''`\$lib/stores/conceptMesh'';' "import type { ConceptDiff, ConceptDiffType } from '`$lib/stores/conceptMesh';" "types.ts: import type"
  Replace-InFile $types '^\s*timestamp\?:\s*number;\s*$' '' "types.ts: removed stray 'timestamp' line"
  Replace-InFile $types '^\s*loopId\?:\s*string;\s*$' '' "types.ts: removed stray 'loopId' line"
  $ct = ReadText $types
  if($ct -notmatch 'type\s+ConceptDiffState'){
    $ct += "`r`nexport type ConceptDiffState = ConceptDiff;`r`n"
  }
  if($ct -notmatch 'interface\s+LoopRecord'){
    $ct += @"
export interface LoopRecord {
  timestamp?: number;
  loopId?: string;
}
"@
  }
  Backup $types; WriteText $types $ct
}

# B6) global.d.ts — ensure browser globals
$global = Join-Path $ProjectRoot "src\lib\types\global.d.ts"
$globalText = @'
export {};

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
    TORI?: {
      toggleHologramAudio?: (enabled: boolean) => void;
      toggleHologramVideo?: (enabled: boolean) => void;
      [k: string]: any;
    };
    TORI_DISPLAY_TYPE?: 'webgpu_only' | 'webgpu_then_canvas' | 'canvas_fallback' | string;
    ghostMemoryDemo?: (...args: any[]) => any;
  }
}
'@
if(-not (Test-Path $global)){ WriteText $global $globalText } else {
  $gt = ReadText $global
  if($gt -ne $globalText){ Backup $global; WriteText $global $globalText }
}

# B7) app.d.ts — ensure Locals/PageData declare user.name
$appd = Join-Path $ProjectRoot "src\app.d.ts"
$appText = @'
declare namespace App {
  interface Locals {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
  interface PageData {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
}
'@
WriteText $appd $appText

# B8) Create stub for ConceptDebugPanel if missing
$cdp = Join-Path $ProjectRoot "src\lib\components\ConceptDebugPanel.svelte"
if(-not (Test-Path $cdp)){
  $stub = @'
<script lang="ts">
  export let enabled: boolean = false;
</script>
{#if enabled}
  <div class="p-2 text-xs text-gray-500 border rounded">ConceptDebugPanel (stub)</div>
{/if}
'@
  WriteText $cdp $stub
}

Ok "Done. Next: pnpm run check; pnpm run build."
