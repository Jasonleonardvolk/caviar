# Write a corrected Apply-PackC.ps1 that fixes the PowerShell RegexOptions bug,
# removes unicode arrows in log tags, and includes stronger Svelte/TS cleanups.
content = r'''Param(
    [switch]$DryRun,
    [switch]$NoBackup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($msg){ Write-Host "[PACK C] $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[PACK C] WARN: $msg" -ForegroundColor Yellow }
function Ok($msg){ Write-Host "[PACK C] OK: $msg" -ForegroundColor Green }
function Err($msg){ Write-Host "[PACK C] ERROR: $msg" -ForegroundColor Red }

# --- Locate repo root and project root ---
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Candidates = @(
    (Join-Path $RepoRoot "tori_ui_svelte"),
    (Join-Path $RepoRoot "frontend"),
    $RepoRoot
)
$ProjectRoot = $null
foreach($c in $Candidates){
    if(Test-Path (Join-Path $c "src")){
        $ProjectRoot = $c
        break
    }
}
if(-not $ProjectRoot){
    Err "Could not locate a project with a 'src' directory. Checked: $($Candidates -join '; ')"
    exit 2
}
Info "RepoRoot    : $RepoRoot"
Info "ProjectRoot : $ProjectRoot"

# --- Helpers ---
$RegexOpts = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline

function Ensure-Dir([string]$Path){
    if(-not (Test-Path $Path)){ 
        if(-not $DryRun){ New-Item -ItemType Directory -Force -Path $Path | Out-Null }
        Info "Created directory: $Path"
    }
}

function Read-File([string]$Path){
    if(-not (Test-Path $Path)){ return $null }
    return [System.IO.File]::ReadAllText($Path)
}

function Write-File([string]$Path, [string]$Text){
    Ensure-Dir ([System.IO.Path]::GetDirectoryName($Path))
    if(-not $DryRun){
        [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
    }
    Ok "Wrote: $Path"
}

function Backup-File([string]$Path){
    if($NoBackup){ return }
    if(Test-Path $Path){
        $bak = "$Path.bak"
        if(-not (Test-Path $bak)){
            if(-not $DryRun){ Copy-Item $Path $bak -Force }
            Info "Backup: $bak"
        }
    }
}

function Apply-RegexPatch([string]$Path, [string]$Pattern, [string]$Replacement, [string]$Tag){
    if(-not (Test-Path $Path)){ Warn "SKIP ($Tag): file not found $Path"; return $false }
    $text = Read-File $Path
    if($null -eq $text){ Warn "SKIP ($Tag): cannot read $Path"; return $false }
    if([System.Text.RegularExpressions.Regex]::IsMatch($text, $Pattern, $RegexOpts)){
        Backup-File $Path
        $newText = [System.Text.RegularExpressions.Regex]::Replace($text, $Pattern, $Replacement, $RegexOpts)
        if($text -ne $newText){
            if(-not $DryRun){ Write-File $Path $newText }
            Ok "Patched: $Tag"
            return $true
        } else {
            Info "Already patched (no change): $Tag"
            return $true
        }
    } else {
        Warn "Pattern not found: $Tag"
        return $false
    }
}

function Replace-All([string]$Path, [string]$Pattern, [string]$Replacement, [string]$Tag){
    if(-not (Test-Path $Path)){ Warn "SKIP ($Tag): file not found $Path"; return $false }
    $text = Read-File $Path
    if($null -eq $text){ Warn "SKIP ($Tag): cannot read $Path"; return $false }
    $newText = [System.Text.RegularExpressions.Regex]::Replace($text, $Pattern, $Replacement, $RegexOpts)
    if($text -ne $newText){
        Backup-File $Path
        if(-not $DryRun){ Write-File $Path $newText }
        Ok "Replaced (global): $Tag"
        return $true
    } else {
        Info "No changes (global): $Tag"
        return $false
    }
}

# --- Phase 1: Syntax unblockers ---
# 1A) +page.svelte: remove stray 'await' before for()
$pageSvelte = Join-Path $ProjectRoot "src\routes\+page.svelte"
Apply-RegexPatch $pageSvelte `
    'await\s*//[^\r\n]*\r?\n\s*for\s*\(' `
    "// Multiple docs save - adapt as needed`r`nfor (" `
    "+page.svelte -> remove 'await // ...' before for()" | Out-Null

Apply-RegexPatch $pageSvelte `
    ';\s*await\s*\r?\n\s*for\s*\(' `
    ";`r`nfor (" `
    "+page.svelte -> generic await-scrub before for()" | Out-Null

# Cleanup malformed inline hover attributes that break parsing
Apply-RegexPatch $pageSvelte `
  "on:mouseover=\{\{[^}]+\}\}" `
  "" `
  "+page.svelte -> remove malformed on:mouseover {{...}}" | Out-Null
Apply-RegexPatch $pageSvelte `
  "on:mouseout=\{[^}]+\}" `
  "" `
  "+page.svelte -> remove malformed on:mouseout" | Out-Null

# --- Phase 2: Types & Globals ---
$appDtsPath = Join-Path $ProjectRoot "src\app.d.ts"
$appDts = @"
// src/app.d.ts
declare namespace App {
  interface Locals {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
  interface PageData {
    user: { id: string; username: string; name: string; role: 'admin' | 'user' } | null;
  }
}
"@
Ensure-FileExact $appDtsPath $appDts

$globalDtsPath = Join-Path $ProjectRoot "src\lib\types\global.d.ts"
$globalDts = @"
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
"@
Ensure-FileExact $globalDtsPath $globalDts

# --- Interpreter wrappers: revert to unknown->cast pattern ---
$interpreterTs = Join-Path $ProjectRoot "src\lib\elfin\interpreter.ts"
Replace-All $interpreterTs `
  "this\.scripts\['onUpload'\]\s*=\s*\(ctx:[^)]+\)\s*=>\s*onUpload\(ctx\);" `
  "this.scripts['onUpload'] = (ctx) => onUpload(ctx as UploadContext);" `
  "interpreter.ts -> onUpload cast wrapper" | Out-Null
Replace-All $interpreterTs `
  "this\.scripts\['onConceptChange'\]\s*=\s*\(ctx:[^)]+\)\s*=>\s*onConceptChange\(ctx\);" `
  "this.scripts['onConceptChange'] = (ctx) => onConceptChange(ctx as ConceptChangeContext);" `
  "interpreter.ts -> onConceptChange cast wrapper" | Out-Null
Replace-All $interpreterTs `
  "this\.scripts\['onGhostStateChange'\]\s*=\s*\(ctx:[^)]+\)\s*=>\s*onGhostStateChange\(ctx\);" `
  "this.scripts['onGhostStateChange'] = (ctx) => onGhostStateChange(ctx as GhostStateChangeContext);" `
  "interpreter.ts -> onGhostStateChange cast wrapper" | Out-Null

# --- toriInit.ts getExecutionStats guard using any-cast ---
$toriInit = Join-Path $ProjectRoot "src\lib\toriInit.ts"
Replace-All $toriInit `
  "checkStats:\s*\(\)\s*=>\s*\(activeEngine.*?getExecutionStats.*?\)\s*\?\s*activeEngine.*?getExecutionStats\(\)\s*:\s*'Engine not available'" `
  "checkStats: () => { const eng = activeEngine as any; return (eng && typeof eng.getExecutionStats === 'function') ? eng.getExecutionStats() : 'Engine not available'; }" `
  "toriInit.ts -> robust checkStats guard (any-cast)" | Out-Null

# --- dynamicApi.ts: safe error message usage ---
$dynamicApi = Join-Path $ProjectRoot "src\lib\dynamicApi.ts"
Replace-All $dynamicApi `
  "error\.message" `
  "(error instanceof Error ? error.message : String(error))" `
  "dynamicApi.ts -> safe error.message guards" | Out-Null

# --- conceptScoring_enhanced.ts: add Bench type + casts ---
$conceptScore = Join-Path $ProjectRoot "src\lib\cognitive\conceptScoring_enhanced.ts"
$text = Read-File $conceptScore
if($text){
  if(-not ($text -match "type\s+Bench\s*=\s*\{")){
    $new = "type Bench = { labels: string[]; method_used?: string; quality_metrics?: Record<string, number>; performance_stats?: Record<string, number>; status?: string; };`r`n" + $text
    Backup-File $conceptScore
    if(-not $DryRun){ Write-File $conceptScore $new }
    Ok "conceptScoring_enhanced.ts -> inserted Bench type"
  }
}
Replace-All $conceptScore "benchmarkResult\." "(benchmarkResult as Bench)." "conceptScoring_enhanced.ts -> cast benchmarkResult.*" | Out-Null
Replace-All $conceptScore "healthResult\.status" "(healthResult as any).status" "conceptScoring_enhanced.ts -> cast healthResult.status" | Out-Null

# --- types.ts: import type + remove stray artifacts + ensure aliases ---
$typesTs = Join-Path $ProjectRoot "src\lib\cognitive\types.ts"
Apply-RegexPatch $typesTs `
  "import\s+\{\s*ConceptDiff\s*,\s*ConceptDiffType\s*\}\s+from\s+'`$lib/stores/conceptMesh';" `
  "import type { ConceptDiff, ConceptDiffType } from '$lib/stores/conceptMesh';" `
  "types.ts -> import type" | Out-Null

# Remove orphaned lines (top-level) if they exist
Replace-All $typesTs "^\s*timestamp\?:\s*number;\s*$" "" "types.ts -> remove stray 'timestamp?: number;'" | Out-Null
Replace-All $typesTs "^\s*loopId\?:\s*string;\s*$" "" "types.ts -> remove stray 'loopId?: string;'" | Out-Null
# Ensure LoopRecord and ConceptDiffState exist
$existing = Read-File $typesTs
if($existing){
  if($existing -notmatch "interface\s+LoopRecord"){
    $append = @"
export interface LoopRecord {
  timestamp?: number;
  loopId?: string;
}
"@
    $final = ($existing.TrimEnd() + "`r`n`r`n" + $append)
    Backup-File $typesTs
    if(-not $DryRun){ Write-File $typesTs $final }
    Ok "types.ts -> appended LoopRecord interface"
  }
  $existing2 = Read-File $typesTs
  if($existing2 -and ($existing2 -notmatch "type\s+ConceptDiffState")){
    $append2 = @"
export type ConceptDiffState = ConceptDiff;
"@
    $final2 = ($existing2.TrimEnd() + "`r`n" + $append2)
    Backup-File $typesTs
    if(-not $DryRun){ Write-File $typesTs $final2 }
    Ok "types.ts -> added ConceptDiffState alias"
  }
}

# --- Phase 3: Svelte markup repair (global + focused) ---
# Global pass: fix on:on:*, and (e)=>fn(e) wrappers
$allSvelte = Get-ChildItem -Path (Join-Path $ProjectRoot "src") -Recurse -Filter *.svelte | Select-Object -ExpandProperty FullName
foreach($file in $allSvelte){
    $text = Read-File $file
    if($null -eq $text){ continue }
    $patched = $text.Replace("on:on:", "on:")
    $patched = [regex]::Replace($patched, "on:(\w+)\s*=\s*\{\s*\(\s*e\s*\)\s*=>\s*([A-Za-z0-9_]+)\s*\(\s*e\s*\)\s*\}", "on:`$1={`$2}", $RegexOpts)
    $patched = [regex]::Replace($patched, "on:mouseover=\{\{[^}]+\}\}", "", $RegexOpts)
    $patched = [regex]::Replace($patched, "on:mouseout=\{[^}]+\}", "", $RegexOpts)
    if($patched -ne $text){
        Backup-File $file
        if(-not $DryRun){ Write-File $file $patched }
        Ok "Svelte handler cleanup: $file"
    }
}

# Focused: modal markup in three components
function Fix-ModalTags([string]$Path, [string]$backdrop, [string]$modal){
    Apply-RegexPatch $Path "<div class=""modal-backdrop""[^>]*>" $backdrop "$([System.IO.Path]::GetFileName($Path)) -> fixed modal-backdrop" | Out-Null
    Apply-RegexPatch $Path "<div class=""modal""[^>]*>" $modal "$([System.IO.Path]::GetFileName($Path)) -> fixed modal" | Out-Null
}
Fix-ModalTags (Join-Path $ProjectRoot "src\lib\components\GroupSelector.svelte") `
  '<div class="modal-backdrop" role="button" tabindex="0" on:click={() => (showCreateModal = false)} on:keydown={(e) => e.key === ''Escape'' && (showCreateModal = false)}>' `
  '<div class="modal" on:click|stopPropagation>'

Fix-ModalTags (Join-Path $ProjectRoot "src\lib\components\InviteButton.svelte") `
  '<div class="modal-backdrop" role="button" tabindex="0" on:click={toggleInviteModal} on:keydown={(e) => e.key === ''Escape'' && toggleInviteModal()}>' `
  '<div class="modal" on:click|stopPropagation>'

Fix-ModalTags (Join-Path $ProjectRoot "src\lib\components\InviteModal.svelte") `
  '<div class="modal-backdrop" role="button" tabindex="0" on:click={close} on:keydown={(e) => e.key === ''Escape'' && close()}>' `
  '<div class="modal" on:click|stopPropagation>'

# InviteModalSecure.svelte: unwrap wrappers
$inviteSecure = Join-Path $ProjectRoot "src\lib\components\InviteModalSecure.svelte"
Replace-All $inviteSecure "on:click=\{\s*\(\s*e\s*\)\s*=>\s*close\(\s*e\s*\)\s*\}" "on:click={close}" "InviteModalSecure.svelte -> close wrapper" | Out-Null
Replace-All $inviteSecure "on:click=\{\s*\(\s*e\s*\)\s*=>\s*joinGroup\(\s*e\s*\)\s*\}" "on:click={joinGroup}" "InviteModalSecure.svelte -> joinGroup wrapper" | Out-Null
Replace-All $inviteSecure "on:beforeunload=\{\s*\(\s*e\s*\)\s*=>\s*cleanup\(\s*e\s*\)\s*\}" "on:beforeunload={cleanup}" "InviteModalSecure.svelte -> beforeunload wrapper" | Out-Null

# HolographicDisplay.svelte: persona normalization to handle string vs object
$holo = Join-Path $ProjectRoot "src\lib\components\HolographicDisplay.svelte"
Apply-RegexPatch $holo `
  "\$:\s*if\s*\(ghostEngine\s*&&\s*\$ghostPersona\?\.\s*activePersona\s*&&\s*\$ghostPersona\.activePersona\.id\s*!==\s*currentPersona\?\.\s*id\)\s*\{[\s\S]*?\}" `
  "$: if (ghostEngine && $ghostPersona?.activePersona) {`r`n  const ap: any = $ghostPersona.activePersona;`r`n  const normalized = typeof ap === 'string' ? { id: ap, name: ap } : ap;`r`n  if (!currentPersona || normalized.id !== currentPersona.id) {`r`n    console.log('Switching hologram to:', normalized.name);`r`n    currentPersona = normalized;`r`n  }`r`n}" `
  "HolographicDisplay.svelte -> persona normalization" | Out-Null

# HealthGate.svelte: safe error.message
$healthGate = Join-Path $ProjectRoot "src\lib\components\HealthGate.svelte"
Replace-All $healthGate "error\.message" "(error instanceof Error ? error.message : String(error))" "HealthGate.svelte -> safe error.message" | Out-Null

# --- Optional: create stub ConceptDebugPanel to satisfy imports ---
$conceptDebug = Join-Path $ProjectRoot "src\lib\components\ConceptDebugPanel.svelte"
if(-not (Test-Path $conceptDebug)){
  $stub = @"
<script lang="ts">
  export let enabled: boolean = false;
</script>

{#if enabled}
  <div class="p-2 text-xs text-gray-500 border rounded">ConceptDebugPanel (stub)</div>
{/if}
"@
  Write-File $conceptDebug $stub
  Ok "Created stub: src\lib\components\ConceptDebugPanel.svelte"
}

Ok "PACK C (fixed) applied. Next steps:"
Write-Host "1) cd `"$ProjectRoot`"" -ForegroundColor Magenta
Write-Host "2) pnpm i" -ForegroundColor Magenta
Write-Host "3) pnpm run check" -ForegroundColor Magenta
Write-Host "4) pnpm run build" -ForegroundColor Magenta
'''
path = "/mnt/data/Apply-PackC.ps1"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
path
