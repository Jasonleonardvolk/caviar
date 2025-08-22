<#

Apply-Tori-FullFix-03.ps1
Targeted fix for: D:\Dev\kha\tori_ui_svelte\src\lib\components\vault\MemoryVaultDashboard.svelte
Error: Unexpected ")" at or near the IIFE close inside onMount (line ~137)

What this does:
  * Deletes any existing onMount(...) { ... } block inside <script lang="ts">
  * Removes stray ')();' / '})();' closers within the <script> block
  * Injects a stable onMount() that triggers an internal async run() with proper cleanup

Backups:
  D:\Dev\kha\patches\backup-YYYYMMDD-HHmmss\...

Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Apply-Tori-FullFix-03.ps1 -Root "D:\Dev\kha\tori_ui_svelte"

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
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $rel = ($path).Replace($Root, "").TrimStart("\","/")
    $backup = "D:\Dev\kha\patches\backup-$timestamp\$rel"
    Ensure-Dir $backup
    Copy-Item -LiteralPath $path -Destination $backup -Force
    Write-TextFile $path $mutated
    return $true
  }
  return $false
}

$path = Join-Path $Root "src\lib\components\vault\MemoryVaultDashboard.svelte"
if (-not (Test-Path $path)) {
  Write-Host "File not found: $path"
  exit 1
}

$changed = Mutate-File $path {
  param($t)

  # Only operate within the primary <script lang="ts"> ... </script> block
  $scriptRegex = [regex]'<script\s+lang="ts">([\s\S]*?)</script>'
  $m = $scriptRegex.Match($t)
  if (-not $m.Success) { return $t }

  $body = $m.Groups[1].Value

  # 1) Remove any loose IIFE closers and ")();" that cause parse errors
  $body = [regex]::Replace($body, "^\s*\)\(\);\s*$", "", 'Multiline')
  $body = [regex]::Replace($body, "^\s*\}\)\(\);\s*$", "", 'Multiline')
  $body = [regex]::Replace($body, "^\s*\)\s*;\s*$", "", 'Multiline')
  $body = [regex]::Replace($body, "^\s*\}\)\s*;\s*$", "", 'Multiline')

  # 2) Remove any existing onMount(...) block (non-greedy; stops at first '});')
  $body = [regex]::Replace($body, "onMount\s*\(\s*\(\)?\s*=>\s*\{[\s\S]*?\}\);\s*", "", 'Singleline')
  $body = [regex]::Replace($body, "onMount\s*\(\s*[^)]*\)\s*=>\s*\{[\s\S]*?\}\);\s*", "", 'Singleline')
  $body = [regex]::Replace($body, "onMount\s*\(\s*[\s\S]*?\)\s*;\s*", "", 'Singleline')

  # 3) Append a safe onMount() that uses a named async run() (no IIFE)
  $safeOnMount = @"
onMount(() => {
  if (!browser) return;
  const unsubscribes: Array<() => void> = [];

  const run = async () => {
    try {
      if (typeof loadMemoryVault === 'function') {
        await loadMemoryVault();
      }
      if (typeof autoRefresh !== 'undefined' && autoRefresh) {
        // refresh every 5s (idempotent)
        try { if (refreshInterval) clearInterval(refreshInterval); } catch {}
        refreshInterval = setInterval(() => {
          try { (loadMemoryVault && loadMemoryVault()); } catch {}
        }, 5000);
      }
      // Optional subscriptions if the stores exist
      try {
        (solitonMemory && solitonMemory.subscribe) && unsubscribes.push(solitonMemory.subscribe(updateFromSoliton));
        (conceptStore && conceptStore.subscribe) && unsubscribes.push(conceptStore.subscribe(updateFromConcepts));
        (ghostPersona && ghostPersona.subscribe) && unsubscribes.push(ghostPersona.subscribe(updateFromGhost));
      } catch {}
    } catch {}
  };

  run();

  return () => {
    try { unsubscribes.forEach((u) => { try { u() } catch {} }); } catch {}
    try { if (refreshInterval) clearInterval(refreshInterval); } catch {}
  };
});
"@

  $newBody = $body.TrimEnd() + "`r`n`r`n" + $safeOnMount + "`r`n"
  $newContent = $t.Substring(0, $m.Index) + "<script lang=`"ts`">" + $newBody + "</script>" + $t.Substring($m.Index + $m.Length)

  return $newContent
}

if ($changed) {
  Write-Host "Patched: $path"
} else {
  Write-Host "No changes applied (already normalized)"
}

Write-Host "`nNext:"
Write-Host "  cd $Root"
Write-Host "  pnpm run check"
Write-Host "  pnpm run build"
