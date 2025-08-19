Param(
  [switch]$DryRun,
  [switch]$NoBackup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[PackC:Safe v2] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[PackC:Safe v2] OK: $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[PackC:Safe v2] WARN: $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[PackC:Safe v2] ERROR: $m" -ForegroundColor Red }

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
$RegexOpts = [Text.RegularExpressions.RegexOptions]::Singleline -bor [Text.RegularExpressions.RegexOptions]::Multiline
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
  $new = [Regex]::Replace($t, $pattern, $replacement, $RegexOpts)
  if($t -ne $new){
    Backup $p
    WriteText $p $new
    Ok $label
  } else { Info "nochange $label" }
}

# --------- PART 1: Kill remaining parser errors ----------
$allSvelte = Get-ChildItem -Path (Join-Path $ProjectRoot "src") -Recurse -Filter *.svelte | % { $_.FullName }

foreach($f in $allSvelte){
  $t = ReadText $f; if($null -eq $t){ continue }
  $orig = $t

  # 1A) Remove double-curly mouseover/mouseout attributes (illegal in Svelte)
  $t = [Regex]::Replace($t, 'on:mouse(?:over|out)=\{\{[\s\S]*?\}\}', '', $RegexOpts)

  # 1B) Remove mouseover/mouseout handlers that mutate this.style directly
  $t = [Regex]::Replace($t, 'on:mouse(?:over|out)=\{[^}]*this\.style[^}]*\}', '', $RegexOpts)

  # 1C) Remove corrupted trailing "Enter click" garbage which duplicates attributes
  $t = [Regex]::Replace($t, '>\s*e\.key\s*===\s*["'']Enter["'']\s*&&\s*e\.currentTarget\.click\(\)\}\s*tabindex="0"\s*role="button">', '>', $RegexOpts)

  # 1D) Remove corrupted trailing "(e.key === 'Escape') && foo()" garbage
  $t = [Regex]::Replace($t, '>\s*\(e\.key\s*===\s*["'']Escape["'']\)\s*&&\s*.*?\}>', '>', $RegexOpts)

  if($t -ne $orig){
    Backup $f; WriteText $f $t; Ok "parser cleanup: $([IO.Path]::GetFileName($f))"
  }
}

# 1E) Explicitly hit ScholarSpherePanel + +page for stubborn hover attrs
$scholar = Join-Path $ProjectRoot "src\lib\components\ScholarSpherePanel.svelte"
Replace-InFile $scholar 'on:mouse(?:over|out)=\{\{[\s\S]*?\}\}' '' "ScholarSpherePanel: removed all {{...}} hover attrs"
Replace-InFile $scholar 'on:mouse(?:over|out)=\{[^}]*\}' '' "ScholarSpherePanel: removed remaining on:mouse* handlers mutating style"

$page = Join-Path $ProjectRoot "src\routes\+page.svelte"
Replace-InFile $page 'on:mouse(?:over|out)=\{[^}]*this\.style[^}]*\}' '' "+page.svelte: removed inline style hover handlers"

# --------- PART 2: Types.ts import + brace sanity ----------
$types = Join-Path $ProjectRoot "src\lib\cognitive\types.ts"
if(Test-Path $types){
  # Replace value import with type-only (escape `$` so PS doesn't interpolate)
  Replace-InFile $types 'import\s+\{\s*ConceptDiff\s*,\s*ConceptDiffType\s*\}\s+from\s+''`\$lib/stores/conceptMesh'';' "import type { ConceptDiff, ConceptDiffType } from '`$lib/stores/conceptMesh';" "types.ts: import type"
  # If still unconverted (single quotes vs double), do a looser pass
  Replace-InFile $types 'import\s+\{\s*ConceptDiff\s*,\s*ConceptDiffType\s*\}\s+from\s+["'']`\$lib/stores/conceptMesh["''];' "import type { ConceptDiff, ConceptDiffType } from '`$lib/stores/conceptMesh';" "types.ts: import type (alt)"
}

# --------- PART 3: HealthGate error guard ----------
$healthGate = Join-Path $ProjectRoot "src\lib\components\HealthGate.svelte"
Replace-InFile $healthGate 'error\.message' '(error instanceof Error ? error.message : String(error))' "HealthGate: safe error.message"

# --------- PART 4: HolographicDisplay persona normalization ----------
$holo = Join-Path $ProjectRoot "src\lib\components\HolographicDisplay.svelte"
if(Test-Path $holo){
  $t = ReadText $holo
  if($t -match '\$:\s*if\s*\(ghostEngine\s*&&\s*\$ghostPersona\?\.\s*activePersona'){
    $patched = [Regex]::Replace($t,
      '\$:\s*if\s*\(ghostEngine\s*&&\s*\$ghostPersona\?\.\s*activePersona[\s\S]*?\{[\s\S]*?currentPersona\s*=\s*\$ghostPersona\.activePersona;[\s\S]*?\}',
      '$: if (ghostEngine && $ghostPersona?.activePersona) { const ap: any = $ghostPersona.activePersona; const normalized = typeof ap === "string" ? { id: ap, name: ap } : ap; if (!currentPersona || normalized.id !== currentPersona.id) { console.log("Switching hologram to:", normalized.name); currentPersona = normalized; } }',
      $RegexOpts
    )
    if($patched -ne $t){
      Backup $holo; WriteText $holo $patched; Ok "HolographicDisplay: persona normalization applied"
    } else {
      Info "nochange HolographicDisplay normalization"
    }
  }
}

Ok "Safe v2 complete. Now run: pnpm run check; pnpm run build"
