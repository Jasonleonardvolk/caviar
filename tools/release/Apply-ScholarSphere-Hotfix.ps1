# Create a clean, parse-safe hotfix script the user can download and run.
content = r'''Param(
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[Hotfix] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[Hotfix] OK: $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[Hotfix] WARN: $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[Hotfix] ERROR: $m" -ForegroundColor Red }

# Common regex options
$RegexOpts = [Text.RegularExpressions.RegexOptions]::Singleline -bor [Text.RegularExpressions.RegexOptions]::Multiline

# Locate project root
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$candidates = @(
  (Join-Path $RepoRoot "tori_ui_svelte"),
  (Join-Path $RepoRoot "frontend"),
  $RepoRoot
)
$ProjectRoot = $null
foreach($c in $candidates){
  if(Test-Path (Join-Path $c "src")){
    $ProjectRoot = $c; break
  }
}
if(-not $ProjectRoot){ Err "Couldn't find project root containing 'src' under: $($candidates -join '; ')"; exit 2 }
Info "ProjectRoot: $ProjectRoot"

$target = Join-Path $ProjectRoot "src\lib\components\ScholarSpherePanel.svelte"
if(-not (Test-Path $target)){ Err "Missing file: $target"; exit 3 }

# Read file
$text = [IO.File]::ReadAllText($target)
$orig = $text

# 1) Remove inline `{#if isUploading}` block that was injected inside an attribute handler:
#    on:keydown={(e) =>
#      {#if isUploading} ... {/if}
#    }
$pat1 = '(on:\w+\s*=\s*\{\s*\([^}]*\)\s*=>)[\s\r\n]*\{#if\s+isUploading\}[\s\S]*?\{/if\}'
$text = [Regex]::Replace($text, $pat1, '$1 {}', $RegexOpts)

# 2) Clean any dangling `{#if isUploading} ... {/if}` sitting in the middle of an attribute list
$pat2 = '\s+\{#if\s+isUploading\}[\s\S]*?\{/if\}\s*(?=>)'
$text = [Regex]::Replace($text, $pat2, '', $RegexOpts)

# 3) Sanity: remove any double-brace hover handlers like: on:mouseover={{ ... }}
$pat3 = 'on:mouse(?:over|out)=\{\{[^}]+\}\}'
$text = [Regex]::Replace($text, $pat3, '', $RegexOpts)

if($text -ne $orig){
  if(-not $DryRun){
    [IO.File]::WriteAllText($target, $text, (New-Object System.Text.UTF8Encoding($false)))
  }
  Ok "Patched $target — removed inline {#if isUploading} from handler attrs"
} else {
  Info "No changes; pattern not found in $target"
}

Ok "Now run: pnpm run build"
'''
path = "/mnt/data/Apply-ScholarSphere-Hotfix_v2.ps1"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(path)
