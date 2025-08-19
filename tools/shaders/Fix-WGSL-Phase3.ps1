# Fix-WGSL-Phase3.ps1
<#
Purpose:
  Phase-3 WGSL fixer focusing on:
    A) Uniform alignment and padding for structs used in var<uniform>.
       - Add @align(16) and @size(16) to scalar/vector members where needed.
       - Add @stride(16) to array<T> members inside uniform structs.
    B) Rename reserved identifiers that break Naga/ WGSL (e.g., "filter").
       - Applies to declarations only (let/var/const/struct member/alias/fn name).
       - Current map: filter -> rc_filter

Notes:
  - This pass is conservative. It only edits structs that are referenced by var<uniform>.
  - It does not attempt algorithmic changes or buffer layout rewrites beyond alignment/stride.
  - Review diffs in frontend\shaders.bak\phase3\<timestamp>\ before commit.

Usage:
  From repo root C:\Users\jason\Desktop\tori\kha\
    tools\shaders\Fix-WGSL-Phase3.cmd
  or
    powershell -ExecutionPolicy Bypass -File tools\shaders\Fix-WGSL-Phase3.ps1 -Apply

Outputs:
  - Backups: frontend\shaders.bak\phase3\<timestamp>\*
  - Reports: tools\shaders\reports\Fix-WGSL-Phase3_<timestamp>.{txt,json}
#>

param(
  [switch]$Apply = $false,
  [string[]]$Roots = @("frontend\lib\webgpu\shaders", "frontend\public\hybrid\wgsl")
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $here "..\..")
$timestamp = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"
$backupRoot = Join-Path $repoRoot "frontend\shaders.bak\phase3\$timestamp"
$reportDir = Join-Path $here "reports"
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$results = @()

# Reserved identifier rename map
$ReservedMap = @{
  "filter" = "rc_filter"
}

function Save-BackupAndWrite {
  param($filePath, $newText, [ref]$changed)
  if ($newText -ne (Get-Content $filePath -Raw)) {
    $rel = Resolve-Path $filePath | ForEach-Object {
      $_.Path.Replace((Resolve-Path $repoRoot).Path + [System.IO.Path]::DirectorySeparatorChar, "")
    }
    $backupPath = Join-Path $backupRoot $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $backupPath) | Out-Null
    Copy-Item $filePath $backupPath -Force
    if ($Apply) {
      Set-Content -Path $filePath -Value $newText -Encoding UTF8
    }
    $changed.Value = $true
  }
}

function Find-UniformStructNames {
  param($text)
  # Captures patterns like:
  #   var<uniform> foo: FooParams;
  #   @group(0) @binding(1) var<uniform> params: Params;
  $names = New-Object System.Collections.Generic.HashSet[string]
  $pattern = '(?m)var<uniform>\s+[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*)\s*;'
  [System.Text.RegularExpressions.Regex]::Matches($text, $pattern) | ForEach-Object {
    $null = $names.Add($_.Groups[1].Value)
  }
  return $names
}

function Add-AlignmentHints {
  param($text, $uniformStructNames, [ref]$changes)

  if ($uniformStructNames.Count -eq 0) { return $text }

  # Rewrite each matched struct body
  $pattern = '(?s)struct\s+([A-Za-z_]\w*)\s*\{\s*(.*?)\s*\}\s*;'
  $newText = [System.Text.RegularExpressions.Regex]::Replace($text, $pattern, {
    param($m)
    $sname = $m.Groups[1].Value
    $body  = $m.Groups[2].Value

    if (-not $uniformStructNames.Contains($sname)) {
      return $m.Value
    }

    $changes.Value += @("align_struct:$sname")

    # Process lines inside struct body
    $lines = $body -split "(\r\n|\n)"
    for ($i=0; $i -lt $lines.Count; $i++) {
      $line = $lines[$i]

      # Skip comments
      if ($line -match '^\s*//') { continue }

      # Member with type array<...>
      $line = [System.Text.RegularExpressions.Regex]::Replace($line,
        '^\s*([A-Za-z_]\w*)\s*:\s*array\s*<\s*([A-Za-z_]\w*(?:<[^>]+>)?)\s*>\s*;',
        { param($mm)
          $mname = $mm.Groups[1].Value
          $elem  = $mm.Groups[2].Value
          "@stride(16) ${mname}`: array<${elem}>;"
        })

      # Member with scalar or vector
      # If already has @align or @size, do not duplicate
      if ($line -match '^\s*@') {
        # leave as-is
      } else {
        # Types that often need 16-byte alignment in uniforms
        $line = [System.Text.RegularExpressions.Regex]::Replace($line,
          '^\s*([A-Za-z_]\w*)\s*:\s*(f32|i32|u32|vec2<\s*(?:f32|i32|u32)\s*>|vec3<\s*(?:f32|i32|u32)\s*>|mat[234]x[234]<\s*f32\s*>)\s*;',
          { param($mm)
            $mname = $mm.Groups[1].Value
            $mtype = $mm.Groups[2].Value
            "@align(16) @size(16) ${mname}`: ${mtype};"
          })
      }

      $lines[$i] = $line
    }

    $newBody = ($lines -join "")
    "struct $sname {`r`n$newBody`r`n};"
  })

  return $newText
}

function Rename-ReservedIdentifiers {
  param($text, [ref]$changes)

  foreach ($k in $ReservedMap.Keys) {
    $v = $ReservedMap[$k]

    # let NAME =
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)(\blet\s+)$k(\b)",
      "`$1$v")

    # var NAME :
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)(\bvar\s+)$k(\b)",
      "`$1$v")

    # const NAME =
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)(\bconst\s+)$k(\b)",
      "`$1$v")

    # struct member NAME :
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)^(\s*)([A-Za-z_]\w*\s*:\s*)",
      { param($m)
        # This generic rule cannot target "filter" specifically, so skip
        $m.Value
      })

    # function name: fn NAME(
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)(\bfn\s+)$k(\s*\()",
      "`$1$v$2")

    # parameter names: fn f(..., NAME: T, ...)
    $text = [System.Text.RegularExpressions.Regex]::Replace($text,
      "(?m)([:,(]\s*)$k(\s*:)",
      "`$1$v$2")

    if ($text -match "\b$v\b") {
      $changes.Value += @("rename:$k->$v")
    }
  }

  return $text
}

$scanPaths = @()
foreach ($root in $Roots) {
  $abs = Join-Path $repoRoot $root
  if (Test-Path $abs) { $scanPaths += $abs }
}
$files = @()
foreach ($p in $scanPaths) {
  $files += Get-ChildItem -Path $p -Recurse -Filter *.wgsl | Where-Object { $_.FullName -notmatch '\\shaders\.bak\\' }
}

Write-Host "Phase-3 WGSL fixer starting"
Write-Host "Repository root: $repoRoot"
Write-Host "Scanning:"
$scanPaths | ForEach-Object { Write-Host "  - $_" }
Write-Host "Found $($files.Count) WGSL files"

foreach ($f in $files) {
  $orig = Get-Content $f.FullName -Raw
  $changedFlags = New-Object System.Collections.Generic.List[string]

  $uniformNames = Find-UniformStructNames $orig
  $txt = $orig
  if ($uniformNames.Count -gt 0) {
    $txt = Add-AlignmentHints -text $txt -uniformStructNames $uniformNames ([ref]$changedFlags)
  }
  $txt = Rename-ReservedIdentifiers -text $txt ([ref]$changedFlags)

  $fileChanged = $false
  Save-BackupAndWrite -filePath $f.FullName -newText $txt -changed ([ref]$fileChanged)

  $results += [PSCustomObject]@{
    file = $f.FullName
    changed = $fileChanged
    changes = $changedFlags
  }
}

# Write reports
$reportJson = Join-Path $reportDir "Fix-WGSL-Phase3_$timestamp.json"
$reportTxt  = Join-Path $reportDir "Fix-WGSL-Phase3_$timestamp.txt"

$results | ConvertTo-Json -Depth 5 | Out-File $reportJson -Encoding UTF8

$summary = @()
$summary += "Fix-WGSL-Phase3 Report  $timestamp"
$summary += "Repo: $repoRoot"
$summary += "Roots: " + ($Roots -join ", ")
$summary += ""
foreach ($r in $results) {
  if ($r.changed) {
    $summary += "$($r.file) : " + (($r.changes | ForEach-Object { $_ }) -join ", ")
  }
}
$summary -join "`r`n" | Out-File $reportTxt -Encoding UTF8

Write-Host "Reports:"
Write-Host "  $reportJson"
Write-Host "  $reportTxt"
if (-not $Apply) {
  Write-Host "Dry run complete. Re-run with -Apply to write changes."
} else {
  Write-Host "Apply complete. Backups in: $backupRoot"
}