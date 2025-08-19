param(
  [string]$InputDir = "D:\Dev\kha\assets\textures",
  [string]$OutDir   = "D:\Dev\kha\exports\textures_ktx2",
  [string]$Config   = "D:\Dev\kha\tools\exporters\ktx2.config.json"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Config)) { throw "Config not found: $Config" }
$cfg = Get-Content $Config | ConvertFrom-Json

$basis = Join-Path $PWD $cfg.basisu
if (!(Test-Path $basis)) {
  Write-Host "❌ basisu not found at $basis"
  Write-Host "→ Download BasisU CLI and place at D:\Dev\kha\tools\bin\basisu.exe"
  exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$images = Get-ChildItem -Path $InputDir -Include *.png,*.jpg,*.jpeg -Recurse

foreach ($img in $images) {
  $rel = $img.FullName.Substring($InputDir.Length).TrimStart('\','/')
  $dst = Join-Path $OutDir ([System.IO.Path]::ChangeExtension($rel, ".ktx2"))
  $dstDir = Split-Path $dst
  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

  $isNormal = ($img.BaseName -match '(_n|_nrml|_normal)$')
  $flip = $cfg.y_flip -eq $true ? 1 : 0

  $args = @(
    "-ktx2",
    "-uastc", $cfg.uastc_level,
    "-uastc_rdo_l", $cfg.rdo_lambda,
    "-jobs", $cfg.jobs,
    "-y_flip", $flip,
    "-mipmap"
  )
  if (-not $cfg.mipmap) { $args = $args | Where-Object { $_ -ne "-mipmap" } }
  if ($isNormal) { $args += "-normal_map" }

  $args += @("-output_file", $dst, $img.FullName)

  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  & $basis @args
  if ($LASTEXITCODE -ne 0) { throw "basisu failed for $($img.FullName)" }
  Write-Host "✅ $rel → $(Resolve-Path $dst -Relative)"
}
Write-Host "Done. Output: $OutDir"