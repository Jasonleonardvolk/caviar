<#
.SYNOPSIS
  Mirror your canonical shaders into the bundler folder without mangling paths.
#>

param(
  [string]$SourceDir = ".\frontend\shaders",
  [string]$TargetDir = ".\frontend\lib\webgpu\shaders"
)

# Resolve full absolute paths
$srcRoot = (Resolve-Path $SourceDir).Path
$dstRoot = (Resolve-Path $TargetDir -ErrorAction SilentlyContinue).Path

# 1) Ensure target exists
if (-not (Test-Path $dstRoot)) {
    New-Item -ItemType Directory -Path $dstRoot -Force | Out-Null
}

# 2) Clear out old mirror
Get-ChildItem $dstRoot -Recurse | Remove-Item -Force -Recurse

# 3) Copy every .wgsl, preserving subfolder structure
Get-ChildItem -Path $srcRoot -Filter *.wgsl -Recurse |
ForEach-Object {
    # Compute path relative to $srcRoot
    $relative = $_.FullName.Substring($srcRoot.Length).TrimStart('\','/')
    $dest    = Join-Path $dstRoot $relative
    $destDir = Split-Path $dest -Parent

    # Ensure its folder exists
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    Copy-Item -Path $_.FullName -Destination $dest -Force
    Write-Host "Copied $($_.Name) → $dest"
}
