$ErrorActionPreference = "Stop"
$project = "D:\Dev\kha"
$frontend = Join-Path $project "frontend"
$templates = Join-Path $project "exports\templates"
$outDir = Join-Path $frontend "static\templates"
$out = Join-Path $outDir "index.json"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$items = @()

if (Test-Path $templates) {
  Get-ChildItem $templates -Filter *.glb | ForEach-Object {
    $metaPath = "$($_.FullName.Substring(0, $_.FullName.Length-4)).template.json"
    $meta = @{}
    if (Test-Path $metaPath) {
      try { $meta = Get-Content $metaPath | ConvertFrom-Json } catch {}
    }
    $items += [PSCustomObject]@{
      name  = $_.Name
      size  = $_.Length
      mtime = $_.LastWriteTimeUtc.ToString("o")
      meta  = $meta
    }
  }
}

$payload = @{ items = $items } | ConvertTo-Json -Depth 6
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Content -Path $out -Value $payload -Encoding UTF8
Write-Host "✅ templates index → $out"