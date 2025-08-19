# Unify-ShaderConstantImports.ps1
# Normalize all imports of shaderConstantManager to static imports
$ErrorActionPreference = "Stop"
$root = "D:\Dev\kha\frontend"

Write-Host "== Normalizing imports of shaderConstantManager to static ==" -ForegroundColor Cyan

# 1) Find any dynamic import(...) of shaderConstantManager (robust regex)
$dynMatches = Get-ChildItem -Path $root -Recurse -Include *.ts,*.tsx,*.mts,*.mjs -File |
  Select-String -AllMatches -Pattern '(?is)\bimport\s*\(\s*[`'']\.{1,2}\/(?:webgpu\/)?shaderConstantManager(?:\.ts)?[`'']\s*\)'

if ($dynMatches) {
  $files = $dynMatches | Select-Object -ExpandProperty Path -Unique
  foreach ($file in $files) {
    $src = Get-Content $file -Raw

    # Replace destructured dynamic import -> static import
    $src = $src -replace '(?is)const\s*\{\s*([^}]+?)\s*\}\s*=\s*await\s*import\s*\(\s*[`'']\.{1,2}\/(?:webgpu\/)?shaderConstantManager(?:\.ts)?[`'']\s*\)\s*;?',
                               'import { $1 } from ''./shaderConstantManager'';'

    # Replace namespace dynamic import -> static namespace (rare)
    $src = $src -replace '(?is)const\s+(\w+)\s*=\s*await\s*import\s*\(\s*[`'']\.{1,2}\/(?:webgpu\/)?shaderConstantManager(?:\.ts)?[`'']\s*\)\s*;?',
                               'import * as $1 from ''./shaderConstantManager'';'

    Set-Content -Path $file -Value $src -NoNewline
    Write-Host "  Fixed dynamic import -> static: $file" -ForegroundColor Green
  }
} else {
  Write-Host "  No dynamic imports found" -ForegroundColor Gray
}

# 2) Verify there are no stragglers
$leftovers = Get-ChildItem -Path $root -Recurse -Include *.ts,*.tsx,*.mts,*.mjs -File |
  Select-String -AllMatches -Pattern '(?is)\bimport\s*\(' |
  Where-Object { $_.Line -match 'shaderConstantManager' }

if ($leftovers) {
  Write-Host "  WARNING: Still saw dynamic import text, please inspect below" -ForegroundColor Yellow
  $leftovers | Select-Object -First 10 | ForEach-Object { "$($_.Path):$($_.LineNumber)  $($_.Line.Trim())" }
} else {
  Write-Host "  Confirmed: no dynamic import calls remain" -ForegroundColor Green
}

Write-Host "== Done ==" -ForegroundColor Cyan
