# Fix-WGSL.ps1
# Targeted auto-fixer for WGSL shader issues flagged by Naga validation.

param(
    [switch]$Apply = $false,
    [switch]$AutoClampWorkgroup = $false,
    [switch]$FixTextureLoad = $true,
    [switch]$FixInlineUniform = $true,
    [switch]$FixSwizzle = $true,
    [switch]$StripLeadingBrace = $true
)

# Bootstrap: Relaunch with ExecutionPolicy Bypass if not already
if (-not $env:FIXWGSL_BOOTSTRAPPED) {
    $env:FIXWGSL_BOOTSTRAPPED = "1"
    powershell -ExecutionPolicy Bypass -File $PSCommandPath @args
    exit $LASTEXITCODE
}

$Timestamp = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"
$BackupDir = Join-Path $PSScriptRoot "..\..\frontend\shaders.bak\auto_fixes\$Timestamp"
$ReportDir = Join-Path $PSScriptRoot "reports"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null

$ReportJson = @()
$ReportTxt = @()

$shaderFiles = Get-ChildItem -Recurse -Filter *.wgsl (Join-Path $PSScriptRoot "..\..") | Where-Object { $_.FullName -notmatch '\\shaders\.bak\\' }

foreach ($file in $shaderFiles) {
    $original = Get-Content $file.FullName -Raw
    $fixed = $original
    $changed = $false

    if ($FixTextureLoad) {
        $pattern = 'textureLoad\(\s*([A-Za-z0-9_]+)\s*,\s*vec2<i32>\([^)]+\)\s*,\s*i32\([^)]+\)\s*\)'
        if ($fixed -match $pattern) {
            $fixed = [regex]::Replace($fixed, $pattern, '$0, 0')
            $changed = $true
            $ReportTxt += "$($file.FullName): Added mip level to textureLoad"
        }
    }

    if ($FixInlineUniform) {
        $pattern = 'var<uniform>\s+(\w+)\s*:\s*struct\s*\{'
        if ($fixed -match $pattern) {
            $structName = "$($matches[1])_Struct"
            $fixed = $fixed -replace $pattern, "struct $structName {\r\n"
            $fixed += "`n@group(0) @binding(0) var<uniform> $($matches[1]): $structName;"
            $changed = $true
            $ReportTxt += "$($file.FullName): Converted inline uniform struct to named struct"
        }
    }

    if ($FixSwizzle) {
        $pattern = '(\w+)\.rgb\s*=\s*(.+?);'
        if ($fixed -match $pattern) {
            $var = $matches[1]
            $expr = $matches[2]
            $replacement = "let __tmp = $expr;\n$var.r = __tmp.r;\n$var.g = __tmp.g;\n$var.b = __tmp.b;"
            $fixed = [regex]::Replace($fixed, $pattern, [regex]::Escape($replacement))
            $changed = $true
            $ReportTxt += "$($file.FullName): Expanded swizzle assignment"
        }
    }

    if ($StripLeadingBrace) {
        if ($fixed.TrimStart().StartsWith('{')) {
            $fixed = $fixed -replace '^\s*\{', ''
            $changed = $true
            $ReportTxt += "$($file.FullName): Removed leading brace"
        }
    }

    if ($changed -and $Apply) {
        $backupPath = Join-Path $BackupDir ($file.FullName.Substring((Join-Path $PSScriptRoot "..\..").Length).TrimStart('\'))
        New-Item -ItemType Directory -Path (Split-Path $backupPath) -Force | Out-Null
        Copy-Item $file.FullName $backupPath
        Set-Content $file.FullName $fixed -Encoding UTF8
    }

    $ReportJson += [PSCustomObject]@{
        File = $file.FullName
        Changed = $changed
    }
}

$reportJsonPath = Join-Path $ReportDir "Fix-WGSL_$Timestamp.json"
$reportTxtPath = Join-Path $ReportDir "Fix-WGSL_$Timestamp.txt"
$ReportJson | ConvertTo-Json -Depth 3 | Out-File $reportJsonPath -Encoding UTF8
$ReportTxt | Out-File $reportTxtPath -Encoding UTF8

Write-Host "Report written to $reportJsonPath and $reportTxtPath"
if (-not $Apply) {
    Write-Host "Dry run complete. Use -Apply to make changes."
}
