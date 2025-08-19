# Audit-ShaderLoaderUsage.ps1
# Audits all ShaderLoader.load() calls in the codebase

param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$Fix = $false
)

Write-Host "=== ShaderLoader.load() Usage Audit ===" -ForegroundColor Cyan

# Search for all TypeScript/JavaScript files
$files = Get-ChildItem -Path $ProjectRoot -Include "*.ts","*.tsx","*.js","*.jsx" -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "node_modules|\.git|dist|build" }

$usages = @()
$shaderPathPattern = 'ShaderLoader\.load\s*\(\s*[''"`]([^''"`]+)[''"`]'

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $matches = [regex]::Matches($content, $shaderPathPattern)
    
    if ($matches.Count -gt 0) {
        $lines = $content -split "`n"
        
        foreach ($match in $matches) {
            $lineNumber = 1
            for ($i = 0; $i -lt $lines.Length; $i++) {
                if ($lines[$i] -match [regex]::Escape($match.Value)) {
                    $lineNumber = $i + 1
                    break
                }
            }
            
            $usages += [PSCustomObject]@{
                File = $file.FullName.Replace($ProjectRoot, "").TrimStart("\")
                Line = $lineNumber
                ShaderPath = $match.Groups[1].Value
                FullMatch = $match.Value
            }
        }
    }
}

# Display results
if ($usages.Count -eq 0) {
    Write-Host "No ShaderLoader.load() usages found!" -ForegroundColor Green
} else {
    Write-Host "`nFound $($usages.Count) ShaderLoader.load() calls:" -ForegroundColor Yellow
    
    $usages | Format-Table -AutoSize
    
    # Analyze shader paths
    Write-Host "`n=== Path Analysis ===" -ForegroundColor Cyan
    
    $pathPrefixes = $usages.ShaderPath | ForEach-Object {
        if ($_ -match '^([^/]+)/') { $matches[1] } else { "(root)" }
    } | Group-Object | Sort-Object Count -Descending
    
    Write-Host "Path prefixes used:" -ForegroundColor White
    $pathPrefixes | Format-Table Name, Count -AutoSize
    
    # Check if paths match canonical location
    $canonicalPrefix = "lib/webgpu/shaders/"
    $incorrectPaths = $usages | Where-Object { 
        -not ($_.ShaderPath.StartsWith($canonicalPrefix) -or $_.ShaderPath -match '\$\{')
    }
    
    if ($incorrectPaths.Count -gt 0) {
        Write-Host "`n⚠️  Found $($incorrectPaths.Count) calls with non-canonical paths:" -ForegroundColor Yellow
        $incorrectPaths | Format-Table File, Line, ShaderPath -AutoSize
        
        if ($Fix) {
            Write-Host "`nWould you like to fix these paths? (Y/N)" -ForegroundColor Yellow
            $response = Read-Host
            
            if ($response -eq 'Y') {
                foreach ($usage in $incorrectPaths) {
                    $filePath = Join-Path $ProjectRoot $usage.File
                    $content = Get-Content $filePath -Raw
                    
                    # Extract shader filename
                    if ($usage.ShaderPath -match '([^/]+\.wgsl)$') {
                        $shaderFile = $matches[1]
                        $newPath = "lib/webgpu/shaders/$shaderFile"
                        
                        $oldPattern = [regex]::Escape($usage.ShaderPath)
                        $content = $content -replace $oldPattern, $newPath
                        
                        Set-Content -Path $filePath -Value $content -NoNewline
                        Write-Host "  Fixed: $($usage.File):$($usage.Line) → $newPath" -ForegroundColor Green
                    }
                }
            }
        }
    } else {
        Write-Host "`n✅ All shader paths appear to use canonical location!" -ForegroundColor Green
    }
}

# Check for dynamic paths
$dynamicPaths = $usages | Where-Object { $_.ShaderPath -match '\$\{' }
if ($dynamicPaths.Count -gt 0) {
    Write-Host "`n📍 Found $($dynamicPaths.Count) dynamic shader paths:" -ForegroundColor Cyan
    $dynamicPaths | Format-Table File, Line, ShaderPath -AutoSize
    Write-Host "These use template literals and should be verified manually." -ForegroundColor Yellow
}

# Export results
$global:ShaderLoaderAudit = @{
    TotalUsages = $usages.Count
    IncorrectPaths = $incorrectPaths.Count
    DynamicPaths = $dynamicPaths.Count
    Usages = $usages
}

Write-Host "`n=== Recommendations ===" -ForegroundColor Cyan
Write-Host "1. All shader paths should use: 'lib/webgpu/shaders/[shader].wgsl'" -ForegroundColor White
Write-Host "2. Consider using shaderSources imports instead of runtime loading" -ForegroundColor White
Write-Host "3. If using bundled shaders, ShaderLoader.load() may be redundant" -ForegroundColor White
