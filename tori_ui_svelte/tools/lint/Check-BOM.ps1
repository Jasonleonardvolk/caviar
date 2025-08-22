# Check-BOM.ps1
# Standalone BOM checker for JSON files
# Can be integrated into pre-push hooks

param(
    [string]$Path = (Get-Location).Path,
    [switch]$AutoFix,
    [switch]$Quiet
)

$patterns = @("*.json", "*.jsonc", "tsconfig*.json", "*.config.json")
$exclude  = @("\node_modules\", "\build\", "\.svelte-kit\", "\dist\", "\.vercel\", "\var\uploads\", "\.git\")

if (-not $Quiet) {
    Write-Host "Checking for BOM in JSON files..." -ForegroundColor Cyan
    Write-Host "Path: $Path" -ForegroundColor Gray
}

$filesWithBom = New-Object System.Collections.Generic.List[string]

Get-ChildItem -Path $Path -Recurse -File -Include $patterns -ErrorAction SilentlyContinue | Where-Object {
    $full = $_.FullName
    foreach ($ex in $exclude) { 
        if ($full -like "*$ex*") { return $false } 
    }
    return $true
} | ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        if ($AutoFix) {
            # Remove BOM
            $trimmed = $bytes[3..($bytes.Length-1)]
            $text = [System.Text.Encoding]::UTF8.GetString($trimmed)
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText($_.FullName, $text, $utf8NoBom)
            if (-not $Quiet) {
                Write-Host "  Fixed: $($_.FullName)" -ForegroundColor Yellow
            }
        } else {
            $filesWithBom.Add($_.FullName) | Out-Null
            if (-not $Quiet) {
                Write-Host "  BOM found: $($_.FullName)" -ForegroundColor Red
            }
        }
    }
}

if ($filesWithBom.Count -gt 0) {
    if (-not $Quiet) {
        Write-Host ""
        Write-Host "Found BOM in $($filesWithBom.Count) file(s):" -ForegroundColor Red
        $filesWithBom | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        Write-Host ""
        Write-Host "Run with -AutoFix to remove BOMs automatically" -ForegroundColor Yellow
    }
    exit 1
} else {
    if (-not $Quiet) {
        if ($AutoFix) {
            Write-Host "All JSON files are BOM-free (fixed any that had BOM)" -ForegroundColor Green
        } else {
            Write-Host "All JSON files are BOM-free!" -ForegroundColor Green
        }
    }
    exit 0
}
