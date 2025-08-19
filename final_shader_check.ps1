# Final check and fix for all shaders
Write-Host "=== FINAL SHADER STATUS CHECK ===" -ForegroundColor Cyan

# Check main propagation.wgsl first
$mainProp = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"
Write-Host "`nChecking main propagation.wgsl..." -ForegroundColor Yellow
$result = & naga $mainProp 2>&1 | Out-String

if ($result -match "error.*line.*476") {
    Write-Host "Found error at line 476. Fixing now..." -ForegroundColor Red
    
    # Apply any of the fixes
    & ".\auto_fix_propagation_476.ps1"
}

# Check all shaders in both directories
$dirs = @(
    "C:\Users\jason\Desktop\tori\kha\frontend\shaders",
    "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
)

$totalValid = 0
$totalShaders = 0
$invalidShaders = @()

foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "`nDirectory: $dir" -ForegroundColor Yellow
        
        Get-ChildItem "$dir\*.wgsl" | ForEach-Object {
            $totalShaders++
            $result = & naga $_.FullName 2>&1 | Out-String
            
            if (-not ($result -match "error")) {
                Write-Host "  ✅ $($_.Name)" -ForegroundColor Green
                $totalValid++
            } else {
                Write-Host "  ❌ $($_.Name)" -ForegroundColor Red
                $invalidShaders += $_.FullName
                
                # Show first error
                $firstError = ($result -split "`n" | Where-Object { $_ -match "error:" }) | Select-Object -First 1
                Write-Host "     $firstError" -ForegroundColor DarkRed
            }
        }
    }
}

Write-Host "`n📊 SUMMARY: $totalValid out of $totalShaders shaders are valid" -ForegroundColor Cyan

if ($totalValid -eq $totalShaders) {
    Write-Host "`n🎉 ALL SHADERS ARE VALID! 🎉" -ForegroundColor Green
    Write-Host "You can now run TORI without shader compilation errors!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Invalid shaders:" -ForegroundColor Red
    $invalidShaders | ForEach-Object { 
        Write-Host "  - $(Split-Path $_ -Leaf)" -ForegroundColor Red 
    }
    
    Write-Host "`nTo fix remaining issues:" -ForegroundColor Yellow
    Write-Host "1. For propagation.wgsl: .\manual_fix_propagation.ps1" -ForegroundColor Cyan
    Write-Host "2. For other shaders: Check the specific error messages above" -ForegroundColor Cyan
}
