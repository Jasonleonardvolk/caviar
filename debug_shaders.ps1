# Debug problematic shaders
Write-Host "`n=== Debugging Problematic Shaders ===" -ForegroundColor Cyan

$shaders = @(
    "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\avatarShader.wgsl",
    "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
)

foreach ($shader in $shaders) {
    if (Test-Path $shader) {
        $name = Split-Path $shader -Leaf
        Write-Host "`nChecking $name..." -ForegroundColor Yellow
        
        # Get first 5 lines
        $firstLines = Get-Content $shader -TotalCount 5
        Write-Host "First 5 lines:" -ForegroundColor Gray
        $firstLines | ForEach-Object { Write-Host "  $_" }
        
        # Check for leading {
        $content = Get-Content $shader -Raw
        if ($content -match '^\s*{') {
            Write-Host "`nFOUND ISSUE: File starts with '{'" -ForegroundColor Red
            Write-Host "This is likely a JSON file or corrupted WGSL file" -ForegroundColor Red
            
            # Check if it's JSON
            if ($content -match '"') {
                Write-Host "Appears to be JSON content!" -ForegroundColor Yellow
            }
        }
    }
}

Write-Host "`nRun the enhanced fixer script to auto-fix these issues:" -ForegroundColor Green
Write-Host "  .\fix_shaders_enhanced.ps1" -ForegroundColor Cyan
