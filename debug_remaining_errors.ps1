# Debug remaining shader errors
Write-Host "`n=== Debugging Remaining Shader Errors ===" -ForegroundColor Cyan

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$naga = "naga"

# Check specific shaders that had issues
$problemShaders = @(
    "avatarShader.wgsl",
    "lenticularInterlace.wgsl", 
    "multiViewSynthesis.wgsl",
    "propagation.wgsl",
    "velocityField.wgsl"
)

foreach ($shaderName in $problemShaders) {
    $shaderPath = Join-Path $shaderDir $shaderName
    
    if (Test-Path $shaderPath) {
        Write-Host "`n--- $shaderName ---" -ForegroundColor Yellow
        
        # Show first 10 lines
        Write-Host "First 10 lines:" -ForegroundColor Gray
        Get-Content $shaderPath -TotalCount 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
        
        # Run validator and capture full error
        Write-Host "`nValidation errors:" -ForegroundColor Red
        $error = & $naga $shaderPath 2>&1
        $error | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
    }
}

Write-Host "`n=== Manual Fix Instructions ===" -ForegroundColor Cyan
Write-Host @"
1. avatarShader.wgsl - Check if it starts with '{' or is JSON
2. propagation.wgsl - Check if it starts with '{' or is JSON  
3. lenticularInterlace.wgsl - May need manual swizzle fix
4. multiViewSynthesis.wgsl - Check if RenderParams struct exists
5. velocityField.wgsl - Check storage buffer parameter syntax
"@ -ForegroundColor Yellow
