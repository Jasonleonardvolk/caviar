# Quick fix for specific shader issues
Write-Host "Applying specific shader fixes..." -ForegroundColor Cyan

# Fix avatarShader.wgsl if it starts with {
$avatarShader = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\avatarShader.wgsl"
if (Test-Path $avatarShader) {
    $content = Get-Content $avatarShader -Raw
    if ($content -match '^\s*{') {
        Write-Host "Fixing avatarShader.wgsl - removing leading '{'" -ForegroundColor Yellow
        $content = $content -replace '^\s*{\s*', ''
        $content | Set-Content $avatarShader -NoNewline
    }
}

# Fix propagation.wgsl in webgpu/shaders if it exists and starts with {
$propagationShader = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders\propagation.wgsl"
if (Test-Path $propagationShader) {
    $content = Get-Content $propagationShader -Raw
    if ($content -match '^\s*{') {
        Write-Host "Fixing propagation.wgsl - removing leading '{'" -ForegroundColor Yellow
        $content = $content -replace '^\s*{\s*', ''
        $content | Set-Content $propagationShader -NoNewline
    }
}

Write-Host "Quick fixes applied!" -ForegroundColor Green
