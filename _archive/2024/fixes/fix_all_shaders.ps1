# Complete shader fix workflow
Write-Host "`n🚀 COMPLETE SHADER FIX WORKFLOW" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Step 1: Debug to see what's wrong
Write-Host "`nStep 1: Debugging shader errors..." -ForegroundColor Yellow
& ".\debug_remaining_errors.ps1"

# Wait for user to see results
Write-Host "`nPress any key to continue with targeted fixes..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

# Step 2: Apply targeted fixes
Write-Host "`nStep 2: Applying targeted fixes..." -ForegroundColor Yellow
& ".\targeted_shader_fixes.ps1"

# Step 3: Re-validate all shaders
Write-Host "`nStep 3: Final validation..." -ForegroundColor Yellow
& ".\fix_shaders.ps1"

Write-Host "`n✅ WORKFLOW COMPLETE!" -ForegroundColor Green
Write-Host @"

If any shaders still have errors:
1. velocityField.wgsl - Run: .\manual_velocity_fix.ps1
2. Other shaders - Check the error messages above

For shaders that were JSON files (avatarShader.wgsl, propagation.wgsl):
- The placeholder shaders need to be replaced with actual implementations
- Check for backup files with .json_backup extension

"@ -ForegroundColor Yellow
