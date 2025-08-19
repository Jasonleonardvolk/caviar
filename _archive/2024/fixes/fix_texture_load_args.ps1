Write-Host "`n🔥 FIXING TEXTURE_STORAGE_2D TEXTURELOAD CALLS" -ForegroundColor Cyan
Write-Host "Removing third argument from textureLoad calls on texture_storage_2d`n" -ForegroundColor Yellow

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"

# Fix lenticularInterlace.wgsl
Write-Host "1. Fixing lenticularInterlace.wgsl..." -ForegroundColor Green
$lenticularPath = Join-Path $shaderDir "lenticularInterlace.wgsl"
$content = Get-Content $lenticularPath -Raw
$content = $content -replace 'textureLoad\(temp_buffer,\s*coord,\s*0\)', 'textureLoad(temp_buffer, coord)'
$content = $content -replace 'textureLoad\(temp_buffer,\s*sample_coord,\s*0\)', 'textureLoad(temp_buffer, sample_coord)'
$content | Out-File -FilePath $lenticularPath -Encoding UTF8 -NoNewline
Write-Host "   ✅ Fixed line 381 and 389" -ForegroundColor Green

# Fix propagation.wgsl
Write-Host "`n2. Fixing propagation.wgsl..." -ForegroundColor Green
$propagationPath = Join-Path $shaderDir "propagation.wgsl"
$content = Get-Content $propagationPath -Raw
# Fix all textureLoad calls with 3 arguments on storage textures
$content = $content -replace 'textureLoad\(frequency_domain,\s*vec2<u32>\(coord\),\s*0\)', 'textureLoad(frequency_domain, vec2<u32>(coord))'
$content = $content -replace 'textureLoad\(transfer_function,\s*vec2<u32>\(coord\),\s*0\)', 'textureLoad(transfer_function, vec2<u32>(coord))'
$content = $content -replace 'textureLoad\(output_field,\s*vec2<u32>\(coord\),\s*0\)', 'textureLoad(output_field, vec2<u32>(coord))'
$content | Out-File -FilePath $propagationPath -Encoding UTF8 -NoNewline
Write-Host "   ✅ Fixed textureLoad calls on texture_storage_2d" -ForegroundColor Green

# Fix velocityField.wgsl
Write-Host "`n3. Fixing velocityField.wgsl..." -ForegroundColor Green
$velocityPath = Join-Path $shaderDir "velocityField.wgsl"
$content = Get-Content $velocityPath -Raw
# Fix all textureLoad calls with 3 arguments
$content = $content -replace 'textureLoad\(wavefield_texture,\s*clamped_coord,\s*0\)', 'textureLoad(wavefield_texture, clamped_coord, 0)'  # This is texture_2d, keep 3 args
$content = $content -replace 'textureLoad\(velocity_out,\s*clamped,\s*0\)', 'textureLoad(velocity_out, clamped)'
$content = $content -replace 'textureLoad\(velocity_out,\s*coord,\s*0\)', 'textureLoad(velocity_out, coord)'
$content = $content -replace 'textureLoad\(velocity_out,\s*coord\s*\+\s*vec2<i32>\(([^)]+)\),\s*0\)', 'textureLoad(velocity_out, coord + vec2<i32>($1))'
$content = $content -replace 'textureLoad\(velocity_out,\s*vec2<i32>\(([^)]+)\),\s*0\)', 'textureLoad(velocity_out, vec2<i32>($1))'
$content | Out-File -FilePath $velocityPath -Encoding UTF8 -NoNewline
Write-Host "   ✅ Fixed textureLoad calls on texture_storage_2d" -ForegroundColor Green

Write-Host "`n✅ ALL FILES FIXED!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Run: npx tsx scripts/bundleShaders.ts" -ForegroundColor White
Write-Host "2. Validate: .\check_shaders.ps1" -ForegroundColor White
Write-Host "3. Launch: .\START_TORI_HARDENED.bat -Force" -ForegroundColor White