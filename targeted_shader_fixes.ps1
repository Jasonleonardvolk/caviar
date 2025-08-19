# Targeted shader fixes for specific issues
Write-Host "`n=== Targeted Shader Fixes ===" -ForegroundColor Cyan

$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"

# Fix 1: Check and fix avatarShader.wgsl and propagation.wgsl if they are JSON
$jsonShaders = @("avatarShader.wgsl", "propagation.wgsl")

foreach ($shader in $jsonShaders) {
    $path = Join-Path $shaderDir $shader
    if (Test-Path $path) {
        $content = Get-Content $path -Raw
        
        # Check if it's JSON (starts with { and has quotes)
        if ($content -match '^\s*{' -and $content -match '"') {
            Write-Host "`n$shader appears to be JSON!" -ForegroundColor Red
            Write-Host "This needs to be replaced with actual WGSL shader code." -ForegroundColor Yellow
            
            # Create a basic placeholder shader
            $placeholderWGSL = @"
// Placeholder shader - replace with actual implementation
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Basic triangle vertices
    let x = f32((vertex_index << 1u) & 2u) - 1.0;
    let y = f32(vertex_index & 2u) - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv, 0.5, 1.0);
}
"@
            
            # Backup and replace
            $backupPath = "$path.json_backup"
            Move-Item $path $backupPath -Force
            $placeholderWGSL | Set-Content $path -NoNewline
            Write-Host "Backed up JSON to: $backupPath" -ForegroundColor DarkGray
            Write-Host "Created placeholder WGSL shader" -ForegroundColor Green
        }
    }
}

# Fix 2: More aggressive swizzle fix for lenticularInterlace.wgsl
$lenticularPath = Join-Path $shaderDir "lenticularInterlace.wgsl"
if (Test-Path $lenticularPath) {
    Write-Host "`nFixing lenticularInterlace.wgsl swizzles..." -ForegroundColor Yellow
    $content = Get-Content $lenticularPath -Raw
    
    # Fix any remaining swizzle assignments
    # Pattern: something.rgb = expression;
    $pattern = '(\s*)(\w+)\.rgb\s*=\s*([^;]+);'
    $replacement = @'
$1let temp_rgb = $3;
$1$2.r = temp_rgb.x;
$1$2.g = temp_rgb.y;
$1$2.b = temp_rgb.z;
'@
    
    $newContent = $content -replace $pattern, $replacement
    
    # Also handle vec3 returns
    $newContent = $newContent -replace 'color\.rgb\s*=\s*apply_calibration_fast\(color\.rgb\);', @'
    let calibrated = apply_calibration_fast(vec3<f32>(color.r, color.g, color.b));
    color.r = calibrated.x;
    color.g = calibrated.y;
    color.b = calibrated.z;
'@
    
    if ($newContent -ne $content) {
        $newContent | Set-Content $lenticularPath -NoNewline
        Write-Host "Applied additional swizzle fixes" -ForegroundColor Green
    }
}

# Fix 3: Add RenderParams struct if missing in multiViewSynthesis.wgsl
$multiViewPath = Join-Path $shaderDir "multiViewSynthesis.wgsl"
if (Test-Path $multiViewPath) {
    Write-Host "`nChecking multiViewSynthesis.wgsl for RenderParams struct..." -ForegroundColor Yellow
    $content = Get-Content $multiViewPath -Raw
    
    # Check if RenderParams exists
    if ($content -notmatch 'struct\s+RenderParams') {
        Write-Host "RenderParams struct not found, adding it..." -ForegroundColor Red
        
        # Add struct at the beginning after comments
        $renderParamsStruct = @"

struct RenderParams {
    view_count: u32,
    view_index: u32,
    time: f32,
    aberration_strength: f32,
}

"@
        
        # Insert after first comment block or at beginning
        if ($content -match '((?://.*\n)+)(.*)') {
            $content = $matches[1] + $renderParamsStruct + $matches[2]
        } else {
            $content = $renderParamsStruct + $content
        }
        
        $content | Set-Content $multiViewPath -NoNewline
        Write-Host "Added RenderParams struct" -ForegroundColor Green
    } else {
        # Struct exists, make sure it has aberration_strength
        if ($content -match 'struct\s+RenderParams\s*{([^}]+)}') {
            $structContent = $matches[1]
            if ($structContent -notmatch 'aberration_strength') {
                Write-Host "Adding aberration_strength to existing RenderParams" -ForegroundColor Yellow
                $content = $content -replace '(struct\s+RenderParams\s*{[^}]+)(})', '$1,    aberration_strength: f32`n}'
                $content | Set-Content $multiViewPath -NoNewline
                Write-Host "Added aberration_strength field" -ForegroundColor Green
            }
        }
    }
}

Write-Host "`n=== Fixes Applied ===" -ForegroundColor Green
Write-Host "Now run validation again to check results:" -ForegroundColor Yellow
Write-Host "  .\fix_shaders.ps1" -ForegroundColor Cyan
