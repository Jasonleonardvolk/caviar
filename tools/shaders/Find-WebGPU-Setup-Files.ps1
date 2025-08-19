# Find-WebGPU-Setup-Files.ps1
# Helps locate TypeScript/JavaScript files that need buffer type changes

param(
    [string]$SearchRoot = "."
)

Write-Host "Searching for WebGPU setup files that need updating..." -ForegroundColor Cyan
Write-Host "Root: $SearchRoot"
Write-Host ""

# Search patterns that indicate WebGPU buffer setup
$patterns = @(
    "wavefield",
    "osc_data", 
    "oscillator",
    "createBindGroup",
    "bindGroupLayout",
    "GPUBuffer",
    "device.createBuffer",
    "\.queue\.writeBuffer",
    "GPUBindGroupLayout"
)

$extensions = @("*.ts", "*.tsx", "*.js", "*.jsx", "*.mjs")
$foundFiles = @{}

foreach ($ext in $extensions) {
    Write-Host "Scanning $ext files..." -ForegroundColor Yellow
    
    Get-ChildItem -Path $SearchRoot -Filter $ext -Recurse -ErrorAction SilentlyContinue | 
    Where-Object { 
        $_.FullName -notmatch "node_modules|\.git|dist|build|shaders\.bak" 
    } | ForEach-Object {
        $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            foreach ($pattern in $patterns) {
                if ($content -match $pattern) {
                    if (-not $foundFiles.ContainsKey($_.FullName)) {
                        $foundFiles[$_.FullName] = @()
                    }
                    $foundFiles[$_.FullName] += $pattern
                }
            }
        }
    }
}

Write-Host ""
Write-Host "=== RESULTS ===" -ForegroundColor Green
Write-Host ""

if ($foundFiles.Count -eq 0) {
    Write-Host "No WebGPU setup files found." -ForegroundColor Red
    Write-Host "Try searching in parent directories or check if files are in .git-ignored paths."
} else {
    Write-Host "Found $($foundFiles.Count) potential files to update:" -ForegroundColor Green
    Write-Host ""
    
    foreach ($file in $foundFiles.GetEnumerator()) {
        $relativePath = Resolve-Path -Relative $file.Key -ErrorAction SilentlyContinue
        if (-not $relativePath) { $relativePath = $file.Key }
        
        Write-Host "  $relativePath" -ForegroundColor Cyan
        Write-Host "    Matches: $($file.Value -join ', ')" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "In these files, look for:" -ForegroundColor Yellow
    Write-Host "  1. GPUBufferUsage.UNIFORM -> Change to GPUBufferUsage.STORAGE"
    Write-Host "  2. buffer: { type: 'uniform' } -> Change to buffer: { type: 'read-only-storage' }"
    Write-Host "  3. Any references to wavefield_params or osc_data buffers"
}

Write-Host ""
Write-Host "=== WHAT TO SEARCH FOR IN YOUR IDE ===" -ForegroundColor Magenta
Write-Host "If the automated search didn't find your files, search for these in your IDE:"
Write-Host '  - "wavefield_params"'
Write-Host '  - "GPUBufferUsage.UNIFORM"'  
Write-Host '  - "type: \"uniform\""'
Write-Host '  - "type: '\''uniform'\''"'
Write-Host '  - "createBuffer"'
Write-Host '  - "bindGroupLayout"'
Write-Host ""
Write-Host "The files are likely named something like:"
Write-Host "  - WavefieldCompute.ts"
Write-Host "  - HologramRenderer.ts"
Write-Host "  - ComputePipeline.ts"
Write-Host "  - WebGPURenderer.ts"
