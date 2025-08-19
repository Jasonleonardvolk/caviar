#!/usr/bin/env pwsh
# C:\Users\jason\Desktop\tori\kha\scripts\migrate_wave_files.ps1
# Migrates FFT/wave files to the wave directory for conditional loading

param(
    [switch]$DryRun = $false,
    [switch]$Rollback = $false
)

$projectRoot = "C:\Users\jason\Desktop\tori\kha"
$waveDir = "$projectRoot\frontend\lib\webgpu\wave"

Write-Host "Wave File Migration Script" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

if ($Rollback) {
    Write-Host "ROLLBACK MODE: Moving files back to original locations" -ForegroundColor Yellow
} elseif ($DryRun) {
    Write-Host "DRY RUN MODE: No files will be moved" -ForegroundColor Yellow
} else {
    Write-Host "MIGRATION MODE: Files will be moved to wave directory" -ForegroundColor Green
}

# Define file migrations
$migrations = @(
    @{
        From = "$projectRoot\frontend\lib\webgpu\fftCompute.ts"
        To = "$waveDir\fft\fftCompute.ts"
    },
    @{
        From = "$projectRoot\frontend\lib\webgpu\fftDispatchValidator.ts"
        To = "$waveDir\fft\fftDispatchValidator.ts"
    },
    @{
        From = "$projectRoot\frontend\lib\webgpu\fftOptimizations.ts"
        To = "$waveDir\fft\fftOptimizations.ts"
    },
    @{
        From = "$projectRoot\frontend\lib\webgpu\fftPrecomputed.ts"
        To = "$waveDir\fft\fftPrecomputed.ts"
    },
    @{
        From = "$projectRoot\frontend\lib\webgpu\hologramPropagation.ts"
        To = "$waveDir\propagation\hologramPropagation.ts"
    }
)

# Process migrations
foreach ($migration in $migrations) {
    $source = if ($Rollback) { $migration.To } else { $migration.From }
    $dest = if ($Rollback) { $migration.From } else { $migration.To }
    
    if (Test-Path $source) {
        $destDir = Split-Path $dest -Parent
        
        if ($DryRun) {
            Write-Host "  Would move: $(Split-Path $source -Leaf)" -ForegroundColor Cyan
            Write-Host "         to: $dest" -ForegroundColor Gray
        } else {
            # Create destination directory if needed
            if (!(Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
                Write-Host "  Created directory: $destDir" -ForegroundColor Green
            }
            
            # Move the file
            Move-Item -Path $source -Destination $dest -Force
            Write-Host "  Moved: $(Split-Path $source -Leaf) -> $dest" -ForegroundColor Green
        }
    } else {
        Write-Host "  Skip: $(Split-Path $source -Leaf) (not found)" -ForegroundColor Yellow
    }
}

# Handle shader directories
$shaderMigrations = @(
    @{
        From = "$projectRoot\frontend\lib\webgpu\shaders\fft"
        To = "$waveDir\shaders\fft"
    }
)

foreach ($migration in $shaderMigrations) {
    $source = if ($Rollback) { $migration.To } else { $migration.From }
    $dest = if ($Rollback) { $migration.From } else { $migration.To }
    
    if (Test-Path $source) {
        if ($DryRun) {
            Write-Host "  Would move directory: $(Split-Path $source -Leaf)" -ForegroundColor Cyan
            Write-Host "                    to: $dest" -ForegroundColor Gray
        } else {
            # Create parent directory if needed
            $destParent = Split-Path $dest -Parent
            if (!(Test-Path $destParent)) {
                New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            }
            
            # Move the directory
            Move-Item -Path $source -Destination $dest -Force
            Write-Host "  Moved directory: $(Split-Path $source -Leaf) -> $dest" -ForegroundColor Green
        }
    }
}

# Create .npmignore in wave directory to exclude from npm package
if (!$Rollback -and !$DryRun) {
    $npmIgnorePath = "$waveDir\.npmignore"
    if (!(Test-Path $npmIgnorePath)) {
        @"
# Exclude wave processing from production npm package
*
!NullWaveBackend.ts
!WaveBackend.ts
"@ | Set-Content $npmIgnorePath
        Write-Host "  Created .npmignore in wave directory" -ForegroundColor Green
    }
}

Write-Host "`nMigration Complete!" -ForegroundColor Cyan

if (!$DryRun) {
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Update imports in holographicEngine.ts to use conditional loading" -ForegroundColor White
    Write-Host "2. Run: .\scripts\validate_wave_exclusion.ps1" -ForegroundColor White
    Write-Host "3. Test with: npm run dev" -ForegroundColor White
    Write-Host "4. Build production: VITE_IRIS_ENABLE_WAVE=0 npm run build" -ForegroundColor White
}
