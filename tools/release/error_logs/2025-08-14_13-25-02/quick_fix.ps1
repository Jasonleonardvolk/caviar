# Quick fix script for common TypeScript errors in the KHA project (Windows PowerShell)
# Generated from photoMorphPipeline.ts fix analysis

Write-Host "KHA TypeScript Error Quick Fix Script" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Function to fix GPUTexture.size references
function Fix-GPUTextureSize {
    Write-Host "Fixing GPUTexture.size references..." -ForegroundColor Yellow
    
    Get-ChildItem -Path . -Filter *.ts -Recurse | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        $original = $content
        
        # Replace common patterns
        $content = $content -replace '\.size\[0\]', '.width'
        $content = $content -replace '\.size\[1\]', '.height'
        $content = $content -replace 'size: ([a-zA-Z]+)\.size,', 'size: [$1.width, $1.height],'
        
        if ($content -ne $original) {
            Set-Content -Path $_.FullName -Value $content
            Write-Host "  Fixed: $($_.Name)" -ForegroundColor Green
        }
    }
}

# Function to remove writeTimestamp calls
function Fix-WriteTimestamp {
    Write-Host "Removing deprecated writeTimestamp calls..." -ForegroundColor Yellow
    
    Get-ChildItem -Path . -Filter *.ts -Recurse | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        $original = $content
        
        # Comment out writeTimestamp lines
        $content = $content -replace '^(.*)\.writeTimestamp', '// DEPRECATED: $1.writeTimestamp'
        
        if ($content -ne $original) {
            Set-Content -Path $_.FullName -Value $content
            Write-Host "  Fixed: $($_.Name)" -ForegroundColor Green
        }
    }
}

# Function to add fragment targets
function Fix-FragmentTargets {
    Write-Host "Checking fragment targets..." -ForegroundColor Yellow
    Write-Host "Note: Fragment targets need manual review - check render pipelines" -ForegroundColor Magenta
    
    # Find files that might need fragment target fixes
    Get-ChildItem -Path . -Filter *.ts -Recurse | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        
        if ($content -match 'fragment:\s*\{[^}]*entryPoint[^}]*\}' -and $content -notmatch 'targets:') {
            Write-Host "  Review needed: $($_.Name)" -ForegroundColor Yellow
        }
    }
}

# Function to report uninitialized properties
function Report-UninitializedProperties {
    Write-Host "Checking for uninitialized properties..." -ForegroundColor Yellow
    Write-Host "Add '!' to properties initialized in async methods:" -ForegroundColor Magenta
    Write-Host "  private myProperty!: Type;" -ForegroundColor Gray
    
    Get-ChildItem -Path . -Filter *.ts -Recurse | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        
        if ($content -match 'Property .* has no initializer') {
            Write-Host "  Check: $($_.Name)" -ForegroundColor Yellow
        }
    }
}

# Main execution
Write-Host "Starting fixes..." -ForegroundColor Green
Write-Host ""

# Create backup
$backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "Creating backup in $backupDir..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

if (Test-Path "./frontend") {
    Copy-Item -Path "./frontend" -Destination "$backupDir/frontend" -Recurse
}
if (Test-Path "./tori_ui_svelte") {
    Copy-Item -Path "./tori_ui_svelte" -Destination "$backupDir/tori_ui_svelte" -Recurse
}

# Apply fixes
Fix-GPUTextureSize
Fix-WriteTimestamp
Fix-FragmentTargets
Report-UninitializedProperties

Write-Host ""
Write-Host "Quick fixes applied!" -ForegroundColor Green
Write-Host "Please review the changes and run:" -ForegroundColor Cyan
Write-Host "  npm run build" -ForegroundColor White
Write-Host "  npm run type-check" -ForegroundColor White
Write-Host ""
Write-Host "Backups created in ./$backupDir directory" -ForegroundColor Gray
