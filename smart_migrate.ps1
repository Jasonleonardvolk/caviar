# Smart Migration with Automatic Path Fixing
# ==========================================
# This PowerShell script migrates files and automatically fixes hardcoded paths

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("PigpenToTori", "ToriToPigpen")]
    [string]$Direction,
    
    [string[]]$FilesToMigrate = @(),
    
    [switch]$DryRun = $false,
    
    [switch]$FixPaths = $true
)

# Define base paths
$TORI_PATH = "C:\Users\jason\Desktop\tori\kha"
$PIGPEN_PATH = "C:\Users\jason\Desktop\pigpen"

# Common hardcoded path patterns to fix
$PATH_REPLACEMENTS = @{
    # File path patterns
    'C:\\Users\\jason\\Desktop\\tori\\kha' = @{
        'tori' = 'C:\\Users\\jason\\Desktop\\tori\\kha'
        'pigpen' = 'C:\\Users\\jason\\Desktop\\pigpen'
    }
    'C:/Users/jason/Desktop/tori/kha' = @{
        'tori' = 'C:/Users/jason/Desktop/tori/kha'
        'pigpen' = 'C:/Users/jason/Desktop/pigpen'
    }
    '/home/user/tori/kha' = @{
        'tori' = '/home/user/tori/kha'
        'pigpen' = '/home/user/pigpen'
    }
    # Variable patterns
    '${TORI_ROOT}' = @{
        'tori' = '${TORI_ROOT}'
        'pigpen' = '${PIGPEN_ROOT}'
    }
    '${PIGPEN_ROOT}' = @{
        'pigpen' = '${PIGPEN_ROOT}'
        'tori' = '${TORI_ROOT}'
    }
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Test-TextFile {
    param([string]$FilePath)
    
    $textExtensions = @('.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml', 
                       '.md', '.txt', '.bat', '.ps1', '.sh', '.env', '.config')
    
    $extension = [System.IO.Path]::GetExtension($FilePath).ToLower()
    return $textExtensions -contains $extension
}

function Fix-HardcodedPaths {
    param(
        [string]$FilePath,
        [string]$FromEnv,
        [string]$ToEnv
    )
    
    if (-not (Test-TextFile -FilePath $FilePath)) {
        return $false
    }
    
    try {
        $content = Get-Content -Path $FilePath -Raw -ErrorAction Stop
        $originalContent = $content
        $changesMade = $false
        
        foreach ($pattern in $PATH_REPLACEMENTS.Keys) {
            if ($PATH_REPLACEMENTS[$pattern].ContainsKey($FromEnv) -and 
                $PATH_REPLACEMENTS[$pattern].ContainsKey($ToEnv)) {
                
                $fromPath = $PATH_REPLACEMENTS[$pattern][$FromEnv]
                $toPath = $PATH_REPLACEMENTS[$pattern][$ToEnv]
                
                if ($content -match [regex]::Escape($fromPath)) {
                    $content = $content -replace [regex]::Escape($fromPath), $toPath
                    $changesMade = $true
                    Write-ColorOutput "  Fixed: $fromPath → $toPath" -Color "Yellow"
                }
            }
        }
        
        # Also check for relative imports that might need adjustment
        if ($FilePath -match '\.py$') {
            # Python relative imports
            if ($FromEnv -eq 'pigpen' -and $ToEnv -eq 'tori') {
                # Pigpen might have different import paths
                $pythonPatterns = @{
                    'from tonka_' = 'from ingest_pdf.tonka_'
                    'import tonka_' = 'import ingest_pdf.tonka_'
                }
                
                foreach ($pattern in $pythonPatterns.Keys) {
                    if ($content -match $pattern) {
                        $content = $content -replace $pattern, $pythonPatterns[$pattern]
                        $changesMade = $true
                        Write-ColorOutput "  Fixed Python import: $pattern" -Color "Yellow"
                    }
                }
            }
        }
        
        if ($changesMade) {
            Set-Content -Path $FilePath -Value $content -NoNewline
            return $true
        }
        
        return $false
    }
    catch {
        Write-ColorOutput "  Error processing file: $_" -Color "Red"
        return $false
    }
}

function Copy-WithPathFix {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$FromEnv,
        [string]$ToEnv
    )
    
    # Ensure destination directory exists
    $destDir = Split-Path -Parent $Destination
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    
    # Copy the file
    Copy-Item -Path $Source -Destination $Destination -Force
    Write-ColorOutput "Copied: $(Split-Path -Leaf $Source)" -Color "Green"
    
    # Fix paths if enabled
    if ($FixPaths) {
        $fixed = Fix-HardcodedPaths -FilePath $Destination -FromEnv $FromEnv -ToEnv $ToEnv
        if ($fixed) {
            Write-ColorOutput "  Paths fixed in: $(Split-Path -Leaf $Destination)" -Color "Cyan"
        }
    }
}

function Start-Migration {
    Write-ColorOutput "`n========================================" -Color "Cyan"
    Write-ColorOutput "Smart File Migration with Path Fixing" -Color "Cyan"
    Write-ColorOutput "========================================`n" -Color "Cyan"
    
    Write-ColorOutput "Direction: $Direction" -Color "White"
    Write-ColorOutput "Fix Paths: $($FixPaths)" -Color "White"
    Write-ColorOutput "Dry Run: $($DryRun)`n" -Color "White"
    
    # Determine source and destination
    if ($Direction -eq "PigpenToTori") {
        $sourcePath = $PIGPEN_PATH
        $destPath = $TORI_PATH
        $fromEnv = "pigpen"
        $toEnv = "tori"
        Write-ColorOutput "Migrating: Pigpen → TORI`n" -Color "Yellow"
    }
    else {
        $sourcePath = $TORI_PATH
        $destPath = $PIGPEN_PATH
        $fromEnv = "tori"
        $toEnv = "pigpen"
        Write-ColorOutput "Migrating: TORI → Pigpen`n" -Color "Yellow"
    }
    
    # Create backup
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = Join-Path $destPath "backup_${Direction}_${timestamp}"
    
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        Write-ColorOutput "Backup directory: $backupDir`n" -Color "Gray"
    }
    
    # Process each file
    $successCount = 0
    $errorCount = 0
    
    foreach ($file in $FilesToMigrate) {
        $sourceFile = Join-Path $sourcePath $file
        $destFile = Join-Path $destPath $file
        
        Write-ColorOutput "`nProcessing: $file" -Color "White"
        
        if (-not (Test-Path $sourceFile)) {
            Write-ColorOutput "  Source not found: $sourceFile" -Color "Red"
            $errorCount++
            continue
        }
        
        # Backup existing file if it exists
        if ((Test-Path $destFile) -and -not $DryRun) {
            $backupFile = Join-Path $backupDir $file
            $backupFileDir = Split-Path -Parent $backupFile
            if (-not (Test-Path $backupFileDir)) {
                New-Item -ItemType Directory -Path $backupFileDir -Force | Out-Null
            }
            Copy-Item -Path $destFile -Destination $backupFile -Force
            Write-ColorOutput "  Backed up existing file" -Color "Gray"
        }
        
        if ($DryRun) {
            Write-ColorOutput "  [DRY RUN] Would copy: $sourceFile → $destFile" -Color "Magenta"
            if ($FixPaths -and (Test-TextFile -FilePath $sourceFile)) {
                Write-ColorOutput "  [DRY RUN] Would fix paths in this file" -Color "Magenta"
            }
        }
        else {
            try {
                Copy-WithPathFix -Source $sourceFile -Destination $destFile `
                    -FromEnv $fromEnv -ToEnv $toEnv
                $successCount++
            }
            catch {
                Write-ColorOutput "  Error: $_" -Color "Red"
                $errorCount++
            }
        }
    }
    
    # Summary
    Write-ColorOutput "`n========================================" -Color "Cyan"
    Write-ColorOutput "Migration Summary" -Color "Cyan"
    Write-ColorOutput "========================================" -Color "Cyan"
    Write-ColorOutput "Success: $successCount files" -Color "Green"
    Write-ColorOutput "Errors: $errorCount files" -Color "Red"
    if (-not $DryRun) {
        Write-ColorOutput "Backup: $backupDir" -Color "Gray"
    }
}

# Auto-detect files if none specified
if ($FilesToMigrate.Count -eq 0) {
    Write-ColorOutput "No files specified. Loading from recent changes..." -Color "Yellow"
    
    # This would be populated by your analyzer output
    # For now, showing the structure
    Write-ColorOutput @"
    
Please run with specific files, for example:

.\smart_migrate.ps1 -Direction PigpenToTori -FilesToMigrate @(
    'api/tonka_api.py',
    'test_tonka_integration.py',
    'bulk_pdf_processor.py',
    'prajna/api/prajna_api.py'
)

Or for dry run:
.\smart_migrate.ps1 -Direction PigpenToTori -FilesToMigrate @('api/tonka_api.py') -DryRun

"@ -Color "Cyan"
    exit
}

# Execute migration
Start-Migration
