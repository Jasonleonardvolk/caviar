# Setup Symbolic Link for Google Drive Sync
# Creates a symbolic link from G:\My Drive\kha to your local kha folder

$ErrorActionPreference = "Stop"

Write-Host "`n===== GOOGLE DRIVE SYMBOLIC LINK SETUP =====" -ForegroundColor Cyan
Write-Host "This will create a symbolic link for automatic syncing" -ForegroundColor Yellow

# Define paths
$googleDrivePath = "G:\My Drive"
$localKhaPath = "C:\Users\jason\Desktop\tori\kha"
$linkPath = "G:\My Drive\kha"

Write-Host "`nPaths:" -ForegroundColor Gray
Write-Host "  Local kha folder: $localKhaPath" -ForegroundColor White
Write-Host "  Google Drive: $googleDrivePath" -ForegroundColor White
Write-Host "  Symlink location: $linkPath" -ForegroundColor White

# Check if Google Drive exists
if (-not (Test-Path $googleDrivePath)) {
    Write-Host "`nERROR: Google Drive not found at G:\My Drive" -ForegroundColor Red
    Write-Host "Please ensure Google Drive is mounted at G:\" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✓ Google Drive found at G:\My Drive" -ForegroundColor Green

# Check if local kha exists
if (-not (Test-Path $localKhaPath)) {
    Write-Host "`nERROR: Local kha folder not found at:" -ForegroundColor Red
    Write-Host "  $localKhaPath" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Local kha folder exists" -ForegroundColor Green

# Check if something already exists at the link path
if (Test-Path $linkPath) {
    $item = Get-Item $linkPath -Force
    
    if ($item.LinkType -eq "SymbolicLink") {
        Write-Host "`nFound existing symbolic link at $linkPath" -ForegroundColor Yellow
        $target = $item.Target
        Write-Host "Currently points to: $target" -ForegroundColor Gray
        
        if ($target -eq $localKhaPath) {
            Write-Host "✓ Symbolic link already correctly configured!" -ForegroundColor Green
            Write-Host "`nNo changes needed. Your kha folder is already syncing." -ForegroundColor Cyan
        } else {
            Write-Host "`nThe symlink points to a different location." -ForegroundColor Yellow
            $response = Read-Host "Do you want to update it? (y/n)"
            if ($response -eq 'y') {
                Write-Host "Removing old symbolic link..." -ForegroundColor Yellow
                Remove-Item $linkPath -Force -Recurse
                $createNew = $true
            } else {
                Write-Host "Keeping existing symbolic link." -ForegroundColor Gray
                $createNew = $false
            }
        }
    } else {
        Write-Host "`nWARNING: A regular folder exists at $linkPath" -ForegroundColor Yellow
        Write-Host "This is NOT a symbolic link." -ForegroundColor Yellow
        
        # Check if it has content
        $fileCount = (Get-ChildItem $linkPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($fileCount -gt 0) {
            Write-Host "It contains $fileCount files." -ForegroundColor Red
            Write-Host "`nOPTIONS:" -ForegroundColor Cyan
            Write-Host "1. Backup and replace with symbolic link" -ForegroundColor White
            Write-Host "2. Keep the existing folder (no symlink)" -ForegroundColor White
            Write-Host "3. Cancel" -ForegroundColor White
            
            $choice = Read-Host "Enter choice (1-3)"
            
            switch ($choice) {
                "1" {
                    $backupPath = "G:\My Drive\kha_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                    Write-Host "Creating backup at: $backupPath" -ForegroundColor Yellow
                    Move-Item $linkPath $backupPath
                    Write-Host "✓ Backup created" -ForegroundColor Green
                    $createNew = $true
                }
                "2" {
                    Write-Host "Keeping existing folder." -ForegroundColor Gray
                    $createNew = $false
                }
                default {
                    Write-Host "Cancelled." -ForegroundColor Gray
                    exit 0
                }
            }
        } else {
            Write-Host "The folder is empty. Safe to replace." -ForegroundColor Yellow
            Remove-Item $linkPath -Force
            $createNew = $true
        }
    }
} else {
    Write-Host "`nNo existing kha folder in Google Drive" -ForegroundColor Yellow
    $createNew = $true
}

# Create the symbolic link if needed
if ($createNew) {
    Write-Host "`n===== CREATING SYMBOLIC LINK =====" -ForegroundColor Cyan
    Write-Host "This requires Administrator privileges..." -ForegroundColor Yellow
    
    # Create the mklink command
    $mklinkCmd = "mklink /D `"$linkPath`" `"$localKhaPath`""
    
    Write-Host "`nCommand to run:" -ForegroundColor Gray
    Write-Host "  $mklinkCmd" -ForegroundColor White
    
    # Check if running as admin
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
    
    if ($isAdmin) {
        Write-Host "`n✓ Running as Administrator" -ForegroundColor Green
        
        # Execute mklink via cmd
        $result = cmd /c $mklinkCmd 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Symbolic link created successfully!" -ForegroundColor Green
            Write-Host "$result" -ForegroundColor Gray
        } else {
            Write-Host "ERROR: Failed to create symbolic link" -ForegroundColor Red
            Write-Host "$result" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "`nNOT running as Administrator!" -ForegroundColor Red
        Write-Host "`nTo create the symbolic link, either:" -ForegroundColor Yellow
        Write-Host "1. Run PowerShell as Administrator and execute this script again" -ForegroundColor White
        Write-Host "2. Open Command Prompt as Administrator and run:" -ForegroundColor White
        Write-Host "   $mklinkCmd" -ForegroundColor Green
        
        # Create a batch file for easy admin execution
        $batchContent = "@echo off`r`n$mklinkCmd`r`npause"
        $batchContent | Out-File -FilePath ".\CREATE_SYMLINK_AS_ADMIN.bat" -Encoding ASCII
        Write-Host "`n3. Or right-click 'CREATE_SYMLINK_AS_ADMIN.bat' and 'Run as Administrator'" -ForegroundColor White
        
        exit 1
    }
}

# Verify the setup
Write-Host "`n===== VERIFICATION =====" -ForegroundColor Cyan

if (Test-Path $linkPath) {
    $item = Get-Item $linkPath -Force
    if ($item.LinkType -eq "SymbolicLink") {
        Write-Host "✓ Symbolic link exists at: $linkPath" -ForegroundColor Green
        Write-Host "✓ Points to: $($item.Target)" -ForegroundColor Green
        
        # Test by creating a file
        $testFile = "$linkPath\SYNC_TEST_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        "Test file created at $(Get-Date)" | Out-File $testFile
        
        if (Test-Path $testFile) {
            Write-Host "✓ Successfully created test file through symlink" -ForegroundColor Green
            
            # Check if it appears in local folder
            $localTestFile = $testFile.Replace($linkPath, $localKhaPath)
            if (Test-Path $localTestFile) {
                Write-Host "✓ Test file appears in local folder" -ForegroundColor Green
                Write-Host "`n========== SUCCESS! ==========" -ForegroundColor Green
                Write-Host "Your kha folder is now syncing via symbolic link!" -ForegroundColor Cyan
                Write-Host "Any changes in $localKhaPath" -ForegroundColor White
                Write-Host "will automatically sync through Google Drive" -ForegroundColor White
            }
        }
    }
}

Write-Host "`n===== SYNC STATUS =====" -ForegroundColor Cyan
Write-Host "Google Drive will now sync:" -ForegroundColor Yellow
Write-Host "  FROM: $localKhaPath" -ForegroundColor White
Write-Host "  TO: $linkPath (symbolic link)" -ForegroundColor White
Write-Host "  Which syncs to: Google Drive cloud" -ForegroundColor White

Write-Host "`n===== MONITORING =====" -ForegroundColor Cyan
Write-Host "To monitor sync status:" -ForegroundColor Yellow
Write-Host "  1. Check system tray for Google Drive icon" -ForegroundColor White
Write-Host "  2. Look for green checkmarks in File Explorer" -ForegroundColor White
Write-Host "  3. Visit: https://drive.google.com/drive/recent" -ForegroundColor White

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")