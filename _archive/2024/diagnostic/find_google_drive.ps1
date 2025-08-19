# Find Google Drive Path Script
# Detects your actual Google Drive location

$foundPath = $null
$khaInDrive = $null

Write-Host "`n===== DETECTING GOOGLE DRIVE PATH =====" -ForegroundColor Cyan

# Common Google Drive locations to check
$possiblePaths = @(
    "G:\My Drive",
    "G:\Shared drives",
    "$env:USERPROFILE\Google Drive",
    "$env:USERPROFILE\My Drive",
    "C:\Google Drive",
    "D:\Google Drive",
    "E:\Google Drive",
    "$env:USERPROFILE\GoogleDrive",
    "$env:USERPROFILE\Desktop\Google Drive"
)

# Check each possible path
foreach ($path in $possiblePaths) {
    Write-Host "Checking: $path" -ForegroundColor Gray
    if (Test-Path $path) {
        Write-Host "  Found Drive at: $path" -ForegroundColor Green
        
        # Check for My Laptop folder
        $myLaptopPath = Join-Path $path "My Laptop"
        if (Test-Path $myLaptopPath) {
            Write-Host "  Found 'My Laptop' folder!" -ForegroundColor Green
            $khaPath = Join-Path $myLaptopPath "kha"
            if (Test-Path $khaPath) {
                Write-Host "  Found kha folder in Drive!" -ForegroundColor Green
                $foundPath = $path
                $khaInDrive = $khaPath
                break
            }
        }
        
        # Also check directly for kha
        $directKha = Join-Path $path "kha"
        if (Test-Path $directKha) {
            Write-Host "  Found kha folder directly in Drive!" -ForegroundColor Green
            $foundPath = $path
            $khaInDrive = $directKha
            break
        }
    }
}

# If not found in common locations, search for GoogleDriveFS.exe process
if (-not $foundPath) {
    Write-Host "`nSearching for Google Drive process..." -ForegroundColor Yellow
    $driveProcess = Get-Process -Name "GoogleDriveFS" -ErrorAction SilentlyContinue
    if ($driveProcess) {
        Write-Host "Google Drive is running. Checking registry..." -ForegroundColor Yellow
        
        # Check registry for Google Drive path
        $regPath = "HKCU:\Software\Google\Drive"
        if (Test-Path $regPath) {
            $installPath = (Get-ItemProperty -Path $regPath -ErrorAction SilentlyContinue).InstallLocation
            if ($installPath) {
                Write-Host "Found install location: $installPath" -ForegroundColor Yellow
            }
        }
    }
}

# Manual search for any folder containing "kha"
if (-not $foundPath) {
    Write-Host "`nSearching all drives for Google Drive folders..." -ForegroundColor Yellow
    $drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -gt 0 }
    
    foreach ($drive in $drives) {
        $searchPath = "$($drive.Name):\"
        Write-Host "Searching $searchPath..." -ForegroundColor Gray
        
        # Look for Google Drive folders
        $googleFolders = Get-ChildItem -Path $searchPath -Directory -ErrorAction SilentlyContinue | 
            Where-Object { $_.Name -like "*Drive*" -or $_.Name -like "*Google*" }
        
        foreach ($folder in $googleFolders) {
            $testPath = Join-Path $folder.FullName "My Laptop\kha"
            if (Test-Path $testPath) {
                Write-Host "  Found kha at: $testPath" -ForegroundColor Green
                $foundPath = $folder.FullName
                $khaInDrive = $testPath
                break
            }
        }
        if ($foundPath) { break }
    }
}

Write-Host "`n===== RESULTS =====" -ForegroundColor Cyan

if ($foundPath) {
    Write-Host "SUCCESS! Google Drive found at: $foundPath" -ForegroundColor Green
    Write-Host "Kha folder in Drive: $khaInDrive" -ForegroundColor Green
    
    # Save the path for other scripts
    $configPath = ".\drive_config.json"
    $config = @{
        GoogleDrivePath = $foundPath
        KhaInDrive = $khaInDrive
        MyLaptopPath = if ($khaInDrive -match "My Laptop") { Split-Path $khaInDrive -Parent } else { $null }
        DetectedOn = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    
    $config | ConvertTo-Json | Out-File -FilePath $configPath
    Write-Host "`nConfiguration saved to: $configPath" -ForegroundColor Green
    
    Write-Host "`n===== NEXT STEPS =====" -ForegroundColor Yellow
    Write-Host "1. Run: .\update_sync_scripts.ps1" -ForegroundColor White
    Write-Host "   This will update all sync scripts with the correct path" -ForegroundColor Gray
    Write-Host "2. Then run: .\verify_drive_sync.ps1" -ForegroundColor White
    Write-Host "   To verify your 8-day sync" -ForegroundColor Gray
    
} else {
    Write-Host "WARNING: Could not automatically find Google Drive!" -ForegroundColor Red
    Write-Host "`nPlease check:" -ForegroundColor Yellow
    Write-Host "1. Is Google Drive installed and running?" -ForegroundColor White
    Write-Host "2. Is the kha folder synced to Google Drive?" -ForegroundColor White
    Write-Host "3. Can you see the Drive folder in File Explorer?" -ForegroundColor White
    
    Write-Host "`nMANUAL SETUP:" -ForegroundColor Cyan
    Write-Host "If you know where your Google Drive folder is, create drive_config.json with:" -ForegroundColor White
    Write-Host '{
    "GoogleDrivePath": "YOUR_DRIVE_PATH_HERE",
    "KhaInDrive": "YOUR_DRIVE_PATH_HERE\\My Laptop\\kha"
}' -ForegroundColor Gray
}

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")