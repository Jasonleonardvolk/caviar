# Hunt-BridgePersistence.ps1
# Find persistence mechanisms (service, task, Run key) for BridgeCommunication.exe

param([string]$ExeName = "BridgeCommunication.exe")

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Persistence Hunter for: $ExeName" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Search for the executable in common locations
Write-Host "SEARCHING FOR EXECUTABLE FILES:" -ForegroundColor Yellow
Write-Host "--------------------------------" -ForegroundColor Yellow

$searchPaths = @(
    "C:\Program Files",
    "C:\Program Files (x86)",
    "C:\ProgramData",
    "$env:LOCALAPPDATA",
    "$env:APPDATA"
)

$paths = @()
foreach ($searchPath in $searchPaths) {
    if (Test-Path $searchPath) {
        $found = Get-ChildItem -Path $searchPath -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ieq $ExeName } | Select-Object -Expand FullName
        $paths += $found
    }
}

if ($paths.Count -gt 0) {
    $paths | ForEach-Object { 
        Write-Host "Found: $_" -ForegroundColor Cyan
        
        # Get file details
        if (Test-Path $_) {
            $file = Get-Item $_
            Write-Host "  Size: $([math]::Round($file.Length / 1MB, 2)) MB" -ForegroundColor Gray
            Write-Host "  Modified: $($file.LastWriteTime)" -ForegroundColor Gray
            
            $vi = $file.VersionInfo
            if ($vi.CompanyName) {
                Write-Host "  Company: $($vi.CompanyName)" -ForegroundColor Gray
            }
            if ($vi.ProductName) {
                Write-Host "  Product: $($vi.ProductName)" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "No files found with name: $ExeName" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "CHECKING WINDOWS SERVICES:" -ForegroundColor Yellow
Write-Host "---------------------------" -ForegroundColor Yellow

# Services pointing to the exe
$services = Get-CimInstance Win32_Service | Where-Object { 
    $_.PathName -match [regex]::Escape($ExeName) 
}

if ($services) {
    $services | Select-Object Name, DisplayName, StartMode, State, PathName | Format-Table -AutoSize
} else {
    Write-Host "No services found referencing $ExeName" -ForegroundColor Gray
}

Write-Host ""
Write-Host "CHECKING SCHEDULED TASKS:" -ForegroundColor Yellow
Write-Host "--------------------------" -ForegroundColor Yellow

# Scheduled Tasks referencing the exe
$tasks = @()
Get-ScheduledTask -ErrorAction SilentlyContinue | ForEach-Object {
    $taskName = $_.TaskName
    $_.Actions | Where-Object {
        $_.Execute -match $ExeName -or $_.Arguments -match $ExeName
    } | ForEach-Object {
        $tasks += [pscustomobject]@{ 
            TaskName = $taskName
            Execute = $_.Execute
            Args = $_.Arguments 
        }
    }
}

if ($tasks.Count -gt 0) {
    $tasks | Format-Table -AutoSize
} else {
    Write-Host "No scheduled tasks found referencing $ExeName" -ForegroundColor Gray
}

Write-Host ""
Write-Host "CHECKING REGISTRY RUN KEYS:" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow

# Run / RunOnce keys (HKLM / HKCU)
$runKeys = @(
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\RunOnce',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
)

$runKeyEntries = @()
foreach($rk in $runKeys) {
    if (Test-Path $rk) {
        $props = Get-ItemProperty -Path $rk -ErrorAction SilentlyContinue
        if ($props) {
            $props.PSObject.Properties | Where-Object {
                $_.Name -notlike "PS*" -and $_.Value -match $ExeName
            } | ForEach-Object {
                $runKeyEntries += [pscustomobject]@{
                    Key = $rk
                    Name = $_.Name
                    Value = $_.Value
                }
            }
        }
    }
}

if ($runKeyEntries.Count -gt 0) {
    $runKeyEntries | Format-Table -AutoSize
} else {
    Write-Host "No registry Run keys found referencing $ExeName" -ForegroundColor Gray
}

Write-Host ""
Write-Host "CHECKING STARTUP FOLDERS:" -ForegroundColor Yellow
Write-Host "--------------------------" -ForegroundColor Yellow

# Check startup folders
$startupPaths = @(
    "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup",
    "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"
)

$startupItems = @()
foreach ($sp in $startupPaths) {
    if (Test-Path $sp) {
        $items = Get-ChildItem -Path $sp -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match $ExeName -or (Test-Path $_.FullName -PathType Leaf -ErrorAction SilentlyContinue) }
        
        foreach ($item in $items) {
            if ($item.Extension -eq ".lnk") {
                $shell = New-Object -ComObject WScript.Shell
                $shortcut = $shell.CreateShortcut($item.FullName)
                if ($shortcut.TargetPath -match $ExeName) {
                    $startupItems += [pscustomobject]@{
                        Location = $sp
                        Name = $item.Name
                        Target = $shortcut.TargetPath
                    }
                }
            } elseif ($item.Name -match $ExeName) {
                $startupItems += [pscustomobject]@{
                    Location = $sp
                    Name = $item.Name
                    Target = $item.FullName
                }
            }
        }
    }
}

if ($startupItems.Count -gt 0) {
    $startupItems | Format-Table -AutoSize
} else {
    Write-Host "No startup folder items found referencing $ExeName" -ForegroundColor Gray
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Persistence Hunt Complete" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Summary
Write-Host ""
Write-Host "SUMMARY:" -ForegroundColor Magenta
Write-Host "--------" -ForegroundColor Magenta
Write-Host "Files found: $($paths.Count)" -ForegroundColor White
Write-Host "Services found: $(if($services){$services.Count}else{0})" -ForegroundColor White
Write-Host "Scheduled tasks found: $($tasks.Count)" -ForegroundColor White
Write-Host "Registry Run keys found: $($runKeyEntries.Count)" -ForegroundColor White
Write-Host "Startup items found: $($startupItems.Count)" -ForegroundColor White