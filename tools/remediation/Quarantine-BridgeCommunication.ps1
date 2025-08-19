# Quarantine-BridgeCommunication.ps1
# Kill and quarantine BridgeCommunication.exe processes safely

param(
    [switch]$DryRun,
    [switch]$SkipFirewall,
    [switch]$Verbose
)

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "BridgeCommunication.exe Quarantine Tool" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "*** DRY RUN MODE - No changes will be made ***" -ForegroundColor Yellow
    Write-Host ""
}

# Step 1: Kill existing processes
Write-Host ""
Write-Host "STEP 1: TERMINATING PROCESSES" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

$procs = Get-Process -Name BridgeCommunication -ErrorAction SilentlyContinue
if ($procs) {
    $procCount = ($procs | Measure-Object).Count
    Write-Host "Found $procCount BridgeCommunication.exe process(es)" -ForegroundColor Red
    
    if ($Verbose) {
        $procs | ForEach-Object {
            Write-Host "  PID: $($_.Id) | Memory: $([math]::Round($_.WorkingSet64/1MB, 2)) MB" -ForegroundColor Gray
        }
    }
    
    if (-not $DryRun) {
        Write-Host "Stopping all BridgeCommunication.exe processes..." -ForegroundColor Yellow
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        
        # Verify they're gone
        $remaining = Get-Process -Name BridgeCommunication -ErrorAction SilentlyContinue
        if ($remaining) {
            Write-Host "WARNING: Some processes could not be killed" -ForegroundColor Red
        } else {
            Write-Host "All processes terminated successfully" -ForegroundColor Green
        }
    } else {
        Write-Host "[DRY RUN] Would terminate $procCount process(es)" -ForegroundColor Cyan
    }
} else {
    Write-Host "No BridgeCommunication.exe processes currently running" -ForegroundColor Green
}

# Step 2: Identify and disable parent services
Write-Host ""
Write-Host "STEP 2: DISABLING RELATED SERVICES" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow

# Get processes before we killed them (if any remain)
$parentInfo = @()
$remainingProcs = Get-CimInstance Win32_Process -Filter "Name='BridgeCommunication.exe'" -ErrorAction SilentlyContinue
if ($remainingProcs) {
    $parents = $remainingProcs | Select-Object -Expand ParentProcessId -Unique | ForEach-Object {
        Get-CimInstance Win32_Process -Filter "ProcessId=$_" -ErrorAction SilentlyContinue
    }
    $parentInfo = $parents | Select-Object Name, ProcessId, ExecutablePath
}

# Find services that reference BridgeCommunication.exe
$services = Get-CimInstance Win32_Service | Where-Object {
    $_.PathName -match 'BridgeCommunication\.exe'
}

if ($services) {
    Write-Host "Found $($services.Count) service(s) referencing BridgeCommunication.exe" -ForegroundColor Red
    
    foreach ($svc in $services) {
        Write-Host "Service: $($svc.Name) [$($svc.DisplayName)]" -ForegroundColor White
        Write-Host "  State: $($svc.State) | Start Mode: $($svc.StartMode)" -ForegroundColor Gray
        Write-Host "  Path: $($svc.PathName)" -ForegroundColor Gray
        
        if (-not $DryRun) {
            Write-Host "  Stopping and disabling service..." -ForegroundColor Yellow
            
            # Stop service
            if ($svc.State -eq 'Running') {
                $result = sc.exe stop $svc.Name 2>&1
                Start-Sleep -Seconds 2
            }
            
            # Disable service
            $result = sc.exe config $svc.Name start= disabled 2>&1
            
            # Verify
            $updatedSvc = Get-CimInstance Win32_Service -Filter "Name='$($svc.Name)'"
            if ($updatedSvc.StartMode -eq 'Disabled') {
                Write-Host "  Service disabled successfully" -ForegroundColor Green
            } else {
                Write-Host "  WARNING: Failed to disable service" -ForegroundColor Red
            }
        } else {
            Write-Host "  [DRY RUN] Would stop and disable this service" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "No services found referencing BridgeCommunication.exe" -ForegroundColor Green
}

# Step 3: Block execution via NTFS ACL
Write-Host ""
Write-Host "STEP 3: BLOCKING FILE EXECUTION" -ForegroundColor Yellow
Write-Host "--------------------------------" -ForegroundColor Yellow

$searchPaths = @("C:\Program Files", "C:\Program Files (x86)", "C:\ProgramData", "$env:LOCALAPPDATA", "$env:APPDATA")
$exePaths = @()

foreach ($searchPath in $searchPaths) {
    if (Test-Path $searchPath) {
        $found = Get-ChildItem -Path $searchPath -Filter "BridgeCommunication.exe" -Recurse -ErrorAction SilentlyContinue
        $exePaths += $found.FullName
    }
}

$exePaths = $exePaths | Select-Object -Unique

if ($exePaths.Count -gt 0) {
    Write-Host "Found $($exePaths.Count) BridgeCommunication.exe file(s)" -ForegroundColor Red
    
    foreach($exePath in $exePaths) {
        Write-Host "File: $exePath" -ForegroundColor White
        
        if (Test-Path $exePath) {
            $file = Get-Item $exePath
            Write-Host "  Size: $([math]::Round($file.Length / 1MB, 2)) MB" -ForegroundColor Gray
            
            if (-not $DryRun) {
                Write-Host "  Applying Deny-Execute ACL..." -ForegroundColor Yellow
                
                # Deny execute permissions for Everyone
                $result = icacls $exePath /deny "*S-1-1-0:(RX)" 2>&1
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  Execution blocked successfully" -ForegroundColor Green
                    
                    # Rename file as additional safety measure
                    $backupPath = "$exePath.quarantine"
                    try {
                        Rename-Item -Path $exePath -NewName "$($file.Name).quarantine" -ErrorAction Stop
                        Write-Host "  File renamed to: $($file.Name).quarantine" -ForegroundColor Green
                    } catch {
                        Write-Host "  Could not rename file (may be in use)" -ForegroundColor Yellow
                    }
                } else {
                    Write-Host "  WARNING: Failed to block execution" -ForegroundColor Red
                }
            } else {
                Write-Host "  [DRY RUN] Would block execution and rename file" -ForegroundColor Cyan
            }
        }
    }
} else {
    Write-Host "No BridgeCommunication.exe files found" -ForegroundColor Green
}

# Step 4: Optional firewall rules
if (-not $SkipFirewall) {
    Write-Host ""
    Write-Host "STEP 4: CREATING FIREWALL RULES" -ForegroundColor Yellow
    Write-Host "--------------------------------" -ForegroundColor Yellow
    
    if ($exePaths.Count -gt 0) {
        foreach($exePath in $exePaths) {
            if (Test-Path $exePath) {
                $ruleName = "Block-BridgeCommunication-$([System.IO.Path]::GetFileNameWithoutExtension($exePath))-$(Get-Random -Maximum 999)"
                
                Write-Host "Creating firewall rule: $ruleName" -ForegroundColor White
                Write-Host "  Program: $exePath" -ForegroundColor Gray
                
                if (-not $DryRun) {
                    # Check if rule already exists
                    $existingRule = Get-NetFirewallRule -DisplayName "Block-BridgeCommunication-*" -ErrorAction SilentlyContinue |
                        Where-Object { (Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $_).Program -eq $exePath }
                    
                    if (-not $existingRule) {
                        try {
                            New-NetFirewallRule -DisplayName $ruleName `
                                -Direction Outbound `
                                -Action Block `
                                -Program $exePath `
                                -Profile Any `
                                -ErrorAction Stop | Out-Null
                            
                            Write-Host "  Outbound traffic blocked" -ForegroundColor Green
                        } catch {
                            Write-Host "  WARNING: Could not create firewall rule: $_" -ForegroundColor Yellow
                        }
                    } else {
                        Write-Host "  Firewall rule already exists" -ForegroundColor Gray
                    }
                } else {
                    Write-Host "  [DRY RUN] Would create outbound block rule" -ForegroundColor Cyan
                }
            }
        }
    } else {
        Write-Host "No files to create firewall rules for" -ForegroundColor Gray
    }
}

# Step 5: Clean up scheduled tasks
Write-Host ""
Write-Host "STEP 5: CHECKING SCHEDULED TASKS" -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Yellow

$tasks = @()
Get-ScheduledTask -ErrorAction SilentlyContinue | ForEach-Object {
    $taskName = $_.TaskName
    $taskPath = $_.TaskPath
    $_.Actions | Where-Object {
        $_.Execute -match 'BridgeCommunication\.exe' -or $_.Arguments -match 'BridgeCommunication\.exe'
    } | ForEach-Object {
        $tasks += [pscustomobject]@{ 
            TaskName = $taskName
            TaskPath = $taskPath
            Execute = $_.Execute
        }
    }
}

if ($tasks.Count -gt 0) {
    Write-Host "Found $($tasks.Count) scheduled task(s) referencing BridgeCommunication.exe" -ForegroundColor Red
    
    foreach ($task in $tasks) {
        Write-Host "Task: $($task.TaskName)" -ForegroundColor White
        Write-Host "  Path: $($task.TaskPath)" -ForegroundColor Gray
        Write-Host "  Execute: $($task.Execute)" -ForegroundColor Gray
        
        if (-not $DryRun) {
            Write-Host "  Disabling task..." -ForegroundColor Yellow
            try {
                Disable-ScheduledTask -TaskName $task.TaskName -ErrorAction Stop | Out-Null
                Write-Host "  Task disabled successfully" -ForegroundColor Green
            } catch {
                Write-Host "  WARNING: Could not disable task: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "  [DRY RUN] Would disable this task" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "No scheduled tasks found referencing BridgeCommunication.exe" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "QUARANTINE SUMMARY" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "DRY RUN COMPLETE - No changes were made" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To apply changes, run without -DryRun parameter:" -ForegroundColor White
    Write-Host "  .\Quarantine-BridgeCommunication.ps1" -ForegroundColor Gray
} else {
    Write-Host "Quarantine actions completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Actions taken:" -ForegroundColor White
    Write-Host "  - Processes terminated" -ForegroundColor Gray
    Write-Host "  - Services disabled" -ForegroundColor Gray
    Write-Host "  - Files blocked from execution" -ForegroundColor Gray
    if (-not $SkipFirewall) {
        Write-Host "  - Firewall rules created" -ForegroundColor Gray
    }
    Write-Host "  - Scheduled tasks disabled" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run Inspect-BridgeCommunication.ps1 to confirm no respawns" -ForegroundColor White
    Write-Host "2. Reboot system to verify persistence is broken" -ForegroundColor White
    Write-Host "3. Consider uninstalling the parent application if not needed" -ForegroundColor White
}

Write-Host ""
Write-Host "To reverse quarantine actions:" -ForegroundColor Magenta
Write-Host "  - Re-enable services: sc.exe config [ServiceName] start= auto" -ForegroundColor Gray
Write-Host "  - Remove ACL blocks: icacls [FilePath] /remove:d *S-1-1-0" -ForegroundColor Gray
Write-Host "  - Rename .quarantine files back to .exe" -ForegroundColor Gray
Write-Host "  - Delete firewall rules via Windows Defender Firewall console" -ForegroundColor Gray
Write-Host "  - Re-enable scheduled tasks via Task Scheduler" -ForegroundColor Gray