# PowerShell script to schedule TORI compaction
# More robust than batch file with better error handling

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptPath "compact_all_meshes.py"
$logDir = Join-Path (Split-Path -Parent $scriptPath) "logs"

# Ensure log directory exists
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Find Python executable
$pythonExe = Join-Path (Split-Path -Parent $scriptPath) "python.exe"
if (!(Test-Path $pythonExe)) {
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (!$pythonExe) {
        Write-Error "Python not found! Please install Python or set the path."
        exit 1
    }
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Green
Write-Host "Script path: $pythonScript" -ForegroundColor Green

# Function to create scheduled task
function Create-CompactionTask {
    param(
        [string]$TaskName,
        [string]$TriggerType,
        [string]$TriggerTime = $null,
        [string]$Description
    )
    
    $action = New-ScheduledTaskAction -Execute $pythonExe -Argument """$pythonScript""" -WorkingDirectory $scriptPath
    
    # Create trigger based on type
    switch ($TriggerType) {
        "Daily" {
            $trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime
        }
        "Startup" {
            $trigger = New-ScheduledTaskTrigger -AtStartup
            # Add 5 minute delay
            $trigger.Delay = "PT5M"
        }
        "Logon" {
            $trigger = New-ScheduledTaskTrigger -AtLogon
            $trigger.Delay = "PT2M"
        }
    }
    
    # Task settings
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable:$false `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    
    # Principal (run with highest privileges)
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Highest
    
    try {
        # Remove existing task if it exists
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        
        # Register new task
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description $Description `
            -Force | Out-Null
            
        Write-Host "[OK] Created task: $TaskName" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to create task: $TaskName" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        return $false
    }
}

# Create tasks
Write-Host "`nCreating TORI Compaction scheduled tasks..." -ForegroundColor Cyan

$tasks = @(
    @{
        Name = "TORI Compaction Midnight"
        Type = "Daily"
        Time = "00:00"
        Description = "Compact TORI concept meshes at midnight"
    },
    @{
        Name = "TORI Compaction Noon"
        Type = "Daily"  
        Time = "12:00"
        Description = "Compact TORI concept meshes at noon"
    },
    @{
        Name = "TORI Compaction Startup"
        Type = "Startup"
        Description = "Compact TORI concept meshes 5 minutes after system startup"
    }
)

$successCount = 0
foreach ($task in $tasks) {
    if (Create-CompactionTask @task) {
        $successCount++
    }
}

Write-Host "`nCreated $successCount of $($tasks.Count) tasks successfully" -ForegroundColor $(if ($successCount -eq $tasks.Count) { "Green" } else { "Yellow" })

# Show task status
Write-Host "`nTask Status:" -ForegroundColor Cyan
Get-ScheduledTask -TaskName "TORI Compaction*" | Select-Object TaskName, State, LastRunTime, NextRunTime | Format-Table

Write-Host "`nUseful commands:" -ForegroundColor Yellow
Write-Host "  Run now:     Start-ScheduledTask -TaskName 'TORI Compaction Midnight'"
Write-Host "  View tasks:  Get-ScheduledTask -TaskName 'TORI Compaction*'"
Write-Host "  View logs:   Get-Content '$logDir\compaction.log' -Tail 50"
Write-Host "  Delete task: Unregister-ScheduledTask -TaskName 'TORI Compaction Midnight' -Confirm:`$false"

# Test run option
$response = Read-Host "`nWould you like to test run the compaction now? (y/n)"
if ($response -eq 'y') {
    Write-Host "Running compaction test..." -ForegroundColor Cyan
    & $pythonExe $pythonScript --force
}
