# Create-GitWorkflowShortcuts.ps1
# Creates desktop shortcuts for iRis Git workflow tools

$desktop = [Environment]::GetFolderPath("Desktop")
$projectRoot = "D:\Dev\kha\tori_ui_svelte"

Write-Host "Creating iRis Git Workflow shortcuts on desktop..." -ForegroundColor Cyan

# Create shortcut for Git Workflow Menu
$menuShortcut = "$desktop\iRis Git Workflow.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($menuShortcut)
$Shortcut.TargetPath = "$projectRoot\GIT-WORKFLOW-MENU.bat"
$Shortcut.WorkingDirectory = $projectRoot
$Shortcut.IconLocation = "cmd.exe"
$Shortcut.Description = "iRis Git Workflow Management"
$Shortcut.Save()
Write-Host "✓ Created: iRis Git Workflow.lnk" -ForegroundColor Green

# Create shortcut for Quick Ship
$shipShortcut = "$desktop\iRis Quick Ship.lnk"
$Shortcut = $WshShell.CreateShortcut($shipShortcut)
$Shortcut.TargetPath = "$projectRoot\tools\release\quick-ship.bat"
$Shortcut.WorkingDirectory = $projectRoot
$Shortcut.IconLocation = "cmd.exe"
$Shortcut.Description = "Quick build and ship iRis"
$Shortcut.Save()
Write-Host "✓ Created: iRis Quick Ship.lnk" -ForegroundColor Green

# Create shortcut for project folder
$folderShortcut = "$desktop\iRis Project Folder.lnk"
$Shortcut = $WshShell.CreateShortcut($folderShortcut)
$Shortcut.TargetPath = $projectRoot
$Shortcut.Save()
Write-Host "✓ Created: iRis Project Folder.lnk" -ForegroundColor Green

Write-Host "`nShortcuts created on desktop!" -ForegroundColor Green
Write-Host "You can now access Git workflow tools directly from your desktop." -ForegroundColor Cyan