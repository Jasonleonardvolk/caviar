# Patch IrisOneButton.ps1 to add NonInteractive support
param(
  [string]$ScriptPath = "D:\Dev\kha\tools\release\IrisOneButton.ps1",
  [switch]$Restore
)

if ($Restore) {
  # Restore from backup
  $backupPath = "$ScriptPath.backup"
  if (Test-Path $backupPath) {
    Copy-Item $backupPath $ScriptPath -Force
    Write-Host "Restored IrisOneButton.ps1 from backup" -ForegroundColor Green
  } else {
    Write-Host "No backup found at $backupPath" -ForegroundColor Red
  }
  exit
}

# Create backup
$backupPath = "$ScriptPath.backup"
if (-not (Test-Path $backupPath)) {
  Copy-Item $ScriptPath $backupPath
  Write-Host "Created backup at $backupPath" -ForegroundColor Green
}

# Read the script
$content = Get-Content $ScriptPath -Raw

# Check if already patched
if ($content -match '\$NonInteractive') {
  Write-Host "Script already has NonInteractive support" -ForegroundColor Yellow
  exit 0
}

# Add NonInteractive parameter
$paramPattern = 'param\s*\(([\s\S]*?)\)'
if ($content -match $paramPattern) {
  $params = $Matches[1]
  # Add the new parameter
  $newParams = $params.TrimEnd() + ",`r`n  [switch]`$NonInteractive = `$false"
  $content = $content -replace $paramPattern, "param(`r`n$newParams`r`n)"
  Write-Host "Added NonInteractive parameter" -ForegroundColor Green
} else {
  Write-Host "Could not find param block" -ForegroundColor Red
  exit 1
}

# Wrap the prompt
$promptPattern = 'if\s*\(\s*\$Host\.UI\.PromptForChoice.*?\)\s*-eq\s*0\)\s*\{[^}]*explorer\.exe[^}]*\}'
if ($content -match $promptPattern) {
  $originalPrompt = $Matches[0]
  $wrappedPrompt = @"
if (-not `$NonInteractive) {
    $originalPrompt
} else {
    Write-Host "Non-interactive mode: Skipping folder prompt" -ForegroundColor Gray
}
"@
  $content = $content -replace [regex]::Escape($originalPrompt), $wrappedPrompt
  Write-Host "Wrapped prompt with NonInteractive check" -ForegroundColor Green
} else {
  Write-Host "Could not find prompt pattern - manual fix may be needed" -ForegroundColor Yellow
}

# Save the patched script
$content | Out-File $ScriptPath -Encoding UTF8
Write-Host "Successfully patched IrisOneButton.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "To restore original:" -ForegroundColor Yellow
Write-Host "  .\Patch-IrisOneButton.ps1 -Restore" -ForegroundColor White
