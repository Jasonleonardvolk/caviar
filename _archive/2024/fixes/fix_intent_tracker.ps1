# PowerShell script to fix the intentTracker error

# First, let's find the exact line causing the issue
Write-Host "Finding $intentTracker usage in +page.svelte..." -ForegroundColor Yellow

$filePath = "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte\src\routes\+page.svelte"

# Create backup
$backupPath = $filePath + ".backup_" + (Get-Date -Format "yyyyMMdd_HHmmss")
Copy-Item $filePath $backupPath
Write-Host "Backup created: $backupPath" -ForegroundColor Green

# Read the file
$content = Get-Content $filePath -Raw

# Find all occurrences of $intentTracker
$pattern = '\$intentTracker'
$matches = [regex]::Matches($content, $pattern)

Write-Host "`nFound $($matches.Count) occurrences of `$intentTracker" -ForegroundColor Cyan

# Show context around each match
foreach ($match in $matches) {
    $position = $match.Index
    $lineNumber = ($content.Substring(0, $position) -split "`n").Count
    $startContext = [Math]::Max(0, $position - 100)
    $endContext = [Math]::Min($content.Length, $position + 100)
    $context = $content.Substring($startContext, $endContext - $startContext)
    
    Write-Host "`nLine ~$lineNumber context:" -ForegroundColor Yellow
    Write-Host $context -ForegroundColor Gray
}

# Perform replacements
Write-Host "`nApplying fixes..." -ForegroundColor Yellow

# Replace $intentTracker with $currentIntent (when used alone)
$newContent = $content -replace '\$intentTracker(?!\.)', '$currentIntent'

# Replace $intentTracker.property with appropriate store references
$newContent = $newContent -replace '\$intentTracker\.currentIntent', '$currentIntent'
$newContent = $newContent -replace '\$intentTracker\.conversationInsights', '$conversationInsights'
$newContent = $newContent -replace '\$intentTracker\.intentHistory', '$intentHistory'

# Write the fixed content
[System.IO.File]::WriteAllText($filePath, $newContent, [System.Text.Encoding]::UTF8)

Write-Host "`nFixes applied successfully!" -ForegroundColor Green
Write-Host "If Vite is still running, it should hot-reload the changes." -ForegroundColor Cyan
Write-Host "If not, restart with: npm run dev" -ForegroundColor Cyan
