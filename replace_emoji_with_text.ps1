# Replace emoji with text labels in all Svelte files
$replacements = @{
    '🚀' = '[ROCKET]'
    '🌊' = '[WAVE]'
    '🧬' = '[DNA]'
    '💻' = '[COMPUTER]'
    '🎯' = '[TARGET]'
    '🌌' = '[GALAXY]'
    '⬇️' = '[DOWN]'
    '⬆️' = '[UP]'
    '🧠' = '[BRAIN]'
    '⚡' = '[BOLT]'
    '📚' = '[BOOKS]'
    '🔽' = '[v]'
    '▶️' = '[>]'
    '✅' = '[OK]'
    '❌' = '[X]'
    '📤' = '[UPLOAD]'
    '📕' = '[PDF]'
    '📄' = '[DOC]'
    '🔧' = '[TOOL]'
    '📖' = '[BOOK]'
    '🗑️' = '[TRASH]'
}

$sveltePath = "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte\src"
$files = Get-ChildItem -Path $sveltePath -Filter "*.svelte" -Recurse

foreach ($file in $files) {
    Write-Host "Processing: $($file.Name)" -ForegroundColor Cyan
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    $changed = $false
    
    foreach ($emoji in $replacements.Keys) {
        if ($content -match [regex]::Escape($emoji)) {
            $content = $content -replace [regex]::Escape($emoji), $replacements[$emoji]
            $changed = $true
            Write-Host "  Replaced $emoji with $($replacements[$emoji])" -ForegroundColor Green
        }
    }
    
    if ($changed) {
        # Create backup
        $backupPath = "$($file.FullName).backup_emoji"
        Copy-Item $file.FullName $backupPath
        
        # Save with text labels
        Set-Content -Path $file.FullName -Value $content -Encoding UTF8
        Write-Host "  Saved with text labels" -ForegroundColor Green
    }
}

Write-Host "`nDone! Restart your dev server to see the changes." -ForegroundColor Yellow
Write-Host "To restore emoji later, rename the .backup_emoji files back." -ForegroundColor Gray
