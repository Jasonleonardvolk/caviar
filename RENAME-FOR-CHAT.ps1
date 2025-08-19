# RENAME-FOR-CHAT.ps1
# Renames AI_QUICK_REFERENCE.md to something unique Chat can find

$oldName = "AI_QUICK_REFERENCE.md"
$newName = "CAVIAR_AI_REFERENCE_JASONVOLK.md"

if (Test-Path $oldName) {
    # Rename the file
    Move-Item $oldName $newName
    Write-Host "Renamed: $oldName -> $newName" -ForegroundColor Green
    
    # Stage for git
    git add $oldName
    git add $newName
    
    # Quick commit and push
    git commit -m "renamed AI reference to unique name for Chat search"
    git push
    
    Write-Host "`nPushed! Tell Chat to search for:" -ForegroundColor Cyan
    Write-Host "  CAVIAR_AI_REFERENCE_JASONVOLK.md" -ForegroundColor Yellow
    Write-Host "  in Jasonleonardvolk/caviar repo" -ForegroundColor Yellow
} else {
    Write-Host "File not found: $oldName" -ForegroundColor Red
}
