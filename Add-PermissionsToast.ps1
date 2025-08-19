# Add-PermissionsToast.ps1
# Adds the PermissionsToast component to HolographicDisplay.svelte

$filePath = "D:\Dev\kha\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte"
$content = Get-Content $filePath -Raw

# Check if already added
if ($content -match "PermissionsToast") {
    Write-Host "PermissionsToast already integrated!" -ForegroundColor Green
    exit 0
}

# Add import after other imports
$importLine = "  import PermissionsToast from './PermissionsToast.svelte';"
$content = $content -replace "(import .* from .*show.*)", "`$1`n$importLine"

# Add component in template (at the very top of the component)
$componentTag = "`n<PermissionsToast />`n"
$content = $content -replace "(<canvas)", "$componentTag`$1"

# Write back
$content | Set-Content $filePath -Encoding UTF8

Write-Host "✅ PermissionsToast integrated into HolographicDisplay!" -ForegroundColor Green
Write-Host "The toast will show automatically when permissions are needed." -ForegroundColor Cyan