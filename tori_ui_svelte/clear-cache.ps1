# Clear SvelteKit cache - PowerShell version
Write-Host "🧹 Clearing SvelteKit cache..."
Remove-Item .svelte-kit -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cache cleared successfully"
