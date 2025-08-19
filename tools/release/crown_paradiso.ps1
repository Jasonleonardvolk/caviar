# tools/release/crown_paradiso.ps1
Param(
  [string]$Tag = "shaders-pass-2025-08-08",
  [string]$Message = "First green run"
)
git add -A
git commit -m "Shaders: green gate; canonicalized; storage-read; specializations" | Out-Null
git tag -a $Tag -m $Message
git push
git push --tags
Write-Host "Crowned Paradiso: $Tag"
