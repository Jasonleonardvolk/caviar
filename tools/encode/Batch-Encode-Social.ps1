# Batch-Encode-Social.ps1
# Batch process all videos for social media export
param(
  [string]$InputDir = "D:\Dev\kha\content\socialpack\input",
  [int]$Framerate = 60,
  [int]$VideoBitrateMbps = 10,
  [int]$MaxFileSizeMB = 250
)

$ErrorActionPreference='Stop'
$build = "D:\Dev\kha\tools\encode\Build-SocialPack.ps1"
if(-not (Test-Path $build)){ throw "Missing $build" }

$files = Get-ChildItem $InputDir -File -Include *.mp4,*.mov
if(-not $files){ Write-Host "No inputs in $InputDir" -ForegroundColor Yellow; exit }

Write-Host "==> Social batch ($($files.Count) files)" -ForegroundColor Cyan
$rows=@()
foreach($f in $files){
  Write-Host ("-- {0}" -f $f.Name) -ForegroundColor Gray
  & $build -Input $f.FullName -Framerate $Framerate -VideoBitrateMbps $VideoBitrateMbps -MaxFileSizeMB $MaxFileSizeMB
  $rows += [PSCustomObject]@{ File=$f.Name; Date=(Get-Date).ToString("s") }
}

# Write a tiny checklist
$check = "D:\Dev\kha\content\socialpack\out\checklist.md"
"# SocialPack Checklist — $(Get-Date -f s)`n" | Out-File $check -Encoding UTF8
$rows | ForEach-Object { "- $($_.File) ✅ $($_.Date)" } | Add-Content $check
Write-Host "Checklist → $check" -ForegroundColor Green