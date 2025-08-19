# Share-File.ps1
# Auto-shares any file with your team
param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath,
    
    [string]$Message = "shared file"
)

$fileName = Split-Path $FilePath -Leaf
$extension = [System.IO.Path]::GetExtension($fileName)

# Auto-sort into correct folder
$destFolder = switch -Wildcard ($extension) {
    ".pdf"     { "shared\pdfs" }
    ".wgsl"    { "shared\shaders" }
    ".md"      { "shared\conversations" }
    ".txt"     { "shared\articles" }
    default    { "shared" }
}

# Copy file
$dest = Join-Path "D:\Dev\kha" $destFolder
Copy-Item $FilePath $dest -Force

# Auto commit and push
Set-Location D:\Dev\kha
git add -A
git commit -m "$Message - $fileName"
git push

Write-Host "✓ Shared $fileName - Your friend can see it now!" -ForegroundColor Green
Write-Host "Location: $destFolder\$fileName" -ForegroundColor Cyan
