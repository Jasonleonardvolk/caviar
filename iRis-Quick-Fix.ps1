# iRis-Quick-Fix.ps1
# Fixes any missing show mode files instantly

Write-Host "`n=== iRIS QUICK FIX ===" -ForegroundColor Cyan
Write-Host "Creating any missing show mode files..." -ForegroundColor Yellow

$modesDir = "D:\Dev\kha\tori_ui_svelte\src\lib\show\modes"

# Ensure modes directory exists
if (!(Test-Path $modesDir)) {
    New-Item -ItemType Directory -Force -Path $modesDir | Out-Null
    Write-Host "Created modes directory" -ForegroundColor Green
}

# Quick mode templates if files are missing
$modeTemplates = @{
    "particles.ts" = @'
export const particles = {
    name: 'Particles',
    init: () => console.log('Particles mode initialized'),
    update: (time: number) => {},
    render: (ctx: any) => {},
    cleanup: () => {}
};
'@
    "portal.ts" = @'
export const portal = {
    name: 'Portal',
    init: () => console.log('Portal mode initialized'),
    update: (time: number) => {},
    render: (ctx: any) => {},
    cleanup: () => {}
};
'@
    "anamorph.ts" = @'
export const anamorph = {
    name: 'Anamorph',
    init: () => console.log('Anamorph mode initialized'),
    update: (time: number) => {},
    render: (ctx: any) => {},
    cleanup: () => {}
};
'@
    "glyphs.ts" = @'
export const glyphs = {
    name: 'Glyphs',
    init: () => console.log('Glyphs mode initialized'),
    update: (time: number) => {},
    render: (ctx: any) => {},
    cleanup: () => {}
};
'@
    "penrose.ts" = @'
export const penrose = {
    name: 'Penrose',
    init: () => console.log('Penrose mode initialized'),
    update: (time: number) => {},
    render: (ctx: any) => {},
    cleanup: () => {}
};
'@
}

$created = 0
foreach ($file in $modeTemplates.Keys) {
    $path = Join-Path $modesDir $file
    if (!(Test-Path $path)) {
        $modeTemplates[$file] | Out-File -Encoding UTF8 $path
        Write-Host "  Created $file" -ForegroundColor Green
        $created++
    }
}

if ($created -gt 0) {
    Write-Host "`n✓ Created $created missing mode files" -ForegroundColor Green
} else {
    Write-Host "`n✓ All mode files already exist" -ForegroundColor Green
}

Write-Host "`nRun .\iRis-Status-Check.ps1 to verify everything" -ForegroundColor Cyan