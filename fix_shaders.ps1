$shaderDir = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$naga = "naga"

Get-ChildItem "$shaderDir\*.wgsl" | ForEach-Object {
    $file = $_.FullName
    $name = $_.Name
    Write-Host ""
    Write-Host "Validating shader file:" -ForegroundColor Cyan
    Write-Host $name

    $result = & $naga $file 2>&1
    $modified = $false

    # Check for 'filter' reserved keyword
    if ($result -match 'name `filter` is a reserved keyword') {
        Write-Host "Reserved keyword 'filter' found in $name. Replacing with 'filter_val'." -ForegroundColor Yellow
        (Get-Content $file) -replace '\bfilter\b', 'filter_val' | Set-Content $file
        $modified = $true
    }

    # Check for '@group' in function parameters
    if ($result -match 'unknown attribute: `group`') {
        Write-Host "WARNING: '@group' found in function parameters in $name. Manual fix needed." -ForegroundColor Red
    }

    # Check for storage buffer as function parameter
    if ($result -match 'expected one of') {
        Write-Host "WARNING: Possible storage buffer in function parameters in $name." -ForegroundColor Red
    }

    if ($modified) {
        Write-Host "Re-validating patched file:" -ForegroundColor Green
        $result2 = & $naga $file 2>&1
        if ($result2 -match "error") {
            Write-Host "Still has errors:" -ForegroundColor Red
            Write-Host $result2
        } else {
            Write-Host "$name is now valid after patch." -ForegroundColor Green
        }
    } else {
        if ($result -match "error") {
            Write-Host "$name has errors:" -ForegroundColor Red
            Write-Host $result
        } else {
            Write-Host "$name is valid." -ForegroundColor Green
        }
    }
}

Write-Host "`nShader validation complete!" -ForegroundColor Cyan
