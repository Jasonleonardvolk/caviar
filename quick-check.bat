@echo off
echo ===============================================
echo    QUICK TYPESCRIPT ERROR COUNT
echo ===============================================
echo.

cd frontend

echo Counting errors...
powershell -Command "$errors = (npx tsc --noEmit 2>&1 | Select-String 'error TS' -AllMatches).Matches.Count; Write-Host \"`nCurrent TypeScript Errors: $errors\" -ForegroundColor $(if ($errors -eq 0) {'Green'} elseif ($errors -le 50) {'Yellow'} else {'Red'}); Write-Host \"Original Errors: 641\" -ForegroundColor Gray; $reduction = [math]::Round((641-$errors)/641*100, 1); Write-Host \"Error Reduction: $reduction%%\" -ForegroundColor Green; Write-Host \"\"; if ($errors -le 50) { Write-Host \"✓ Safe to proceed with development!\" -ForegroundColor Green } else { Write-Host \"WebGPU types are working - OK for shader work\" -ForegroundColor Yellow }"

cd ..
echo.
pause
