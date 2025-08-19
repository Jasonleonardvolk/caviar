@echo off
echo.
echo Checking TypeScript error count...
cd frontend
powershell -Command "$count = (npx tsc --noEmit 2>&1 | Select-String 'error TS' -AllMatches).Matches.Count; Write-Host \"TypeScript Errors: $count\" -ForegroundColor $(if ($count -eq 0) {'Green'} elseif ($count -le 10) {'Yellow'} else {'Red'}); Write-Host \"Original: 641 errors\" -ForegroundColor Gray; Write-Host \"Reduction: $(Math.Round((641-$count)/641*100, 1))%%\" -ForegroundColor Green"
cd ..
pause
