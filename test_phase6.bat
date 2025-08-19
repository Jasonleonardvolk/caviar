@echo off
echo Testing Phase 6 - ScholarSphere diff route
echo ==========================================
echo.

echo Testing with PowerShell Invoke-RestMethod...
powershell -Command "try { $result = Invoke-RestMethod -Uri 'http://localhost:5173/api/concept-mesh/record_diff' -Method POST -Headers @{'Content-Type'='application/json'} -Body '{\"record_id\":\"smoke\"}' -ErrorAction Stop; Write-Host 'Success! Response:' -ForegroundColor Green; $result | ConvertTo-Json } catch { Write-Host 'Error:' $_.Exception.Message -ForegroundColor Red; Write-Host 'Status Code:' $_.Exception.Response.StatusCode.value__ -ForegroundColor Yellow }"

echo.
pause
