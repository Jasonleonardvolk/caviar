@echo off
echo Finding and stopping process on port 5173...
echo.

REM Find the process using port 5173
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173') do (
    set PID=%%a
    goto :found
)

echo No process found using port 5173
goto :end

:found
echo Found process with PID: %PID%
echo.

REM Get process name
for /f "tokens=1" %%b in ('tasklist /FI "PID eq %PID%" /FO TABLE /NH') do (
    echo Process name: %%b
)

echo.
echo Stopping process...
taskkill /PID %PID% /F

if %ERRORLEVEL% == 0 (
    echo.
    echo Successfully stopped process on port 5173!
) else (
    echo.
    echo Failed to stop process. You may need to run this as Administrator.
)

:end
echo.
pause
