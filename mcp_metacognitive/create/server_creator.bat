@echo off
REM MCP Server Creator - Windows Helper
REM Easy commands for creating and managing MCP servers

echo.
echo ===== MCP Server Creator =====
echo.
echo Commands:
echo   1. Create new server (no PDFs)
echo   2. Create new server (with PDFs)
echo   3. Add PDFs to existing server
echo   4. List PDFs for a server
echo   5. Batch add PDFs from directory
echo   6. Remove PDF from server
echo   7. Refresh server seed.txt
echo   8. Show statistics
echo   9. Exit
echo.

:menu
set /p choice="Select option (1-9): "

if "%choice%"=="1" goto create_simple
if "%choice%"=="2" goto create_with_pdfs
if "%choice%"=="3" goto add_pdfs
if "%choice%"=="4" goto list_pdfs
if "%choice%"=="5" goto batch_add
if "%choice%"=="6" goto remove_pdf
if "%choice%"=="7" goto refresh
if "%choice%"=="8" goto stats
if "%choice%"=="9" goto end

echo Invalid choice. Please try again.
goto menu

:create_simple
set /p name="Enter server name: "
set /p desc="Enter description: "
python mk_server.py create "%name%" "%desc%"
pause
goto menu

:create_with_pdfs
set /p name="Enter server name: "
set /p desc="Enter description: "
set /p pdfs="Enter PDF paths (space-separated): "
python mk_server.py create "%name%" "%desc%" %pdfs%
pause
goto menu

:add_pdfs
set /p name="Enter server name: "
set /p pdfs="Enter PDF paths to add (space-separated): "
python mk_server.py add-pdf "%name%" %pdfs%
pause
goto menu

:list_pdfs
set /p name="Enter server name: "
python mk_server.py list-pdfs "%name%"
pause
goto menu

:batch_add
set /p name="Enter server name: "
set /p dir="Enter directory path containing PDFs: "
python pdf_manager.py batch-add "%name%" "%dir%"
pause
goto menu

:remove_pdf
set /p name="Enter server name: "
set /p pdf="Enter PDF filename to remove: "
python pdf_manager.py remove-pdf "%name%" "%pdf%"
pause
goto menu

:refresh
set /p name="Enter server name: "
python pdf_manager.py refresh "%name%"
pause
goto menu

:stats
echo.
echo 1. Show all servers
echo 2. Show specific server
set /p statschoice="Select (1-2): "

if "%statschoice%"=="1" (
    python pdf_manager.py stats
) else (
    set /p name="Enter server name: "
    python pdf_manager.py stats "%name%"
)
pause
goto menu

:end
echo.
echo Thank you for using MCP Server Creator!
pause
