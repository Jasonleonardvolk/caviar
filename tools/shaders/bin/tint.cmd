@echo off
REM tint.exe - Wrapper that redirects to naga.exe
REM This allows existing scripts expecting tint to work with naga

set NAGA=%~dp0naga.exe

if not exist "%NAGA%" (
    echo Error: naga.exe not found in %~dp0
    exit /b 1
)

REM Translate common tint arguments to naga equivalents
REM Naga is simpler - it mainly just validates, doesn't have all of Tint's options

REM Just pass the WGSL file to naga for validation
"%NAGA%" %*
