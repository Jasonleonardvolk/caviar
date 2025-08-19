@echo off
REM Quick test that Naga is working

echo === NAGA WGSL VALIDATOR TEST ===
echo.

cd /d "%~dp0"

if exist naga.exe (
    echo [OK] naga.exe found
    echo.
    
    naga.exe --version
    echo.
    
    echo Creating test shader...
    (
        echo @vertex
        echo fn vs_main^(@builtin^(vertex_index^) idx: u32^) -^> @builtin^(position^) vec4^<f32^> {
        echo     return vec4^<f32^>^(0.0, 0.0, 0.0, 1.0^);
        echo }
        echo.
        echo @fragment  
        echo fn fs_main^(^) -^> @location^(0^) vec4^<f32^> {
        echo     return vec4^<f32^>^(1.0, 0.0, 0.0, 1.0^);
        echo }
    ) > test.wgsl
    
    echo Testing validation...
    naga.exe test.wgsl
    
    if %errorlevel% equ 0 (
        echo.
        echo [SUCCESS] Naga is working!
        del test.wgsl 2>nul
    ) else (
        echo.
        echo [ERROR] Validation failed
    )
) else (
    echo [ERROR] naga.exe not found!
)

echo.
echo === NEXT STEPS ===
echo 1. Go back to repo root: cd ..\..\..\
echo 2. Run Virgil: npm run virgil
echo    Or: node tools/shaders/virgil_summon.mjs --strict
echo.
pause
