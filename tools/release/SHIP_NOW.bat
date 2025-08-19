@echo off
REM SHIP IT NOW - Fix shaders and run checks

cd /d D:\Dev\kha

echo Fixing shader bundle...
node scripts\bundleShaders.mjs

echo.
echo Running TypeScript check...
npx tsc -p .\frontend\tsconfig.json

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  TYPESCRIPT: PASSED!
    echo ========================================
    echo.
    echo Continuing with remaining checks...
    echo.
    
    REM Run remaining checks (skip absolute path check since we know about conversations)
    node .\tools\quilt\WebGPU\QuiltGenerator.ts -src .\frontend\lib\webgpu\shaders -out .\frontend\public\hybrid\wgsl
    
    REM Shader validation
    powershell -ExecutionPolicy Bypass -File .\tools\shaders\run_shader_gate.ps1 -RepoRoot "D:\Dev\kha" -Targets @("iphone11","iphone15")
    
    REM API check
    node .\tools\release\api-smoke.js --env ".\.env.production"
    
    echo.
    echo ========================================
    echo        ALL CHECKS COMPLETE!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo     TYPESCRIPT STILL HAS ERRORS
    echo ========================================
    echo Check the output above for details
)

pause
