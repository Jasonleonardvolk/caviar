@echo off
echo Getting all TypeScript errors...
npx tsc --noEmit > all_typescript_errors.txt 2>&1
echo Errors saved to all_typescript_errors.txt
echo.
echo First 50 lines of errors:
echo ========================================
type all_typescript_errors.txt | more
pause