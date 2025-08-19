@echo off
echo.
echo ========================================
echo TORI Dependency Emergency Fix
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha

echo Step 1: Removing broken scipy directories...
rmdir /s /q ".venv\Lib\site-packages\~cipy" 2>nul
rmdir /s /q ".venv\Lib\site-packages\~cipy-1.16.1.dist-info" 2>nul
rmdir /s /q ".venv\Lib\site-packages\~cipy.libs" 2>nul
rmdir /s /q ".venv\Lib\site-packages\~pds" 2>nul
echo Done.

echo.
echo Step 2: Uninstalling problematic packages...
pip uninstall -y numpy scipy scikit-learn transformers sentence-transformers spacy thinc blis

echo.
echo Step 3: Installing compatible versions...
echo.

echo Installing numpy 1.26.4...
pip install numpy==1.26.4
if errorlevel 1 goto error

echo Installing scipy 1.13.1...
pip install scipy==1.13.1
if errorlevel 1 goto error

echo Installing scikit-learn 1.5.1...
pip install scikit-learn==1.5.1
if errorlevel 1 goto error

echo Installing transformers 4.44.2...
pip install transformers==4.44.2
if errorlevel 1 goto error

echo Installing sentence-transformers 3.0.1...
pip install sentence-transformers==3.0.1
if errorlevel 1 goto error

echo Installing spacy 3.7.5...
pip install spacy==3.7.5
if errorlevel 1 goto error

echo.
echo Step 4: Testing imports...
python test_entropy_state.py

echo.
echo ========================================
echo Fix complete! Check the results above.
echo ========================================
echo.
pause
goto end

:error
echo.
echo ERROR: Failed to install packages!
echo Try running as Administrator or use:
echo   python comprehensive_dependency_fix.py
echo.
pause

:end
