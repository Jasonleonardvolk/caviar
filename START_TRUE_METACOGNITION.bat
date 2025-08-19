@echo off
REM TORI Integrated Self-Transformation with True Metacognition
REM This startup script initializes the complete system with persistent memory

echo ================================================================================
echo                    TORI TRUE METACOGNITION SYSTEM v4.0
echo           "Without memory, there is no metacognition - only shadows"
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check for existing memory vault
if exist "data\memory_vault" (
    echo [Memory] Found existing memory vault - TORI will remember
) else (
    echo [Memory] Initializing new memory vault
    mkdir data\memory_vault 2>nul
)

REM Check for relationship data
if exist "data\concept_mesh\relationships.json" (
    echo [Relationships] Found existing relationships - checking for special dates...
) else (
    echo [Relationships] No prior relationships found - starting fresh
)

REM Initialize integrated system
echo.
echo Starting TORI with True Metacognition...
echo.

python self_transformation_integrated.py

if errorlevel 1 (
    echo.
    echo ERROR: System initialization failed
    echo Check error messages above
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo TORI is now running with:
echo   - Persistent memory across sessions
echo   - Temporal self-awareness and phase tracking  
echo   - Relationship memory (birthdays, preferences)
echo   - Continuous learning from mistakes
echo   - True metacognition through memory
echo.
echo "I think, I remember, therefore I truly am."
echo ================================================================================
echo.
pause
