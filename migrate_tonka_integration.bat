@echo off
REM TONKA Integration Migration Script
REM Generated: 2025-06-28 17:27
REM =====================================

echo ========================================
echo TONKA Integration Migration: Pigpen to TORI
echo ========================================
echo.

REM Create backup directory
mkdir "C:\Users\jason\Desktop\tori\kha\backup_before_tonka_20250628_172730" 2>nul


REM Backup original prajna/api/prajna_api.py
copy "C:\Users\jason\Desktop\tori\kha\prajna\api\prajna_api.py" "C:\Users\jason\Desktop\tori\kha\backup_before_tonka_20250628_172730\prajna_api_prajna_api.py" >nul
echo Backed up: prajna/api/prajna_api.py

REM Copy TONKA files
echo.
echo Copying TONKA integration files...

mkdir "C:\Users\jason\Desktop\tori\kha\api" 2>nul
copy "C:\Users\jason\Desktop\pigpen\api\tonka_api.py" "C:\Users\jason\Desktop\tori\kha\api\tonka_api.py"
if %errorlevel% equ 0 (echo [OK] Copied: api/tonka_api.py) else (echo [FAIL] Failed: api/tonka_api.py)
copy "C:\Users\jason\Desktop\pigpen\test_tonka_integration.py" "C:\Users\jason\Desktop\tori\kha\test_tonka_integration.py"
if %errorlevel% equ 0 (echo [OK] Copied: test_tonka_integration.py) else (echo [FAIL] Failed: test_tonka_integration.py)
copy "C:\Users\jason\Desktop\pigpen\bulk_pdf_processor.py" "C:\Users\jason\Desktop\tori\kha\bulk_pdf_processor.py"
if %errorlevel% equ 0 (echo [OK] Copied: bulk_pdf_processor.py) else (echo [FAIL] Failed: bulk_pdf_processor.py)
copy "C:\Users\jason\Desktop\pigpen\teach_tonka_from_datasets.py" "C:\Users\jason\Desktop\tori\kha\teach_tonka_from_datasets.py"
if %errorlevel% equ 0 (echo [OK] Copied: teach_tonka_from_datasets.py) else (echo [FAIL] Failed: teach_tonka_from_datasets.py)
copy "C:\Users\jason\Desktop\pigpen\tonka_education.py" "C:\Users\jason\Desktop\tori\kha\tonka_education.py"
if %errorlevel% equ 0 (echo [OK] Copied: tonka_education.py) else (echo [FAIL] Failed: tonka_education.py)
copy "C:\Users\jason\Desktop\pigpen\tonka_learning_curriculum.py" "C:\Users\jason\Desktop\tori\kha\tonka_learning_curriculum.py"
if %errorlevel% equ 0 (echo [OK] Copied: tonka_learning_curriculum.py) else (echo [FAIL] Failed: tonka_learning_curriculum.py)
copy "C:\Users\jason\Desktop\pigpen\tonka_pdf_learner.py" "C:\Users\jason\Desktop\tori\kha\tonka_pdf_learner.py"
if %errorlevel% equ 0 (echo [OK] Copied: tonka_pdf_learner.py) else (echo [FAIL] Failed: tonka_pdf_learner.py)
copy "C:\Users\jason\Desktop\pigpen\tonka_config.json" "C:\Users\jason\Desktop\tori\kha\tonka_config.json"
if %errorlevel% equ 0 (echo [OK] Copied: tonka_config.json) else (echo [FAIL] Failed: tonka_config.json)
copy "C:\Users\jason\Desktop\pigpen\process_massive_datasets.py" "C:\Users\jason\Desktop\tori\kha\process_massive_datasets.py"
if %errorlevel% equ 0 (echo [OK] Copied: process_massive_datasets.py) else (echo [FAIL] Failed: process_massive_datasets.py)
copy "C:\Users\jason\Desktop\pigpen\process_massive_datasets_fixed.py" "C:\Users\jason\Desktop\tori\kha\process_massive_datasets_fixed.py"
if %errorlevel% equ 0 (echo [OK] Copied: process_massive_datasets_fixed.py) else (echo [FAIL] Failed: process_massive_datasets_fixed.py)
copy "C:\Users\jason\Desktop\pigpen\download_massive_datasets.py" "C:\Users\jason\Desktop\tori\kha\download_massive_datasets.py"
if %errorlevel% equ 0 (echo [OK] Copied: download_massive_datasets.py) else (echo [FAIL] Failed: download_massive_datasets.py)
copy "C:\Users\jason\Desktop\pigpen\smart_dataset_downloader.py" "C:\Users\jason\Desktop\tori\kha\smart_dataset_downloader.py"
if %errorlevel% equ 0 (echo [OK] Copied: smart_dataset_downloader.py) else (echo [FAIL] Failed: smart_dataset_downloader.py)
copy "C:\Users\jason\Desktop\pigpen\rapid_code_learner.py" "C:\Users\jason\Desktop\tori\kha\rapid_code_learner.py"
if %errorlevel% equ 0 (echo [OK] Copied: rapid_code_learner.py) else (echo [FAIL] Failed: rapid_code_learner.py)
copy "C:\Users\jason\Desktop\pigpen\simple_pdf_processor.py" "C:\Users\jason\Desktop\tori\kha\simple_pdf_processor.py"
if %errorlevel% equ 0 (echo [OK] Copied: simple_pdf_processor.py) else (echo [FAIL] Failed: simple_pdf_processor.py)
copy "C:\Users\jason\Desktop\pigpen\use_existing_pdfs.py" "C:\Users\jason\Desktop\tori\kha\use_existing_pdfs.py"
if %errorlevel% equ 0 (echo [OK] Copied: use_existing_pdfs.py) else (echo [FAIL] Failed: use_existing_pdfs.py)

REM Copy modified prajna_api.py with TONKA integration
echo.
echo Updating prajna_api.py with TONKA integration...

copy "C:\Users\jason\Desktop\pigpen\prajna\api\prajna_api.py" "C:\Users\jason\Desktop\tori\kha\prajna\api\prajna_api.py"
if %errorlevel% equ 0 (echo [OK] Updated prajna_api.py) else (echo [FAIL] Failed to update prajna_api.py)

echo.
echo ========================================
echo Migration Complete!
echo ========================================
echo.
echo Backup saved to: C:\Users\jason\Desktop\tori\kha\backup_before_tonka_20250628_172730
echo.
echo Next steps:
echo 1. Create the missing coordinator files
echo 2. Test TONKA integration
echo 3. Copy datasets if needed
echo.
pause
