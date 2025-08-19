@echo off
REM Quick Analysis: TORI vs Pigpen
REM ==============================

echo ========================================
echo Quick Analysis: TORI vs Pigpen
echo ========================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha

echo Running comparison analysis...
echo.

REM Create a temporary Python script
echo import os > temp_analysis.py
echo import json >> temp_analysis.py
echo from pathlib import Path >> temp_analysis.py
echo from datetime import datetime >> temp_analysis.py
echo. >> temp_analysis.py
echo tori = Path(r'C:\Users\jason\Desktop\tori\kha') >> temp_analysis.py
echo pigpen = Path(r'C:\Users\jason\Desktop\pigpen') >> temp_analysis.py
echo. >> temp_analysis.py
echo print('🔍 QUICK ANALYSIS RESULTS:') >> temp_analysis.py
echo print('='*50) >> temp_analysis.py
echo. >> temp_analysis.py
echo # Check for key differences >> temp_analysis.py
echo key_files = [ >> temp_analysis.py
echo     'enhanced_launcher.py', >> temp_analysis.py
echo     'ingest_pdf/pipeline.py', >> temp_analysis.py
echo     'ingest_pdf/cognitive_interface.py', >> temp_analysis.py
echo     'concept_mesh_data.json', >> temp_analysis.py
echo     'concepts.json', >> temp_analysis.py
echo     'concepts.npz' >> temp_analysis.py
echo ] >> temp_analysis.py
echo. >> temp_analysis.py
echo print('\n📊 Key File Status:') >> temp_analysis.py
echo for file in key_files: >> temp_analysis.py
echo     t = tori / file >> temp_analysis.py
echo     p = pigpen / file >> temp_analysis.py
echo     if t.exists() and p.exists(): >> temp_analysis.py
echo         t_size = t.stat().st_size >> temp_analysis.py
echo         p_size = p.stat().st_size >> temp_analysis.py
echo         if t_size != p_size: >> temp_analysis.py
echo             print(f'  ⚡ {file}: Different (TORI: {t_size} bytes, Pigpen: {p_size} bytes)') >> temp_analysis.py
echo         else: >> temp_analysis.py
echo             print(f'  ✓ {file}: Same size') >> temp_analysis.py
echo     elif p.exists() and not t.exists(): >> temp_analysis.py
echo         print(f'  🆕 {file}: Only in Pigpen') >> temp_analysis.py
echo     elif t.exists() and not p.exists(): >> temp_analysis.py
echo         print(f'  📦 {file}: Only in TORI') >> temp_analysis.py
echo. >> temp_analysis.py
echo # Check for datasets >> temp_analysis.py
echo print('\n💾 Dataset Files in Pigpen:') >> temp_analysis.py
echo datasets_found = False >> temp_analysis.py
echo for ext in ['*.npz', '*.pkl', '*.h5', '*.csv']: >> temp_analysis.py
echo     for f in pigpen.rglob(ext): >> temp_analysis.py
echo         if f.stat().st_size ^> 1024*1024:  # ^> 1MB >> temp_analysis.py
echo             size_mb = f.stat().st_size / (1024*1024) >> temp_analysis.py
echo             print(f'  - {f.name} ({size_mb:.1f} MB)') >> temp_analysis.py
echo             datasets_found = True >> temp_analysis.py
echo. >> temp_analysis.py
echo if not datasets_found: >> temp_analysis.py
echo     print('  No large dataset files found') >> temp_analysis.py
echo. >> temp_analysis.py
echo # Recent modifications >> temp_analysis.py
echo print('\n🕐 Recent Modifications (last 24h):') >> temp_analysis.py
echo now = datetime.now().timestamp() >> temp_analysis.py
echo cutoff = now - (24 * 3600) >> temp_analysis.py
echo. >> temp_analysis.py
echo recent_pigpen = [] >> temp_analysis.py
echo recent_tori = [] >> temp_analysis.py
echo. >> temp_analysis.py
echo for f in pigpen.rglob('*.py'): >> temp_analysis.py
echo     if f.stat().st_mtime ^> cutoff: >> temp_analysis.py
echo         recent_pigpen.append(f.relative_to(pigpen)) >> temp_analysis.py
echo. >> temp_analysis.py
echo for f in tori.rglob('*.py'): >> temp_analysis.py
echo     if f.stat().st_mtime ^> cutoff: >> temp_analysis.py
echo         recent_tori.append(f.relative_to(tori)) >> temp_analysis.py
echo. >> temp_analysis.py
echo print(f'\nPigpen: {len(recent_pigpen)} files modified') >> temp_analysis.py
echo for f in recent_pigpen[:5]: >> temp_analysis.py
echo     print(f'  - {f}') >> temp_analysis.py
echo. >> temp_analysis.py
echo print(f'\nTORI: {len(recent_tori)} files modified') >> temp_analysis.py
echo for f in recent_tori[:5]: >> temp_analysis.py
echo     print(f'  - {f}') >> temp_analysis.py
echo. >> temp_analysis.py
echo print('\n'+'='*50) >> temp_analysis.py
echo print('💡 Recommendations:') >> temp_analysis.py
echo print('1. Run: python compare_tori_pigpen.py for detailed analysis') >> temp_analysis.py
echo print('2. Run: python smart_merge_to_pigpen.py to merge TORI updates') >> temp_analysis.py
echo print('3. Check MERGE_CONFLICTS_REPORT.md after merge') >> temp_analysis.py

REM Run the temporary Python script
python temp_analysis.py

REM Clean up
del temp_analysis.py

echo.
echo ========================================
echo For detailed comparison, run:
echo   python compare_tori_pigpen.py
echo.
echo To merge TORI updates to pigpen:
echo   python smart_merge_to_pigpen.py
echo ========================================
pause
