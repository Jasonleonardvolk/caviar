#!/usr/bin/env python3
"""
TONKA Integration Migrator
==========================
Identifies and migrates all TONKA-related changes from pigpen to TORI
"""

import os
import shutil
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime

class TONKAMigrator:
    def __init__(self):
        self.pigpen = Path(r"C:\Users\jason\Desktop\pigpen")
        self.tori = Path(r"{PROJECT_ROOT}")
        self.backup_dir = self.tori / f"backup_before_tonka_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # TONKA files that exist in pigpen
        self.tonka_files = [
            "api/tonka_api.py",
            "test_tonka_integration.py",
            "bulk_pdf_processor.py",
            "teach_tonka_from_datasets.py",
            "tonka_education.py",
            "tonka_learning_curriculum.py",
            "tonka_pdf_learner.py",
            "tonka_config.json",
            "process_massive_datasets.py",
            "process_massive_datasets_fixed.py",
            "download_massive_datasets.py",
            "smart_dataset_downloader.py",
            "rapid_code_learner.py",
            "simple_pdf_processor.py",
            "use_existing_pdfs.py"
        ]
        
        # Files that need to be created (coordinator)
        self.missing_files = [
            "api/prajna_tonka_coordinator.py",
            "test_prajna_tonka_coordination.py"
        ]
        
        # Modified files
        self.modified_files = [
            "prajna/api/prajna_api.py"  # Has TONKA integration
        ]
        
    def analyze(self):
        """Analyze what needs to be migrated"""
        print("🔍 TONKA Integration Analysis")
        print("="*60)
        
        # Check existing TONKA files
        print("\n📦 TONKA Files in Pigpen:")
        existing_tonka = []
        for file in self.tonka_files:
            pigpen_file = self.pigpen / file
            if pigpen_file.exists():
                size_kb = pigpen_file.stat().st_size / 1024
                print(f"  ✅ {file} ({size_kb:.1f} KB)")
                existing_tonka.append(file)
            else:
                print(f"  ❌ {file} (not found)")
        
        # Check missing coordinator files
        print("\n⚠️  Missing Coordinator Files:")
        for file in self.missing_files:
            print(f"  ❓ {file} (needs to be created)")
        
        # Check modified files
        print("\n🔧 Modified Files:")
        for file in self.modified_files:
            pigpen_file = self.pigpen / file
            tori_file = self.tori / file
            
            if pigpen_file.exists() and tori_file.exists():
                pigpen_size = pigpen_file.stat().st_size
                tori_size = tori_file.stat().st_size
                diff_kb = (pigpen_size - tori_size) / 1024
                
                pigpen_time = datetime.fromtimestamp(pigpen_file.stat().st_mtime)
                tori_time = datetime.fromtimestamp(tori_file.stat().st_mtime)
                
                print(f"  📝 {file}")
                print(f"     Pigpen: {pigpen_size:,} bytes (modified {pigpen_time.strftime('%Y-%m-%d %H:%M')})")
                print(f"     TORI:   {tori_size:,} bytes (modified {tori_time.strftime('%Y-%m-%d %H:%M')})")
                print(f"     Difference: +{diff_kb:.1f} KB in pigpen (TONKA integration)")
        
        # Check datasets directory
        datasets_dir = self.pigpen / "datasets"
        if datasets_dir.exists():
            dataset_files = list(datasets_dir.glob("*"))
            print(f"\n📊 Datasets Directory: {len(dataset_files)} files")
            total_size = sum(f.stat().st_size for f in dataset_files if f.is_file()) / (1024**2)
            print(f"   Total size: {total_size:.1f} MB")
        
        return existing_tonka
    
    def create_migration_script(self):
        """Create a batch script for migration"""
        script = f"""@echo off
REM TONKA Integration Migration Script
REM Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
REM =====================================

echo ========================================
echo TONKA Integration Migration: Pigpen to TORI
echo ========================================
echo.

REM Create backup directory
mkdir "{self.backup_dir}" 2>nul

"""
        # Add backup commands for modified files
        for file in self.modified_files:
            tori_file = self.tori / file
            if tori_file.exists():
                rel_path = file.replace('/', '\\')
                backup_filename = rel_path.replace('\\', '_')
            script += f"""
REM Backup original {file}
copy "{self.tori}\\{rel_path}" "{self.backup_dir}\\{backup_filename}" >nul
echo Backed up: {file}
"""

        # Add copy commands for TONKA files
        script += """
REM Copy TONKA files
echo.
echo Copying TONKA integration files...
"""
        
        for file in self.tonka_files:
            pigpen_file = self.pigpen / file
            if pigpen_file.exists():
                rel_path = file.replace('/', '\\')
                tori_path = self.tori / file
                
                # Create directory if needed
                if '\\' in rel_path:
                    dir_path = str(tori_path.parent).replace('/', '\\')
                    script += f'\nmkdir "{dir_path}" 2>nul'
                
                script += f'\ncopy "{self.pigpen}\\{rel_path}" "{self.tori}\\{rel_path}"'
                script += f'\nif %errorlevel% equ 0 (echo [OK] Copied: {file}) else (echo [FAIL] Failed: {file})'
        
        # Add modified file copy
        script += """

REM Copy modified prajna_api.py with TONKA integration
echo.
echo Updating prajna_api.py with TONKA integration...
"""
        script += f'\ncopy "{self.pigpen}\\prajna\\api\\prajna_api.py" "{self.tori}\\prajna\\api\\prajna_api.py"'
        script += '\nif %errorlevel% equ 0 (echo [OK] Updated prajna_api.py) else (echo [FAIL] Failed to update prajna_api.py)'
        
        script += """

echo.
echo ========================================
echo Migration Complete!
echo ========================================
echo.
echo Backup saved to: """ + str(self.backup_dir) + """
echo.
echo Next steps:
echo 1. Create the missing coordinator files
echo 2. Test TONKA integration
echo 3. Copy datasets if needed
echo.
pause
"""
        
        script_path = self.tori / "migrate_tonka_integration.bat"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"\n✅ Created migration script: {script_path}")
        return script_path
    
    def suggest_coordinator_creation(self):
        """Suggest how to create the missing coordinator files"""
        print("\n📝 Missing Coordinator Files Need Creation:")
        print("="*60)
        print("\nThe coordinator files mentioned in your list don't exist yet.")
        print("Based on the pattern, they should:")
        print("\n1. api/prajna_tonka_coordinator.py should:")
        print("   - Import both Prajna and TONKA modules")
        print("   - Analyze user requests for code generation patterns")
        print("   - Route requests to appropriate service")
        print("   - Return combined responses")
        print("\n2. test_prajna_tonka_coordination.py should:")
        print("   - Test the coordinator functionality")
        print("   - Test multi-part requests")
        print("   - Verify routing logic")

def main():
    print("🚀 TONKA Integration Migrator")
    print("="*60)
    
    migrator = TONKAMigrator()
    
    # Analyze current state
    existing_files = migrator.analyze()
    
    # Create migration script
    if existing_files:
        script_path = migrator.create_migration_script()
        
        print("\n🎯 Ready to Migrate!")
        print(f"Run: {script_path.name}")
    
    # Suggest coordinator creation
    migrator.suggest_coordinator_creation()
    
    print("\n💡 Summary:")
    print(f"  - {len(existing_files)} TONKA files ready to migrate")
    print(f"  - 1 modified file (prajna_api.py) with TONKA integration")
    print(f"  - 2 coordinator files need to be created")
    print(f"  - Datasets directory available for migration")

if __name__ == "__main__":
    main()
