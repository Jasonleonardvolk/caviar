#!/usr/bin/env python3
"""
Precise Bidirectional Change Analyzer
=====================================
Shows EXACTLY what changed in both directions with timestamps
"""

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime, timedelta

class PreciseChangeAnalyzer:
    def __init__(self):
        self.tori = Path(r"{PROJECT_ROOT}")
        self.pigpen = Path(r"C:\Users\jason\Desktop\pigpen")
        
        # Time boundaries
        self.now = datetime.now()
        self.yesterday = self.now - timedelta(days=1)
        self.two_days_ago = self.now - timedelta(days=2)
        
    def get_file_time(self, filepath):
        """Get modification time of a file"""
        if filepath.exists():
            return datetime.fromtimestamp(filepath.stat().st_mtime)
        return None
    
    def scan_recent_changes(self, base_path, name, since_time):
        """Scan for files modified since a specific time"""
        changes = []
        
        for filepath in base_path.rglob('*'):
            if filepath.is_file():
                # Skip system files
                if any(skip in str(filepath) for skip in ['.git', '__pycache__', 'node_modules', '.pyc']):
                    continue
                    
                mtime = self.get_file_time(filepath)
                if mtime and mtime > since_time:
                    relative = filepath.relative_to(base_path)
                    changes.append({
                        'path': str(relative),
                        'mtime': mtime,
                        'size': filepath.stat().st_size,
                        'full_path': filepath
                    })
        
        return sorted(changes, key=lambda x: x['mtime'], reverse=True)
    
    def compare_files(self, file_path):
        """Compare the same file in both locations"""
        tori_file = self.tori / file_path
        pigpen_file = self.pigpen / file_path
        
        tori_exists = tori_file.exists()
        pigpen_exists = pigpen_file.exists()
        
        if not tori_exists and not pigpen_exists:
            return None
            
        result = {
            'path': file_path,
            'status': '',
            'action': ''
        }
        
        if tori_exists and not pigpen_exists:
            result['status'] = 'TORI_ONLY'
            result['action'] = 'Copy to Pigpen'
            result['tori_time'] = self.get_file_time(tori_file)
            
        elif pigpen_exists and not tori_exists:
            result['status'] = 'PIGPEN_ONLY'
            result['action'] = 'Copy to TORI'
            result['pigpen_time'] = self.get_file_time(pigpen_file)
            
        else:  # Both exist
            tori_time = self.get_file_time(tori_file)
            pigpen_time = self.get_file_time(pigpen_file)
            
            if tori_time > pigpen_time:
                result['status'] = 'TORI_NEWER'
                result['action'] = 'Copy TORI → Pigpen'
            elif pigpen_time > tori_time:
                result['status'] = 'PIGPEN_NEWER'
                result['action'] = 'Copy Pigpen → TORI'
            else:
                result['status'] = 'SAME_TIME'
                result['action'] = 'Check content'
                
            result['tori_time'] = tori_time
            result['pigpen_time'] = pigpen_time
            result['time_diff'] = abs((tori_time - pigpen_time).total_seconds())
            
        return result
    
    def analyze(self):
        """Run the precise analysis"""
        print("🔍 PRECISE BIDIRECTIONAL ANALYSIS")
        print("="*70)
        print(f"Current time: {self.now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analyzing changes since: {self.yesterday.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Get recent changes from both sides
        pigpen_changes = self.scan_recent_changes(self.pigpen, "Pigpen", self.yesterday)
        tori_changes = self.scan_recent_changes(self.tori, "TORI", self.yesterday)
        
        # Separate by time periods
        print(f"\n📅 PIGPEN CHANGES (Last 24 hours):")
        print("-"*70)
        pigpen_yesterday = [c for c in pigpen_changes if c['mtime'] > self.yesterday]
        
        if pigpen_yesterday:
            for change in pigpen_yesterday[:20]:  # Show top 20
                time_str = change['mtime'].strftime('%H:%M:%S')
                size_kb = change['size'] / 1024
                print(f"  {time_str} | {change['path']:<50} | {size_kb:>6.1f} KB")
        else:
            print("  No changes in last 24 hours")
            
        print(f"\n📅 TORI CHANGES (Last 24 hours):")
        print("-"*70)
        tori_yesterday = [c for c in tori_changes if c['mtime'] > self.yesterday]
        
        if tori_yesterday:
            for change in tori_yesterday[:20]:  # Show top 20
                time_str = change['mtime'].strftime('%H:%M:%S')
                size_kb = change['size'] / 1024
                print(f"  {time_str} | {change['path']:<50} | {size_kb:>6.1f} KB")
        else:
            print("  No changes in last 24 hours")
        
        # Identify conflicts (files changed in both)
        pigpen_files = {c['path'] for c in pigpen_yesterday}
        tori_files = {c['path'] for c in tori_yesterday}
        
        conflicts = pigpen_files & tori_files
        
        if conflicts:
            print(f"\n⚠️  CONFLICTS (Changed in BOTH):")
            print("-"*70)
            for file_path in sorted(conflicts):
                comparison = self.compare_files(file_path)
                if comparison:
                    print(f"  {file_path}")
                    if 'tori_time' in comparison:
                        print(f"    TORI:   {comparison['tori_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if 'pigpen_time' in comparison:
                        print(f"    Pigpen: {comparison['pigpen_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"    Action: {comparison['action']}")
        
        # Migration recommendations
        print(f"\n🎯 MIGRATION RECOMMENDATIONS:")
        print("-"*70)
        
        # Pigpen → TORI (your valuable work from yesterday)
        pigpen_only = pigpen_files - tori_files
        if pigpen_only:
            print(f"\n1. PIGPEN → TORI (Your work from yesterday):")
            for path in sorted(pigpen_only)[:15]:
                print(f"   - {path}")
        
        # TORI → Pigpen (recent TORI improvements)
        tori_only = tori_files - pigpen_files
        if tori_only:
            print(f"\n2. TORI → PIGPEN (Recent TORI improvements):")
            for path in sorted(tori_only)[:15]:
                print(f"   - {path}")
        
        return {
            'pigpen_to_tori': list(pigpen_only),
            'tori_to_pigpen': list(tori_only),
            'conflicts': list(conflicts)
        }
    
    def create_precise_migration_scripts(self, analysis_results):
        """Create separate scripts for each direction"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Script 1: Pigpen → TORI
        if analysis_results['pigpen_to_tori']:
            script1 = f"""@echo off
REM Pigpen to TORI Migration (Your yesterday's work)
REM Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
REM ================================================

echo ========================================
echo MIGRATING: Pigpen to TORI
echo (Your valuable work from yesterday)
echo ========================================
echo.

set BACKUP_DIR=str(PROJECT_ROOT / "backup_pigpen_to_tori_{timestamp}
mkdir "%BACKUP_DIR%" 2>nul

"""
            for file_path in analysis_results['pigpen_to_tori']:
                rel_path = file_path.replace('/', '\\')
                script1 += f"""
echo Copying: {file_path}
copy "C:\\Users\\jason\\Desktop\\pigpen\\{rel_path}" "str(PROJECT_ROOT / "{rel_path}" /Y
if %errorlevel% equ 0 (echo [OK]) else (echo [FAILED])
"""
            
            script1 += "\necho.\necho Migration complete!\npause"
            
            script1_path = self.tori / f"migrate_pigpen_to_tori_{timestamp}.bat"
            with open(script1_path, 'w', encoding='utf-8') as f:
                f.write(script1)
            print(f"\n✅ Created: {script1_path.name}")
        
        # Script 2: TORI → Pigpen
        if analysis_results['tori_to_pigpen']:
            script2 = f"""@echo off
REM TORI to Pigpen Migration (Recent TORI improvements)
REM Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
REM ==================================================

echo ========================================
echo MIGRATING: TORI to Pigpen
echo (Recent TORI improvements)
echo ========================================
echo.

set BACKUP_DIR=C:\\Users\\jason\\Desktop\\pigpen\\backup_tori_to_pigpen_{timestamp}
mkdir "%BACKUP_DIR%" 2>nul

"""
            for file_path in analysis_results['tori_to_pigpen']:
                rel_path = file_path.replace('/', '\\')
                script2 += f"""
echo Copying: {file_path}
copy "str(PROJECT_ROOT / "{rel_path}" "C:\\Users\\jason\\Desktop\\pigpen\\{rel_path}" /Y
if %errorlevel% equ 0 (echo [OK]) else (echo [FAILED])
"""
            
            script2 += "\necho.\necho Migration complete!\npause"
            
            script2_path = self.tori / f"migrate_tori_to_pigpen_{timestamp}.bat"
            with open(script2_path, 'w', encoding='utf-8') as f:
                f.write(script2)
            print(f"✅ Created: {script2_path.name}")

def main():
    analyzer = PreciseChangeAnalyzer()
    results = analyzer.analyze()
    
    if results['pigpen_to_tori'] or results['tori_to_pigpen']:
        analyzer.create_precise_migration_scripts(results)
        
        print("\n📋 SUMMARY:")
        print(f"  - Files to migrate Pigpen → TORI: {len(results['pigpen_to_tori'])}")
        print(f"  - Files to migrate TORI → Pigpen: {len(results['tori_to_pigpen'])}")
        print(f"  - Conflicts to resolve manually: {len(results['conflicts'])}")

if __name__ == "__main__":
    main()
