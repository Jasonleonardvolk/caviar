#!/usr/bin/env python3
"""
Smart Merge Tool for TORI → Pigpen
==================================

This tool helps you:
1. Identify valuable changes in pigpen
2. Apply recent TORI updates to pigpen
3. Keep your pigpen customizations intact

Usage: python smart_merge_to_pigpen.py
"""

import os
import shutil
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime
import filecmp
import subprocess

class SmartMerger:
    def __init__(self):
        self.tori_path = Path(r"{PROJECT_ROOT}")
        self.pigpen_path = Path(r"C:\Users\jason\Desktop\pigpen")
        self.backup_dir = Path(r"C:\Users\jason\Desktop\pigpen_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Files that should not be merged (pigpen-specific changes)
        self.protect_files = set()
        
    def create_backup(self):
        """Create a backup of pigpen before any changes"""
        print(f"📁 Creating backup of pigpen to: {self.backup_dir}")
        
        if self.backup_dir.exists():
            print("❌ Backup directory already exists!")
            return False
            
        try:
            shutil.copytree(self.pigpen_path, self.backup_dir, 
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'node_modules'))
            print("✅ Backup created successfully!")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def find_pigpen_customizations(self):
        """Identify files that have pigpen-specific changes"""
        print("\n🔍 Analyzing pigpen customizations...")
        
        customizations = {
            'new_files': [],
            'modified_files': [],
            'datasets': [],
            'configs': []
        }
        
        # Check all files in pigpen
        for pigpen_file in self.pigpen_path.rglob('*'):
            if pigpen_file.is_file():
                relative_path = pigpen_file.relative_to(self.pigpen_path)
                tori_file = self.tori_path / relative_path
                
                # Skip system files
                if any(part.startswith('.') for part in relative_path.parts):
                    continue
                if '__pycache__' in str(relative_path):
                    continue
                
                # New file (only in pigpen)
                if not tori_file.exists():
                    customizations['new_files'].append(str(relative_path))
                    
                    # Check if it's a dataset
                    if pigpen_file.suffix in ['.npz', '.json', '.csv', '.pkl', '.h5', '.txt']:
                        size_mb = pigpen_file.stat().st_size / (1024 * 1024)
                        if size_mb > 1:  # Files larger than 1MB
                            customizations['datasets'].append({
                                'file': str(relative_path),
                                'size_mb': round(size_mb, 2)
                            })
                    
                    # Check if it's a config
                    if pigpen_file.suffix in ['.json', '.yaml', '.yml', '.ini']:
                        customizations['configs'].append(str(relative_path))
                
                # Modified file
                elif tori_file.exists() and not filecmp.cmp(pigpen_file, tori_file, shallow=False):
                    customizations['modified_files'].append(str(relative_path))
        
        return customizations
    
    def find_recent_tori_updates(self, hours=48):
        """Find files recently updated in TORI"""
        print(f"\n🔍 Finding TORI updates from last {hours} hours...")
        
        recent_updates = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        for tori_file in self.tori_path.rglob('*'):
            if tori_file.is_file():
                relative_path = tori_file.relative_to(self.tori_path)
                
                # Skip system files
                if any(part.startswith('.') for part in relative_path.parts):
                    continue
                if '__pycache__' in str(relative_path):
                    continue
                
                # Check modification time
                if tori_file.stat().st_mtime > cutoff_time:
                    pigpen_file = self.pigpen_path / relative_path
                    
                    # Only include if file exists in pigpen and is different
                    if pigpen_file.exists() and not filecmp.cmp(tori_file, pigpen_file, shallow=False):
                        recent_updates.append({
                            'file': str(relative_path),
                            'modified': datetime.fromtimestamp(tori_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                        })
        
        return recent_updates
    
    def show_merge_plan(self, customizations, recent_updates):
        """Show the merge plan to the user"""
        print("\n" + "="*60)
        print("📋 MERGE PLAN SUMMARY")
        print("="*60)
        
        print(f"\n🎯 Pigpen Customizations Found:")
        print(f"   - New files: {len(customizations['new_files'])}")
        print(f"   - Modified files: {len(customizations['modified_files'])}")
        print(f"   - Datasets: {len(customizations['datasets'])}")
        
        if customizations['datasets']:
            print("\n📊 Datasets in pigpen:")
            for ds in customizations['datasets'][:5]:
                print(f"   - {ds['file']} ({ds['size_mb']} MB)")
        
        if customizations['new_files']:
            print("\n✨ New files in pigpen (will be preserved):")
            for f in customizations['new_files'][:10]:
                if not any(f.endswith(ext) for ext in ['.npz', '.json', '.csv', '.pkl', '.h5']):
                    print(f"   - {f}")
        
        print(f"\n🔄 Recent TORI updates to apply: {len(recent_updates)}")
        if recent_updates:
            print("Recent TORI changes:")
            for update in recent_updates[:10]:
                print(f"   - {update['file']} (modified {update['modified']})")
        
        # Files to protect
        self.protect_files = set(customizations['modified_files'])
        if self.protect_files:
            print(f"\n🛡️ Files with pigpen changes (need manual review): {len(self.protect_files)}")
            for f in list(self.protect_files)[:5]:
                print(f"   - {f}")
    
    def apply_updates(self, recent_updates):
        """Apply TORI updates to pigpen"""
        print("\n🔄 Applying TORI updates to pigpen...")
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for update in recent_updates:
            relative_path = Path(update['file'])
            tori_file = self.tori_path / relative_path
            pigpen_file = self.pigpen_path / relative_path
            
            # Skip protected files
            if str(relative_path) in self.protect_files:
                print(f"⏭️ Skipping protected file: {relative_path}")
                skip_count += 1
                continue
            
            try:
                # Create directory if needed
                pigpen_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(tori_file, pigpen_file)
                print(f"✅ Updated: {relative_path}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error updating {relative_path}: {e}")
                error_count += 1
        
        print(f"\n📊 Update Summary:")
        print(f"   - Successfully updated: {success_count}")
        print(f"   - Skipped (protected): {skip_count}")
        print(f"   - Errors: {error_count}")
    
    def create_conflict_report(self, customizations):
        """Create a report of files that need manual review"""
        report_file = self.pigpen_path / "MERGE_CONFLICTS_REPORT.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Pigpen Merge Conflicts Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Files Requiring Manual Review\n\n")
            f.write("These files have changes in both TORI and pigpen:\n\n")
            
            for file in sorted(self.protect_files):
                f.write(f"- `{file}`\n")
            
            f.write("\n## Your Customizations\n\n")
            
            if customizations['new_files']:
                f.write("### New Files (Preserved)\n\n")
                for file in sorted(customizations['new_files']):
                    f.write(f"- `{file}`\n")
            
            if customizations['datasets']:
                f.write("\n### Datasets (Preserved)\n\n")
                for ds in customizations['datasets']:
                    f.write(f"- `{ds['file']}` ({ds['size_mb']} MB)\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Review the protected files listed above\n")
            f.write("2. Manually merge any important changes\n")
            f.write("3. Test your pigpen installation\n")
            f.write(f"4. If something went wrong, restore from: `{self.backup_dir}`\n")
        
        print(f"\n📄 Conflict report saved to: {report_file}")

def main():
    """Main merge process"""
    print("🔧 Smart Merge Tool: TORI → Pigpen")
    print("="*60)
    
    merger = SmartMerger()
    
    # Step 1: Create backup
    if not merger.create_backup():
        print("❌ Cannot proceed without backup!")
        return
    
    # Step 2: Analyze customizations
    customizations = merger.find_pigpen_customizations()
    
    # Step 3: Find recent TORI updates
    recent_updates = merger.find_recent_tori_updates()
    
    # Step 4: Show merge plan
    merger.show_merge_plan(customizations, recent_updates)
    
    # Step 5: Ask for confirmation
    print("\n" + "="*60)
    response = input("Proceed with merge? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Merge cancelled")
        return
    
    # Step 6: Apply updates
    merger.apply_updates(recent_updates)
    
    # Step 7: Create conflict report
    merger.create_conflict_report(customizations)
    
    print("\n✅ Merge complete!")
    print(f"📁 Backup saved to: {merger.backup_dir}")
    print("📋 Review MERGE_CONFLICTS_REPORT.md for files needing manual attention")

if __name__ == "__main__":
    main()
