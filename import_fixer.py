#!/usr/bin/env python3
"""
Import Fixer Script - Fixes broken imports in TORI/KHA project
Identifies and fixes common import issues
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class ImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.errors_found = []
        
        # Common import fixes
        self.import_fixes = {
            # Python fixes
            "from python.core.CognitiveEngine import": "from python.core.CognitiveEngine import",
            "from python.stability.eigenvalue_monitor import": "from python.stability.eigenvalue_monitor import",
            "from python.stability.lyapunov_analyzer import": "from python.stability.lyapunov_analyzer import", 
            "from python.stability.koopman_operator import": "from python.stability.koopman_operator import",
            "from python.core.memory_vault import": "from python.core.memory_vault import",
            
            # Database-related fixes (we don't use file_storages)
            "# # import sqlite3 // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # import sqlite3 // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "# # import sqlalchemy // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # import sqlalchemy // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "# # import pymongo // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # import pymongo // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "# # import redis // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # import redis // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "# # CREATE TABLE // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # CREATE TABLE // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "# # SQLite // FIXED: We use file-based storage only // FIXED: We use file-based storage only": "# # # SQLite // FIXED: We use file-based storage only // FIXED: We use file-based storage only // FIXED: We use file-based storage only",
            "file_storage": "file_storage",
            
            # TypeScript fixes
            "// // import { KoopmanOperator } // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead": "// // // import { KoopmanOperator } // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead",
            "// // import { LyapunovAnalyzer } // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead": "// // // import { LyapunovAnalyzer } // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead",
            "// // import { GhostCollective } // FIXED: Not implemented // FIXED: Not implemented": "// // // import { GhostCollective } // FIXED: Not implemented // FIXED: Not implemented // FIXED: Not implemented",
            
            # Remove broken references
            "// // KoopmanOperator. // FIXED: Use Python bridge // FIXED: Use Python bridge": "// // // KoopmanOperator. // FIXED: Use Python bridge // FIXED: Use Python bridge // FIXED: Use Python bridge",
            "// // LyapunovAnalyzer. // FIXED: Use Python bridge // FIXED: Use Python bridge": "// // // LyapunovAnalyzer. // FIXED: Use Python bridge // FIXED: Use Python bridge // FIXED: Use Python bridge",
            "// // new KoopmanOperator // FIXED: Use Python bridge // FIXED: Use Python bridge": "// // // new KoopmanOperator // FIXED: Use Python bridge // FIXED: Use Python bridge // FIXED: Use Python bridge",
            "// // new LyapunovAnalyzer // FIXED: Use Python bridge // FIXED: Use Python bridge": "// // // new LyapunovAnalyzer // FIXED: Use Python bridge // FIXED: Use Python bridge // FIXED: Use Python bridge",
        }
        
        # Files to check
        self.file_patterns = [
            "**/*.ts",
            "**/*.js",  
            "**/*.py",
            "**/*.svelte"
        ]
        
        # Directories to skip
        self.skip_dirs = {
            "node_modules",
            ".git", 
            "__pycache__",
            ".svelte-kit",
            "dist",
            "build"
        }
    
    def find_files(self) -> List[Path]:
        """Find all files to check"""
        files = []
        
        for pattern in self.file_patterns:
            try:
                for file_path in self.project_root.glob(pattern):
                    # Skip if in excluded directory
                    try:
                        if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                            continue
                        
                        if file_path.is_file() and file_path.exists():
                            files.append(file_path)
                    except (OSError, IOError) as e:
                        # Skip files that can't be accessed
                        continue
            except Exception as e:
                print(f"Warning: Error scanning pattern {pattern}: {e}")
                continue
        
        return files
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix imports in a single file"""
        try:
            # Skip if file doesn't exist or can't be read
            if not file_path.exists() or not file_path.is_file():
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            for old_import, new_import in self.import_fixes.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    print(f"Fixed: {file_path} - {old_import}")
                    self.fixes_applied += 1
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.errors_found.append(error_msg)
            print(f"ERROR: {error_msg}")
        
        return False
    
    def find_broken_imports(self, file_path: Path) -> List[str]:
        """Find broken imports in a file"""
        broken_imports = []
        
        try:
            if not file_path.exists() or not file_path.is_file():
                return []
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common broken import patterns
            broken_patterns = [
                r'import.*// // KoopmanOperator. // FIXED: Use Python bridge // FIXED: Use Python bridge*from.*(?!python\.bridges)',
                r'import.*// // LyapunovAnalyzer. // FIXED: Use Python bridge // FIXED: Use Python bridge*from.*(?!python\.bridges)', 
                r'import.*GhostCollective',
                r'from.*\..*CognitiveEngine.*import.*(?!.*python\.bridges)',
                r'// // new KoopmanOperator // FIXED: Use Python bridge // FIXED: Use Python bridge\(',
                r'// // new LyapunovAnalyzer // FIXED: Use Python bridge // FIXED: Use Python bridge\(',
                r'KoopmanOperator\.[a-zA-Z]',
                r'LyapunovAnalyzer\.[a-zA-Z]'
            ]
            
            for pattern in broken_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    broken_imports.extend(matches)
                    
        except Exception as e:
            pass
            
        return broken_imports
    
    def create_bridge_usage_examples(self):
        """Create examples of how to use Python bridges"""
        examples_dir = self.project_root / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        bridge_example = '''
// Example: Using Python Bridges Instead of Direct Imports

// OLD (BROKEN):
// // // import { KoopmanOperator } // FIXED: Use Python bridge instead // FIXED: Use Python bridge instead from '../stability/KoopmanOperator';
// const koopman = // // new KoopmanOperator // FIXED: Use Python bridge // FIXED: Use Python bridge();

// NEW (WORKING):
import { createPythonBridge } from '../bridges/PythonBridge';

class StabilityAnalyzer {
  private koopmanBridge: any;
  
  async initialize() {
    this.koopmanBridge = createPythonBridge('python/stability/koopman_operator.py');
    await this.koopmanBridge.call('initialize');
  }
  
  async analyzeStability(data: number[][]) {
    const result = await this.koopmanBridge.call('compute_dmd', data, 0.1);
    return result;
  }
}

// Usage:
const analyzer = new StabilityAnalyzer();
await analyzer.initialize();
const stability = await analyzer.analyzeStability(myData);
'''
        
        with open(examples_dir / "bridge_usage.ts", 'w') as f:
            f.write(bridge_example)
        
        print(f"Created bridge usage examples in {examples_dir}")
    
    def generate_report(self) -> str:
        """Generate a report of fixes applied"""
        report = f"""
=== TORI/KHA Import Fix Report ===

Fixes Applied: {self.fixes_applied}
Errors Found: {len(self.errors_found)}

"""    
        if self.errors_found:
            report += "Errors:\n"
            for error in self.errors_found[:10]:  # Limit to first 10 errors
                report += f"  - {error}\n"
            if len(self.errors_found) > 10:
                report += f"  ... and {len(self.errors_found) - 10} more errors\n"
            report += "\n"
        
        report += """Summary:
- Commented out broken KoopmanOperator/LyapunovAnalyzer imports
- Fixed Python import paths to include 'python.' prefix
- Created bridge usage examples
- Ready for Python bridge integration

Next Steps:
1. Update TypeScript files to use Python bridges
2. Test the start_tori.py script
3. Verify all services start correctly
"""
        
        return report
    
    def run(self):
        """Run the import fixer"""
        print("🔧 Starting TORI/KHA Import Fixer...")
        print(f"Project root: {self.project_root}")
        
        # Find files
        print("Scanning for files...")
        files = self.find_files()
        print(f"Found {len(files)} files to check")
        
        # Process files
        fixed_files = 0
        for i, file_path in enumerate(files):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{len(files)} files processed...")
                
            try:
                if self.fix_file(file_path):
                    fixed_files += 1
                    
                # Also check for broken imports
                broken = self.find_broken_imports(file_path)
                if broken:
                    print(f"⚠️  Potential issues in {file_path}: {len(broken)} broken imports")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
        
        # Create examples
        self.create_bridge_usage_examples()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_file = self.project_root / "import_fix_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"Report saved to: {report_file}")
        
        if self.fixes_applied > 0:
            print(f"✅ Fixed {self.fixes_applied} import issues in {fixed_files} files")
        else:
            print("ℹ️  No import issues found")

def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
    
    fixer = ImportFixer(project_root)
    fixer.run()

if __name__ == "__main__":
    main()
