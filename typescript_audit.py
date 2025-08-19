#!/usr/bin/env python3
"""
TypeScript Error Audit for IRIS v1.0.0
Comprehensive analysis of all TypeScript errors
"""

import subprocess
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class TypeScriptAuditor:
    def __init__(self):
        self.root = Path(r"D:\Dev\kha")
        self.errors = []
        self.error_patterns = defaultdict(list)
        self.file_errors = defaultdict(list)
        self.error_codes = defaultdict(int)
        
    def run_audit(self):
        print("=" * 60)
        print("  IRIS TypeScript Comprehensive Audit")
        print("=" * 60)
        
        # Step 1: Collect all errors
        self.collect_errors()
        
        # Step 2: Analyze patterns
        self.analyze_patterns()
        
        # Step 3: Check project structure
        self.check_project_structure()
        
        # Step 4: Generate report
        self.generate_report()
        
    def collect_errors(self):
        """Run tsc and collect all errors"""
        print("\n[1/4] Collecting TypeScript errors...")
        
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            capture_output=True,
            text=True,
            shell=True,
            cwd=self.root
        )
        
        # Parse errors
        lines = (result.stdout + result.stderr).split('\n')
        current_file = None
        
        for line in lines:
            # Match error pattern: file.ts:line:col - error TScode: message
            match = re.match(r'(.+?):(\d+):(\d+) - error (TS\d+): (.+)', line)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                col_num = int(match.group(3))
                error_code = match.group(4)
                message = match.group(5)
                
                error = {
                    'file': file_path,
                    'line': line_num,
                    'column': col_num,
                    'code': error_code,
                    'message': message
                }
                
                self.errors.append(error)
                self.file_errors[file_path].append(error)
                self.error_codes[error_code] += 1
                
        print(f"  Found {len(self.errors)} total errors in {len(self.file_errors)} files")
        
    def analyze_patterns(self):
        """Analyze error patterns"""
        print("\n[2/4] Analyzing error patterns...")
        
        # Group by error type
        for error in self.errors:
            code = error['code']
            msg = error['message']
            
            # Categorize by common patterns
            if "Cannot find module" in msg:
                self.error_patterns['MODULE_NOT_FOUND'].append(error)
            elif "Property" in msg and "does not exist" in msg:
                self.error_patterns['PROPERTY_MISSING'].append(error)
            elif "has no exported member" in msg:
                self.error_patterns['NO_EXPORT'].append(error)
            elif "Cannot find name" in msg:
                self.error_patterns['NAME_NOT_FOUND'].append(error)
            elif "Type" in msg and "is not assignable" in msg:
                self.error_patterns['TYPE_MISMATCH'].append(error)
            elif "Duplicate" in msg:
                self.error_patterns['DUPLICATE'].append(error)
            else:
                self.error_patterns['OTHER'].append(error)
        
        print(f"  Error patterns identified:")
        for pattern, errors in sorted(self.error_patterns.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"    {pattern}: {len(errors)} errors")
            
    def check_project_structure(self):
        """Check which directories contain errors"""
        print("\n[3/4] Checking project structure...")
        
        dir_errors = defaultdict(int)
        file_types = defaultdict(int)
        
        for file_path in self.file_errors:
            # Get directory
            parts = Path(file_path).parts
            if len(parts) > 0:
                top_dir = parts[0] if len(parts) > 1 else "root"
                dir_errors[top_dir] += len(self.file_errors[file_path])
            
            # Get file extension
            ext = Path(file_path).suffix
            file_types[ext] += len(self.file_errors[file_path])
        
        print(f"  Errors by directory:")
        for dir_name, count in sorted(dir_errors.items(), key=lambda x: -x[1])[:10]:
            print(f"    {dir_name}: {count} errors")
            
        print(f"\n  Errors by file type:")
        for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
            print(f"    {ext}: {count} errors")
            
    def generate_report(self):
        """Generate comprehensive report and action plan"""
        print("\n[4/4] Generating report and action plan...")
        
        report = {
            'summary': {
                'total_errors': len(self.errors),
                'files_with_errors': len(self.file_errors),
                'top_error_codes': dict(sorted(self.error_codes.items(), key=lambda x: -x[1])[:10])
            },
            'patterns': {k: len(v) for k, v in self.error_patterns.items()},
            'top_problem_files': [],
            'action_plan': []
        }
        
        # Top problem files
        for file_path, errors in sorted(self.file_errors.items(), key=lambda x: -len(x[1]))[:10]:
            report['top_problem_files'].append({
                'file': file_path,
                'error_count': len(errors),
                'error_types': list(set(e['code'] for e in errors))
            })
        
        # Generate action plan
        if self.error_patterns['MODULE_NOT_FOUND']:
            report['action_plan'].append({
                'priority': 1,
                'action': 'Fix missing module imports',
                'count': len(self.error_patterns['MODULE_NOT_FOUND']),
                'solution': 'Install missing packages or create type declarations'
            })
            
        if self.error_patterns['PROPERTY_MISSING']:
            report['action_plan'].append({
                'priority': 2,
                'action': 'Fix missing properties on types',
                'count': len(self.error_patterns['PROPERTY_MISSING']),
                'solution': 'Add property declarations or fix type definitions'
            })
            
        if self.error_codes.get('TS2307', 0) > 0:  # Cannot find module
            # Find which modules are missing
            missing_modules = set()
            for error in self.errors:
                if error['code'] == 'TS2307':
                    match = re.search(r"'([^']+)'", error['message'])
                    if match:
                        missing_modules.add(match.group(1))
            
            report['missing_modules'] = list(missing_modules)
        
        # Save report
        report_path = self.root / "typescript_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n  Report saved to: typescript_audit_report.json")
        
        # Print action plan
        print("\n" + "=" * 60)
        print("  ACTION PLAN")
        print("=" * 60)
        
        for i, action in enumerate(report['action_plan'], 1):
            print(f"\n  [{i}] {action['action']}")
            print(f"      Affects: {action['count']} errors")
            print(f"      Solution: {action['solution']}")
            
        if 'missing_modules' in report and report['missing_modules']:
            print(f"\n  Missing modules to address:")
            for module in report['missing_modules'][:10]:
                print(f"    - {module}")
                
        print("\n  Top problem files to fix first:")
        for file_info in report['top_problem_files'][:5]:
            print(f"    - {file_info['file']}: {file_info['error_count']} errors")

if __name__ == "__main__":
    auditor = TypeScriptAuditor()
    auditor.run_audit()
