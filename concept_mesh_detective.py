#!/usr/bin/env python3
"""
Concept Mesh Detective - Find Your Import Fixes!
==============================================

This tool helps identify what you fixed in pigpen that made
the concept mesh imports work (0 → 100 → 1095 concepts).

It focuses on:
1. Concept mesh related files
2. Import/ingestion pipeline files  
3. Changes made after the clone
4. Current concept counts
"""

import json
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime
import re

class ConceptMeshDetective:
    def __init__(self):
        self.tori_path = Path(r"{PROJECT_ROOT}")
        self.pigpen_path = Path(r"C:\Users\jason\Desktop\pigpen")
        
        # Files likely involved in concept mesh import
        self.mesh_related_patterns = [
            "*concept*",
            "*mesh*", 
            "*ingest*",
            "*pipeline*",
            "*extract*",
            "*import*",
            "*cognitive*",
            "main.py",
            "app.py",
            "*loader*",
            "*reader*"
        ]
        
    def find_concept_data(self):
        """Find and analyze concept storage files"""
        print("🔍 Analyzing concept storage files...")
        
        concept_files = {
            'tori': {},
            'pigpen': {}
        }
        
        # Check both directories for concept data files
        for name, base_path in [('tori', self.tori_path), ('pigpen', self.pigpen_path)]:
            # Common concept storage files
            potential_files = [
                'concepts.json',
                'concepts.npz',
                'concept_mesh_data.json',
                'concept_registry.json',
                'concept_registry_enhanced.json',
                'data/concepts.json',
                'data/concept_mesh.json'
            ]
            
            for file_path in potential_files:
                full_path = base_path / file_path
                if full_path.exists():
                    try:
                        # Get file info
                        stat = full_path.stat()
                        info = {
                            'path': str(file_path),
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime),
                            'concept_count': 0
                        }
                        
                        # Try to count concepts
                        if file_path.endswith('.json'):
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                # Different structures for concept storage
                                if isinstance(data, list):
                                    info['concept_count'] = len(data)
                                elif isinstance(data, dict):
                                    if 'concepts' in data:
                                        info['concept_count'] = len(data['concepts'])
                                    elif 'items' in data:
                                        info['concept_count'] = len(data['items'])
                                    else:
                                        # Count top-level keys as concepts
                                        info['concept_count'] = len(data)
                            except:
                                pass
                        
                        concept_files[name][file_path] = info
                    except Exception as e:
                        print(f"  Error reading {full_path}: {e}")
        
        return concept_files
    
    def find_mesh_related_files(self):
        """Find all files related to concept mesh operations"""
        print("\n🔍 Finding concept mesh related files...")
        
        mesh_files = {'tori': [], 'pigpen': []}
        
        for name, base_path in [('tori', self.tori_path), ('pigpen', self.pigpen_path)]:
            for pattern in self.mesh_related_patterns:
                for file_path in base_path.rglob(pattern):
                    if file_path.is_file() and file_path.suffix in ['.py', '.js', '.json']:
                        relative = file_path.relative_to(base_path)
                        if '__pycache__' not in str(relative):
                            mesh_files[name].append(str(relative))
        
        # Find files only in pigpen (potential fixes)
        pigpen_only = set(mesh_files['pigpen']) - set(mesh_files['tori'])
        modified_files = []
        
        # Check for modified files
        for file in set(mesh_files['pigpen']) & set(mesh_files['tori']):
            tori_file = self.tori_path / file
            pigpen_file = self.pigpen_path / file
            
            if tori_file.stat().st_size != pigpen_file.stat().st_size:
                modified_files.append(file)
        
        return {
            'pigpen_only': sorted(list(pigpen_only)),
            'modified': sorted(modified_files)
        }
    
    def analyze_import_code(self, file_path):
        """Analyze a file for concept import related code"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for concept-related patterns
            patterns = [
                r'concept.*=.*\[\]',  # Empty concept list initialization
                r'concepts\.append',   # Adding concepts
                r'add_concept',       # Concept addition function
                r'extract.*concept',  # Concept extraction
                r'concept.*count',    # Concept counting
                r'len\(.*concept',    # Concept length checks
                r'migrate.*concept',  # Concept migration
                r'import.*concept',   # Concept imports
                r'concept.*=.*\{',    # Concept dict initialization
                r'concept.*fix',      # Concept fixes
                r'concept.*error',    # Error handling
            ]
            
            findings = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'line_num': i + 1,
                            'line': line.strip(),
                            'pattern': pattern
                        })
            
            return findings
        except:
            return []
    
    def find_recent_logs(self):
        """Find recent log files that might contain import info"""
        print("\n🔍 Checking for recent log files...")
        
        log_files = []
        
        # Check pigpen for log files
        for log_pattern in ['*.log', '*concept*.txt', '*import*.txt', '*ingest*.txt']:
            for log_file in self.pigpen_path.rglob(log_pattern):
                if log_file.is_file():
                    stat = log_file.stat()
                    # Only recent logs (last 48 hours)
                    if (datetime.now().timestamp() - stat.st_mtime) < 48 * 3600:
                        log_files.append({
                            'path': str(log_file.relative_to(self.pigpen_path)),
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                        })
        
        return sorted(log_files, key=lambda x: x['modified'], reverse=True)
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\n" + "="*60)
        print("🔍 CONCEPT MESH DETECTIVE REPORT")
        print("="*60)
        
        # 1. Analyze concept data
        concept_data = self.find_concept_data()
        
        print("\n📊 CONCEPT COUNTS:")
        print("-"*40)
        for location in ['tori', 'pigpen']:
            print(f"\n{location.upper()}:")
            if concept_data[location]:
                for file, info in concept_data[location].items():
                    print(f"  📁 {file}")
                    print(f"     Concepts: {info['concept_count']}")
                    print(f"     Size: {info['size']:,} bytes")
                    print(f"     Modified: {info['modified'].strftime('%Y-%m-%d %H:%M')}")
            else:
                print("  No concept files found")
        
        # 2. Find mesh-related files
        mesh_files = self.find_mesh_related_files()
        
        if mesh_files['pigpen_only']:
            print(f"\n✨ FILES ONLY IN PIGPEN (potential fixes):")
            print("-"*40)
            for file in mesh_files['pigpen_only'][:10]:
                print(f"  - {file}")
        
        if mesh_files['modified']:
            print(f"\n🔧 MODIFIED FILES (check these for fixes):")
            print("-"*40)
            for file in mesh_files['modified'][:10]:
                print(f"  - {file}")
                
                # Analyze the modified file
                pigpen_file = self.pigpen_path / file
                findings = self.analyze_import_code(pigpen_file)
                if findings:
                    print(f"    Found {len(findings)} concept-related lines")
        
        # 3. Check for logs
        logs = self.find_recent_logs()
        if logs:
            print(f"\n📝 RECENT LOG FILES:")
            print("-"*40)
            for log in logs[:5]:
                print(f"  - {log['path']} ({log['modified']})")
        
        # 4. Key files to check
        print(f"\n🎯 KEY FILES TO CHECK FOR YOUR FIXES:")
        print("-"*40)
        key_files = [
            'ingest_pdf/pipeline.py',
            'ingest_pdf/cognitive_interface.py',
            'ingest_pdf/concept_extractor.py',
            'ingest_pdf/main.py',
            'concept_mesh_data.json',
            'main.py',
            'enhanced_launcher.py'
        ]
        
        for file in key_files:
            pigpen_file = self.pigpen_path / file
            tori_file = self.tori_path / file
            
            if pigpen_file.exists():
                status = "✓ Exists"
                if not tori_file.exists():
                    status += " (ONLY in pigpen!)"
                elif pigpen_file.stat().st_size != tori_file.stat().st_size:
                    status += " (MODIFIED!)"
                print(f"  {file}: {status}")
        
        print("\n💡 NEXT STEPS:")
        print("-"*40)
        print("1. Check the MODIFIED files above - they likely contain your fixes")
        print("2. Look for error handling or validation code you added")
        print("3. Check if you modified concept extraction logic")
        print("4. Look for changes in how concepts are stored/formatted")
        print(f"5. Your pigpen has {concept_data['pigpen'].get('concepts.json', {}).get('concept_count', 0)} concepts")
        print("6. Run: python compare_tori_pigpen.py to see detailed diffs")

if __name__ == "__main__":
    detective = ConceptMeshDetective()
    detective.generate_report()
    
    print("\n" + "="*60)
    input("Press Enter to continue...")
