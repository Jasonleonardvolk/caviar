#!/usr/bin/env python3
"""
Find all camelCase vs snake_case mismatches in Soliton code
============================================================
"""

import os
import re
from pathlib import Path

def search_for_field_mismatches(directory):
    """Search for potential field name mismatches"""
    
    # Common patterns where mismatches occur
    camel_fields = ['userId', 'conceptId', 'targetPhase', 'maxResults', 'vaultLevel']
    snake_fields = ['user_id', 'concept_id', 'target_phase', 'max_results', 'vault_level']
    
    issues_found = []
    files_checked = 0
    
    # Search patterns
    patterns = [
        # Model definitions
        (r'class\s+\w*Request.*?:\s*\n(.*?)(?=\n\s*class|\n\s*def|\n\s*@|\Z)', 'model_definition'),
        # Field access patterns
        (r'request\.\w+', 'field_access'),
        # JSON/Dict creation
        (r'["\'](?:userId|user_id|conceptId|concept_id)["\']', 'json_key'),
    ]
    
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv_tori_prod', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                files_checked += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip if not Soliton-related
                    if 'soliton' not in content.lower():
                        continue
                    
                    # Check for SolitonInitRequest and similar models
                    if 'SolitonInitRequest' in content:
                        print(f"\n📄 Found SolitonInitRequest in: {filepath}")
                        
                        # Extract the model definition
                        model_match = re.search(r'class\s+SolitonInitRequest.*?(?=\n\s*class|\n\s*def|\n\s*@|\Z)', content, re.DOTALL)
                        if model_match:
                            model_text = model_match.group(0)
                            print("  Model fields:")
                            
                            # Find field definitions
                            field_pattern = r'(\w+):\s*(?:str|int|float|bool|Optional\[.*?\]|List\[.*?\])'
                            fields = re.findall(field_pattern, model_text)
                            for field in fields:
                                print(f"    - {field}")
                        
                        # Find how the request is accessed
                        access_pattern = r'request\.(\w+)'
                        accesses = re.findall(access_pattern, content)
                        if accesses:
                            print("  Field accesses:")
                            for access in set(accesses):
                                print(f"    - request.{access}")
                                
                    # Check for other Soliton models
                    for model_name in ['SolitonStoreRequest', 'SolitonPhaseRequest', 'SolitonVaultRequest']:
                        if model_name in content:
                            print(f"\n📄 Found {model_name} in: {filepath}")
                            
                            # Extract model and check fields
                            model_match = re.search(rf'class\s+{model_name}.*?(?=\n\s*class|\n\s*def|\n\s*@|\Z)', content, re.DOTALL)
                            if model_match:
                                model_text = model_match.group(0)
                                field_pattern = r'(\w+):\s*(?:str|int|float|bool|Optional\[.*?\]|List\[.*?\])'
                                fields = re.findall(field_pattern, model_text)
                                if fields:
                                    print(f"  Model fields: {', '.join(fields)}")
                    
                    # Look for potential mismatches in function definitions
                    func_pattern = r'def\s+(\w*soliton\w*)\s*\([^)]*\):'
                    functions = re.findall(func_pattern, content)
                    for func_name in functions:
                        # Get the function body
                        func_match = re.search(rf'def\s+{func_name}\s*\([^)]*\):.*?(?=\ndef|\nclass|\Z)', content, re.DOTALL)
                        if func_match:
                            func_text = func_match.group(0)
                            
                            # Check for both camelCase and snake_case usage
                            has_camel = any(field in func_text for field in camel_fields)
                            has_snake = any(field in func_text for field in snake_fields)
                            
                            if has_camel and has_snake:
                                print(f"\n⚠️  Mixed case in {filepath} - function {func_name}")
                                issues_found.append((filepath, func_name, "mixed_case"))
                            
                            # Check for specific mismatches
                            if 'request.user_id' in func_text and 'userId' in content:
                                print(f"❌ Mismatch in {filepath} - {func_name}: request.user_id but model has userId")
                                issues_found.append((filepath, func_name, "user_id_mismatch"))
                                
                            if 'request.concept_id' in func_text and 'conceptId' in content:
                                print(f"❌ Mismatch in {filepath} - {func_name}: request.concept_id but model has conceptId")
                                issues_found.append((filepath, func_name, "concept_id_mismatch"))
                                
                except Exception as e:
                    print(f"⚠️  Error reading {filepath}: {e}")
    
    return files_checked, issues_found

def check_specific_files():
    """Check specific files that are likely to have issues"""
    
    files_to_check = [
        "prajna/api/prajna_api.py",
        "api/routes/soliton.py",
        "api/routes/soliton_production.py",
        "alan_backend/routes/soliton.py",
        "mcp_metacognitive/core/soliton_memory.py"
    ]
    
    print("🔍 Checking specific Soliton files for field mismatches...\n")
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"Checking: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for request field accesses
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # Check for request.field patterns
                if 'request.' in line and any(field in line for field in ['user_id', 'userId', 'concept_id', 'conceptId']):
                    print(f"  Line {i+1}: {line.strip()}")
                    
                # Check for dictionary/JSON creation
                if any(field in line for field in ['userId', 'user_id', 'conceptId', 'concept_id']) and '{' in line:
                    print(f"  Line {i+1} (dict): {line.strip()}")

def main():
    """Main function"""
    print("🔍 Searching for camelCase vs snake_case mismatches in Soliton code...")
    print("=" * 60)
    
    # First check specific files
    check_specific_files()
    
    # Then do a full search
    print("\n" + "=" * 60)
    print("🔍 Full codebase search...")
    files_checked, issues = search_for_field_mismatches(".")
    
    print(f"\n📊 Summary:")
    print(f"  Files checked: {files_checked}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print(f"\n❌ Issues requiring attention:")
        for filepath, location, issue_type in issues:
            print(f"  - {filepath} ({location}): {issue_type}")
    
    print("\n💡 Recommendations:")
    print("1. Ensure all Pydantic models use consistent field naming")
    print("2. When accessing request fields, use the exact name from the model")
    print("3. Consider using snake_case everywhere for Python consistency")

if __name__ == "__main__":
    main()
