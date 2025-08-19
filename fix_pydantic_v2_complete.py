#!/usr/bin/env python3
"""
Comprehensive Pydantic v2 Migration Fix
Fixes all Pydantic v1 to v2 migration issues
"""

import os
import re
from pathlib import Path

def fix_pydantic_v2_issues(file_path):
    """Fix all Pydantic v2 migration issues in a file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. Fix BaseSettings import
        if 'from pydantic import' in content and 'BaseSettings' in content:
            # Remove BaseSettings from pydantic import
            content = re.sub(
                r'from pydantic import (.*?)BaseSettings(.*?)(?=\n)',
                lambda m: f"from pydantic import {m.group(1).strip().rstrip(',')}{m.group(2).strip().lstrip(',')}".strip().rstrip(','),
                content
            )
            # Add pydantic_settings import if not present
            if 'from pydantic_settings import' not in content:
                # Find where to insert
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'from pydantic import' in line:
                        lines.insert(i + 1, 'from pydantic_settings import BaseSettings, SettingsConfigDict')
                        content = '\n'.join(lines)
                        break
        
        # 2. Fix validator to field_validator
        if '@validator' in content:
            # Replace @validator with @field_validator
            content = re.sub(r'@validator\(', '@field_validator(', content)
            
            # Add @classmethod after @field_validator if not present
            lines = content.split('\n')
            for i in range(len(lines)):
                if '@field_validator' in lines[i] and i + 1 < len(lines):
                    if '@classmethod' not in lines[i + 1] and 'def ' in lines[i + 1]:
                        # Insert @classmethod
                        indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                        lines.insert(i + 1, ' ' * indent + '@classmethod')
            content = '\n'.join(lines)
            
            # Fix pre=True to mode="before"
            content = re.sub(r'@field_validator\((.*?),\s*pre\s*=\s*True(.*?)\)', 
                           r'@field_validator(\1, mode="before"\2)', content)
            
            # Add field_validator import if needed
            if 'field_validator' in content and 'from pydantic import' in content:
                content = re.sub(
                    r'from pydantic import (.*?)(?=\n)',
                    lambda m: f"from pydantic import {m.group(1)}, field_validator" if 'field_validator' not in m.group(1) else m.group(0),
                    content,
                    count=1
                )
        
        # 3. Fix Config class to model_config
        if 'class Config:' in content and 'BaseSettings' in content:
            # Extract Config class content
            config_match = re.search(r'class Config:(.*?)(?=\n\s*\n|\n\s*#|\Z)', content, re.DOTALL)
            if config_match:
                config_content = config_match.group(1)
                
                # Parse config values
                config_dict = {}
                for line in config_content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        config_dict[key] = value
                
                # Build model_config
                model_config_lines = ['    model_config = SettingsConfigDict(']
                
                for key, value in config_dict.items():
                    if key == 'env_prefix':
                        model_config_lines.append(f'        env_prefix={value},')
                    elif key == 'env_file':
                        model_config_lines.append(f'        env_file={value},')
                    elif key == 'case_sensitive':
                        model_config_lines.append(f'        case_sensitive={value},')
                
                # Add extra="allow" for flexibility
                model_config_lines.append('        extra="allow"')
                model_config_lines.append('    )')
                
                # Replace Config class with model_config
                content = re.sub(
                    r'class Config:.*?(?=\n\s*\n|\n\s*#|\Z)',
                    '\n'.join(model_config_lines),
                    content,
                    flags=re.DOTALL
                )
        
        # 4. Clean up imports
        # Remove empty imports
        content = re.sub(r'from pydantic import\s*\n', '', content)
        content = re.sub(r'from pydantic import\s*,', 'from pydantic import', content)
        
        # Remove duplicate imports
        lines = content.split('\n')
        seen_imports = set()
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('from pydantic') or line.strip().startswith('from pydantic_settings'):
                if line.strip() not in seen_imports:
                    seen_imports.add(line.strip())
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        content = '\n'.join(cleaned_lines)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("🚀 Pydantic v2 Migration Fix")
    print("=" * 50)
    
    # Get all Python files
    fixed_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv_tori_prod'}]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Check if file needs fixing
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if any(pattern in content for pattern in ['BaseSettings', '@validator', 'class Config:']):
                        print(f"🔧 Checking: {file_path}")
                        if fix_pydantic_v2_issues(file_path):
                            print(f"✅ Fixed: {file_path}")
                            fixed_count += 1
                            
                except Exception as e:
                    print(f"❌ Could not process {file_path}: {e}")
    
    print(f"\n✅ Fixed {fixed_count} files")
    print("\n🎯 Now try running: python enhanced_launcher.py")

if __name__ == "__main__":
    main()
