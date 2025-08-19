#!/usr/bin/env python3
"""
TypeScript Error Fixer Script
Automatically fixes common TypeScript compilation errors
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import json

class TypeScriptErrorFixer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixes_applied = []
        self.files_modified = set()
        
    def backup_file(self, filepath):
        """Create a backup of the file before modifying"""
        rel_path = Path(filepath).relative_to(self.project_root)
        backup_path = self.backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, backup_path)
        
    def read_file(self, filepath):
        """Read file content"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
            
    def write_file(self, filepath, content):
        """Write content to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def fix_arraybuffer_issues(self, content, filepath):
        """Fix ArrayBufferLike vs ArrayBuffer type issues"""
        patterns = [
            # Fix writeBuffer calls with typed arrays
            (r'(device\.queue\.writeBuffer\([^,]+,\s*\d+,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\)', 
             r'\1\2.slice()'),
            # Alternative: wrap in new typed array
            (r'(device\.queue\.writeBuffer\([^,]+,\s*\d+,\s*)(amp|phi|data)(\))',
             r'\1new Float32Array(\2.buffer.slice(0))\3'),
        ]
        
        modified = False
        original = content
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                
        if modified:
            self.fixes_applied.append(f"{filepath}: Fixed ArrayBuffer type issues")
            
        return content, modified
        
    def fix_static_let_syntax(self, content, filepath):
        """Fix 'static let' syntax error"""
        pattern = r'static\s+let\s+'
        if re.search(pattern, content):
            content = re.sub(pattern, 'static ', content)
            self.fixes_applied.append(f"{filepath}: Fixed 'static let' syntax")
            return content, True
        return content, False
        
    def fix_waveBufferSize_undefined(self, content, filepath):
        """Fix undefined waveBufferSize variable"""
        if 'waveBufferSize' in content and 'wavefrontReconstructor' in str(filepath):
            # Check if it's already defined
            if not re.search(r'(const|let|var)\s+waveBufferSize', content):
                # Find where to insert the definition
                # Look for the function or class containing the usage
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'waveBufferSize' in line and 'device.createBuffer' in line:
                        # Insert definition before this line
                        # Find the beginning of the function/method
                        indent = len(line) - len(line.lstrip())
                        for j in range(i-1, -1, -1):
                            if 'async' in lines[j] or 'function' in lines[j] or '{' in lines[j]:
                                # Insert after the opening brace
                                insert_line = j + 1
                                while insert_line < len(lines) and '{' not in lines[j]:
                                    insert_line += 1
                                if insert_line < len(lines):
                                    # Calculate buffer size based on context
                                    new_line = ' ' * (indent + 4) + 'const waveBufferSize = 1024 * 1024 * 4; // 4MB buffer, adjust as needed'
                                    lines.insert(insert_line + 1, new_line)
                                    content = '\n'.join(lines)
                                    self.fixes_applied.append(f"{filepath}: Added waveBufferSize definition")
                                    return content, True
                                break
        return content, False
        
    def fix_gpu_limits_access(self, content, filepath):
        """Fix GPU device limits access pattern"""
        if 'capabilities.ts' in str(filepath):
            # Fix the limits access pattern
            replacements = [
                ('l.maxComputeInvocationsPerWorkgroup', 'l.maxComputeInvocationsPerWorkgroup || 256'),
                ('l.maxComputeWorkgroupSizeX', 'l.maxComputeWorkgroupSizeX || 256'),
                ('l.maxComputeWorkgroupSizeY', 'l.maxComputeWorkgroupSizeY || 256'),
                ('l.maxComputeWorkgroupSizeZ', 'l.maxComputeWorkgroupSizeZ || 64'),
                ('l.maxSampledTexturesPerShaderStage', 'l.maxSampledTexturesPerShaderStage || 16'),
                ('l.maxSamplersPerShaderStage', 'l.maxSamplersPerShaderStage || 16'),
            ]
            
            modified = False
            for old, new in replacements:
                if old in content and new not in content:
                    content = content.replace(old, f'(l as any).{old.split(".")[1]}')
                    modified = True
                    
            if modified:
                self.fixes_applied.append(f"{filepath}: Fixed GPU limits access with type assertions")
                
            return content, modified
        return content, False
        
    def fix_ktx2_properties(self, content, filepath):
        """Fix KTX2Level property access"""
        if 'ktx2Loader.ts' in str(filepath):
            # Fix the KTX2 level access
            pattern = r'new Uint8Array\(buf\.buffer,\s*l\.byteOffset,\s*l\.byteLength\)'
            replacement = r'new Uint8Array(buf.buffer, (l as any).byteOffset || l.levelDataByteOffset, (l as any).byteLength || l.levelDataByteLength)'
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.fixes_applied.append(f"{filepath}: Fixed KTX2Level property access")
                return content, True
        return content, False
        
    def add_device_orientation_types(self, content, filepath):
        """Add DeviceOrientationEvent type augmentation"""
        if 'DeviceOrientationEvent.requestPermission' in content:
            # Check if augmentation already exists
            if 'interface DeviceOrientationEvent' not in content:
                # Add type augmentation at the top of the file
                augmentation = '''// Type augmentation for iOS DeviceOrientationEvent
declare global {
    interface DeviceOrientationEventStatic {
        requestPermission?: () => Promise<PermissionState>;
    }
}

'''
                content = augmentation + content
                self.fixes_applied.append(f"{filepath}: Added DeviceOrientationEvent type augmentation")
                return content, True
        return content, False
        
    def fix_module_imports(self, content, filepath):
        """Fix common import issues"""
        replacements = []
        
        # Fix @/lib imports
        if '@/lib/webgpu/context/device' in content:
            # Try relative import
            if 'hybrid/lib/post' in str(filepath):
                replacements.append(
                    ('@/lib/webgpu/context/device', '../../webgpu/context/device')
                )
                
        # Fix @hybrid imports
        if '@hybrid/' in content:
            replacements.append(
                ('@hybrid/lib', '../lib')
            )
            replacements.append(
                ('@hybrid/wgsl', '../wgsl')
            )
            
        modified = False
        for old, new in replacements:
            if old in content:
                content = content.replace(f'"{old}"', f'"{new}"')
                content = content.replace(f"'{old}'", f"'{new}'")
                modified = True
                
        if modified:
            self.fixes_applied.append(f"{filepath}: Fixed module import paths")
            
        return content, modified
        
    def process_file(self, filepath):
        """Process a single file and apply all fixes"""
        filepath = self.project_root / filepath
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return
            
        content = self.read_file(filepath)
        original_content = content
        any_modified = False
        
        # Apply all fixes
        fixes = [
            self.fix_arraybuffer_issues,
            self.fix_static_let_syntax,
            self.fix_waveBufferSize_undefined,
            self.fix_gpu_limits_access,
            self.fix_ktx2_properties,
            self.add_device_orientation_types,
            self.fix_module_imports,
        ]
        
        for fix_func in fixes:
            content, modified = fix_func(content, filepath)
            any_modified = any_modified or modified
            
        if any_modified:
            self.backup_file(filepath)
            self.write_file(filepath, content)
            self.files_modified.add(str(filepath))
            
    def run(self):
        """Run the fixer on all error files"""
        error_files = [
            'frontend/hybrid/lib/integration/quickStart.ts',
            'frontend/hybrid/lib/ktx2Loader.ts',
            'frontend/hybrid/lib/memory/iosPressure.ts',
            'frontend/hybrid/lib/post/applyPhaseLUT.ts',
            'frontend/hybrid/lib/post/phasePolisher.ts',
            'frontend/hybrid/lib/post/zernikeApply.ts',
            'frontend/hybrid/main.ts',
            'frontend/hybrid/pipelines/render_select_example.ts',
            'frontend/hybrid/src/lightFieldComposerPipeline.ts',
            'frontend/hybrid/src/main.ts',
            'frontend/hybrid/wavefrontReconstructor.ts',
            'frontend/lib/webgpu/capabilities.ts',
            'frontend/lib/webgpu/kernels/index.ts',
            'frontend/lib/webgpu/minimalGPUTest.ts',
            'frontend/lib/webgpu/shaderLoader.ts',
            'frontend/src/lib/featureFlags.ts',
            'frontend/src/lib/stores/psiTelemetry.ts',
            'standalone-holo/src/pipelines/SLMEncoderPipeline.ts',
        ]
        
        print(f"TypeScript Error Fixer")
        print(f"=" * 50)
        print(f"Processing {len(error_files)} files...")
        print(f"Backup directory: {self.backup_dir}")
        print()
        
        for filepath in error_files:
            self.process_file(filepath)
            
        # Generate report
        print(f"\nFixes Applied:")
        print(f"-" * 50)
        for fix in self.fixes_applied:
            print(f"  - {fix}")
            
        print(f"\nSummary:")
        print(f"-" * 50)
        print(f"Files modified: {len(self.files_modified)}")
        print(f"Total fixes applied: {len(self.fixes_applied)}")
        
        # Create a fix report
        report = {
            'timestamp': datetime.now().isoformat(),
            'files_modified': list(self.files_modified),
            'fixes_applied': self.fixes_applied,
            'backup_directory': str(self.backup_dir)
        }
        
        report_path = self.project_root / 'typescript_fixes_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nReport saved to: {report_path}")
        print(f"\nNext steps:")
        print("1. Run 'npx tsc -p frontend/tsconfig.json --noEmit' to check remaining errors")
        print("2. Manually fix any import path issues that couldn't be auto-fixed")
        print("3. Check if any type definitions need to be installed")
        print("\nBackup created at: {self.backup_dir}")
        

def main():
    # Get the project root
    project_root = r"D:\Dev\kha"
    
    fixer = TypeScriptErrorFixer(project_root)
    fixer.run()
    

if __name__ == "__main__":
    main()
