#!/usr/bin/env python3
"""
Comprehensive TypeScript Fix Script
Fixes all remaining TypeScript issues in the project
"""

import os
import re
from pathlib import Path

def fix_buffer_writes(content):
    """Fix buffer write patterns to use .buffer property"""
    # Fix patterns like writeBuffer(buffer, 0, someData)
    pattern = r'writeBuffer\(([^,]+),\s*(\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*(?:Data|Array)?)\)'
    
    def replace_func(match):
        buffer_var = match.group(1)
        offset = match.group(2)
        data_var = match.group(3)
        
        # Check if it already has .buffer
        if '.buffer' in data_var:
            return match.group(0)
        
        return f'writeBuffer({buffer_var}, {offset}, {data_var}.buffer)'
    
    return re.sub(pattern, replace_func, content)

def fix_write_timestamp(content):
    """Fix writeTimestamp calls to use compute passes"""
    # Pattern to find encoder.writeTimestamp calls
    pattern = r'(\s*)encoder\.writeTimestamp\(([^,]+),\s*([^)]+)\);?'
    
    def replace_func(match):
        indent = match.group(1)
        query_set = match.group(2)
        index = match.group(3)
        
        # Create a compute pass for the timestamp
        replacement = f"""{indent}{{
{indent}    const timestampPass = encoder.beginComputePass();
{indent}    if (device.features.has('timestamp-query')) {{
{indent}        timestampPass.writeTimestamp({query_set}, {index});
{indent}    }}
{indent}    timestampPass.end();
{indent}}}"""
        
        return replacement
    
    return re.sub(pattern, replace_func, content)

def fix_onnx_runtime(content):
    """Fix ONNX Runtime usage patterns"""
    # Remove IOBinding references
    content = re.sub(r'ort\.IOBinding', 'null', content)
    content = re.sub(r'new\s+ort\.IOBinding\([^)]*\)', 'null', content)
    
    # Fix InferenceSession references
    content = re.sub(r'ort\.InferenceSession', 'InferenceSession', content)
    
    # Fix Tensor references
    content = re.sub(r'ort\.Tensor', 'Tensor', content)
    
    # Ensure proper imports if needed
    if 'InferenceSession' in content and 'import' in content:
        if 'from "onnxruntime-web"' in content:
            # Ensure proper named imports
            content = re.sub(
                r'import\s*\*\s*as\s+ort\s+from\s+"onnxruntime-web"',
                'import { InferenceSession, Tensor } from "onnxruntime-web"',
                content
            )
    
    return content

def fix_performance_shadowing(content):
    """Fix performance variable shadowing issues"""
    # Look for local variables named 'performance'
    pattern = r'(const|let|var)\s+performance\s*='
    
    if re.search(pattern, content):
        # Rename local performance variables to perf
        content = re.sub(r'(const|let|var)\s+performance\s*=', r'\1 perf =', content)
        # Update references to the renamed variable
        content = re.sub(r'(?<!window\.)(?<!globalThis\.)(?<!\.)performance\.', 'perf.', content)
    
    return content

def fix_implementation_property(content):
    """Fix missing implementation property in schrodingerKernelRegistry"""
    if 'schrodingerKernelRegistry' in content:
        # Add implementation property if missing
        pattern = r'(interface\s+[^{]+\{[^}]*?)(\})'
        
        def add_implementation(match):
            interface_body = match.group(1)
            if 'implementation' not in interface_body:
                return f'{interface_body}    implementation?: any;\n{match.group(2)}'
            return match.group(0)
        
        content = re.sub(pattern, add_implementation, content, flags=re.DOTALL)
    
    return content

def process_file(file_path):
    """Process a single file with all fixes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all fixes
        content = fix_buffer_writes(content)
        content = fix_write_timestamp(content)
        content = fix_onnx_runtime(content)
        content = fix_performance_shadowing(content)
        content = fix_implementation_property(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all TypeScript issues"""
    
    # List of files that need buffer fixes
    buffer_fix_files = [
        "frontend/lib/holographicEngine.ts",
        "frontend/lib/webgpu/fftCompute.ts",
        "frontend/lib/webgpu/indirect.ts",
        "frontend/lib/webgpu/kernels/schrodingerBenchmark.ts",
        "frontend/lib/webgpu/kernels/schrodingerEvolution.ts",
        "frontend/lib/webgpu/pipelines/phaseLUT.ts",
        "frontend/lib/webgpu/quilt/WebGPUQuiltGenerator.ts",
        "frontend/lib/webgpu/utils/bufferHelpers.ts",
        "tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts"
    ]
    
    # Files that need writeTimestamp fixes
    timestamp_fix_files = [
        "frontend/lib/webgpu/fftCompute.ts",
        "frontend/lib/webgpu/kernels/splitStepOrchestrator.ts"
    ]
    
    # Files that need ONNX fixes
    onnx_fix_files = [
        "frontend/lib/ml/onnxWaveOpRunner.ts",
        "frontend/lib/ml/ort.ts"
    ]
    
    # Files with performance shadowing issues
    performance_fix_files = [
        "frontend/lib/webgpu/kernels/schrodingerBenchmark.ts"
    ]
    
    # Files with implementation property issues
    implementation_fix_files = [
        "frontend/lib/webgpu/kernels/schrodingerKernelRegistry.ts"
    ]
    
    # Combine all files
    all_files = set(
        buffer_fix_files + 
        timestamp_fix_files + 
        onnx_fix_files + 
        performance_fix_files +
        implementation_fix_files
    )
    
    fixed_count = 0
    
    print("Starting TypeScript fixes...")
    print("=" * 50)
    
    for file_path in all_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"Processing: {file_path}")
            if process_file(full_path):
                fixed_count += 1
                print(f"  ✓ Fixed")
            else:
                print(f"  - No changes needed")
        else:
            print(f"  ✗ File not found: {file_path}")
    
    print("=" * 50)
    print(f"Fixed {fixed_count} files")
    
    # Additional manual fix for specific complex patterns
    print("\nApplying additional targeted fixes...")
    
    # Fix specific writeBuffer patterns with immediate data
    specific_fixes = [
        ("frontend/lib/holographicEngine.ts", [
            (r'writeBuffer\(uniformBuffer,\s*0,\s*new Float32Array\(', 
             r'writeBuffer(uniformBuffer, 0, new Float32Array('),
            (r'\)\)', r').buffer)'),
        ]),
        ("frontend/lib/webgpu/quilt/WebGPUQuiltGenerator.ts", [
            (r'writeBuffer\(([^,]+),\s*0,\s*new (Float32Array|Uint32Array)\(([^)]+)\)\)',
             r'writeBuffer(\1, 0, new \2(\3).buffer)'),
        ])
    ]
    
    for file_path, patterns in specific_fixes:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✓ Applied specific fixes to {file_path}")
            except Exception as e:
                print(f"  ✗ Error with specific fixes for {file_path}: {e}")
    
    print("\nAll TypeScript fixes complete!")
    print("\nNext steps:")
    print("1. Run: npx tsc -p frontend/tsconfig.json --noEmit")
    print("2. Check for any remaining errors")
    print("3. If clean, run: npm run build")

if __name__ == "__main__":
    main()
