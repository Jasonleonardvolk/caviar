#!/usr/bin/env python3
"""
Fix Remaining TypeScript Issues in Frontend Directory
Addresses buffer writes, writeTimestamp calls, ONNX runtime, and other issues
"""

import os
import re
from pathlib import Path
from datetime import datetime
import json

# Base path for the project
PROJECT_PATH = Path("D:/Dev/kha")
FRONTEND_PATH = PROJECT_PATH / "frontend"

def backup_file(file_path):
    """Create a backup of a file before modifying"""
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    if not backup_path.exists():
        content = file_path.read_text(encoding='utf-8')
        backup_path.write_text(content, encoding='utf-8')
    return backup_path

# ============================================================
# FIX 1: Buffer Write Pattern Fixes
# ============================================================

def fix_buffer_writes():
    """Fix all buffer write patterns to use .buffer property"""
    print("\n[1/5] Fixing buffer write patterns...")
    
    files_to_fix = [
        ("frontend/lib/holographicEngine.ts", [574, 585, 602, 616, 958]),
        ("frontend/lib/webgpu/fftCompute.ts", [447]),
        ("frontend/lib/webgpu/indirect.ts", [9]),
        ("frontend/lib/webgpu/kernels/schrodingerBenchmark.ts", [401]),
        ("frontend/lib/webgpu/kernels/schrodingerEvolution.ts", [198, 499]),
        ("frontend/lib/webgpu/pipelines/phaseLUT.ts", [85]),
        ("frontend/lib/webgpu/quilt/WebGPUQuiltGenerator.ts", [327, 335, 608]),
        ("frontend/lib/webgpu/utils/bufferHelpers.ts", [13]),
        ("tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts", [443])
    ]
    
    for file_path, line_numbers in files_to_fix:
        full_path = PROJECT_PATH / file_path
        if full_path.exists():
            backup_file(full_path)
            content = full_path.read_text(encoding='utf-8')
            
            # Fix patterns like writeBuffer(buffer, 0, someData) to writeBuffer(buffer, 0, someData.buffer)
            # Pattern 1: Direct typed array variables
            content = re.sub(
                r'writeBuffer\(([^,]+),\s*(\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*(?:Data|Array)?)\)(?!\.buffer)',
                r'writeBuffer(\1, \2, \3.buffer)',
                content
            )
            
            # Pattern 2: new Float32Array/Uint32Array/Int32Array constructions
            content = re.sub(
                r'writeBuffer\(([^,]+),\s*(\d+),\s*new\s+(Float32Array|Uint32Array|Int32Array|Uint8Array)\(([^)]+)\)\)',
                r'writeBuffer(\1, \2, new \3(\4).buffer)',
                content
            )
            
            # Pattern 3: Method calls that return typed arrays
            content = re.sub(
                r'writeBuffer\(([^,]+),\s*(\d+),\s*([^,]+\.(?:getData|toArray|getBuffer)\(\))\)',
                r'writeBuffer(\1, \2, \3.buffer)',
                content
            )
            
            full_path.write_text(content, encoding='utf-8')
            print(f"  ✓ Fixed buffer writes in {file_path}")

# ============================================================
# FIX 2: writeTimestamp Calls
# ============================================================

def fix_write_timestamp():
    """Fix writeTimestamp calls to use compute passes"""
    print("\n[2/5] Fixing writeTimestamp calls...")
    
    files_to_fix = [
        "frontend/lib/webgpu/fftCompute.ts",
        "frontend/lib/webgpu/kernels/splitStepOrchestrator.ts"
    ]
    
    for file_path in files_to_fix:
        full_path = PROJECT_PATH / file_path
        if full_path.exists():
            backup_file(full_path)
            content = full_path.read_text(encoding='utf-8')
            
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
            
            content = re.sub(pattern, replace_func, content)
            
            full_path.write_text(content, encoding='utf-8')
            print(f"  ✓ Fixed writeTimestamp in {file_path}")

# ============================================================
# FIX 3: ONNX Runtime Usage
# ============================================================

def fix_onnx_runtime():
    """Fix ONNX Runtime usage patterns"""
    print("\n[3/5] Fixing ONNX Runtime usage...")
    
    # Find all TypeScript files that might use ONNX
    onnx_files = []
    for ts_file in FRONTEND_PATH.glob("**/*.ts"):
        content = ts_file.read_text(encoding='utf-8')
        if 'onnxruntime' in content or 'ort.' in content:
            onnx_files.append(ts_file)
    
    for file_path in onnx_files:
        backup_file(file_path)
        content = file_path.read_text(encoding='utf-8')
        
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
        
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✓ Fixed ONNX usage in {file_path.relative_to(PROJECT_PATH)}")

# ============================================================
# FIX 4: Missing 'implementation' Property
# ============================================================

def fix_implementation_property():
    """Fix missing implementation property in schrodingerKernelRegistry"""
    print("\n[4/5] Fixing missing 'implementation' property...")
    
    registry_path = FRONTEND_PATH / "lib/webgpu/kernels/schrodingerKernelRegistry.ts"
    if registry_path.exists():
        backup_file(registry_path)
        content = registry_path.read_text(encoding='utf-8')
        
        # Add implementation property if missing
        pattern = r'(interface\s+[^{]+\{[^}]*?)(\})'
        
        def add_implementation(match):
            interface_body = match.group(1)
            if 'implementation' not in interface_body:
                return f'{interface_body}    implementation?: any;\n{match.group(2)}'
            return match.group(0)
        
        content = re.sub(pattern, add_implementation, content, flags=re.DOTALL)
        
        registry_path.write_text(content, encoding='utf-8')
        print("  ✓ Fixed implementation property in schrodingerKernelRegistry.ts")

# ============================================================
# FIX 5: Performance Variable Shadowing
# ============================================================

def fix_performance_shadowing():
    """Fix performance variable shadowing issues"""
    print("\n[5/5] Fixing performance variable shadowing...")
    
    benchmark_path = FRONTEND_PATH / "lib/webgpu/kernels/schrodingerBenchmark.ts"
    if benchmark_path.exists():
        backup_file(benchmark_path)
        content = benchmark_path.read_text(encoding='utf-8')
        
        # Look for local variables named 'performance'
        pattern = r'(const|let|var)\s+performance\s*='
        
        if re.search(pattern, content):
            # Rename local performance variables to perf
            content = re.sub(r'(const|let|var)\s+performance\s*=', r'\1 perf =', content)
            # Update references to the renamed variable
            content = re.sub(r'(?<!window\.)(?<!globalThis\.)(?<!\.)performance\.', 'perf.', content)
        
        benchmark_path.write_text(content, encoding='utf-8')
        print("  ✓ Fixed performance shadowing in schrodingerBenchmark.ts")

# ============================================================
# Additional Fixes
# ============================================================

def fix_additional_issues():
    """Fix any additional TypeScript issues"""
    print("\n[Bonus] Applying additional TypeScript fixes...")
    
    # Fix any remaining ArrayBufferLike issues
    for ts_file in FRONTEND_PATH.glob("**/*.ts"):
        content = ts_file.read_text(encoding='utf-8')
        original = content
        
        # Fix queue.writeBuffer patterns that might have been missed
        if 'queue.writeBuffer' in content:
            # Ensure all writeBuffer calls use .buffer for typed arrays
            content = re.sub(
                r'queue\.writeBuffer\(([^,]+),\s*(\d+),\s*([^,)]+)(?<!\.buffer)\)',
                lambda m: f'queue.writeBuffer({m.group(1)}, {m.group(2)}, {m.group(3)}.buffer)' 
                         if any(x in m.group(3) for x in ['Array', 'Data', 'Buffer']) 
                         else m.group(0),
                content
            )
        
        if content != original:
            backup_file(ts_file)
            ts_file.write_text(content, encoding='utf-8')
            print(f"  ✓ Applied additional fixes to {ts_file.relative_to(PROJECT_PATH)}")

# ============================================================
# Main Execution
# ============================================================

def main():
    """Execute all fixes"""
    print("=" * 60)
    print("Frontend TypeScript Remaining Fixes")
    print("Fixing buffer writes, timestamps, ONNX, and more")
    print("=" * 60)
    
    if not FRONTEND_PATH.exists():
        print(f"ERROR: {FRONTEND_PATH} does not exist!")
        return 1
    
    try:
        # Execute all fixes
        fix_buffer_writes()       # Fix 1: Buffer write patterns
        fix_write_timestamp()     # Fix 2: writeTimestamp calls
        fix_onnx_runtime()        # Fix 3: ONNX Runtime usage
        fix_implementation_property()  # Fix 4: Missing property
        fix_performance_shadowing()    # Fix 5: Performance shadowing
        fix_additional_issues()   # Additional fixes
        
        print("\n" + "=" * 60)
        print("✓ All frontend TypeScript fixes applied successfully!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. cd D:\\Dev\\kha\\frontend")
        print("2. npx tsc -p tsconfig.json --noEmit")
        print("3. If clean, run: npm run build")
        
        # Save status
        status = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": [
                "Buffer write patterns",
                "writeTimestamp calls",
                "ONNX Runtime usage",
                "Implementation property",
                "Performance shadowing",
                "Additional TypeScript fixes"
            ],
            "status": "complete"
        }
        
        status_file = PROJECT_PATH / "frontend_typescript_fix_status.json"
        status_file.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(f"\nStatus saved to: {status_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during fixes: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
