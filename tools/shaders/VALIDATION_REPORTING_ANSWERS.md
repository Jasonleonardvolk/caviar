# ANSWERS FOR VALIDATION REPORTING SETUP

## 1. **Exact output path**: 
YES - Use `${IRIS_ROOT}\tools\shaders\reports\`

Keep everything there with timestamped filenames:
- `shader_validation_YYYY-MM-DDTHH-mm-ss.json` (raw validation data)
- `shader_validation_YYYY-MM-DDTHH-mm-ss.junit.xml` (JUnit format)
- `shader_validation_YYYY-MM-DDTHH-mm-ss_summary.txt` (human-readable)
- `shader_validation_latest.json` (symlink/copy to most recent)

## 2. **Validator command integration**:
YES - Hook it into the existing pipeline AFTER shader_quality_gate_v2.mjs runs.

Create a wrapper script that:
1. Runs the validator
2. Captures output
3. Generates all three report formats
4. Shows summary in console
5. Returns appropriate exit code

## 3. **Validation status per file**:
YES - Absolutely include per-file status! Structure it like:

```
=== SHADER VALIDATION SUMMARY ===
Timestamp: 2025-08-08T12:55:00
Total Files: 42
Status: 38 PASSED | 3 FAILED | 1 WARNING

FAILED (3):
  ✗ wavefieldEncoder.wgsl
    - Line 44: Uniform buffer array stride violation (array<f32>)
    - Line 50: Uniform buffer array stride violation (array<vec2<f32>>)
    
  ✗ particleCompute.wgsl  
    - Line 67: textureLoad missing mip level argument
    
  ✗ raymarch.wgsl
    - Line 234: Workgroup size 1024 exceeds device limit 256

WARNING (1):
  ⚠ blend.wgsl
    - Line 12: Deprecated texture2D usage (use texture2d)

PASSED (38):
  ✓ depth.wgsl
  ✓ tonemap.wgsl
  ✓ ... (remaining files)

=== TOP ISSUES TO FIX ===
1. Uniform buffer array stride (2 files) - Convert to storage buffers
2. Missing mip levels (1 file) - Add ,0 to textureLoad calls
3. Workgroup size limits (1 file) - Reduce to 256 or less
```

## ADDITIONAL REQUIREMENTS:

### A. Error Categories
Group errors by type for easier fixing:
- Uniform/Storage buffer issues
- Texture sampling issues  
- Workgroup/compute issues
- Syntax/semantic errors
- Deprecated features

### B. Fix Suggestions
For each error type, include the fix:
- "Uniform array stride" → "Convert to storage buffer or use vec4 padding"
- "Missing mip level" → "Add ,0 as last parameter to textureLoad"
- "Workgroup exceeds limit" → "Reduce to @workgroup_size(8,8,1) or less"

### C. Exit Codes
- 0 = All passed
- 1 = Has warnings but no errors
- 2 = Has errors
- 3 = Validator itself failed

### D. Color Output (if terminal supports)
- Red for FAILED
- Yellow for WARNING  
- Green for PASSED
- Cyan for fix suggestions

This gives us actionable, scannable output that directly points to what needs fixing!