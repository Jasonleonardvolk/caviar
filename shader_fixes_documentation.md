# WGSL Shader Fixes Documentation

## Overview
This document details the fixes applied to resolve WGSL validation errors in three shader files for the holographic rendering system.

## Fixed Shaders

### 1. propagation.wgsl
**Location:** `${IRIS_ROOT}\frontend\lib\webgpu\shaders\propagation.wgsl`

**Issues Fixed:**
- **Storage buffer in function parameters:** The `multiview_buffer` storage texture was incorrectly declared inside the `prepare_for_multiview` function parameter list.
- **Read-only texture being read:** The `output_field` texture was declared as write-only but was being read in the `post_process` function.

**Solutions:**
- Moved `@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;` to module level with other storage declarations
- Fixed the `prepare_for_multiview` function signature to remove the storage parameter
- Changed `post_process`, `visualize_magnitude`, and `visualize_phase` functions to read from `frequency_domain` instead of `output_field`
- Added mip level parameter (0) to all textureLoad calls

### 2. lenticularInterlace.wgsl
**Location:** `${IRIS_ROOT}\frontend\lib\webgpu\shaders\lenticularInterlace.wgsl`

**Issues Fixed:**
- **Missing mip level parameter:** textureLoad calls were missing the required third parameter (mip level)

**Solutions:**
- Added mip level parameter `0` to textureLoad calls on lines 381 and 389 in the `cs_edge_enhance` function:
  - `textureLoad(temp_buffer, coord)` → `textureLoad(temp_buffer, coord, 0)`
  - `textureLoad(temp_buffer, sample_coord)` → `textureLoad(temp_buffer, sample_coord, 0)`

### 3. velocityField.wgsl
**Location:** `${IRIS_ROOT}\frontend\lib\webgpu\shaders\velocityField.wgsl`

**Issues Fixed:**
- **Storage buffers in function parameters:** Both `particles` storage buffer and `flow_vis_out` storage texture were incorrectly declared inside function parameter lists

**Solutions:**
- Moved `@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;` to module level
- Moved `@group(0) @binding(6) var flow_vis_out: texture_storage_2d<rgba8unorm, write>;` to module level
- Fixed the `advect_particles` and `visualize_flow` function signatures to remove storage parameters
- Added mip level parameter (0) to all textureLoad calls

## WGSL Best Practices Applied

1. **Storage declarations must be at module level:** All `@group` and `@binding` declarations must be at the module (global) scope, not inside function parameters.

2. **textureLoad requires mip level:** The `textureLoad` function for texture_2d requires three parameters:
   - Texture handle
   - Coordinate (vec2<i32> or vec2<u32>)
   - Mip level (i32 or u32)

3. **Write-only textures cannot be read:** Textures declared with `texture_storage_2d<format, write>` can only be written to using `textureStore`, not read from using `textureLoad`.

## Validation
Run the validation script to verify all fixes:
```powershell
.\validate_shader_fixes.ps1
```

## File Locations
- Fixed shaders: `${IRIS_ROOT}\frontend\lib\webgpu\shaders\`
- Validation script: `${IRIS_ROOT}\validate_shader_fixes.ps1`
- This documentation: `${IRIS_ROOT}\shader_fixes_documentation.md`