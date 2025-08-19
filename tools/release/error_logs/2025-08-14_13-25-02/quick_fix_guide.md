# Quick Fix Action List
Priority fixes for build errors (2025-08-14_13-25-02)

## CRITICAL - Fix These First

### 1. WebGPU API Updates
Replace in all affected files:
```typescript
// OLD: encoder.writeTimestamp(querySet, index)
// NEW: Remove or use alternative performance monitoring

// OLD: texture.size
// NEW: texture.width, texture.height, texture.depthOrArrayLayers

// OLD: fragment: { module: shader, entryPoint: 'main' }
// NEW: fragment: { module: shader, entryPoint: 'main', targets: [{ format: 'bgra8unorm' }] }
```

### 2. Missing Modules
Check if these files exist:
- [ ] frontend/lib/webgpu/holographicEngine.ts
- [ ] frontend/types/renderer.ts
- [ ] frontend/lib/webgpu/fftCompute.ts
- [ ] frontend/lib/webgpu/fftDispatchValidator.ts

### 3. Property Initialization
Add to constructors or use definite assignment:
```typescript
// Option 1: Initialize in constructor
constructor() {
    this.fftModule = /* initialize */;
}

// Option 2: Definite assignment assertion
private fftModule!: GPUShaderModule;
```

### 4. Fix Private Property Access
In WebGPUEngine class:
```typescript
// Change from:
private device: GPUDevice;

// To:
protected device: GPUDevice;
// OR add getter:
public getDevice(): GPUDevice { return this.device; }
```

### 5. ONNX Runtime Fix
Update package.json:
```json
"onnxruntime-web": "^1.17.0"
```

## Files to Fix (In Order)

1. **tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts** (31 errors)
   - Fix GPUTexture.size references
   - Initialize properties
   - Add missing methods

2. **frontend/lib/webgpu/kernels/splitStepOrchestrator.ts** (20 errors)
   - Remove writeTimestamp calls
   - Initialize properties

3. **frontend/lib/webgpu/enginePerf.ts** (12 errors)
   - Fix device property access

4. **frontend/lib/webgpu/fftCompute.ts** (11 errors)
   - Remove writeTimestamp
   - Initialize properties

5. **frontend/lib/webgpu/kernels/schrodingerKernelRegistry.ts** (9 errors)
   - Add implementation property

## Test Commands
After fixing, run:
```bash
npm run build
npm run type-check
```

## Need Help?
- WebGPU API Changes: https://gpuweb.github.io/gpuweb/
- TypeScript Strict Mode: https://www.typescriptlang.org/tsconfig#strict
- ONNX Runtime Docs: https://onnxruntime.ai/docs/
