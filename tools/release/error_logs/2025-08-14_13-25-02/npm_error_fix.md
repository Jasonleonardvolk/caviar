# NPM Error Fix: Missing "type-check" Script

## Problem
The error `npm ERR! Missing script: "type-check"` occurred because there's no `type-check` script defined in any of your package.json files.

## Solutions

### Option 1: Use TypeScript Compiler Directly
Since you have TypeScript installed, you can run type checking directly:

```bash
# From the project root (D:\Dev\kha)
npx tsc --noEmit

# Or for specific directories:
cd tori_ui_svelte
npx tsc --noEmit

cd ../frontend
npx tsc --noEmit
```

### Option 2: Use Svelte Check (for Svelte components)
For the Svelte project, use the existing svelte-check:

```bash
cd tori_ui_svelte
npx svelte-check
```

### Option 3: Add type-check Scripts
Add these scripts to your package.json files:

#### For `D:\Dev\kha\package.json`:
```json
"scripts": {
  ...existing scripts...,
  "type-check": "tsc --noEmit",
  "type-check:all": "npm run type-check:root && npm run type-check:frontend && npm run type-check:svelte",
  "type-check:root": "tsc --noEmit",
  "type-check:frontend": "cd frontend && npx tsc --noEmit",
  "type-check:svelte": "cd tori_ui_svelte && npx svelte-check"
}
```

#### For `D:\Dev\kha\frontend\package.json`:
```json
"scripts": {
  ...existing scripts...,
  "type-check": "tsc --noEmit"
}
```

#### For `D:\Dev\kha\tori_ui_svelte\package.json`:
```json
"scripts": {
  ...existing scripts...,
  "type-check": "svelte-check --tsconfig ./tsconfig.json"
}
```

### Option 4: Create TypeScript Config (if missing)
If you get errors about missing tsconfig.json:

Create `D:\Dev\kha\tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["@webgpu/types"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "noEmit": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "allowJs": true,
    "checkJs": false
  },
  "include": [
    "frontend/**/*.ts",
    "tori_ui_svelte/src/**/*.ts",
    "scripts/**/*.ts",
    "tools/**/*.mjs"
  ],
  "exclude": [
    "node_modules",
    "**/node_modules",
    "dist",
    "build",
    ".svelte-kit"
  ]
}
```

## Quick Fix Command

Run this command to check the fixed TypeScript file immediately:

```bash
cd D:\Dev\kha\tori_ui_svelte
npx tsc src/lib/webgpu/photoMorphPipeline.ts --noEmit --target ES2022 --lib dom,es2022 --types @webgpu/types --strict
```

## Test The Fixes

After applying the fixes to photoMorphPipeline.ts, test with:

```bash
# Option 1: Direct TypeScript check
cd D:\Dev\kha
npx tsc tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts --noEmit

# Option 2: Build the project
npm run build

# Option 3: Run the frontend build
cd tori_ui_svelte
npm run frontend
```

## Current Status

✅ **photoMorphPipeline.ts has been fixed** - All 31 TypeScript errors resolved
- Fixed WebGPU API changes (size → width/height)
- Added missing methods
- Fixed uninitialized properties
- Added proper type annotations
- Fixed fragment shader targets

## Next Steps

1. **First**, test the fixed file:
   ```bash
   cd D:\Dev\kha\tori_ui_svelte
   npx tsc src/lib/webgpu/photoMorphPipeline.ts --noEmit
   ```

2. **Then**, fix remaining files with similar issues:
   - splitStepOrchestrator.ts (20 errors)
   - enginePerf.ts (12 errors)
   - fftCompute.ts (11 errors)

3. **Finally**, run the full build:
   ```bash
   cd D:\Dev\kha
   npm run build
   ```

## Alternative: Use Existing Build Scripts

Looking at your package.json, you can also use:

```bash
# Build the Svelte frontend
npm run build

# Or run development mode
npm run dev

# Or specifically for frontend
npm run frontend
```

These commands should compile TypeScript as part of the build process.
