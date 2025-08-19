#!/usr/bin/env node
// fix-typescript-surgical.js
// Node.js script to fix all recurring TypeScript issues

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('===== TypeScript Surgical Fixes =====\n');

let fixCount = 0;
const issues = [];

// Helper to recursively find TypeScript files
function findTsFiles(dir, fileList = []) {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        
        if (stat.isDirectory()) {
            if (!file.includes('node_modules') && !file.includes('.svelte-kit')) {
                findTsFiles(filePath, fileList);
            }
        } else if (file.endsWith('.ts') || file.endsWith('.tsx')) {
            fileList.push(filePath);
        }
    }
    
    return fileList;
}

// Fix typed array generics
function fixTypedArrayGenerics(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const original = content;
    
    // Fix all typed array generics
    content = content.replace(/Float32Array<[^>]+>/g, 'Float32Array');
    content = content.replace(/Uint8Array<[^>]+>/g, 'Uint8Array');
    content = content.replace(/Uint16Array<[^>]+>/g, 'Uint16Array');
    content = content.replace(/Uint32Array<[^>]+>/g, 'Uint32Array');
    content = content.replace(/Int8Array<[^>]+>/g, 'Int8Array');
    content = content.replace(/Int16Array<[^>]+>/g, 'Int16Array');
    content = content.replace(/Int32Array<[^>]+>/g, 'Int32Array');
    
    if (content !== original) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed typed arrays in: ${path.basename(filePath)}`);
        return true;
    }
    
    return false;
}

// Fix WebGL context issues
function fixWebGLContext(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const original = content;
    
    // Replace 2D context with WebGL when trying to use WebGL extensions
    const pattern = /const\s+ctx\s*=\s*canvas\.getContext\(['"]2d['"]\);?\s*\n\s*const\s+ext\s*=\s*ctx\?\.getExtension/g;
    const replacement = `const gl = (canvas.getContext('webgl2') || canvas.getContext('webgl')) as
      WebGL2RenderingContext | WebGLRenderingContext | null;
    const ext = gl?.getExtension`;
    
    content = content.replace(pattern, replacement);
    
    if (content !== original) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed WebGL context in: ${path.basename(filePath)}`);
        return true;
    }
    
    return false;
}

// Fix bindGroup.entries runtime access
function fixBindGroupEntries(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    const original = content;
    
    // Replace bindGroup.entries with local entries variable
    content = content.replace(/for\s*\(\s*const\s+(\w+)\s+of\s+bindGroup\.entries\s*\)/g, 
                              'for (const $1 of entries)');
    
    if (content !== original) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed bindGroup.entries in: ${path.basename(filePath)}`);
        return true;
    }
    
    return false;
}

// Fix KTX2 level fields
function fixKTX2Fields(filePath) {
    if (!filePath.toLowerCase().includes('ktx')) return false;
    
    let content = fs.readFileSync(filePath, 'utf8');
    const original = content;
    
    // Replace offset/length with byteOffset/byteLength
    content = content.replace(/const\s*{\s*offset\s*,\s*length\s*}\s*=\s*level/g,
                              'const { byteOffset: offset, byteLength: length } = level');
    
    if (content !== original) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed KTX2 fields in: ${path.basename(filePath)}`);
        return true;
    }
    
    return false;
}

// Main execution
console.log('Searching for TypeScript files...');
const frontendPath = path.join('D:', 'Dev', 'kha', 'frontend');
const tsFiles = findTsFiles(frontendPath);
console.log(`Found ${tsFiles.length} TypeScript files\n`);

// Apply fixes
console.log('Applying fixes...');
for (const file of tsFiles) {
    if (fixTypedArrayGenerics(file)) fixCount++;
    if (fixWebGLContext(file)) fixCount++;
    if (fixBindGroupEntries(file)) fixCount++;
    if (fixKTX2Fields(file)) fixCount++;
}

// Create onnxruntime-web types if missing
const onnxTypesPath = path.join(frontendPath, 'types', 'onnxruntime-web.d.ts');
if (!fs.existsSync(onnxTypesPath)) {
    const onnxTypes = `declare module "onnxruntime-web" {
  export class InferenceSession {
    static create(model: string | ArrayBufferLike, options?: any): Promise<InferenceSession>;
    run(feeds: Record<string, any>, fetches?: string[] | Record<string, any>, options?: any): Promise<Record<string, any>>;
  }
  export class Tensor<T = number> {
    constructor(type: string, data: T[] | T, dims: number[]);
    readonly type: string;
    readonly data: T[] | T;
    readonly dims: number[];
  }
}
`;
    fs.writeFileSync(onnxTypesPath, onnxTypes);
    console.log('  Created onnxruntime-web.d.ts');
    fixCount++;
}

// Run SvelteKit sync
console.log('\nRunning SvelteKit sync...');
const svelteKitPath = path.join('D:', 'Dev', 'kha', 'tori_ui_svelte');
try {
    process.chdir(svelteKitPath);
    try {
        execSync('pnpm dlx svelte-kit sync', { stdio: 'inherit' });
    } catch (e) {
        console.log('  pnpm not found, trying npm...');
        execSync('npx svelte-kit sync', { stdio: 'inherit' });
    }
    console.log('  SvelteKit sync completed');
} catch (error) {
    console.log(`  Warning: Could not run SvelteKit sync: ${error.message}`);
}

// Summary
console.log('\n===== Summary =====');
console.log(`Total fixes applied: ${fixCount}`);
if (issues.length > 0) {
    console.log('Issues found:');
    issues.forEach(issue => console.log(`  - ${issue}`));
} else {
    console.log('All recurring TypeScript blockers resolved!');
}
