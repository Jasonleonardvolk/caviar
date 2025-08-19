const fs = require('fs');
const path = require('path');

// Color codes for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    red: '\x1b[31m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
};

// Helper to read file
function readFile(filePath) {
    try {
        return fs.readFileSync(filePath, 'utf8');
    } catch (error) {
        console.error(`${colors.red}Error reading ${filePath}: ${error.message}${colors.reset}`);
        return null;
    }
}

// Helper to write file
function writeFile(filePath, content) {
    try {
        fs.writeFileSync(filePath, content, 'utf8');
        return true;
    } catch (error) {
        console.error(`${colors.red}Error writing ${filePath}: ${error.message}${colors.reset}`);
        return false;
    }
}

// Final fixes for the last 4 errors
const fileFixes = {
    'frontend/lib/webgpu/kernels/onnxWaveOpRunner.ts': (content) => {
        let fixed = content;
        
        // Replace the problematic InferenceSession.create call with proper type handling
        // Find the pattern and extract the modelPath conversion before the create call
        
        // First pattern - in selectBackend method
        fixed = fixed.replace(
            /const testSession = await ort\.InferenceSession\.create\(\s*typeof this\.config\.modelPath === 'string'[\s\S]*?:\s*this\.config\.modelPath,/gm,
            `// Load model data first
                const modelData: Uint8Array = typeof this.config.modelPath === 'string'
                    ? await fetch(this.config.modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
                    : this.config.modelPath as Uint8Array;
                const testSession = await ort.InferenceSession.create(
                    modelData,`
        );
        
        // Second pattern - if there's another create call that needs fixing
        fixed = fixed.replace(
            /await ort\.InferenceSession\.create\(\s*typeof this\.config\.modelPath[\s\S]*?:\s*this\.config\.modelPath,/gm,
            `// Load model data
                const modelData: Uint8Array = typeof this.config.modelPath === 'string'
                    ? await fetch(this.config.modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
                    : this.config.modelPath as Uint8Array;
                await ort.InferenceSession.create(
                    modelData,`
        );
        
        // Also update the type of modelPath in OnnxConfig to allow Uint8Array
        fixed = fixed.replace(
            /modelPath: string;/g,
            'modelPath: string | Uint8Array;'
        );
        
        return fixed;
    },
    
    'frontend/lib/webgpu/minimalGPUTest.ts': (content) => {
        let fixed = content;
        
        // Fix the FFT execute return type handling
        // The output might be void, so we need proper type guards
        
        // First, let's fix the type of output by adding a check
        fixed = fixed.replace(
            /const output = await fft\.execute\(input\);/g,
            'const output = await fft.execute(input) || new Float32Array(0);'
        );
        
        // Since we're ensuring output is always defined now, the other lines should work
        // But let's add safety checks anyway
        fixed = fixed.replace(
            /outputSample: output\.slice\(0, 4\)/g,
            'outputSample: output ? output.slice(0, 4) : []'
        );
        
        // Update the length check to handle the case where output might be empty
        fixed = fixed.replace(
            /if \(output\.length !== input\.length\) \{/g,
            'if (!output || output.length !== input.length) {'
        );
        
        // Fix the error message to handle undefined output
        fixed = fixed.replace(
            /throw new Error\(`Output length mismatch: \$\{output\.length\} vs \$\{input\.length\}`\);/g,
            'throw new Error(`Output length mismatch: ${output?.length || 0} vs ${input.length}`);'
        );
        
        return fixed;
    }
};

// Main execution
console.log(`${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║    Final TypeScript Error Fixes        ║`);
console.log(`║         (Last 4 Errors)                ║`);
console.log(`╚════════════════════════════════════════╝${colors.reset}\n`);

let totalFixed = 0;
let totalErrors = 0;

for (const [relativePath, fixFunction] of Object.entries(fileFixes)) {
    const fullPath = path.join(process.cwd(), relativePath);
    console.log(`${colors.yellow}📁 Processing: ${relativePath}${colors.reset}`);
    
    const content = readFile(fullPath);
    if (!content) {
        console.log(`${colors.red}   ✗ Could not read file${colors.reset}`);
        totalErrors++;
        continue;
    }
    
    const fixed = fixFunction(content);
    
    if (fixed !== content) {
        if (writeFile(fullPath, fixed)) {
            console.log(`${colors.green}   ✓ Fixed and saved${colors.reset}`);
            totalFixed++;
        } else {
            console.log(`${colors.red}   ✗ Failed to write${colors.reset}`);
            totalErrors++;
        }
    } else {
        console.log(`${colors.blue}   - No changes applied (may need manual review)${colors.reset}`);
    }
}

console.log(`\n${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║              Summary                    ║`);
console.log(`╚════════════════════════════════════════╝${colors.reset}`);
console.log(`${colors.green}✓ Files processed: ${totalFixed}${colors.reset}`);
if (totalErrors > 0) {
    console.log(`${colors.red}✗ Errors: ${totalErrors}${colors.reset}`);
}

console.log(`\n${colors.cyan}Final step:${colors.reset}`);
console.log(`Run: ${colors.yellow}npx tsc -p frontend/tsconfig.json --noEmit${colors.reset}`);
console.log(`\nIf errors persist, they may need manual review.`);
console.log(`The script has applied type fixes for:`);
console.log(`  • ONNX model loading with proper Uint8Array conversion`);
console.log(`  • FFT output handling with null safety checks\n`);
