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
        
        // Fix InferenceSession.create to properly handle the modelPath
        // We need to ensure it's always a Uint8Array for the create method
        // Replace the problematic ternary with a proper conversion
        fixed = fixed.replace(
            /const testSession = await ort\.InferenceSession\.create\(\s*typeof this\.config\.modelPath === 'string'[\s\S]*?\),\s*\{/g,
            `const modelData = typeof this.config.modelPath === 'string'
                        ? await fetch(this.config.modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
                        : this.config.modelPath;
                const testSession = await ort.InferenceSession.create(
                    modelData,
                    {`
        );
        
        // Also fix any other similar patterns
        fixed = fixed.replace(
            /await ort\.InferenceSession\.create\(\s*typeof this\.config\.modelPath[\s\S]*?this\.config\.modelPath,/g,
            `const modelData = typeof this.config.modelPath === 'string'
                    ? await fetch(this.config.modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
                    : this.config.modelPath;
                await ort.InferenceSession.create(
                    modelData,`
        );
        
        return fixed;
    },
    
    'frontend/lib/webgpu/minimalGPUTest.ts': (content) => {
        let fixed = content;
        
        // Fix the output variable type checking - add null check
        // Find the line with output.slice and add a type guard
        fixed = fixed.replace(
            /const output = await fft\.execute\(input\);/g,
            'const output = await fft.execute(input);'
        );
        
        // Fix the outputSample line by adding a null check
        fixed = fixed.replace(
            /outputSample: output\.slice\(0, 4\)/g,
            'outputSample: output ? output.slice(0, 4) : []'
        );
        
        // Fix the length check
        fixed = fixed.replace(
            /if \(output\.length !== input\.length\) \{/g,
            'if (!output || output.length !== input.length) {'
        );
        
        // Fix the error message 
        fixed = fixed.replace(
            /throw new Error\(`Output length mismatch: \$\{output\.length\} vs \$\{input\.length\}`\);/g,
            'throw new Error(`Output length mismatch: ${output?.length || 0} vs ${input.length}`);'
        );
        
        return fixed;
    }
};

// Main execution
console.log(`${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║  Final TypeScript Error Fixes          ║`);
console.log(`╚════════════════════════════════════════╝${colors.reset}\n`);

let totalFixed = 0;
let totalErrors = 0;

for (const [relativePath, fixFunction] of Object.entries(fileFixes)) {
    const fullPath = path.join(process.cwd(), relativePath);
    console.log(`${colors.yellow}📁 ${relativePath}${colors.reset}`);
    
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
        console.log(`${colors.blue}   - No changes needed${colors.reset}`);
    }
}

console.log(`\n${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║              Summary                    ║`);
console.log(`╚════════════════════════════════════════╝${colors.reset}`);
console.log(`${colors.green}✓ Fixed: ${totalFixed} files${colors.reset}`);
if (totalErrors > 0) {
    console.log(`${colors.red}✗ Errors: ${totalErrors} files${colors.reset}`);
}

console.log(`\n${colors.cyan}Next step:${colors.reset}`);
console.log(`Run: ${colors.yellow}npx tsc -p frontend/tsconfig.json --noEmit${colors.reset}`);
console.log(`\nThis should now show 0 errors! 🎉\n`);
