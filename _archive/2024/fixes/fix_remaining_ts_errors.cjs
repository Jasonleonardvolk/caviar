const fs = require('fs');
const path = require('path');

// Color codes for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    red: '\x1b[31m',
    blue: '\x1b[34m'
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

// Remaining fixes for the 5 errors
const fileFixes = {
    'frontend/lib/webgpu/kernels/onnxWaveOpRunner.ts': (content) => {
        let fixed = content;
        
        // Fix powerPreference property (line 220)
        // Remove powerPreference as it's not supported
        fixed = fixed.replace(
            /powerPreference:\s*['"]high-performance['"]/g,
            '// powerPreference: \'high-performance\' // Not supported in current version'
        );
        
        // Fix InferenceSession.create - ensure we handle both string path and Uint8Array
        // Need to check if modelPath is already a Uint8Array or needs fetching
        fixed = fixed.replace(
            /await ort\.InferenceSession\.create\(\s*await fetch\(this\.config\.modelPath\)\.then\(r => r\.arrayBuffer\(\)\)\.then\(b => new Uint8Array\(b\)\),/g,
            `await ort.InferenceSession.create(
                    typeof this.config.modelPath === 'string' 
                        ? await fetch(this.config.modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
                        : this.config.modelPath,`
        );
        
        return fixed;
    },
    
    'frontend/lib/webgpu/minimalGPUTest.ts': (content) => {
        let fixed = content;
        
        // Fix import paths - these should be relative to current file location
        // minimalGPUTest.ts is in frontend/lib/webgpu/
        // So we need to import from the same directory
        fixed = fixed.replace(
            /from ['"]\.\.\/frontend\/lib\/webgpu\/fftCompute['"]/g,
            'from \'./fftCompute\''
        );
        
        fixed = fixed.replace(
            /from ['"]\.\.\/frontend\/lib\/webgpu\/fftDispatchValidator['"]/g,
            'from \'./fftDispatchValidator\''
        );
        
        return fixed;
    },
    
    'frontend/lib/webgpu/utils/bufferHelpers.ts': (content) => {
        let fixed = content;
        
        // Fix SharedArrayBuffer to ArrayBuffer conversion
        // Need to go through unknown first for type safety
        fixed = fixed.replace(
            /device\.queue\.writeBuffer\(buffer,\s*bufferOffset,\s*new Uint8Array\(data\)\.buffer as ArrayBuffer\);/g,
            'device.queue.writeBuffer(buffer, bufferOffset, new Uint8Array(data).buffer as unknown as ArrayBuffer);'
        );
        
        return fixed;
    }
};

// Main execution
console.log(`${colors.blue}========================================`);
console.log(`Fixing Remaining TypeScript Errors`);
console.log(`========================================${colors.reset}\n`);

let totalFixed = 0;
let totalErrors = 0;

for (const [relativePath, fixFunction] of Object.entries(fileFixes)) {
    const fullPath = path.join(process.cwd(), relativePath);
    console.log(`${colors.yellow}Processing: ${relativePath}${colors.reset}`);
    
    const content = readFile(fullPath);
    if (!content) {
        console.log(`${colors.red}  ✗ Could not read file${colors.reset}`);
        totalErrors++;
        continue;
    }
    
    const fixed = fixFunction(content);
    
    if (fixed !== content) {
        if (writeFile(fullPath, fixed)) {
            console.log(`${colors.green}  ✓ Fixed successfully${colors.reset}`);
            totalFixed++;
        } else {
            console.log(`${colors.red}  ✗ Failed to write file${colors.reset}`);
            totalErrors++;
        }
    } else {
        console.log(`  - No changes needed`);
    }
}

console.log(`\n${colors.blue}========================================${colors.reset}`);
console.log(`${colors.green}Fixed: ${totalFixed} files${colors.reset}`);
if (totalErrors > 0) {
    console.log(`${colors.red}Errors: ${totalErrors} files${colors.reset}`);
}
console.log(`${colors.blue}========================================${colors.reset}\n`);

console.log('To verify all fixes are complete, run:');
console.log(`${colors.yellow}npx tsc -p frontend/tsconfig.json --noEmit${colors.reset}\n`);
