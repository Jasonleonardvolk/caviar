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

// Fix for the SessionCache type issues
const fileFixes = {
    'frontend/lib/webgpu/kernels/onnxWaveOpRunner.ts': (content) => {
        let fixed = content;
        
        // Update SessionCache.get method to accept string | Uint8Array
        fixed = fixed.replace(
            /async get\(modelPath: string, options\?\: ort\.InferenceSession\.SessionOptions\): Promise<CachedSession>/g,
            'async get(modelPath: string | Uint8Array, options?: ort.InferenceSession.SessionOptions): Promise<CachedSession>'
        );
        
        // Update SessionCache.release method to accept string | Uint8Array
        fixed = fixed.replace(
            /release\(modelPath: string, options\?\: ort\.InferenceSession\.SessionOptions\): void/g,
            'release(modelPath: string | Uint8Array, options?: ort.InferenceSession.SessionOptions): void'
        );
        
        // Update getCacheKey to handle both types
        fixed = fixed.replace(
            /private getCacheKey\(modelPath: string, options\?\: ort\.InferenceSession\.SessionOptions\): string \{/g,
            'private getCacheKey(modelPath: string | Uint8Array, options?: ort.InferenceSession.SessionOptions): string {'
        );
        
        // Update the getCacheKey implementation to handle Uint8Array
        fixed = fixed.replace(
            /const optionsStr = options \? JSON\.stringify\(options\) : 'default';\s*return `\$\{modelPath\}::\$\{optionsStr\}`;/g,
            `const optionsStr = options ? JSON.stringify(options) : 'default';
        // Convert Uint8Array to a stable string key (use length as simple identifier)
        const pathKey = typeof modelPath === 'string' 
            ? modelPath 
            : \`uint8array_\${modelPath.length}_\${modelPath[0]}_\${modelPath[modelPath.length-1]}\`;
        return \`\${pathKey}::\${optionsStr}\`;`
        );
        
        // Update createSession to accept string | Uint8Array
        fixed = fixed.replace(
            /private async createSession\(\s*modelPath: string,/g,
            'private async createSession(\n        modelPath: string | Uint8Array,'
        );
        
        // Fix the createSession method body to handle both types
        fixed = fixed.replace(
            /const session = await ort\.InferenceSession\.create\(modelPath, options\);/g,
            `const modelData = typeof modelPath === 'string'
            ? await fetch(modelPath).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))
            : modelPath;
        const session = await ort.InferenceSession.create(modelData, options);`
        );
        
        // Update the CachedSession interface to handle both types
        fixed = fixed.replace(
            /modelPath: string;/g,
            'modelPath: string | Uint8Array;'
        );
        
        // Fix console.log statements that expect string
        fixed = fixed.replace(
            /console\.log\(`\[ONNX Cache\] Hit for \$\{modelPath\}/g,
            'console.log(`[ONNX Cache] Hit for ${typeof modelPath === "string" ? modelPath : "Uint8Array"}'
        );
        
        fixed = fixed.replace(
            /console\.log\(`\[ONNX Cache\] Miss for \$\{modelPath\}/g,
            'console.log(`[ONNX Cache] Miss for ${typeof modelPath === "string" ? modelPath : "Uint8Array"}'
        );
        
        fixed = fixed.replace(
            /console\.log\(`\[ONNX Cache\] Released \$\{modelPath\}/g,
            'console.log(`[ONNX Cache] Released ${typeof modelPath === "string" ? modelPath : "Uint8Array"}'
        );
        
        fixed = fixed.replace(
            /console\.log\(`\[ONNX Cache\] Evicting \$\{evicted\.modelPath\}/g,
            'console.log(`[ONNX Cache] Evicting ${typeof evicted.modelPath === "string" ? evicted.modelPath : "Uint8Array"}'
        );
        
        return fixed;
    }
};

// Main execution
console.log(`${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║   Fix SessionCache Type Issues         ║`);
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
        console.log(`${colors.blue}   - No changes needed${colors.reset}`);
    }
}

console.log(`\n${colors.cyan}╔════════════════════════════════════════╗`);
console.log(`║              Summary                    ║`);
console.log(`╚════════════════════════════════════════╝${colors.reset}`);
console.log(`${colors.green}✓ Files processed: ${totalFixed}${colors.reset}`);
if (totalErrors > 0) {
    console.log(`${colors.red}✗ Errors: ${totalErrors}${colors.reset}`);
}

console.log(`\n${colors.cyan}Next step:${colors.reset}`);
console.log(`Run: ${colors.yellow}npx tsc -p frontend/tsconfig.json --noEmit${colors.reset}`);
console.log(`\nThis fix updates the SessionCache to handle both string and Uint8Array model paths.\n`);
