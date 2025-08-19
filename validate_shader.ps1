# Quick WGSL Shader Validator using Node.js
# This script validates your propagation.wgsl shader

$shaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

# Create a simple Node.js validator script
$validatorScript = @'
const fs = require('fs');
const path = require('path');

// Simple WGSL syntax checker
function validateWGSL(code) {
    const errors = [];
    const lines = code.split('\n');
    
    // Check for common issues
    const structRegex = /struct\s+\w+\s*{([^}]+)}/g;
    let match;
    
    // Check struct field separators
    while ((match = structRegex.exec(code)) !== null) {
        const fields = match[1];
        if (fields.includes(';') && !fields.includes('};')) {
            errors.push(`Possible struct field separator issue (use commas, not semicolons) near: ${match[0].substring(0, 50)}...`);
        }
    }
    
    // Check for function parameters with storage buffers
    const funcRegex = /@compute.*\n.*fn\s+(\w+)\s*\([^)]+@group[^)]+\)/gs;
    while ((match = funcRegex.exec(code)) !== null) {
        errors.push(`Error: Storage buffers cannot be function parameters in function '${match[1]}'`);
    }
    
    // Check for reserved keywords
    const reserved = ['texture', 'buffer', 'const', 'step', 'image', 'sample', 'patch', 'smooth'];
    const varRegex = /(?:let|var)\s+(\w+)/g;
    while ((match = varRegex.exec(code)) !== null) {
        if (reserved.includes(match[1])) {
            errors.push(`Warning: Using reserved keyword '${match[1]}' as variable name`);
        }
    }
    
    // Check workgroup sizes
    const workgroupRegex = /@workgroup_size\s*\(([^)]+)\)/g;
    while ((match = workgroupRegex.exec(code)) !== null) {
        const sizes = match[1].split(',').map(s => parseInt(s.trim()));
        const total = sizes.reduce((a, b) => a * b, 1);
        if (total > 256) {
            errors.push(`Error: Workgroup size ${match[1]} exceeds limit (${total} > 256)`);
        }
    }
    
    // Check for old syntax
    if (code.includes('[[')) {
        errors.push('Error: Found old [[binding]] syntax, use @binding instead');
    }
    
    return errors;
}

// Read and validate the shader
const shaderPath = process.argv[2];
if (!shaderPath) {
    console.error('Usage: node validate.js <shader-path>');
    process.exit(1);
}

try {
    const code = fs.readFileSync(shaderPath, 'utf8');
    console.log(`Validating: ${shaderPath}`);
    console.log('=' .repeat(60));
    
    const errors = validateWGSL(code);
    
    if (errors.length === 0) {
        console.log('✅ No obvious WGSL syntax errors found!');
        console.log('\nNote: This is a basic check. For complete validation, use:');
        console.log('- Chrome DevTools with WebGPU');
        console.log('- naga validator (cargo install naga-cli)');
        console.log('- wgsl-analyzer');
    } else {
        console.log('❌ Found potential issues:\n');
        errors.forEach((err, i) => {
            console.log(`${i + 1}. ${err}`);
        });
    }
} catch (err) {
    console.error(`Failed to read shader: ${err.message}`);
}
'@

# Save the validator script
$validatorScript | Out-File -FilePath "validate_wgsl.js" -Encoding UTF8

# Run the validator
Write-Host "Running WGSL validator..." -ForegroundColor Cyan
node validate_wgsl.js $shaderPath

# Cleanup
Remove-Item "validate_wgsl.js" -ErrorAction SilentlyContinue

Write-Host "`nFor more thorough validation, you can:" -ForegroundColor Yellow
Write-Host "1. Install Rust and run: cargo install naga-cli" -ForegroundColor Yellow
Write-Host "2. Then run: naga `"$shaderPath`" --verbose" -ForegroundColor Yellow
