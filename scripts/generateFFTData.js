// scripts/generateFFTData.js
// Build-time script to generate precomputed FFT data

const fs = require('fs');
const path = require('path');

// Common FFT sizes
const SIZES = [256, 512, 1024, 2048, 4096];

function generateTwiddleFactors(size, direction) {
    const sign = direction === 'forward' ? -1 : 1;
    const stages = Math.log2(size);
    const data = [];
    
    for (let stage = 0; stage < stages; stage++) {
        const stageSize = 1 << (stage + 1);
        const halfStageSize = stageSize >> 1;
        
        for (let k = 0; k < halfStageSize; k++) {
            const angle = sign * 2 * Math.PI * k / stageSize;
            data.push(Math.cos(angle), Math.sin(angle));
        }
    }
    
    return data;
}

function generateBitReversalIndices(size) {
    const bits = Math.log2(size);
    const indices = [];
    
    for (let i = 0; i < size; i++) {
        let reversed = 0;
        let temp = i;
        
        for (let j = 0; j < bits; j++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        indices.push(reversed);
    }
    
    return indices;
}

function generateStaticData() {
    let code = `// Auto-generated FFT precomputed data
// Generated on: ${new Date().toISOString()}

export const PRECOMPUTED_TWIDDLES: Record<string, Float32Array> = {
`;
    
    // Generate twiddle factors
    for (const size of SIZES) {
        for (const direction of ['forward', 'inverse']) {
            const key = `${size}_${direction}`;
            const data = generateTwiddleFactors(size, direction);
            
            // Format array data with line breaks for readability
            const formattedData = [];
            for (let i = 0; i < data.length; i += 8) {
                const chunk = data.slice(i, i + 8).map(n => n.toFixed(6)).join(', ');
                formattedData.push(`        ${chunk}`);
            }
            
            code += `    '${key}': new Float32Array([
${formattedData.join(',\n')}
    ]),\n`;
        }
    }
    
    code += `};

export const PRECOMPUTED_BIT_REVERSAL: Record<number, Uint32Array> = {
`;
    
    // Generate bit reversal indices
    for (const size of SIZES) {
        const data = generateBitReversalIndices(size);
        
        // Format array data
        const formattedData = [];
        for (let i = 0; i < data.length; i += 16) {
            const chunk = data.slice(i, i + 16).join(', ');
            formattedData.push(`        ${chunk}`);
        }
        
        code += `    ${size}: new Uint32Array([
${formattedData.join(',\n')}
    ]),\n`;
    }
    
    code += `};

// Helper to get precomputed data with fallback
export function getPrecomputedTwiddles(size: number, direction: 'forward' | 'inverse'): Float32Array | null {
    const key = \`\${size}_\${direction}\`;
    return PRECOMPUTED_TWIDDLES[key] || null;
}

export function getPrecomputedBitReversal(size: number): Uint32Array | null {
    return PRECOMPUTED_BIT_REVERSAL[size] || null;
}
`;
    
    return code;
}

// Main execution
function main() {
    const outputDir = path.join(__dirname, '../frontend/lib/webgpu/generated');
    
    // Create output directory
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Generate and write the file
    const code = generateStaticData();
    const outputPath = path.join(outputDir, 'fftPrecomputedData.ts');
    
    fs.writeFileSync(outputPath, code);
    console.log(`Generated FFT precomputed data at: ${outputPath}`);
    
    // Calculate sizes
    let totalSize = 0;
    for (const size of SIZES) {
        const twiddleSize = size * Math.log2(size) * 2 * 4; // complex f32
        const bitRevSize = size * 4; // u32
        totalSize += twiddleSize * 2 + bitRevSize; // forward + inverse
    }
    
    console.log(`Total precomputed data size: ${(totalSize / 1024).toFixed(2)} KB`);
}

if (require.main === module) {
    main();
}

module.exports = { generateTwiddleFactors, generateBitReversalIndices };
