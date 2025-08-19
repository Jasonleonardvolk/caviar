// scripts/genFftPrecomputed.ts
// Build-time FFT data generator with optimizations
// Usage: npm run generate-fft -- --sizes=256,512,1024 --format=minified --split

import * as fs from 'fs';
import * as path from 'path';
import { Command } from 'commander';
import * as prettier from 'prettier';

// Configuration interface
interface GeneratorConfig {
    sizes: number[];
    outputDir: string;
    format: 'readable' | 'minified';
    split: boolean;
    includeTimestamp: boolean;
    generateBinary: boolean;
}

// Default configuration
const DEFAULT_CONFIG: GeneratorConfig = {
    sizes: [256, 512, 1024, 2048, 4096],
    outputDir: '../frontend/lib/webgpu/generated',
    format: 'readable',
    split: false,
    includeTimestamp: process.env.NODE_ENV !== 'production',
    generateBinary: false
};

// Load configuration from package.json if available
function loadConfig(): Partial<GeneratorConfig> {
    try {
        const packagePath = path.join(__dirname, '../package.json');
        const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf-8'));
        return packageJson.fftGenerator || {};
    } catch {
        return {};
    }
}

// Parse command line arguments
function parseArgs(): GeneratorConfig {
    const program = new Command();
    const packageConfig = loadConfig();
    
    program
        .name('generate-fft')
        .description('Generate precomputed FFT data for WebGPU')
        .option('--sizes <sizes>', 'Comma-separated list of FFT sizes', (value) => 
            value.split(',').map(s => parseInt(s.trim())))
        .option('--output <dir>', 'Output directory', packageConfig.outputDir)
        .option('--format <type>', 'Output format (readable|minified)', packageConfig.format || 'readable')
        .option('--split', 'Split into separate files per size', packageConfig.split || false)
        .option('--no-timestamp', 'Exclude timestamp from generated files')
        .option('--binary', 'Also generate binary files', false)
        .parse();
    
    const options = program.opts();
    
    return {
        sizes: options.sizes || packageConfig.sizes || DEFAULT_CONFIG.sizes,
        outputDir: path.resolve(__dirname, options.output || packageConfig.outputDir || DEFAULT_CONFIG.outputDir),
        format: options.format || DEFAULT_CONFIG.format,
        split: options.split || DEFAULT_CONFIG.split,
        includeTimestamp: options.timestamp !== false,
        generateBinary: options.binary || DEFAULT_CONFIG.generateBinary
    };
}

// FFT data generation functions
function generateBitReversal(size: number): Uint32Array {
    const bits = Math.log2(size);
    const indices = new Uint32Array(size);
    
    for (let i = 0; i < size; i++) {
        let reversed = 0;
        let temp = i;
        
        for (let j = 0; j < bits; j++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        indices[i] = reversed;
    }
    
    return indices;
}

function generateTwiddles(size: number, direction: 'forward' | 'inverse'): Float32Array {
    const sign = direction === 'forward' ? -1 : 1;
    const stages = Math.log2(size);
    const data: number[] = [];
    
    for (let stage = 0; stage < stages; stage++) {
        const stageSize = 1 << (stage + 1);
        const halfStageSize = stageSize >> 1;
        
        for (let k = 0; k < halfStageSize; k++) {
            const angle = sign * 2 * Math.PI * k / stageSize;
            data.push(Math.cos(angle), Math.sin(angle));
        }
    }
    
    return new Float32Array(data);
}

function generateTwiddleOffsets(size: number): Uint32Array {
    const stages = Math.log2(size);
    const offsets = new Uint32Array(stages);
    
    for (let stage = 0; stage < stages; stage++) {
        offsets[stage] = (1 << stage) - 1;
    }
    
    return offsets;
}

// Code generation using arrays for efficiency
class CodeGenerator {
    private lines: string[] = [];
    private indentLevel = 0;
    private format: 'readable' | 'minified';
    
    constructor(format: 'readable' | 'minified') {
        this.format = format;
    }
    
    indent(): void {
        this.indentLevel++;
    }
    
    dedent(): void {
        this.indentLevel = Math.max(0, this.indentLevel - 1);
    }
    
    writeLine(line: string = ''): void {
        if (this.format === 'readable' && line) {
            const indent = '    '.repeat(this.indentLevel);
            this.lines.push(indent + line);
        } else if (line) {
            this.lines.push(line);
        } else if (this.format === 'readable') {
            this.lines.push('');
        }
    }
    
    writeArray(name: string, arr: Uint32Array | Float32Array, itemsPerLine: number = 16): void {
        const values = Array.from(arr);
        const isFloat = arr instanceof Float32Array;
        
        if (this.format === 'minified') {
            const formatted = values.map(v => isFloat ? v.toFixed(6) : v.toString()).join(',');
            this.writeLine(`${name}: new ${arr.constructor.name}([${formatted}]),`);
            return;
        }
        
        this.writeLine(`${name}: new ${arr.constructor.name}([`);
        this.indent();
        
        for (let i = 0; i < values.length; i += itemsPerLine) {
            const chunk = values.slice(i, i + itemsPerLine)
                .map(v => isFloat ? v.toFixed(6) : v.toString())
                .join(', ');
            
            const isLast = i + itemsPerLine >= values.length;
            this.writeLine(chunk + (isLast ? '' : ','));
        }
        
        this.dedent();
        this.writeLine(']),');
    }
    
    toString(): string {
        return this.lines.join('\n');
    }
}

// Generate TypeScript code for a single size
function generateSizeModule(size: number, config: GeneratorConfig): string {
    const gen = new CodeGenerator(config.format);
    
    // Generate data
    const bitReversal = generateBitReversal(size);
    const twiddlesForward = generateTwiddles(size, 'forward');
    const twiddlesInverse = generateTwiddles(size, 'inverse');
    const twiddleOffsets = generateTwiddleOffsets(size);
    
    // Header
    gen.writeLine('// Auto-generated FFT precomputed data');
    if (config.includeTimestamp) {
        gen.writeLine(`// Generated on: ${new Date().toISOString()}`);
    }
    gen.writeLine(`// Size: ${size}`);
    gen.writeLine('// DO NOT EDIT - Run \'npm run generate-fft\' to regenerate');
    gen.writeLine();
    
    gen.writeLine('import { PrecomputedFFTData } from \'./types\';');
    gen.writeLine();
    
    gen.writeLine(`export const FFT_DATA_${size}: PrecomputedFFTData = {`);
    gen.indent();
    
    gen.writeArray('bitReversal', bitReversal);
    gen.writeArray('twiddlesForward', twiddlesForward, 8);
    gen.writeArray('twiddlesInverse', twiddlesInverse, 8);
    gen.writeArray('twiddleOffsets', twiddleOffsets);
    
    gen.dedent();
    gen.writeLine('};');
    gen.writeLine();
    
    // Memory stats
    const bitRevSize = bitReversal.byteLength;
    const twiddleSize = twiddlesForward.byteLength;
    const totalSize = bitRevSize + twiddleSize * 2 + twiddleOffsets.byteLength;
    
    gen.writeLine(`export const MEMORY_STATS_${size} = {`);
    gen.indent();
    gen.writeLine(`bitReversal: ${bitRevSize},`);
    gen.writeLine(`twiddles: ${twiddleSize},`);
    gen.writeLine(`total: ${totalSize}`);
    gen.dedent();
    gen.writeLine('};');
    
    return gen.toString();
}

// Generate main index file
function generateIndexModule(sizes: number[], config: GeneratorConfig): string {
    const gen = new CodeGenerator(config.format);
    
    gen.writeLine('// Auto-generated FFT precomputed data index');
    if (config.includeTimestamp) {
        gen.writeLine(`// Generated on: ${new Date().toISOString()}`);
    }
    gen.writeLine('// DO NOT EDIT - Run \'npm run generate-fft\' to regenerate');
    gen.writeLine();
    
    gen.writeLine('export interface PrecomputedFFTData {');
    gen.indent();
    gen.writeLine('bitReversal: Uint32Array;');
    gen.writeLine('twiddlesForward: Float32Array;');
    gen.writeLine('twiddlesInverse: Float32Array;');
    gen.writeLine('twiddleOffsets: Uint32Array;');
    gen.dedent();
    gen.writeLine('}');
    gen.writeLine();
    
    if (config.split) {
        // Dynamic imports for code splitting
        gen.writeLine('// Lazy-loaded FFT data');
        gen.writeLine('const loaders: Record<number, () => Promise<PrecomputedFFTData>> = {');
        gen.indent();
        
        for (const size of sizes) {
            gen.writeLine(`${size}: async () => {`);
            gen.indent();
            gen.writeLine(`const module = await import('./fft_${size}');`);
            gen.writeLine(`return module.FFT_DATA_${size};`);
            gen.dedent();
            gen.writeLine('},');
        }
        
        gen.dedent();
        gen.writeLine('};');
        gen.writeLine();
        
        gen.writeLine('export async function getPrecomputedData(size: number): Promise<PrecomputedFFTData | null> {');
        gen.indent();
        gen.writeLine('const loader = loaders[size];');
        gen.writeLine('return loader ? await loader() : null;');
        gen.dedent();
        gen.writeLine('}');
    } else {
        // Static imports
        for (const size of sizes) {
            gen.writeLine(`import { FFT_DATA_${size} } from './fft_${size}';`);
        }
        gen.writeLine();
        
        gen.writeLine('export const FFT_PRECOMPUTED: Record<number, PrecomputedFFTData> = {');
        gen.indent();
        
        for (const size of sizes) {
            gen.writeLine(`${size}: FFT_DATA_${size},`);
        }
        
        gen.dedent();
        gen.writeLine('};');
        gen.writeLine();
        
        gen.writeLine('export function getPrecomputedData(size: number): PrecomputedFFTData | null {');
        gen.indent();
        gen.writeLine('return FFT_PRECOMPUTED[size] || null;');
        gen.dedent();
        gen.writeLine('}');
    }
    
    gen.writeLine();
    gen.writeLine('export function isPrecomputedSize(size: number): boolean {');
    gen.indent();
    gen.writeLine(`return [${sizes.join(', ')}].includes(size);`);
    gen.dedent();
    gen.writeLine('}');
    gen.writeLine();
    gen.writeLine(`export const PRECOMPUTED_SIZES = [${sizes.join(', ')}];`);
    
    return gen.toString();
}

// Generate binary files for even smaller bundles
async function generateBinaryFiles(sizes: number[], outputDir: string): Promise<void> {
    const binaryDir = path.join(outputDir, 'binary');
    
    if (!fs.existsSync(binaryDir)) {
        fs.mkdirSync(binaryDir, { recursive: true });
    }
    
    for (const size of sizes) {
        const data = {
            bitReversal: Array.from(generateBitReversal(size)),
            twiddlesForward: Array.from(generateTwiddles(size, 'forward')),
            twiddlesInverse: Array.from(generateTwiddles(size, 'inverse')),
            twiddleOffsets: Array.from(generateTwiddleOffsets(size))
        };
        
        const buffer = Buffer.from(JSON.stringify(data));
        const filePath = path.join(binaryDir, `fft_${size}.json`);
        
        fs.writeFileSync(filePath, buffer);
        console.log(`  Generated binary: fft_${size}.json (${(buffer.length / 1024).toFixed(2)} KB)`);
    }
}

// Format code with Prettier
async function formatCode(code: string): Promise<string> {
    try {
        const formatted = await prettier.format(code, {
            parser: 'typescript',
            singleQuote: true,
            trailingComma: 'es5',
            printWidth: 100,
            tabWidth: 2,
        });
        return formatted;
    } catch (error) {
        console.warn('Prettier formatting failed, using unformatted code');
        return code;
    }
}

// Main execution
async function main() {
    const config = parseArgs();
    
    console.log('🔧 FFT Data Generator');
    console.log('====================');
    console.log(`Sizes: ${config.sizes.join(', ')}`);
    console.log(`Output: ${config.outputDir}`);
    console.log(`Format: ${config.format}`);
    console.log(`Split files: ${config.split}`);
    console.log();
    
    // Create output directory
    if (!fs.existsSync(config.outputDir)) {
        fs.mkdirSync(config.outputDir, { recursive: true });
    }
    
    try {
        // Generate individual size modules
        let totalSize = 0;
        
        for (const size of config.sizes) {
            console.log(`Generating FFT data for size ${size}...`);
            
            const code = generateSizeModule(size, config);
            const formatted = await formatCode(code);
            const filePath = path.join(config.outputDir, `fft_${size}.ts`);
            
            fs.writeFileSync(filePath, formatted);
            
            const fileSize = Buffer.byteLength(formatted, 'utf-8');
            totalSize += fileSize;
            console.log(`  ✓ Generated fft_${size}.ts (${(fileSize / 1024).toFixed(2)} KB)`);
        }
        
        // Generate types file
        const typesCode = `export interface PrecomputedFFTData {
    bitReversal: Uint32Array;
    twiddlesForward: Float32Array;
    twiddlesInverse: Float32Array;
    twiddleOffsets: Uint32Array;
}`;
        
        fs.writeFileSync(path.join(config.outputDir, 'types.ts'), await formatCode(typesCode));
        
        // Generate index file
        console.log('\nGenerating index file...');
        const indexCode = generateIndexModule(config.sizes, config);
        const formattedIndex = await formatCode(indexCode);
        fs.writeFileSync(path.join(config.outputDir, 'index.ts'), formattedIndex);
        
        console.log(`  ✓ Generated index.ts`);
        
        // Generate binary files if requested
        if (config.generateBinary) {
            console.log('\nGenerating binary files...');
            await generateBinaryFiles(config.sizes, config.outputDir);
        }
        
        // Summary
        console.log('\n✅ Generation complete!');
        console.log(`Total size: ${(totalSize / 1024).toFixed(2)} KB`);
        
        if (config.split) {
            console.log('\n💡 Using dynamic imports for code splitting.');
            console.log('   FFT data will be loaded on-demand.');
        }
        
    } catch (error) {
        console.error('\n❌ Generation failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

// Export for testing
export { generateBitReversal, generateTwiddles, generateTwiddleOffsets, CodeGenerator };
