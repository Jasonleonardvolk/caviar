/**
 * Golden Image Test for Holographic Quilt Output
 * Ensures visual consistency across changes
 */

import { SpectralHologramEngine, getDefaultCalibration } from '../../frontend/lib/holographicEngine';
import { createHash } from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// ESM compatibility - fixed naming
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test configuration
const TEST_CONFIG = {
    hologramSize: 512,
    numViews: 9,
    quiltCols: 3,
    quiltRows: 3,
    wavelength: 0.000633,
    propagationDistance: 0.3
};

// Expected hashes for known good outputs
// Note: These are placeholder values. In a real test, you would:
// 1. Run the test once to get the actual hash
// 2. Verify the output is correct
// 3. Update these values with the actual hashes
const GOLDEN_HASHES = {
    'simple_wavefield': 'af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc',
    'multi_oscillator': 'af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc', 
    'with_depth': 'af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc'
};

// Set to true to update golden hashes with current results
const UPDATE_GOLDEN_HASHES = process.env.UPDATE_GOLDEN === 'true';

interface TestCase {
    name: string;
    setup: () => Promise<any>;
    expectedHash?: string;
}

export class GoldenImageTest {
    private engine!: SpectralHologramEngine;
    private canvas!: HTMLCanvasElement;
    private context!: CanvasRenderingContext2D;
    
    async setup() {
        // Create offscreen canvas
        const { createCanvas } = await import('canvas');
        this.canvas = createCanvas(
            TEST_CONFIG.quiltCols * 256,
            TEST_CONFIG.quiltRows * 256
        );
        
        // Initialize engine
        this.engine = new SpectralHologramEngine();
        await this.engine.initialize(
            this.canvas as any,
            getDefaultCalibration('webgpu_only')
        );
        
        // Override config for testing
        await this.engine.setQuality('low');
    }
    
    async teardown() {
        this.engine.destroy();
    }
    
    /**
     * Run a test case and compare output
     */
    async runTest(testCase: TestCase): Promise<boolean> {
        console.log(`Running test: ${testCase.name}`);
        
        try {
            // Setup test data
            const testData = await testCase.setup();
            
            // Update engine state
            if (testData.oscillatorState) {
                this.engine.updateFromOscillator(testData.oscillatorState);
            }
            if (testData.wavefieldParams) {
                this.engine.updateFromWavefieldParams(testData.wavefieldParams);
            }
            if (testData.depthData) {
                this.engine.updateDepthTexture(testData.depthData);
            }
            
            // Render
            await this.engine.render({
                propagationDistance: TEST_CONFIG.propagationDistance
            });
            
            // Get rendered output
            const imageData = await this.captureOutput();
            
            // Calculate hash
            const hash = this.calculateHash(imageData);
            
            // Compare with expected
            if (testCase.expectedHash) {
                const passed = hash === testCase.expectedHash;
                
                if (!passed) {
                    console.error(`Hash mismatch for ${testCase.name}:`);
                    console.error(`  Expected: ${testCase.expectedHash}`);
                    console.error(`  Actual:   ${hash}`);
                    
                    // Save difference image
                    await this.saveDifferenceImage(
                        testCase.name,
                        imageData,
                        testCase.expectedHash
                    );
                }
                
                return passed;
            } else {
                // No expected hash - save as new golden
                console.log(`New golden hash for ${testCase.name}: ${hash}`);
                await this.saveGoldenImage(testCase.name, imageData, hash);
                return true;
            }
            
        } catch (error) {
            console.error(`Test ${testCase.name} failed with error:`, error);
            return false;
        }
    }
    
    /**
     * Capture rendered output
     */
    private async captureOutput(): Promise<Buffer> {
        // Read pixels from WebGPU
        const texture = this.engine['context'].getCurrentTexture();
        const buffer = this.engine['device'].createBuffer({
            size: texture.width * texture.height * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const commandEncoder = this.engine['device'].createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture },
            { 
                buffer,
                bytesPerRow: texture.width * 4,
                rowsPerImage: texture.height
            },
            { 
                width: texture.width, 
                height: texture.height 
            }
        );
        
        this.engine['device'].queue.submit([commandEncoder.finish()]);
        
        await buffer.mapAsync(GPUMapMode.READ);
        const data = Buffer.from(buffer.getMappedRange());
        buffer.unmap();
        
        return data;
    }
    
    /**
     * Calculate hash of image data
     */
    private calculateHash(data: Buffer): string {
        return createHash('sha256').update(data).digest('hex');
    }
    
    /**
     * Save golden image for future comparison
     */
    private async saveGoldenImage(
        name: string, 
        data: Buffer, 
        hash: string
    ): Promise<void> {
        const goldenDir = path.join(__dirname, 'golden');
        await fs.mkdir(goldenDir, { recursive: true });
        
        // Save image
        const imagePath = path.join(goldenDir, `${name}.png`);
        // Would use sharp or similar to save as PNG
        
        // Save hash
        const hashPath = path.join(goldenDir, `${name}.hash`);
        await fs.writeFile(hashPath, hash);
    }
    
    /**
     * Save difference image for debugging
     */
    private async saveDifferenceImage(
        name: string,
        actual: Buffer,
        expectedHash: string
    ): Promise<void> {
        const diffDir = path.join(__dirname, 'diffs');
        await fs.mkdir(diffDir, { recursive: true });
        
        const diffPath = path.join(diffDir, `${name}_diff.png`);
        // Would use pixelmatch or similar to create diff image
    }
}

// Define test cases
export const TEST_CASES: TestCase[] = [
    {
        name: 'simple_wavefield',
        setup: async () => ({
            oscillatorState: {
                psi_phase: 0.5,
                phase_coherence: 0.8,
                oscillator_phases: [0, Math.PI/4, Math.PI/2],
                oscillator_frequencies: [440, 880, 1320],
                coupling_strength: 0.5,
                dominant_frequency: 440
            }
        }),
        expectedHash: GOLDEN_HASHES.simple_wavefield
    },
    {
        name: 'multi_oscillator',
        setup: async () => ({
            oscillatorState: {
                psi_phase: 0.7,
                phase_coherence: 0.9,
                oscillator_phases: Array(32).fill(0).map((_, i) => i * Math.PI / 16),
                oscillator_frequencies: Array(32).fill(0).map((_, i) => 220 * (i + 1)),
                coupling_strength: 0.7,
                dominant_frequency: 660
            }
        }),
        expectedHash: GOLDEN_HASHES.multi_oscillator
    },
    {
        name: 'with_depth',
        setup: async () => {
            // Generate synthetic depth map
            const size = TEST_CONFIG.hologramSize;
            const depth = new Float32Array(size * size);
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const idx = y * size + x;
                    const cx = (x - size/2) / size;
                    const cy = (y - size/2) / size;
                    depth[idx] = Math.exp(-(cx*cx + cy*cy) * 4);
                }
            }
            
            return {
                depthData: depth,
                oscillatorState: {
                    psi_phase: 0.3,
                    phase_coherence: 0.7,
                    oscillator_phases: [0, Math.PI/3],
                    oscillator_frequencies: [550, 1100],
                    coupling_strength: 0.4,
                    dominant_frequency: 550
                }
            };
        },
        expectedHash: GOLDEN_HASHES.with_depth
    }
];

// Run all tests
export async function runAllTests(): Promise<void> {
    const test = new GoldenImageTest();
    
    try {
        await test.setup();
        
        let passed = 0;
        let failed = 0;
        
        for (const testCase of TEST_CASES) {
            const result = await test.runTest(testCase);
            if (result) {
                passed++;
            } else {
                failed++;
            }
        }
        
        console.log('\n=== Test Results ===');
        console.log(`Passed: ${passed}`);
        console.log(`Failed: ${failed}`);
        console.log(`Total:  ${passed + failed}`);
        
        process.exit(failed > 0 ? 1 : 0);
        
    } finally {
        await test.teardown();
    }
}

// Run if called directly
if (require.main === module) {
    runAllTests().catch(console.error);
}
