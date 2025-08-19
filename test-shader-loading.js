// Test shader loading independently
import { ShaderLoader } from './frontend/lib/webgpu/shaderLoader.js';

console.log('Testing shader loading...\n');

const testShaders = [
    'shaders/bitReversal.wgsl',
    'shaders/butterflyStage.wgsl',
    'shaders/normalize.wgsl',
    'shaders/fftShift.wgsl',
    'shaders/transpose.wgsl'
];

async function testShaderLoading() {
    for (const shaderPath of testShaders) {
        try {
            console.log(`Loading ${shaderPath}...`);
            const content = await ShaderLoader.load(shaderPath);
            console.log(`  ✓ Loaded successfully (${content.length} bytes)`);
            
            // Basic validation
            if (content.includes('struct FFTUniforms')) {
                console.log(`  ✓ Contains FFTUniforms struct`);
            }
            if (content.includes('@compute')) {
                console.log(`  ✓ Contains compute shader entry point`);
            }
            console.log('');
        } catch (error) {
            console.error(`  ✗ Failed to load: ${error.message}`);
        }
    }
    
    // Test cache
    const stats = ShaderLoader.getCacheStats();
    console.log(`\nCache stats: ${stats.size} shaders loaded`);
    console.log('Cached shaders:', stats.shaders.join(', '));
}

testShaderLoading().catch(console.error);
