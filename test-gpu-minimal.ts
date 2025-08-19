// test-gpu-minimal.ts
// Minimal test to isolate GPU issues

import { FFTCompute } from './frontend/lib/webgpu/fftCompute.js';

async function testMinimalGPU() {
    console.log('=== Minimal GPU Test ===');
    
    // Get adapter with validation
    const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: 'high-performance'
    });
    
    if (!adapter) {
        throw new Error('No GPU adapter available');
    }
    
    console.log('Adapter features:', [...adapter.features]);
    
    // Request device with error handling
    const device = await adapter.requestDevice({
        label: 'Test Device',
        requiredFeatures: [],
        requiredLimits: {
            maxComputeWorkgroupSizeX: 256,
            maxComputeWorkgroupSizeY: 256,
            maxComputeWorkgroupSizeZ: 64
        }
    });
    
    console.log('Device limits:', device.limits);
    
    // Add device lost handler
    device.lost.then((info) => {
        console.error('DEVICE LOST:', info);
    });
    
    // Test minimal FFT
    try {
        const fft = new FFTCompute(device, {
            size: 64,  // Start small
            precision: 'f32',
            normalization: 'forward',
            direction: 'forward',
            dimensions: 1,
            batchSize: 1,
            workgroupSize: 64
        });
        
        await fft.initialize();
        console.log('âœ“ FFT initialized');
        
        // Test with small data
        const input = new Float32Array(64 * 2); // Complex
        for (let i = 0; i < 64; i++) {
            input[i * 2] = Math.sin(i * 0.1);     // Real
            input[i * 2 + 1] = 0;                 // Imaginary
        }
        
        const output = await fft.execute(input);
        console.log('âœ“ FFT executed');
        console.log('Output sample:', output.slice(0, 4));
        
    } catch (error) {
        console.error('FFT Error:', error);
    }
}

testMinimalGPU().catch(console.error);
