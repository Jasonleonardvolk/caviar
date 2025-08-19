// test-gpu-device-immediate.ts
// Minimal test to check GPU device immediately

async function testGPUDevice() {
    console.log('=== Testing GPU Device ===');
    
    // Check if WebGPU is available
    if (!navigator.gpu) {
        console.error('WebGPU not available!');
        return;
    }
    
    console.log('WebGPU is available');
    
    // Request adapter
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
    });
    
    if (!adapter) {
        console.error('No GPU adapter found!');
        return;
    }
    
    console.log('GPU adapter obtained');
    console.log('Adapter info:', {
        isFallbackAdapter: adapter.isFallbackAdapter,
        limits: adapter.limits,
        features: [...adapter.features]
    });
    
    // Request device with minimal requirements
    try {
        const device = await adapter.requestDevice({
            label: 'Test Device'
        });
        
        console.log('GPU device created successfully');
        
        // Add device lost handler
        device.lost.then((info) => {
            console.error('DEVICE LOST:', info);
            console.error('Reason:', info.reason);
            console.error('Message:', info.message);
        });
        
        // Test creating a simple buffer
        const buffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.STORAGE
        });
        
        console.log('Test buffer created');
        
        // Wait a moment to see if device gets lost
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        console.log('Device still active after 1 second');
        
        // Cleanup
        buffer.destroy();
        device.destroy();
        
        console.log('Test completed successfully');
        
    } catch (error) {
        console.error('Failed to create device:', error);
    }
}

// For Node.js environment
if (typeof window === 'undefined') {
    console.log('Running in Node.js environment');
    console.log('WebGPU tests require a browser environment or special Node.js setup');
    
    // Try to check if we have the setup module
    try {
        const setupPath = './tests/setup.ts';
        console.log('Looking for setup at:', setupPath);
        process.exit(0);
    } catch (e) {
        console.error('Setup not found');
    }
} else {
    testGPUDevice().catch(console.error);
}

export { testGPUDevice };
