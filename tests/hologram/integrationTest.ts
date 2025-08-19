/**
 * Integration Test for Holographic Pipeline
 * Tests the full flow from oscillator input to quilt output
 */

import { SpectralHologramEngine, getDefaultCalibration } from '../../frontend/lib/holographicEngine';
import { BanksyOscillator } from '../../ingest_bus/audio/spectral_oscillator';
import { FFTCompute } from '../../frontend/lib/webgpu/fftCompute';

export class HologramIntegrationTest {
    private engine!: SpectralHologramEngine;
    private oscillator!: BanksyOscillator;
    private canvas!: HTMLCanvasElement;
    
    async setup() {
        // Create test canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1920;
        this.canvas.height = 1080;
        
        // Initialize engine
        this.engine = new SpectralHologramEngine();
        await this.engine.initialize(
            this.canvas,
            getDefaultCalibration('webgpu_only'),
            'test-session'
        );
        
        // Initialize oscillator
        this.oscillator = new BanksyOscillator({
            num_oscillators: 32,
            base_frequency: 440,
            coupling_strength: 0.5,
            damping: 0.01,
            noise_amount: 0.1
        });
    }
    
    async teardown() {
        this.engine.destroy();
    }
    
    /**
     * Test oscillator to wavefield parameter conversion
     */
    async testOscillatorToWavefield(): Promise<boolean> {
        console.log('Testing oscillator to wavefield conversion...');
        
        try {
            // Generate test audio signal
            const sampleRate = 48000;
            const duration = 1; // 1 second
            const samples = new Float32Array(sampleRate * duration);
            
            // Generate sine wave
            for (let i = 0; i < samples.length; i++) {
                samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate);
            }
            
            // Process through oscillator
            const oscState = this.oscillator.process_audio(samples);
            
            // Get wavefield parameters
            const wavefieldParams = this.oscillator.get_wavefield_params();
            
            // Verify conversion
            console.log('Oscillator state:', {
                psi_phase: oscState.psi_phase.toFixed(3),
                coherence: oscState.phase_coherence.toFixed(3),
                num_phases: oscState.oscillator_phases.length
            });
            
            console.log('Wavefield params:', {
                modulation: wavefieldParams.phase_modulation.toFixed(3),
                coherence: wavefieldParams.coherence.toFixed(3),
                num_frequencies: wavefieldParams.spatial_frequencies.length
            });
            
            // Update engine
            this.engine.updateFromOscillator(oscState);
            this.engine.updateFromWavefieldParams(wavefieldParams);
            
            return true;
        } catch (error) {
            console.error('Oscillator test failed:', error);
            return false;
        }
    }
    
    /**
     * Test full rendering pipeline
     */
    async testFullPipeline(): Promise<boolean> {
        console.log('Testing full rendering pipeline...');
        
        try {
            // Generate complex test pattern
            const testPattern = this.generateTestPattern();
            
            // Update oscillator
            const oscState = this.oscillator.process_audio(testPattern.audio);
            this.engine.updateFromOscillator(oscState);
            
            // Update depth
            this.engine.updateDepthTexture(testPattern.depth);
            
            // Render multiple frames
            const frameTimes: number[] = [];
            
            for (let i = 0; i < 10; i++) {
                const start = performance.now();
                
                await this.engine.render({
                    propagationDistance: 0.1 + i * 0.05,
                    enableVelocityField: true
                });
                
                frameTimes.push(performance.now() - start);
            }
            
            // Check performance
            const avgTime = frameTimes.reduce((a, b) => a + b) / frameTimes.length;
            console.log(`Average frame time: ${avgTime.toFixed(2)}ms (${(1000/avgTime).toFixed(1)} FPS)`);
            
            return avgTime < 33.33; // 30 FPS minimum
            
        } catch (error) {
            console.error('Pipeline test failed:', error);
            return false;
        }
    }
    
    /**
     * Test WebSocket communication
     */
    async testWebSocketUpdates(): Promise<boolean> {
        console.log('Testing WebSocket updates...');
        
        try {
            // Simulate WebSocket messages
            const messages = [
                {
                    type: 'wavefield_update',
                    wavefield_params: {
                        phase_modulation: 0.5,
                        coherence: 0.8,
                        oscillator_phases: [0, Math.PI/2, Math.PI],
                        dominant_freq: 440,
                        spatial_frequencies: [[1, 0], [0, 1], [1, 1]],
                        amplitudes: [1, 0.8, 0.6]
                    }
                },
                {
                    type: 'quality_adjustment',
                    quality: 'medium'
                }
            ];
            
            // Process messages
            for (const msg of messages) {
                this.engine['handleWebSocketMessage'](msg);
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            return true;
            
        } catch (error) {
            console.error('WebSocket test failed:', error);
            return false;
        }
    }
    
    /**
     * Test quality switching
     */
    async testQualitySwitching(): Promise<boolean> {
        console.log('Testing quality switching...');
        
        try {
            const qualities = ['low', 'medium', 'high', 'ultra'] as const;
            
            for (const quality of qualities) {
                console.log(`  Setting quality to ${quality}...`);
                await this.engine.setQuality(quality);
                
                // Render frame
                const start = performance.now();
                await this.engine.render();
                const elapsed = performance.now() - start;
                
                console.log(`    Rendered in ${elapsed.toFixed(2)}ms`);
            }
            
            return true;
            
        } catch (error) {
            console.error('Quality switching test failed:', error);
            return false;
        }
    }
    
    /**
     * Test memory management
     */
    async testMemoryManagement(): Promise<boolean> {
        console.log('Testing memory management...');
        
        try {
            // Get initial memory (if available)
            const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
            
            // Render many frames
            for (let i = 0; i < 100; i++) {
                await this.engine.render();
                
                // Update with new data
                this.engine.updateFromOscillator({
                    psi_phase: Math.random(),
                    phase_coherence: Math.random(),
                    oscillator_phases: Array(32).fill(0).map(() => Math.random() * Math.PI * 2),
                    oscillator_frequencies: Array(32).fill(0).map(() => 200 + Math.random() * 2000),
                    coupling_strength: Math.random(),
                    dominant_frequency: 440
                });
            }
            
            // Check memory growth
            const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
            const memoryGrowth = finalMemory - initialMemory;
            
            console.log(`Memory growth: ${(memoryGrowth / 1024 / 1024).toFixed(2)} MB`);
            
            // Should not grow more than 50MB
            return memoryGrowth < 50 * 1024 * 1024;
            
        } catch (error) {
            console.error('Memory test failed:', error);
            return false;
        }
    }
    
    /**
     * Generate test pattern
     */
    private generateTestPattern(): { audio: Float32Array, depth: Float32Array } {
        const audioLength = 48000; // 1 second
        const audio = new Float32Array(audioLength);
        
        // Multi-frequency test signal
        for (let i = 0; i < audioLength; i++) {
            audio[i] = 
                0.5 * Math.sin(2 * Math.PI * 440 * i / 48000) +
                0.3 * Math.sin(2 * Math.PI * 880 * i / 48000) +
                0.2 * Math.sin(2 * Math.PI * 1320 * i / 48000);
        }
        
        // Depth map - sphere
        const size = 512;
        const depth = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = (x - size/2) / (size/2);
                const cy = (y - size/2) / (size/2);
                const r2 = cx*cx + cy*cy;
                
                if (r2 < 1) {
                    depth[y * size + x] = Math.sqrt(1 - r2);
                }
            }
        }
        
        return { audio, depth };
    }
}

// Run all integration tests
export async function runIntegrationTests(): Promise<void> {
    const test = new HologramIntegrationTest();
    
    try {
        await test.setup();
        
        const tests = [
            { name: 'Oscillator to Wavefield', fn: () => test.testOscillatorToWavefield() },
            { name: 'Full Pipeline', fn: () => test.testFullPipeline() },
            { name: 'WebSocket Updates', fn: () => test.testWebSocketUpdates() },
            { name: 'Quality Switching', fn: () => test.testQualitySwitching() },
            { name: 'Memory Management', fn: () => test.testMemoryManagement() }
        ];
        
        let passed = 0;
        let failed = 0;
        
        for (const { name, fn } of tests) {
            console.log(`\nRunning: ${name}`);
            
            try {
                const result = await fn();
                if (result) {
                    console.log(`✅ ${name} PASSED`);
                    passed++;
                } else {
                    console.log(`❌ ${name} FAILED`);
                    failed++;
                }
            } catch (error) {
                console.log(`❌ ${name} FAILED with error:`, error);
                failed++;
            }
        }
        
        console.log('\n=== Integration Test Summary ===');
        console.log(`Passed: ${passed}`);
        console.log(`Failed: ${failed}`);
        console.log(`Total:  ${passed + failed}`);
        
    } finally {
        await test.teardown();
    }
}

// Run if called directly
if (require.main === module) {
    runIntegrationTests().catch(console.error);
}
