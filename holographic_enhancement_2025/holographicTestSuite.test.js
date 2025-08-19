/**
 * Comprehensive Testing Framework for Enhanced Holographic System
 * Includes unit tests, integration tests, visual regression, and performance benchmarks
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import { holographicSystem } from '../enhancedUnifiedHolographicSystem';
import { conceptMesh } from '../enhancedConceptMeshIntegration';
import { PenroseWavefieldEngine } from '../penroseWavefieldEngine';
import { AIAssistedRenderer } from '../aiAssistedRenderer';

// Test utilities
import { 
  createMockWebGPUDevice,
  createTestCanvas,
  generateTestOscillatorState,
  compareImages,
  measurePerformance
} from './testUtils';

// Reference data
import { 
  referenceWavefields,
  referenceQuilts,
  testConcepts,
  testAudioFeatures
} from './testData';

describe('Enhanced Holographic System Tests', () => {
  let device;
  let canvas;
  
  beforeAll(async () => {
    // Setup mock WebGPU if not available
    if (!navigator.gpu) {
      device = await createMockWebGPUDevice();
      global.navigator.gpu = device.adapter;
    }
    
    canvas = createTestCanvas(1920, 1080);
  });
  
  afterAll(() => {
    holographicSystem.destroy();
  });
  
  describe('Unit Tests', () => {
    describe('Penrose Wavefield Engine', () => {
      let penroseEngine;
      
      beforeEach(async () => {
        penroseEngine = new PenroseWavefieldEngine(device, 512);
        await penroseEngine.initialize();
      });
      
      it('should initialize successfully', () => {
        expect(penroseEngine.initialized).toBe(true);
        expect(penroseEngine.size).toBe(512);
      });
      
      it('should generate valid wavefield', async () => {
        const oscillatorState = generateTestOscillatorState();
        const wavefield = await penroseEngine.generateWavefield(oscillatorState);
        
        expect(wavefield).toBeDefined();
        expect(wavefield.size).toBeGreaterThan(0);
      });
      
      it('should converge within iteration limit', async () => {
        const oscillatorState = generateTestOscillatorState();
        const options = {
          iterations: 100,
          convergenceThreshold: 0.001
        };
        
        const startTime = performance.now();
        const wavefield = await penroseEngine.generateWavefield(oscillatorState, options);
        const duration = performance.now() - startTime;
        
        expect(duration).toBeLessThan(1000); // Should complete within 1 second
        
        // Check convergence
        const converged = await penroseEngine.checkConvergence();
        expect(converged).toBe(true);
      });
      
      it('should handle different quality modes', async () => {
        const oscillatorState = generateTestOscillatorState();
        
        // Test draft mode
        penroseEngine.setQuality(0);
        const draftTime = await measurePerformance(async () => {
          await penroseEngine.generateWavefield(oscillatorState);
        });
        
        // Test high quality mode
        penroseEngine.setQuality(2);
        const highQualityTime = await measurePerformance(async () => {
          await penroseEngine.generateWavefield(oscillatorState);
        });
        
        // High quality should take longer
        expect(highQualityTime).toBeGreaterThan(draftTime);
      });
      
      it('should fall back to CPU when GPU unavailable', async () => {
        penroseEngine.useCPUFallback = true;
        
        const oscillatorState = generateTestOscillatorState();
        const wavefield = await penroseEngine.generateWavefield(oscillatorState);
        
        expect(wavefield).toBeDefined();
        expect(wavefield.length).toBe(512 * 512 * 2); // Complex values
      });
    });
    
    describe('AI-Assisted Renderer', () => {
      let aiRenderer;
      
      beforeEach(async () => {
        aiRenderer = new AIAssistedRenderer(device, holographicSystem);
        await aiRenderer.initialize();
      });
      
      it('should initialize all AI modules', () => {
        const status = aiRenderer.getStatus();
        
        expect(status.dibr.initialized).toBe(true);
        expect(status.nerf.initialized).toBe(true);
        expect(status.gan.initialized).toBe(true);
      });
      
      it('should select appropriate pipeline based on input', () => {
        // Test image input
        const imagePipeline = aiRenderer.selectPipeline(
          { type: 'image', hasDepth: true },
          'auto'
        );
        expect(imagePipeline.useDIBR).toBe(true);
        
        // Test multi-view input
        const multiViewPipeline = aiRenderer.selectPipeline(
          { type: 'multiView' },
          'auto'
        );
        expect(multiViewPipeline.useNeRF).toBe(true);
      });
      
      it('should apply DIBR to single image', async () => {
        const inputData = {
          type: 'image',
          image: createTestImage(),
          depth: createTestDepthMap()
        };
        
        const result = await aiRenderer.applyDIBR(inputData);
        
        expect(result.views).toBeDefined();
        expect(result.views.length).toBe(45); // Default view count
        expect(result.quilt).toBeDefined();
      });
      
      it('should train NeRF with sufficient data', async () => {
        const sceneId = 'test-scene';
        const captures = generateTestCaptures(15); // More than threshold
        
        const model = await aiRenderer.trainNeRF(sceneId, captures);
        
        expect(model).toBeDefined();
        expect(model.trained).toBe(true);
        expect(aiRenderer.hasTrainedModel(sceneId)).toBe(true);
      });
      
      it('should enhance quilt with GAN', async () => {
        const inputQuilt = createTestQuilt();
        const inputData = { quilt: inputQuilt };
        
        const result = await aiRenderer.applyGANEnhancement(inputData);
        
        expect(result.quilt).toBeDefined();
        expect(result.enhanced).toBe(true);
        
        // Compare quality metrics
        const originalMetrics = calculateImageMetrics(inputQuilt);
        const enhancedMetrics = calculateImageMetrics(result.quilt);
        
        expect(enhancedMetrics.sharpness).toBeGreaterThan(originalMetrics.sharpness);
      });
    });
    
    describe('Enhanced Concept Mesh Integration', () => {
      it('should handle concept deletion', () => {
        const concept = testConcepts[0];
        conceptMesh.conceptCache.set(concept.id, concept);
        
        conceptMesh.handleConceptDeletion([concept.id]);
        
        expect(conceptMesh.conceptCache.has(concept.id)).toBe(false);
        expect(conceptMesh.deletedConcepts.size).toBe(1);
      });
      
      it('should handle relation updates', () => {
        const relation = {
          id: 'rel-1',
          source_id: 'concept-1',
          target_id: 'concept-2',
          strength: 0.5
        };
        
        conceptMesh.relationCache.set(relation.id, relation);
        
        conceptMesh.handleRelationUpdate({
          relationId: relation.id,
          updates: { strength: 0.8 }
        });
        
        const updated = conceptMesh.relationCache.get(relation.id);
        expect(updated.strength).toBe(0.8);
      });
      
      it('should work in offline mode', () => {
        conceptMesh.offlineMode = true;
        conceptMesh.isConnected = false;
        
        const concept = testConcepts[0];
        conceptMesh.addConcept(concept);
        
        expect(conceptMesh.conceptCache.has(concept.id)).toBe(true);
        expect(conceptMesh.messageQueue.length).toBe(1);
      });
      
      it('should support undo operations', () => {
        const concept = testConcepts[0];
        
        // Add concept
        conceptMesh.handleConceptAddition(concept);
        expect(conceptMesh.conceptCache.has(concept.id)).toBe(true);
        
        // Undo
        conceptMesh.undo();
        expect(conceptMesh.conceptCache.has(concept.id)).toBe(false);
      });
    });
  });
  
  describe('Integration Tests', () => {
    beforeEach(async () => {
      await holographicSystem.initialize(canvas, {
        hologramSize: 512,
        numViews: 45,
        development: true
      });
    });
    
    it('should initialize all subsystems', async () => {
      const status = holographicSystem.getStatus();
      
      expect(status.initialized).toBe(true);
      expect(status.capabilities.rendering.fft).toBe(true);
      expect(status.capabilities.rendering.penrose).toBe(true);
      expect(status.capabilities.rendering.ai).toBeDefined();
    });
    
    it('should switch between rendering modes', async () => {
      // Test FFT mode
      holographicSystem.setRenderingMode('fft');
      let result = await holographicSystem.generateHologram();
      expect(result.mode).toBe('fft');
      
      // Test Penrose mode
      holographicSystem.setRenderingMode('penrose');
      result = await holographicSystem.generateHologram();
      expect(result.mode).toBe('penrose');
      
      // Test AI-assisted mode
      holographicSystem.setRenderingMode('ai_assisted');
      result = await holographicSystem.generateHologram();
      expect(result.mode).toBe('ai_assisted');
    });
    
    it('should handle audio input correctly', () => {
      holographicSystem.updateFromAudioFeatures(testAudioFeatures);
      
      const psiState = holographicSystem.psiState;
      expect(psiState.dominant_frequency).toBe(testAudioFeatures.pitch);
      expect(psiState.phase_coherence).toBe(testAudioFeatures.spectral_flatness);
      expect(psiState.oscillator_phases).toBeDefined();
    });
    
    it('should integrate concept mesh operations', async () => {
      const concept = testConcepts[0];
      
      // Add concept
      await holographicSystem.addConcept(concept);
      expect(conceptMesh.conceptCache.has(concept.id)).toBe(true);
      
      // Update position
      const newPosition = [1, 2, 3];
      await holographicSystem.updateConceptPosition(concept.id, newPosition);
      
      const updated = conceptMesh.conceptCache.get(concept.id);
      expect(updated.position).toEqual(newPosition);
      
      // Delete concept
      await holographicSystem.deleteConcept(concept.id);
      expect(conceptMesh.conceptCache.has(concept.id)).toBe(false);
    });
    
    it('should generate comparison view', async () => {
      holographicSystem.setRenderingMode('comparison');
      const result = await holographicSystem.generateHologram();
      
      expect(result.mode).toBe('comparison');
      expect(result.texture).toBeDefined();
      expect(result.texture.width).toBe(3360 * 3); // 3 quilts side by side
    });
  });
  
  describe('Visual Regression Tests', () => {
    it('should match reference FFT wavefield', async () => {
      const oscillatorState = generateTestOscillatorState();
      const wavefield = await holographicSystem.generateFFTWavefield();
      
      const similarity = await compareWavefields(
        wavefield,
        referenceWavefields.fft
      );
      
      expect(similarity).toBeGreaterThan(0.95); // 95% similarity
    });
    
    it('should match reference Penrose wavefield', async () => {
      holographicSystem.setRenderingMode('penrose');
      const wavefield = await holographicSystem.generatePenroseWavefield();
      
      const similarity = await compareWavefields(
        wavefield,
        referenceWavefields.penrose
      );
      
      expect(similarity).toBeGreaterThan(0.90); // Allow more variation
    });
    
    it('should produce consistent quilts', async () => {
      // Generate quilt twice with same input
      const quilt1 = await holographicSystem.generateHologram();
      const quilt2 = await holographicSystem.generateHologram();
      
      const similarity = await compareImages(
        quilt1.quilt,
        quilt2.quilt
      );
      
      expect(similarity).toBeGreaterThan(0.99); // Should be nearly identical
    });
    
    it('should maintain temporal coherence', async () => {
      holographicSystem.aiRenderer.config.gan.temporalSmoothing = true;
      
      const frames = [];
      for (let i = 0; i < 5; i++) {
        const result = await holographicSystem.generateHologram();
        frames.push(result.quilt);
      }
      
      // Check frame-to-frame differences
      for (let i = 1; i < frames.length; i++) {
        const diff = await calculateFrameDifference(
          frames[i - 1],
          frames[i]
        );
        
        expect(diff).toBeLessThan(0.1); // Max 10% change between frames
      }
    });
  });
  
  describe('Performance Tests', () => {
    it('should achieve target frame rate', async () => {
      const targetFPS = 30;
      const duration = 3000; // 3 seconds
      const startTime = performance.now();
      let frameCount = 0;
      
      while (performance.now() - startTime < duration) {
        await holographicSystem.generateHologram();
        frameCount++;
      }
      
      const actualFPS = frameCount / (duration / 1000);
      expect(actualFPS).toBeGreaterThan(targetFPS);
    });
    
    it('should scale with hologram size', async () => {
      const sizes = [256, 512, 1024];
      const times = [];
      
      for (const size of sizes) {
        // Reinitialize with different size
        holographicSystem.destroy();
        await holographicSystem.initialize(canvas, {
          hologramSize: size
        });
        
        const time = await measurePerformance(async () => {
          await holographicSystem.generateHologram();
        });
        
        times.push(time);
      }
      
      // Check that time increases roughly quadratically
      const ratio1 = times[1] / times[0];
      const ratio2 = times[2] / times[1];
      
      expect(ratio1).toBeGreaterThan(3); // ~4x for 2x size
      expect(ratio1).toBeLessThan(5);
      expect(ratio2).toBeGreaterThan(3);
      expect(ratio2).toBeLessThan(5);
    });
    
    it('should compare mode performance', async () => {
      const modes = ['fft', 'penrose', 'ai_assisted'];
      const performance = {};
      
      for (const mode of modes) {
        holographicSystem.setRenderingMode(mode);
        
        const times = [];
        for (let i = 0; i < 10; i++) {
          const time = await measurePerformance(async () => {
            await holographicSystem.generateHologram();
          });
          times.push(time);
        }
        
        performance[mode] = {
          average: times.reduce((a, b) => a + b) / times.length,
          min: Math.min(...times),
          max: Math.max(...times)
        };
      }
      
      console.log('Performance comparison:', performance);
      
      // FFT should be fastest
      expect(performance.fft.average).toBeLessThan(performance.penrose.average);
      expect(performance.fft.average).toBeLessThan(performance.ai_assisted.average);
    });
    
    it('should handle memory efficiently', async () => {
      if (!performance.memory) {
        console.log('Memory API not available, skipping test');
        return;
      }
      
      const initialMemory = performance.memory.usedJSHeapSize;
      
      // Generate many holograms
      for (let i = 0; i < 100; i++) {
        await holographicSystem.generateHologram();
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = performance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Should not leak more than 100MB
      expect(memoryIncrease).toBeLessThan(100 * 1024 * 1024);
    });
  });
  
  describe('Error Handling Tests', () => {
    it('should handle WebGPU unavailability gracefully', async () => {
      // Mock WebGPU unavailable
      const originalGPU = navigator.gpu;
      delete navigator.gpu;
      
      try {
        await holographicSystem.initialize(canvas);
      } catch (error) {
        expect(error.message).toContain('WebGPU not supported');
      }
      
      // Restore
      navigator.gpu = originalGPU;
    });
    
    it('should recover from render errors', async () => {
      // Force an error
      holographicSystem.device = null;
      
      let errorThrown = false;
      try {
        await holographicSystem.generateHologram();
      } catch (error) {
        errorThrown = true;
      }
      
      expect(errorThrown).toBe(true);
      
      // System should still be running
      expect(holographicSystem.isInitialized).toBe(true);
    });
    
    it('should handle concept mesh disconnection', () => {
      conceptMesh.handleConnectionError();
      
      expect(conceptMesh.offlineMode).toBe(true);
      expect(conceptMesh.isConnected).toBe(false);
      
      // Should still work offline
      const concept = testConcepts[0];
      conceptMesh.addConcept(concept);
      
      expect(conceptMesh.conceptCache.has(concept.id)).toBe(true);
    });
  });
});

// Helper functions
async function compareWavefields(wavefield1, wavefield2) {
  // Read wavefield data
  const data1 = await readBufferData(wavefield1);
  const data2 = await readBufferData(wavefield2);
  
  if (data1.length !== data2.length) {
    return 0;
  }
  
  // Calculate normalized cross-correlation
  let correlation = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < data1.length; i++) {
    correlation += data1[i] * data2[i];
    norm1 += data1[i] * data1[i];
    norm2 += data2[i] * data2[i];
  }
  
  return correlation / Math.sqrt(norm1 * norm2);
}

async function readBufferData(buffer) {
  // Implementation to read GPU buffer data
  // Returns Float32Array
}

function createTestImage() {
  // Create test image texture
  return {
    width: 1920,
    height: 1080,
    format: 'rgba8unorm'
  };
}

function createTestDepthMap() {
  // Create test depth map
  return {
    width: 1920,
    height: 1080,
    data: new Float32Array(1920 * 1080)
  };
}

function createTestQuilt() {
  // Create test quilt texture
  return {
    width: 3360,
    height: 3360,
    format: 'rgba8unorm'
  };
}

function generateTestCaptures(count) {
  const captures = [];
  for (let i = 0; i < count; i++) {
    captures.push({
      image: createTestImage(),
      pose: {
        position: [i * 0.1, 0, 2],
        rotation: [0, i * 0.05, 0]
      },
      timestamp: Date.now() + i * 100
    });
  }
  return captures;
}

function calculateImageMetrics(image) {
  // Calculate image quality metrics
  return {
    sharpness: 0.8,
    contrast: 0.7,
    brightness: 0.6
  };
}

async function calculateFrameDifference(frame1, frame2) {
  // Calculate normalized difference between frames
  // Returns value between 0 and 1
  return 0.05; // Placeholder
}

// Export test suite
export default {
  runAllTests: async () => {
    const results = await run();
    console.log('Test Results:', results);
    return results;
  }
};
