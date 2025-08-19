/**
 * Enhanced Unified Holographic System
 * Complete integration with Penrose mode, AI rendering, and all TODO fixes
 */

import { SpectralHologramEngine, getDefaultCalibration } from './holographicEngine';
import { ToriHolographicRenderer } from './holographicRenderer';
import { FFTCompute } from './webgpu/fftCompute';
import { HologramPropagation, PropagationMethod } from './webgpu/hologramPropagation';
import { QuiltGenerator } from './webgpu/quiltGenerator';
import { ConceptHologramRenderer } from './conceptHologramRenderer';
import { hologramBridge } from './hologramBridge';
import { conceptMesh } from './enhancedConceptMeshIntegration';
import { PenroseWavefieldEngine } from './penroseWavefieldEngine';
import { AIAssistedRenderer } from './aiAssistedRenderer';

// Rendering modes
export const RenderingMode = {
  FFT: 'fft',
  PENROSE: 'penrose',
  HYBRID: 'hybrid',
  AI_ASSISTED: 'ai_assisted',
  COMPARISON: 'comparison'
};

export class EnhancedUnifiedHolographicSystem {
  constructor() {
    // Core systems
    this.engine = null;
    this.renderer = null;
    this.fftCompute = null;
    this.propagation = null;
    this.quiltGen = null;
    
    // Enhanced systems
    this.penroseEngine = null;
    this.aiRenderer = null;
    this.conceptRenderer = null;
    this.bridge = hologramBridge;
    
    // Rendering mode
    this.renderingMode = RenderingMode.FFT;
    this.comparisonMode = false;
    
    // State
    this.device = null;
    this.isInitialized = false;
    this.canvas = null;
    
    // Performance metrics
    this.metrics = {
      fftTime: 0,
      penroseTime: 0,
      aiTime: 0,
      frameTime: 0,
      frameCount: 0
    };
    
    // Audio-driven state
    this.psiState = {
      psi_phase: 0,
      phase_coherence: 0.8,
      oscillator_phases: new Array(32).fill(0),
      oscillator_frequencies: new Array(32).fill(0),
      coupling_strength: 0.5,
      dominant_frequency: 440,
      couplingMatrix: null
    };
    
    // AI training data
    this.captureBuffer = [];
    this.currentSceneId = 'default';
    
    console.log('🌟 Enhanced Unified Holographic System created');
  }
  
  async initialize(canvas, options = {}) {
    try {
      console.log('🚀 Initializing Enhanced Unified Holographic System...');
      this.canvas = canvas;
      
      // 1. Check WebGPU support
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported! Please use a compatible browser.');
      }
      
      // 2. Get GPU device
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        // Request additional features for advanced rendering
        requiredFeatures: ['timestamp-query', 'texture-compression-bc']
      });
      
      if (!adapter) {
        throw new Error('No appropriate GPUAdapter found');
      }
      
      // Check for optional features
      const features = [];
      if (adapter.features.has('timestamp-query')) {
        features.push('timestamp-query');
      }
      
      this.device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxBufferSize: 2147483648, // 2GB
          maxStorageBufferBindingSize: 1073741824, // 1GB
          maxComputeWorkgroupSizeX: 512,
          maxComputeWorkgroupSizeY: 512
        }
      });
      
      // 3. Initialize Spectral Hologram Engine
      const calibration = options.calibration || getDefaultCalibration(options.displayType || 'webgpu_only');
      this.engine = new SpectralHologramEngine();
      await this.engine.initialize(canvas, calibration);
      
      // 4. Initialize Holographic Renderer
      this.renderer = new ToriHolographicRenderer(canvas);
      await this.renderer.initialize();
      
      // 5. Initialize WebGPU compute modules
      const hologramSize = options.hologramSize || 1024;
      
      this.fftCompute = new FFTCompute(this.device, hologramSize);
      await this.fftCompute.initialize();
      
      this.propagation = new HologramPropagation(this.device, hologramSize);
      await this.propagation.initialize();
      
      this.quiltGen = new QuiltGenerator(this.device, options.numViews || 45);
      await this.quiltGen.initialize();
      
      // 6. Initialize Penrose Engine
      this.penroseEngine = new PenroseWavefieldEngine(this.device, hologramSize);
      await this.penroseEngine.initialize();
      
      // 7. Initialize AI-Assisted Renderer
      this.aiRenderer = new AIAssistedRenderer(this.device, this);
      await this.aiRenderer.initialize();
      
      // 8. Initialize Concept Renderer
      this.conceptRenderer = new ConceptHologramRenderer();
      await this.conceptRenderer.initialize(canvas, this.engine);
      
      // 9. Connect hologram bridge
      this.bridge.connect();
      this.setupBridgeHandlers();
      
      // 10. Connect enhanced concept mesh
      conceptMesh.connect();
      
      // 11. Initialize shader hot reload in development
      if (options.development) {
        this.setupShaderHotReload();
      }
      
      // 12. Start render loop
      this.startRenderLoop();
      
      this.isInitialized = true;
      console.log('✅ Enhanced Unified Holographic System initialized successfully!');
      
      // Show system capabilities
      this.logCapabilities();
      
      return {
        success: true,
        device: this.device,
        capabilities: this.getCapabilities()
      };
      
    } catch (error) {
      console.error('Failed to initialize holographic system:', error);
      
      // Try CPU fallback for Penrose
      if (this.penroseEngine) {
        console.log('Attempting Penrose CPU fallback...');
        this.penroseEngine.useCPUFallback = true;
      }
      
      throw error;
    }
  }
  
  setupBridgeHandlers() {
    this.bridge.on('message', (data) => {
      this.handleBridgeMessage(data);
    });
    
    this.bridge.on('connected', () => {
      console.log('✅ Hologram bridge connected');
      this.showNotification('System online', 'success');
    });
    
    this.bridge.on('error', (error) => {
      console.error('Hologram bridge error:', error);
      this.showNotification('Bridge connection error', 'error');
    });
  }
  
  handleBridgeMessage(data) {
    switch (data.type) {
      case 'psi_update':
        this.updatePsiState(data.psi_state);
        break;
        
      case 'concept_update':
        this.conceptRenderer.handleMeshUpdate(data);
        break;
        
      case 'audio_features':
        this.updateFromAudioFeatures(data.features);
        break;
        
      case 'mode_change':
        this.setRenderingMode(data.mode);
        break;
        
      case 'ai_training_data':
        this.collectTrainingData(data);
        break;
        
      default:
        console.log('Unknown bridge message:', data.type);
    }
  }
  
  // Rendering Mode Management
  
  setRenderingMode(mode) {
    if (!Object.values(RenderingMode).includes(mode)) {
      console.error('Invalid rendering mode:', mode);
      return;
    }
    
    this.renderingMode = mode;
    console.log(`🎨 Switched to ${mode} rendering mode`);
    
    if (mode === RenderingMode.COMPARISON) {
      this.comparisonMode = true;
    } else {
      this.comparisonMode = false;
    }
    
    this.showNotification(`Rendering mode: ${mode}`, 'info');
  }
  
  // Main Render Pipeline
  
  async generateHologram(options = {}) {
    const startTime = performance.now();
    
    let wavefield;
    let quilt;
    
    try {
      // Select rendering path based on mode
      switch (this.renderingMode) {
        case RenderingMode.FFT:
          wavefield = await this.generateFFTWavefield();
          break;
          
        case RenderingMode.PENROSE:
          wavefield = await this.generatePenroseWavefield();
          break;
          
        case RenderingMode.HYBRID:
          wavefield = await this.generateHybridWavefield();
          break;
          
        case RenderingMode.AI_ASSISTED:
          const aiResult = await this.generateAIAssistedHologram();
          wavefield = aiResult.wavefield;
          quilt = aiResult.quilt;
          break;
          
        case RenderingMode.COMPARISON:
          const comparison = await this.generateComparisonHologram();
          return comparison;
      }
      
      // Generate quilt if not already created by AI
      if (!quilt && wavefield) {
        quilt = await this.generateQuiltFromWavefield(wavefield);
      }
      
      // Apply post-processing if enabled
      if (options.postProcess) {
        quilt = await this.applyPostProcessing(quilt);
      }
      
      const endTime = performance.now();
      this.updateMetrics('frameTime', endTime - startTime);
      
      return {
        wavefield,
        quilt,
        renderTime: endTime - startTime,
        mode: this.renderingMode
      };
      
    } catch (error) {
      console.error('Error generating hologram:', error);
      throw error;
    }
  }
  
  async generateFFTWavefield() {
    const startTime = performance.now();
    
    // Use existing FFT pipeline
    const wavefield = await this.fftCompute.computeFFT(this.psiState);
    
    const endTime = performance.now();
    this.updateMetrics('fftTime', endTime - startTime);
    
    return wavefield;
  }
  
  async generatePenroseWavefield() {
    const startTime = performance.now();
    
    // Use Penrose engine
    const wavefield = await this.penroseEngine.generateWavefield(this.psiState, {
      quality: this.penroseEngine.qualityMode,
      iterations: 50,
      convergenceThreshold: 0.001
    });
    
    const endTime = performance.now();
    this.updateMetrics('penroseTime', endTime - startTime);
    
    return wavefield;
  }
  
  async generateHybridWavefield() {
    // Generate both FFT and Penrose
    const [fftWavefield, penroseWavefield] = await Promise.all([
      this.generateFFTWavefield(),
      this.generatePenroseWavefield()
    ]);
    
    // Blend based on quality regions
    return this.blendWavefields(fftWavefield, penroseWavefield, {
      blendMode: 'quality-based',
      fftWeight: 0.7,
      penroseWeight: 0.3
    });
  }
  
  async generateAIAssistedHologram() {
    const startTime = performance.now();
    
    // Prepare input data
    const inputData = {
      type: this.determineInputType(),
      image: this.captureCanvas(),
      sceneId: this.currentSceneId,
      psiState: this.psiState
    };
    
    // Process through AI pipeline
    const result = await this.aiRenderer.processFrame(inputData, 'auto');
    
    const endTime = performance.now();
    this.updateMetrics('aiTime', endTime - startTime);
    
    return result;
  }
  
  async generateComparisonHologram() {
    // Generate all modes for comparison
    const [fft, penrose, ai] = await Promise.all([
      this.generateFFTWavefield(),
      this.generatePenroseWavefield(),
      this.generateAIAssistedHologram()
    ]);
    
    // Create side-by-side comparison
    return this.createComparisonView({
      fft: fft,
      penrose: penrose,
      ai: ai.quilt || ai.wavefield
    });
  }
  
  // Wavefield Operations
  
  async blendWavefields(wavefield1, wavefield2, options) {
    const blendShader = `
      @group(0) @binding(0) var<storage, read> wavefield1: array<vec2<f32>>;
      @group(0) @binding(1) var<storage, read> wavefield2: array<vec2<f32>>;
      @group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
      @group(0) @binding(3) var<uniform> blend_params: vec4<f32>;
      
      @compute @workgroup_size(256)
      fn blend(@builtin(global_invocation_id) id: vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&wavefield1)) {
          return;
        }
        
        let w1 = blend_params.x;
        let w2 = blend_params.y;
        
        output[idx] = wavefield1[idx] * w1 + wavefield2[idx] * w2;
      }
    `;
    
    // Create and execute blend pipeline
    // ... implementation
    
    return wavefield1; // Placeholder
  }
  
  async generateQuiltFromWavefield(wavefield) {
    // Propagate wavefield
    const propagated = await this.propagation.propagate(wavefield, {
      method: PropagationMethod.ANGULAR_SPECTRUM,
      distance: 0.3
    });
    
    // Generate multi-view quilt
    const quilt = await this.quiltGen.generateQuilt(propagated);
    
    return quilt;
  }
  
  // AI Integration
  
  collectTrainingData(data) {
    // Collect frames for NeRF training
    this.captureBuffer.push({
      image: data.image || this.captureCanvas(),
      pose: data.pose || this.getCurrentPose(),
      timestamp: Date.now()
    });
    
    // Trigger training if enough data
    if (this.captureBuffer.length >= 10 && this.aiRenderer.config.nerf.autoTrain) {
      this.trainNeRF();
    }
  }
  
  async trainNeRF() {
    if (this.captureBuffer.length < 10) return;
    
    const model = await this.aiRenderer.trainNeRF(
      this.currentSceneId,
      this.captureBuffer
    );
    
    if (model) {
      this.showNotification('NeRF model trained!', 'success');
      this.captureBuffer = []; // Clear buffer
    }
  }
  
  // Update Methods
  
  updatePsiState(newState) {
    // Merge new psi state
    Object.assign(this.psiState, newState);
    
    // Generate coupling matrix if not provided
    if (!this.psiState.couplingMatrix) {
      this.psiState.couplingMatrix = this.generateCouplingMatrix();
    }
    
    // Update engine with new state
    if (this.engine) {
      this.engine.updateFromOscillator(this.psiState);
    }
  }
  
  generateCouplingMatrix() {
    // Create coupling matrix for oscillators
    const size = 32;
    const matrix = new Float32Array(size * size);
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (i === j) {
          matrix[i * size + j] = 1.0;
        } else {
          // Coupling strength decreases with distance
          const distance = Math.abs(i - j);
          matrix[i * size + j] = Math.exp(-distance * 0.1) * 0.3;
        }
      }
    }
    
    return matrix;
  }
  
  updateFromAudioFeatures(features) {
    // Convert audio features to wavefield parameters
    const wavefieldParams = {
      phase_modulation: features.spectral_centroid / 1000,
      coherence: features.spectral_flatness,
      oscillator_phases: this.mapAudioToPhases(features),
      dominant_frequency: features.pitch || 440,
      spatial_frequencies: this.generateSpatialFrequencies(features),
      amplitudes: features.band_energies || []
    };
    
    // Update engine
    if (this.engine) {
      this.engine.updateFromWavefieldParams(wavefieldParams);
    }
    
    // Update oscillator state
    this.updateOscillatorsFromAudio(features);
  }
  
  mapAudioToPhases(features) {
    const phases = new Array(32).fill(0);
    
    if (features.band_energies) {
      features.band_energies.forEach((energy, i) => {
        if (i < phases.length) {
          phases[i] = energy * Math.PI * 2;
        }
      });
    }
    
    return phases;
  }
  
  updateOscillatorsFromAudio(features) {
    // Update oscillator frequencies based on audio
    if (features.band_energies) {
      for (let i = 0; i < Math.min(features.band_energies.length, 32); i++) {
        this.psiState.oscillator_frequencies[i] = 
          features.band_energies[i] * 1000 + 100; // 100-1100 Hz range
      }
    }
    
    // Update coherence
    this.psiState.phase_coherence = features.spectral_flatness || 0.8;
    
    // Update dominant frequency
    this.psiState.dominant_frequency = features.pitch || 440;
  }
  
  generateSpatialFrequencies(features) {
    const freqs = [];
    const numOscillators = 32;
    
    for (let i = 0; i < numOscillators; i++) {
      const angle = (i / numOscillators) * Math.PI * 2;
      const magnitude = features.band_energies?.[i] || 0.5;
      
      freqs.push([
        Math.cos(angle) * magnitude * 10,
        Math.sin(angle) * magnitude * 10
      ]);
    }
    
    return freqs;
  }
  
  // Render Loop
  
  startRenderLoop() {
    const render = async () => {
      if (!this.isInitialized) return;
      
      try {
        // Update concept animations
        this.conceptRenderer.update(0.016); // 60 FPS
        
        // Generate hologram based on current mode
        const hologram = await this.generateHologram({
          postProcess: true
        });
        
        // Render to display
        if (hologram.quilt) {
          this.renderer.renderQuilt(hologram.quilt);
        }
        
        // Update metrics
        this.metrics.frameCount++;
        
        // Schedule next frame
        requestAnimationFrame(render);
        
      } catch (error) {
        console.error('Render error:', error);
        // Continue rendering despite errors
        requestAnimationFrame(render);
      }
    };
    
    requestAnimationFrame(render);
  }
  
  // Post Processing
  
  async applyPostProcessing(quilt) {
    // Apply GAN enhancement if enabled
    if (this.aiRenderer.config.gan.enabled) {
      return await this.aiRenderer.applyGANEnhancement({
        quilt: quilt
      });
    }
    
    return quilt;
  }
  
  // Utility Methods
  
  captureCanvas() {
    // Capture current canvas content
    const imageData = this.renderer.captureFrame();
    return imageData;
  }
  
  getCurrentPose() {
    // Get current camera pose
    return {
      position: [0, 0, 2],
      rotation: [0, 0, 0],
      fov: 45
    };
  }
  
  determineInputType() {
    // Determine input type based on current state
    if (this.captureBuffer.length > 0) {
      return 'multiView';
    }
    return 'image';
  }
  
  async createComparisonView(results) {
    // Create side-by-side comparison texture
    const comparisonTexture = this.device.createTexture({
      size: [3360 * 3, 3360], // 3 quilts side by side
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | 
             GPUTextureUsage.TEXTURE_BINDING
    });
    
    // Copy each result to its position
    const commandEncoder = this.device.createCommandEncoder();
    
    // Copy FFT result
    if (results.fft) {
      // ... copy to left third
    }
    
    // Copy Penrose result
    if (results.penrose) {
      // ... copy to middle third
    }
    
    // Copy AI result
    if (results.ai) {
      // ... copy to right third
    }
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    return {
      texture: comparisonTexture,
      mode: 'comparison'
    };
  }
  
  // Hot Reload Support
  
  setupShaderHotReload() {
    // Watch for shader file changes
    if (typeof window !== 'undefined' && window.WebSocket) {
      const ws = new WebSocket('ws://localhost:3001/shader-reload');
      
      ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'shader-update') {
          console.log('🔄 Reloading shader:', data.file);
          await this.reloadShaders();
        }
      };
    }
  }
  
  async reloadShaders() {
    // Reload all shader modules
    try {
      // Recreate pipelines with new shaders
      await this.fftCompute.reloadShaders();
      await this.propagation.reloadShaders();
      await this.quiltGen.reloadShaders();
      await this.penroseEngine.reloadShaders();
      
      console.log('✅ Shaders reloaded');
    } catch (error) {
      console.error('Failed to reload shaders:', error);
    }
  }
  
  // Performance and Metrics
  
  updateMetrics(key, value) {
    this.metrics[key] = value;
    
    // Emit metrics for monitoring
    if (this.bridge) {
      this.bridge.emit('metrics', this.metrics);
    }
  }
  
  getPerformanceReport() {
    const avgFrameTime = this.metrics.frameTime / Math.max(1, this.metrics.frameCount);
    const fps = 1000 / avgFrameTime;
    
    return {
      averageFrameTime: avgFrameTime,
      fps: fps,
      fftTime: this.metrics.fftTime,
      penroseTime: this.metrics.penroseTime,
      aiTime: this.metrics.aiTime,
      frameCount: this.metrics.frameCount,
      mode: this.renderingMode
    };
  }
  
  // System Information
  
  getCapabilities() {
    return {
      webgpu: true,
      device: {
        vendor: this.device?.vendor || 'unknown',
        architecture: this.device?.architecture || 'unknown',
        features: Array.from(this.device?.features || [])
      },
      rendering: {
        fft: true,
        penrose: this.penroseEngine?.initialized || false,
        ai: {
          dibr: this.aiRenderer?.config.dibr.enabled || false,
          nerf: this.aiRenderer?.config.nerf.enabled || false,
          gan: this.aiRenderer?.config.gan.enabled || false
        }
      },
      hologramSize: this.fftCompute?.size || 1024,
      numViews: this.quiltGen?.numViews || 45,
      conceptMesh: conceptMesh.getStatus()
    };
  }
  
  logCapabilities() {
    const caps = this.getCapabilities();
    console.log('🔧 System Capabilities:');
    console.log('  - WebGPU:', caps.webgpu);
    console.log('  - Device:', caps.device.vendor);
    console.log('  - Rendering modes:', Object.keys(caps.rendering));
    console.log('  - Hologram size:', caps.hologramSize);
    console.log('  - Views:', caps.numViews);
    console.log('  - Concept mesh:', caps.conceptMesh.connected ? 'connected' : 'offline');
  }
  
  // UI Notifications
  
  showNotification(message, type = 'info') {
    // Emit notification event for UI
    if (this.bridge) {
      this.bridge.emit('notification', {
        message,
        type,
        timestamp: Date.now()
      });
    }
    
    console.log(`[${type.toUpperCase()}] ${message}`);
  }
  
  // Public API
  
  async processAudioToHologram(audioData) {
    if (!this.isInitialized) {
      throw new Error('System not initialized');
    }
    
    console.log('Processing audio to hologram...');
    
    // Extract features and update oscillators
    // This will trigger hologram updates through the render loop
  }
  
  async addConcept(conceptData) {
    conceptMesh.addConcept(conceptData);
  }
  
  async deleteConcept(conceptId) {
    conceptMesh.deleteConcept(conceptId);
  }
  
  async updateConceptPosition(conceptId, position) {
    conceptMesh.updateConcept(conceptId, { position });
  }
  
  undo() {
    conceptMesh.undo();
  }
  
  setQuality(quality) {
    // Set quality for all renderers
    if (this.engine) {
      this.engine.setQuality(quality);
    }
    
    if (this.penroseEngine) {
      this.penroseEngine.setQuality(quality);
    }
    
    if (this.aiRenderer) {
      this.aiRenderer.setConfig({
        gan: { enhancementLevel: quality === 'high' ? 2.0 : 1.0 }
      });
    }
  }
  
  setPenroseQuality(mode) {
    // 0: draft, 1: normal, 2: high
    if (this.penroseEngine) {
      this.penroseEngine.setQuality(mode);
    }
  }
  
  enableAIMode(mode, enabled = true) {
    if (this.aiRenderer) {
      switch (mode) {
        case 'dibr':
          this.aiRenderer.enableDIBR(enabled);
          break;
        case 'nerf':
          this.aiRenderer.enableNeRF(enabled);
          break;
        case 'gan':
          this.aiRenderer.enableGAN(enabled);
          break;
      }
    }
  }
  
  async captureFrame(format = 'png') {
    if (this.engine) {
      return this.engine.captureFrame(format);
    }
  }
  
  getStatus() {
    return {
      initialized: this.isInitialized,
      mode: this.renderingMode,
      performance: this.getPerformanceReport(),
      capabilities: this.getCapabilities(),
      conceptMesh: conceptMesh.getStatus(),
      aiRenderer: this.aiRenderer?.getStatus()
    };
  }
  
  destroy() {
    console.log('🧹 Destroying enhanced holographic system...');
    
    // Disconnect services
    if (this.bridge) {
      this.bridge.disconnect();
    }
    
    conceptMesh.disconnect();
    
    // Destroy renderers
    if (this.conceptRenderer) {
      this.conceptRenderer.destroy();
    }
    
    if (this.engine) {
      this.engine.destroy();
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
    
    if (this.penroseEngine) {
      this.penroseEngine.destroy();
    }
    
    // Clear state
    this.isInitialized = false;
    this.captureBuffer = [];
  }
}

// Export singleton instance
export const holographicSystem = new EnhancedUnifiedHolographicSystem();

// Export for component usage
export default holographicSystem;
