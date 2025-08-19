/**
 * Ghost Engine - Advanced holographic rendering and ψ-state management
 * Enhanced with WebGPU support, real-time synchronization, and predictive capabilities
 */

import { writable, derived, get } from 'svelte/store';
import { psiMemoryStore } from './stores/psiMemory';
import { interpolateHologramStates, getHolographicHighlights } from '../../core/psiMemory/psiFrames';

// Engine state store
const engineState = writable({
  initialized: false,
  running: false,
  mode: 'holographic', // 'holographic', 'preview', 'debug'
  performance: {
    fps: 0,
    frameTime: 0,
    drawCalls: 0,
    gpuMemory: 0,
    cpuUsage: 0
  },
  capabilities: {
    webgpu: false,
    webgl2: false,
    xr: false,
    offscreenCanvas: false,
    videoDecoder: false
  },
  error: null
});

// Configuration
const config = {
  targetFPS: 60,
  adaptiveQuality: true,
  maxDrawCalls: 1000,
  particleLimit: 10000,
  volumeResolution: 64,
  enablePostProcessing: true,
  enablePrediction: true,
  predictionSteps: 5,
  syncInterval: 100, // ms
  renderScale: 1.0,
  antialias: true,
  shadows: true,
  bloom: true,
  motionBlur: true,
  depthOfField: false,
  ambientOcclusion: true
};

// Performance monitoring
class PerformanceMonitor {
  constructor() {
    this.frameCount = 0;
    this.lastTime = performance.now();
    this.lastFPSUpdate = 0;
    this.frameTimes = [];
    this.maxFrameTimes = 60;
    
    // GPU timing (if available)
    this.gpuTimer = null;
    this.cpuTimer = null;
  }
  
  startFrame() {
    this.frameStartTime = performance.now();
    
    if (this.gpuTimer) {
      this.gpuTimer.start();
    }
  }
  
  endFrame() {
    const frameTime = performance.now() - this.frameStartTime;
    this.frameTimes.push(frameTime);
    
    if (this.frameTimes.length > this.maxFrameTimes) {
      this.frameTimes.shift();
    }
    
    this.frameCount++;
    
    // Update FPS every second
    const now = performance.now();
    if (now - this.lastFPSUpdate > 1000) {
      const fps = this.frameCount * 1000 / (now - this.lastFPSUpdate);
      const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      
      engineState.update(state => ({
        ...state,
        performance: {
          ...state.performance,
          fps: Math.round(fps),
          frameTime: avgFrameTime,
          cpuUsage: this.estimateCPUUsage()
        }
      }));
      
      this.frameCount = 0;
      this.lastFPSUpdate = now;
    }
    
    if (this.gpuTimer) {
      this.gpuTimer.end();
    }
  }
  
  estimateCPUUsage() {
    // Estimate based on frame time vs target
    const targetFrameTime = 1000 / config.targetFPS;
    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    return Math.min(100, (avgFrameTime / targetFrameTime) * 100);
  }
  
  getMetrics() {
    return {
      fps: get(engineState).performance.fps,
      frameTime: get(engineState).performance.frameTime,
      consistency: this.calculateConsistency(),
      gpuTime: this.gpuTimer ? this.gpuTimer.getAverage() : 0
    };
  }
  
  calculateConsistency() {
    if (this.frameTimes.length < 2) return 100;
    
    const mean = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const variance = this.frameTimes.reduce((sum, time) => sum + Math.pow(time - mean, 2), 0) / this.frameTimes.length;
    const stdDev = Math.sqrt(variance);
    
    // Convert to percentage (lower std dev = higher consistency)
    return Math.max(0, 100 - (stdDev / mean) * 100);
  }
}

// Resource manager for efficient memory usage
class ResourceManager {
  constructor() {
    this.textures = new Map();
    this.buffers = new Map();
    this.shaders = new Map();
    this.materials = new Map();
    this.geometries = new Map();
    this.refCounts = new Map();
  }
  
  async loadTexture(url, options = {}) {
    if (this.textures.has(url)) {
      this.incrementRef(url);
      return this.textures.get(url);
    }
    
    try {
      const texture = await this.createTexture(url, options);
      this.textures.set(url, texture);
      this.refCounts.set(url, 1);
      return texture;
    } catch (error) {
      console.error(`Failed to load texture: ${url}`, error);
      throw error;
    }
  }
  
  async createTexture(url, options) {
    const response = await fetch(url);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    
    // Create texture based on renderer type
    const renderer = ghostEngine.renderer;
    if (!renderer) throw new Error('Renderer not initialized');
    
    return renderer.createTexture(bitmap, options);
  }
  
  incrementRef(key) {
    const count = this.refCounts.get(key) || 0;
    this.refCounts.set(key, count + 1);
  }
  
  decrementRef(key) {
    const count = this.refCounts.get(key) || 0;
    if (count <= 1) {
      // Release resource
      this.releaseResource(key);
      this.refCounts.delete(key);
    } else {
      this.refCounts.set(key, count - 1);
    }
  }
  
  releaseResource(key) {
    // Release from appropriate map
    if (this.textures.has(key)) {
      const texture = this.textures.get(key);
      texture.destroy?.();
      this.textures.delete(key);
    }
    // Similar for other resource types...
  }
  
  releaseAll() {
    // Release all resources
    this.textures.forEach(texture => texture.destroy?.());
    this.buffers.forEach(buffer => buffer.destroy?.());
    
    this.textures.clear();
    this.buffers.clear();
    this.shaders.clear();
    this.materials.clear();
    this.geometries.clear();
    this.refCounts.clear();
  }
  
  getMemoryUsage() {
    let total = 0;
    
    // Estimate texture memory
    this.textures.forEach(texture => {
      if (texture.width && texture.height) {
        // Assume RGBA8 format (4 bytes per pixel)
        total += texture.width * texture.height * 4;
      }
    });
    
    // Estimate buffer memory
    this.buffers.forEach(buffer => {
      total += buffer.size || 0;
    });
    
    return total;
  }
}

// Scene graph for holographic objects
class SceneGraph {
  constructor() {
    this.root = {
      id: 'root',
      children: [],
      transform: new Transform(),
      visible: true
    };
    this.nodes = new Map();
    this.nodes.set('root', this.root);
    this.dirtyNodes = new Set();
  }
  
  addNode(node, parentId = 'root') {
    const parent = this.nodes.get(parentId);
    if (!parent) throw new Error(`Parent node ${parentId} not found`);
    
    node.id = node.id || `node_${Date.now()}_${Math.random()}`;
    node.parent = parent;
    node.children = node.children || [];
    node.transform = node.transform || new Transform();
    node.worldTransform = new Transform();
    
    parent.children.push(node);
    this.nodes.set(node.id, node);
    this.markDirty(node);
    
    return node.id;
  }
  
  removeNode(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node || nodeId === 'root') return;
    
    // Remove from parent
    const parent = node.parent;
    if (parent) {
      parent.children = parent.children.filter(child => child.id !== nodeId);
    }
    
    // Remove all descendants
    this.removeDescendants(node);
    
    this.nodes.delete(nodeId);
    this.dirtyNodes.delete(nodeId);
  }
  
  removeDescendants(node) {
    node.children.forEach(child => {
      this.removeDescendants(child);
      this.nodes.delete(child.id);
      this.dirtyNodes.delete(child.id);
    });
  }
  
  updateNode(nodeId, updates) {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    
    Object.assign(node, updates);
    this.markDirty(node);
  }
  
  markDirty(node) {
    this.dirtyNodes.add(node.id);
    
    // Mark all descendants as dirty
    node.children.forEach(child => this.markDirty(child));
  }
  
  updateTransforms() {
    // Update only dirty nodes
    this.dirtyNodes.forEach(nodeId => {
      const node = this.nodes.get(nodeId);
      if (node) {
        this.updateNodeTransform(node);
      }
    });
    
    this.dirtyNodes.clear();
  }
  
  updateNodeTransform(node) {
    if (node.parent && node.parent.worldTransform) {
      node.worldTransform.multiplyMatrices(
        node.parent.worldTransform.matrix,
        node.transform.matrix
      );
    } else {
      node.worldTransform.copy(node.transform);
    }
  }
  
  traverse(callback, node = this.root) {
    if (!node.visible) return;
    
    callback(node);
    
    node.children.forEach(child => {
      this.traverse(callback, child);
    });
  }
  
  findNode(predicate) {
    let found = null;
    
    this.traverse(node => {
      if (predicate(node)) {
        found = node;
        return true; // Stop traversal
      }
    });
    
    return found;
  }
}

// Transform class for 3D transformations
class Transform {
  constructor() {
    this.position = { x: 0, y: 0, z: 0 };
    this.rotation = { x: 0, y: 0, z: 0 };
    this.scale = { x: 1, y: 1, z: 1 };
    this.matrix = new Float32Array(16);
    this.dirty = true;
    
    this.updateMatrix();
  }
  
  setPosition(x, y, z) {
    this.position.x = x;
    this.position.y = y;
    this.position.z = z;
    this.dirty = true;
  }
  
  setRotation(x, y, z) {
    this.rotation.x = x;
    this.rotation.y = y;
    this.rotation.z = z;
    this.dirty = true;
  }
  
  setScale(x, y, z) {
    this.scale.x = x;
    this.scale.y = y || x;
    this.scale.z = z || x;
    this.dirty = true;
  }
  
  updateMatrix() {
    if (!this.dirty) return;
    
    // Create transformation matrix
    const c = Math.cos;
    const s = Math.sin;
    const { x: rx, y: ry, z: rz } = this.rotation;
    const { x: sx, y: sy, z: sz } = this.scale;
    const { x: tx, y: ty, z: tz } = this.position;
    
    // Rotation matrices
    const cx = c(rx), sx = s(rx);
    const cy = c(ry), sy = s(ry);
    const cz = c(rz), sz = s(rz);
    
    // Combined rotation matrix
    this.matrix[0] = cy * cz * sx;
    this.matrix[1] = cy * sz * sx;
    this.matrix[2] = -sy * sx;
    this.matrix[3] = 0;
    
    this.matrix[4] = (sx * sy * cz - cx * sz) * sy;
    this.matrix[5] = (sx * sy * sz + cx * cz) * sy;
    this.matrix[6] = sx * cy * sy;
    this.matrix[7] = 0;
    
    this.matrix[8] = (cx * sy * cz + sx * sz) * sz;
    this.matrix[9] = (cx * sy * sz - sx * cz) * sz;
    this.matrix[10] = cx * cy * sz;
    this.matrix[11] = 0;
    
    this.matrix[12] = tx;
    this.matrix[13] = ty;
    this.matrix[14] = tz;
    this.matrix[15] = 1;
    
    this.dirty = false;
  }
  
  copy(other) {
    this.position = { ...other.position };
    this.rotation = { ...other.rotation };
    this.scale = { ...other.scale };
    this.matrix = new Float32Array(other.matrix);
    this.dirty = other.dirty;
  }
  
  multiplyMatrices(a, b) {
    const result = new Float32Array(16);
    
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        result[i * 4 + j] = 
          a[i * 4 + 0] * b[0 * 4 + j] +
          a[i * 4 + 1] * b[1 * 4 + j] +
          a[i * 4 + 2] * b[2 * 4 + j] +
          a[i * 4 + 3] * b[3 * 4 + j];
      }
    }
    
    this.matrix = result;
    this.dirty = false;
  }
}

// ψ-state synchronization
class PsiStateSynchronizer {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.sessionId = `ghost_${Date.now()}`;
    this.lastSync = 0;
    this.syncBuffer = [];
  }
  
  async connect(url = '/api/v2/hologram/ws') {
    try {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${url}/${this.sessionId}`;
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('Ghost Engine WebSocket connected');
        this.reconnectAttempts = 0;
        
        // Send initial state
        this.sendState({
          type: 'ghost_engine_init',
          capabilities: get(engineState).capabilities,
          config: config
        });
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.handleDisconnect();
      };
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.handleDisconnect();
    }
  }
  
  handleMessage(data) {
    switch (data.type) {
      case 'psi_update':
        psiMemoryStore.updateFromServer(data.psi_state, data.hologram_hints);
        break;
        
      case 'config_update':
        Object.assign(config, data.config);
        break;
        
      case 'sync_request':
        this.sendCurrentState();
        break;
        
      case 'performance_warning':
        this.handlePerformanceWarning(data);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  }
  
  handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      engineState.update(state => ({
        ...state,
        error: 'WebSocket connection lost'
      }));
    }
  }
  
  sendState(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // Buffer for later
      this.syncBuffer.push(data);
    }
  }
  
  sendCurrentState() {
    const currentState = get(psiMemoryStore);
    
    this.sendState({
      type: 'state_sync',
      timestamp: Date.now(),
      psi_state: currentState.currentState,
      performance: get(engineState).performance
    });
  }
  
  handlePerformanceWarning(data) {
    console.warn('Performance warning:', data.message);
    
    // Auto-adjust quality if enabled
    if (config.adaptiveQuality) {
      this.adjustQuality(data.metrics);
    }
  }
  
  adjustQuality(metrics) {
    if (metrics.fps < 30) {
      // Reduce quality
      config.renderScale = Math.max(0.5, config.renderScale - 0.1);
      config.particleLimit = Math.max(1000, config.particleLimit - 1000);
      config.shadows = false;
      config.bloom = false;
    } else if (metrics.fps > 55) {
      // Increase quality
      config.renderScale = Math.min(1.0, config.renderScale + 0.1);
      config.particleLimit = Math.min(10000, config.particleLimit + 1000);
      config.shadows = true;
      config.bloom = true;
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Main Ghost Engine class
class GhostEngine {
  constructor() {
    this.renderer = null;
    this.scene = new SceneGraph();
    this.resources = new ResourceManager();
    this.performance = new PerformanceMonitor();
    this.synchronizer = new PsiStateSynchronizer();
    this.animationId = null;
    this.lastFrameTime = 0;
    this.frameCallbacks = new Set();
    this.initialized = false;
  }
  
  async initialize(canvas, options = {}) {
    try {
      // Detect capabilities
      await this.detectCapabilities();
      
      // Create renderer based on capabilities
      const state = get(engineState);
      
      if (state.capabilities.webgpu) {
        const { ToriHolographicRenderer } = await import('./holographicRenderer');
        this.renderer = new ToriHolographicRenderer(canvas);
      } else if (state.capabilities.webgl2) {
        const { WebGLFallbackRenderer } = await import('./webglRenderer');
        this.renderer = new WebGLFallbackRenderer(canvas);
      } else {
        throw new Error('No suitable renderer available');
      }
      
      // Initialize renderer
      await this.renderer.initialize();
      
      // Connect synchronizer
      if (options.enableSync !== false) {
        await this.synchronizer.connect();
      }
      
      // Update state
      engineState.update(state => ({
        ...state,
        initialized: true,
        error: null
      }));
      
      this.initialized = true;
      
      console.log('Ghost Engine initialized successfully');
      
    } catch (error) {
      console.error('Ghost Engine initialization failed:', error);
      
      engineState.update(state => ({
        ...state,
        initialized: false,
        error: error.message
      }));
      
      throw error;
    }
  }
  
  async detectCapabilities() {
    const capabilities = {
      webgpu: false,
      webgl2: false,
      xr: false,
      offscreenCanvas: false,
      videoDecoder: false
    };
    
    // Check WebGPU
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        capabilities.webgpu = !!adapter;
      } catch (e) {
        console.log('WebGPU not available:', e);
      }
    }
    
    // Check WebGL2
    const testCanvas = document.createElement('canvas');
    const gl = testCanvas.getContext('webgl2');
    capabilities.webgl2 = !!gl;
    
    // Check WebXR
    capabilities.xr = 'xr' in navigator;
    
    // Check OffscreenCanvas
    capabilities.offscreenCanvas = 'OffscreenCanvas' in window;
    
    // Check VideoDecoder
    capabilities.videoDecoder = 'VideoDecoder' in window;
    
    engineState.update(state => ({
      ...state,
      capabilities
    }));
    
    return capabilities;
  }
  
  start() {
    if (!this.initialized || get(engineState).running) return;
    
    engineState.update(state => ({
      ...state,
      running: true
    }));
    
    this.lastFrameTime = performance.now();
    this.animate();
  }
  
  stop() {
    engineState.update(state => ({
      ...state,
      running: false
    }));
    
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
  
  animate() {
    if (!get(engineState).running) return;
    
    this.animationId = requestAnimationFrame(() => this.animate());
    
    const now = performance.now();
    const deltaTime = now - this.lastFrameTime;
    
    // Limit delta time to prevent large jumps
    const clampedDelta = Math.min(deltaTime, 100);
    
    this.performance.startFrame();
    
    // Update scene
    this.update(clampedDelta / 1000); // Convert to seconds
    
    // Render
    this.render();
    
    this.performance.endFrame();
    
    this.lastFrameTime = now;
  }
  
  update(deltaTime) {
    // Update scene graph transforms
    this.scene.updateTransforms();
    
    // Update ψ-state interpolation
    const alpha = (performance.now() % 1000) / 1000;
    const interpolatedState = interpolateHologramStates(alpha);
    
    // Apply interpolated state to scene
    this.applyPsiState(interpolatedState);
    
    // Call frame callbacks
    this.frameCallbacks.forEach(callback => {
      callback(deltaTime, interpolatedState);
    });
    
    // Sync state periodically
    if (performance.now() - this.synchronizer.lastSync > config.syncInterval) {
      this.synchronizer.sendCurrentState();
      this.synchronizer.lastSync = performance.now();
    }
  }
  
  render() {
    if (!this.renderer) return;
    
    const psiState = get(psiMemoryStore).currentState;
    const drawCalls = { count: 0 };
    
    // Create render scene
    const renderScene = {
      render: (renderPass) => {
        this.scene.traverse(node => {
          if (node.mesh && node.visible) {
            // Apply world transform
            node.mesh.setTransform(node.worldTransform.matrix);
            
            // Render mesh
            node.mesh.render(renderPass);
            drawCalls.count++;
          }
          
          // Render particles if present
          if (node.particles) {
            node.particles.render(renderPass);
            drawCalls.count++;
          }
          
          // Render volume if present
          if (node.volume) {
            node.volume.render(renderPass);
            drawCalls.count++;
          }
        });
      }
    };
    
    // Render frame
    this.renderer.renderFrame(renderScene, psiState);
    
    // Update performance metrics
    engineState.update(state => ({
      ...state,
      performance: {
        ...state.performance,
        drawCalls: drawCalls.count,
        gpuMemory: this.resources.getMemoryUsage()
      }
    }));
  }
  
  applyPsiState(state) {
    if (!state) return;
    
    // Update holographic objects based on ψ-state
    this.scene.traverse(node => {
      if (node.psiResponsive) {
        // Update position based on ψ-phase
        if (node.psiMapping?.position) {
          const offset = node.psiMapping.position;
          node.transform.setPosition(
            state.position_3d.x + offset.x * Math.sin(state.animationHints?.phaseOffset || 0),
            state.position_3d.y + offset.y,
            state.position_3d.z + offset.z * Math.cos(state.animationHints?.phaseOffset || 0)
          );
          this.scene.markDirty(node);
        }
        
        // Update color based on emotional resonance
        if (node.mesh?.material && state.colorHSL) {
          node.mesh.material.setColor(state.colorHSL);
        }
        
        // Update particle density
        if (node.particles && state.synchronizationVisual) {
          node.particles.setDensity(state.synchronizationVisual.particleDensity);
        }
      }
    });
  }
  
  // Public API
  
  addHolographicObject(options) {
    const node = {
      id: options.id,
      transform: new Transform(),
      visible: true,
      psiResponsive: options.psiResponsive !== false,
      psiMapping: options.psiMapping || {}
    };
    
    // Create mesh if geometry provided
    if (options.geometry) {
      node.mesh = this.createMesh(options.geometry, options.material);
    }
    
    // Create particles if requested
    if (options.particles) {
      node.particles = this.createParticleSystem(options.particles);
    }
    
    // Create volume if requested
    if (options.volume) {
      node.volume = this.createVolume(options.volume);
    }
    
    // Set initial transform
    if (options.position) {
      node.transform.setPosition(options.position.x, options.position.y, options.position.z);
    }
    if (options.rotation) {
      node.transform.setRotation(options.rotation.x, options.rotation.y, options.rotation.z);
    }
    if (options.scale) {
      node.transform.setScale(options.scale.x || options.scale, options.scale.y || options.scale, options.scale.z || options.scale);
    }
    
    return this.scene.addNode(node, options.parent);
  }
  
  removeHolographicObject(id) {
    this.scene.removeNode(id);
  }
  
  updateHolographicObject(id, updates) {
    this.scene.updateNode(id, updates);
  }
  
  createMesh(geometry, material) {
    // Implementation depends on renderer type
    return {
      geometry,
      material,
      setTransform: (matrix) => {
        // Update shader uniforms
      },
      render: (renderPass) => {
        // Render mesh
      }
    };
  }
  
  createParticleSystem(options) {
    return {
      count: options.count || 1000,
      emissionRate: options.emissionRate || 10,
      lifetime: options.lifetime || 2,
      setDensity: (density) => {
        // Update particle density
      },
      render: (renderPass) => {
        // Render particles
      }
    };
  }
  
  createVolume(options) {
    return {
      resolution: options.resolution || 32,
      data: options.data || null,
      render: (renderPass) => {
        // Render volume
      }
    };
  }
  
  onFrame(callback) {
    this.frameCallbacks.add(callback);
    
    return () => {
      this.frameCallbacks.delete(callback);
    };
  }
  
  async captureFrame(format = 'png') {
    if (!this.renderer?.canvas) return null;
    
    return new Promise((resolve) => {
      this.renderer.canvas.toBlob((blob) => {
        resolve(blob);
      }, `image/${format}`);
    });
  }
  
  setRenderMode(mode) {
    if (this.renderer?.setRenderMode) {
      this.renderer.setRenderMode(mode);
    }
    
    engineState.update(state => ({
      ...state,
      mode
    }));
  }
  
  getPerformanceMetrics() {
    return this.performance.getMetrics();
  }
  
  dispose() {
    this.stop();
    
    // Clean up resources
    this.resources.releaseAll();
    
    // Disconnect synchronizer
    this.synchronizer.disconnect();
    
    // Dispose renderer
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer = null;
    }
    
    // Clear scene
    this.scene = new SceneGraph();
    
    // Reset state
    engineState.update(state => ({
      ...state,
      initialized: false,
      running: false,
      error: null
    }));
    
    this.initialized = false;
  }
}

// Create singleton instance
const ghostEngine = new GhostEngine();

// Derived stores for reactive updates
export const isRunning = derived(engineState, $state => $state.running);
export const currentFPS = derived(engineState, $state => $state.performance.fps);
export const renderMode = derived(engineState, $state => $state.mode);

// Export main instance and utilities
export default ghostEngine;
export { engineState, config, SceneGraph, Transform, ResourceManager };