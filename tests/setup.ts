// tests/setup.ts
// Minimal WebGPU shim for Node + JSDOM test runs

console.log('[setup] WebGPU + CommonJS shim loaded');

// -------------------------------------------------------------
// Provide minimal CommonJS globals so legacy `require.main === module` guards work
import { createRequire } from 'node:module';
const customRequire = createRequire(import.meta.url);

// Create a module object that will pass the require.main === module check
const moduleObj = { exports: {}, filename: import.meta.url, id: '.', loaded: true };

// Set up require with main pointing to our module
(customRequire as any).main = moduleObj;
(globalThis as any).require = customRequire;
(globalThis as any).module = moduleObj;
// -------------------------------------------------------------

// ── lightweight mocks ────────────────────────────────────────────
class GPUDeviceMock {
  // Add proper limits
  limits = {
    maxTextureDimension1D: 8192,
    maxTextureDimension2D: 8192,
    maxTextureDimension3D: 2048,
    maxTextureArrayLayers: 256,
    maxBindGroups: 4,
    maxBindingsPerBindGroup: 1000,
    maxDynamicUniformBuffersPerPipelineLayout: 8,
    maxDynamicStorageBuffersPerPipelineLayout: 4,
    maxSampledTexturesPerShaderStage: 16,
    maxSamplersPerShaderStage: 16,
    maxStorageBuffersPerShaderStage: 8,
    maxStorageTexturesPerShaderStage: 4,
    maxUniformBuffersPerShaderStage: 12,
    maxUniformBufferBindingSize: 65536,
    maxStorageBufferBindingSize: 134217728,
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
    maxVertexBuffers: 8,
    maxBufferSize: 268435456,
    maxVertexAttributes: 16,
    maxVertexBufferArrayStride: 2048,
    maxInterStageShaderComponents: 60,
    maxInterStageShaderVariables: 16,
    maxColorAttachments: 8,
    maxColorAttachmentBytesPerSample: 32,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535
  };
  
  createShaderModule() { 
    return {
      getCompilationInfo: async () => ({ messages: [] }),
    };
  }
  createComputePipeline() { 
    return {
      getBindGroupLayout: () => ({})
    };
  }
  createRenderPipeline() { 
    return {
      getBindGroupLayout: () => ({})
    };
  }
  createBuffer() {
    return {
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(8),
      unmap() {},
      destroy() {}
    };
  }
  createTexture(descriptor?: any) {
    return {
      createView: () => ({}),
      destroy() {},
      width: descriptor?.size?.[0] || 512,
      height: descriptor?.size?.[1] || 512,
      format: descriptor?.format || 'rgba8unorm'
    };
  }
  createBindGroup() {
    return {};
  }
  createBindGroupLayout() {
    return {};
  }
  createPipelineLayout() {
    return {};
  }
  createSampler() {
    return {};
  }
  createQuerySet() {
    return { destroy() {} };
  }
  createCommandEncoder() {
    return {
      beginComputePass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {}
        };
      },
      beginRenderPass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          draw() {},
          end() {}
        };
      },
      copyTextureToBuffer() {},
      writeBuffer() {},
      copyBufferToBuffer() {},
      writeTimestamp() {},
      resolveQuerySet() {},
      finish() { return {}; },
    };
  }
  queue = { 
    submit() {},
    writeBuffer() {},
    writeTexture() {},
    copyExternalImageToTexture() {},
    onSubmittedWorkDone: () => Promise.resolve()
  };
  lost = new Promise(() => {}); // Never resolves unless explicitly triggered
  features = new Set(['timestamp-query']);
  
  // Add destroy method
  destroy() {
    // Mock destroy - do nothing
  }
}

class GPUAdapterMock {
  async requestDevice() {
    return new GPUDeviceMock();
  }
  features = new Set(['timestamp-query']);
  limits = {
    maxTextureDimension1D: 8192,
    maxTextureDimension2D: 8192,
    maxTextureDimension3D: 2048,
    maxTextureArrayLayers: 256,
    maxBindGroups: 4,
    maxBindingsPerBindGroup: 1000,
    maxDynamicUniformBuffersPerPipelineLayout: 8,
    maxDynamicStorageBuffersPerPipelineLayout: 4,
    maxSampledTexturesPerShaderStage: 16,
    maxSamplersPerShaderStage: 16,
    maxStorageBuffersPerShaderStage: 8,
    maxStorageTexturesPerShaderStage: 4,
    maxUniformBuffersPerShaderStage: 12,
    maxUniformBufferBindingSize: 65536,
    maxStorageBufferBindingSize: 134217728,
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
    maxVertexBuffers: 8,
    maxBufferSize: 268435456,
    maxVertexAttributes: 16,
    maxVertexBufferArrayStride: 2048,
    maxInterStageShaderComponents: 60,
    maxInterStageShaderVariables: 16,
    maxColorAttachments: 8,
    maxColorAttachmentBytesPerSample: 32,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535
  };
  isFallbackAdapter = false;
}

// ── attach to global navigator ───────────────────────────────────
(globalThis as any).navigator ??= {} as Navigator;
(globalThis as any).navigator.gpu = {
  requestAdapter: async () => new GPUAdapterMock(),
  getPreferredCanvasFormat: () => 'bgra8unorm'
};

// Add minimal enums your code bitwise-ORs
(globalThis as any).GPUBufferUsage = {
  MAP_READ: 1 << 0,
  MAP_WRITE: 1 << 1,
  COPY_SRC: 1 << 2,
  COPY_DST: 1 << 3,
  INDEX: 1 << 4,
  VERTEX: 1 << 5,
  UNIFORM: 1 << 6,
  STORAGE: 1 << 7,
  INDIRECT: 1 << 8,
  QUERY_RESOLVE: 1 << 9
};

(globalThis as any).GPUTextureUsage = {
  COPY_SRC: 1 << 0,
  COPY_DST: 1 << 1,
  TEXTURE_BINDING: 1 << 2,
  STORAGE_BINDING: 1 << 3,
  RENDER_ATTACHMENT: 1 << 4
};

(globalThis as any).GPUMapMode = {
  READ: 1,
  WRITE: 2
};

// Add GPUShaderStage constants - THIS IS THE FIX!
(globalThis as any).GPUShaderStage = {
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 4
};

// Mock WebGPU canvas context
class MockCanvasRenderingContext2D {}

class MockHTMLCanvasElement {
  width = 512;
  height = 512;
  
  getContext(contextType: string) {
    if (contextType === 'webgpu') {
      return {
        configure(config: any) {
          // Mock configure method
        },
        getCurrentTexture() {
          return {
            createView: () => ({}),
            width: this.width,
            height: this.height,
            format: 'bgra8unorm'
          };
        }
      };
    } else if (contextType === '2d') {
      return new MockCanvasRenderingContext2D();
    }
    return null;
  }
}

// Replace HTMLCanvasElement if it doesn't exist
if (typeof HTMLCanvasElement === 'undefined') {
  (globalThis as any).HTMLCanvasElement = MockHTMLCanvasElement;
} else {
  // Patch existing HTMLCanvasElement prototype
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(contextType: string) {
    if (contextType === 'webgpu') {
      const canvas = this;
      return {
        configure(config: any) {
          // Mock configure method
        },
        getCurrentTexture() {
          return {
            createView: () => ({}),
            width: canvas.width || 512,
            height: canvas.height || 512,
            format: 'bgra8unorm'
          };
        }
      };
    }
    // Fall back to original for other context types
    return originalGetContext.call(this, contextType as any);
  };
}

// Mock for createCanvas from canvas package
try {
  const canvas = require('canvas');
  if (canvas && canvas.createCanvas) {
    const originalCreateCanvas = canvas.createCanvas;
    canvas.createCanvas = function(width: number, height: number) {
      const canvasInstance = originalCreateCanvas(width, height);
      
      // Patch the instance to support webgpu context
      const originalGetContext = canvasInstance.getContext.bind(canvasInstance);
      canvasInstance.getContext = function(contextType: string) {
        if (contextType === 'webgpu') {
          return {
            configure(config: any) {},
            getCurrentTexture() {
              return {
                createView: () => ({}),
                width: canvasInstance.width,
                height: canvasInstance.height,
                format: 'bgra8unorm'
              };
            }
          };
        }
        return originalGetContext(contextType);
      };
      
      return canvasInstance;
    };
  }
} catch (e) {
  // canvas package not installed or not available
}

// Mock WebSocket if not available
if (typeof WebSocket === 'undefined') {
  (globalThis as any).WebSocket = class WebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
    
    readyState = WebSocket.CLOSED;
    
    constructor(url: string) {
      // Mock implementation
    }
    
    send(data: any) {}
    close() {}
    
    // Event handlers
    onopen: ((event: any) => void) | null = null;
    onmessage: ((event: any) => void) | null = null;
    onerror: ((event: any) => void) | null = null;
    onclose: ((event: any) => void) | null = null;
  };
}

// Mock window.location if not available
if (typeof window === 'undefined') {
  (globalThis as any).window = {
    location: {
      protocol: 'http:',
      host: 'localhost:3000',
      hostname: 'localhost',
      port: '3000',
      pathname: '/',
      search: '',
      hash: ''
    }
  };
}
