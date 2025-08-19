/**
 * Device Detection Utility
 * Ensures system never fails - always finds a rendering path
 */

export interface DeviceCapabilities {
  webgpu: boolean;
  wasm: boolean;
  webgl2: boolean;
  webgl: boolean;
  cpu: boolean;
  memory: number;
  cores: number;
  isMobile: boolean;
  preferredBackend: 'webgpu' | 'wasm' | 'webgl2' | 'webgl' | 'cpu';
}

export function detectCapabilities(): DeviceCapabilities {
  const caps: DeviceCapabilities = {
    webgpu: false,
    wasm: false,
    webgl2: false,
    webgl: false,
    cpu: true, // Always available
    memory: 2048, // Default 2GB
    cores: 4, // Default 4 cores
    isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),
    preferredBackend: 'cpu'
  };

  // Check WebGPU
  if ('gpu' in navigator) {
    caps.webgpu = true;
    caps.preferredBackend = 'webgpu';
  }

  // Check WASM
  if (typeof WebAssembly !== 'undefined') {
    caps.wasm = true;
    if (!caps.webgpu) {
      caps.preferredBackend = 'wasm';
    }
  }

  // Check WebGL2
  const canvas = document.createElement('canvas');
  const gl2 = canvas.getContext('webgl2');
  if (gl2) {
    caps.webgl2 = true;
    if (!caps.webgpu && !caps.wasm) {
      caps.preferredBackend = 'webgl2';
    }
  }

  // Check WebGL
  const gl = canvas.getContext('webgl');
  if (gl) {
    caps.webgl = true;
    if (!caps.webgpu && !caps.wasm && !caps.webgl2) {
      caps.preferredBackend = 'webgl';
    }
  }

  // Get memory estimate
  if ('deviceMemory' in navigator) {
    caps.memory = (navigator as any).deviceMemory * 1024; // Convert GB to MB
  }

  // Get CPU cores
  if ('hardwareConcurrency' in navigator) {
    caps.cores = navigator.hardwareConcurrency;
  }

  return caps;
}

export function selectQualityLevel(caps: DeviceCapabilities): 'ultra' | 'high' | 'medium' | 'low' | 'minimal' {
  // Mobile always gets reduced quality
  if (caps.isMobile) {
    if (caps.webgpu && caps.memory >= 4096) return 'medium';
    if (caps.wasm && caps.memory >= 2048) return 'low';
    return 'minimal';
  }

  // Desktop quality selection
  if (caps.webgpu && caps.memory >= 8192 && caps.cores >= 8) return 'ultra';
  if (caps.webgpu && caps.memory >= 4096) return 'high';
  if ((caps.wasm || caps.webgl2) && caps.memory >= 2048) return 'medium';
  if (caps.webgl && caps.memory >= 1024) return 'low';
  return 'minimal';
}

export function getRendererConfig(caps: DeviceCapabilities) {
  const quality = selectQualityLevel(caps);
  
  const configs = {
    ultra: {
      views: 45,
      resolution: [1920, 1080] as [number, number],
      samples: 4,
      backend: 'webgpu' as const
    },
    high: {
      views: 45,
      resolution: [1280, 720] as [number, number],
      samples: 2,
      backend: 'webgpu' as const
    },
    medium: {
      views: 9,
      resolution: [1024, 576] as [number, number],
      samples: 1,
      backend: caps.preferredBackend
    },
    low: {
      views: 5,
      resolution: [640, 360] as [number, number],
      samples: 1,
      backend: caps.preferredBackend
    },
    minimal: {
      views: 1,
      resolution: [320, 180] as [number, number],
      samples: 1,
      backend: 'cpu' as const
    }
  };

  return configs[quality];
}

import type { RendererConfig } from './rendererTypes';

export async function loadRenderer(caps: DeviceCapabilities): Promise<any> {
  const config = getRendererConfig(caps) as RendererConfig;
  
  console.log(`[DeviceDetect] Selected backend: ${config.backend}, Quality: ${selectQualityLevel(caps)}`);
  
  try {
    switch (config.backend) {
      case 'webgpu':
        const { WebGPURenderer } = await import('./webgpuRenderer');
        return new WebGPURenderer(config);
      
      case 'wasm':
        const { WASMRenderer } = await import('./wasmFallbackRenderer');
        return new WASMRenderer(config);
      
      case 'webgl2':
      case 'webgl':
        const { WebGLRenderer } = await import('./webglRenderer');
        return new WebGLRenderer(config);
      
      case 'cpu':
      default:
        const { CPURenderer } = await import('./cpuRenderer');
        return new CPURenderer(config);
    }
  } catch (error) {
    console.error(`[DeviceDetect] Failed to load ${config.backend} renderer, falling back to CPU`, error);
    const { CPURenderer } = await import('./cpuRenderer');
    return new CPURenderer(config);
  }
}
