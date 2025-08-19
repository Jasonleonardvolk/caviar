/**
 * Quilt Texture Pipeline
 * Complete pipeline for loading, transcoding, and uploading quilt textures to WebGPU
 */

import { initBasisWorker, pickTargetFormat, transcodeKTX2 } from './basisTranscoder';
import { uploadTexture, getSupportedFormats, createTextureSampler } from './compressedUpload';
import type { TextureLevel } from './compressedUpload';

/**
 * Complete quilt texture result
 */
export interface QuiltTexture {
  texture: GPUTexture;
  sampler: GPUSampler;
  format: 'astc' | 'bc7' | 'etc2' | 'rgba';
  width: number;
  height: number;
  levels: number;
}

/**
 * Create a complete quilt texture from KTX2 file
 */
export async function createQuiltTextureFromKTX2(
  device: GPUDevice,
  adapter: GPUAdapter,
  url: string,
  srgb: boolean = false
): Promise<QuiltTexture> {
  // Initialize transcoder if not already done
  await initBasisWorker();
  
  // Pick the best format for this device
  const targetFormat = pickTargetFormat(adapter);
  
  console.log(`[QuiltTexture] Loading ${url} as ${targetFormat}`);
  
  // Fetch the KTX2 file
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  
  const buffer = await response.arrayBuffer();
  
  // Transcode the texture
  const { levels } = await transcodeKTX2(buffer, targetFormat);
  
  if (levels.length === 0) {
    throw new Error('No texture levels returned from transcoder');
  }
  
  // Upload to GPU
  const texture = uploadTexture(device, levels, targetFormat, srgb);
  
  // Create sampler with anisotropic filtering
  const sampler = createTextureSampler(device, {
    minFilter: 'linear',
    magFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16
  });
  
  return {
    texture,
    sampler,
    format: targetFormat,
    width: levels[0].width,
    height: levels[0].height,
    levels: levels.length
  };
}

/**
 * Quilt grid configuration
 */
export interface QuiltGrid {
  rows: number;
  cols: number;
  tiles: string[];
}

/**
 * Complete quilt with all tiles loaded
 */
export interface QuiltTextureArray {
  textures: QuiltTexture[];
  grid: QuiltGrid;
  totalViews: number;
}

/**
 * Load a complete quilt grid
 */
export async function loadQuiltGrid(
  device: GPUDevice,
  adapter: GPUAdapter,
  baseUrl: string,
  grid: QuiltGrid,
  srgb: boolean = false
): Promise<QuiltTextureArray> {
  // Initialize transcoder once for all tiles
  await initBasisWorker();
  
  const textures: QuiltTexture[] = [];
  
  // Load tiles in parallel with concurrency limit
  const concurrency = 4;
  const queue = [...grid.tiles];
  const inProgress: Promise<void>[] = [];
  
  while (queue.length > 0 || inProgress.length > 0) {
    // Start new loads up to concurrency limit
    while (inProgress.length < concurrency && queue.length > 0) {
      const tile = queue.shift()!;
      const url = `${baseUrl}/${tile}`;
      
      const task = createQuiltTextureFromKTX2(device, adapter, url, srgb)
        .then(texture => {
          textures.push(texture);
        })
        .catch(error => {
          console.error(`[QuiltTexture] Failed to load ${url}:`, error);
        });
      
      inProgress.push(task);
    }
    
    // Wait for at least one to complete
    if (inProgress.length > 0) {
      await Promise.race(inProgress);
      // Remove completed tasks
      for (let i = inProgress.length - 1; i >= 0; i--) {
        const task = inProgress[i];
        if (await Promise.race([task, Promise.resolve('pending')]) !== 'pending') {
          inProgress.splice(i, 1);
        }
      }
    }
  }
  
  return {
    textures,
    grid,
    totalViews: grid.rows * grid.cols
  };
}

/**
 * Create texture array for quilt rendering
 */
export function createQuiltTextureArray(
  device: GPUDevice,
  quilts: QuiltTexture[]
): GPUTexture {
  if (quilts.length === 0) {
    throw new Error('No quilt textures provided');
  }
  
  const firstQuilt = quilts[0];
  
  // Create 2D texture array
  const textureArray = device.createTexture({
    size: {
      width: firstQuilt.width,
      height: firstQuilt.height,
      depthOrArrayLayers: quilts.length
    },
    format: 'rgba8unorm', // Use RGBA for compatibility
    mipLevelCount: firstQuilt.levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | 
           GPUTextureUsage.COPY_DST | 
           GPUTextureUsage.RENDER_ATTACHMENT,
    label: 'Quilt Texture Array'
  });
  
  // Note: In a real implementation, you'd copy each quilt texture
  // to a layer of the array texture. This requires render passes
  // or compute shaders to do the copy.
  
  return textureArray;
}

/**
 * Quilt manifest data
 */
export interface QuiltManifest {
  name: string;
  grid: [number, number];
  views: number;
  tileFormat: string;
  width: number;
  height: number;
  tiles: string[];
  path?: string;
}

/**
 * Load quilt manifest
 */
export async function loadQuiltManifest(url: string): Promise<QuiltManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load quilt manifest: ${response.status}`);
  }
  return response.json();
}

/**
 * Load complete quilt from manifest
 */
export async function loadQuiltFromManifest(
  device: GPUDevice,
  adapter: GPUAdapter,
  manifestUrl: string,
  srgb: boolean = false
): Promise<QuiltTextureArray> {
  // Load manifest
  const manifest = await loadQuiltManifest(manifestUrl);
  
  // Get base URL from manifest URL
  const baseUrl = manifestUrl.substring(0, manifestUrl.lastIndexOf('/'));
  
  // Create grid configuration
  const grid: QuiltGrid = {
    rows: manifest.grid[0],
    cols: manifest.grid[1],
    tiles: manifest.tiles
  };
  
  // Load all tiles
  return loadQuiltGrid(device, adapter, baseUrl, grid, srgb);
}

/**
 * Performance metrics for texture loading
 */
export interface LoadMetrics {
  totalTime: number;
  fetchTime: number;
  transcodeTime: number;
  uploadTime: number;
  textureCount: number;
  totalBytes: number;
}

/**
 * Load quilt with performance metrics
 */
export async function loadQuiltWithMetrics(
  device: GPUDevice,
  adapter: GPUAdapter,
  manifestUrl: string,
  srgb: boolean = false
): Promise<{ quilt: QuiltTextureArray; metrics: LoadMetrics }> {
  const startTime = performance.now();
  let fetchTime = 0;
  let transcodeTime = 0;
  let uploadTime = 0;
  let totalBytes = 0;
  
  // Load manifest
  const fetchStart = performance.now();
  const manifest = await loadQuiltManifest(manifestUrl);
  fetchTime += performance.now() - fetchStart;
  
  const baseUrl = manifestUrl.substring(0, manifestUrl.lastIndexOf('/'));
  
  // Initialize transcoder
  await initBasisWorker();
  
  const textures: QuiltTexture[] = [];
  const targetFormat = pickTargetFormat(adapter);
  
  // Load each tile with timing
  for (const tile of manifest.tiles) {
    const url = `${baseUrl}/${tile}`;
    
    // Fetch
    const fetchStart = performance.now();
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    fetchTime += performance.now() - fetchStart;
    totalBytes += buffer.byteLength;
    
    // Transcode
    const transcodeStart = performance.now();
    const { levels } = await transcodeKTX2(buffer, targetFormat);
    transcodeTime += performance.now() - transcodeStart;
    
    // Upload
    const uploadStart = performance.now();
    const texture = uploadTexture(device, levels, targetFormat, srgb);
    const sampler = createTextureSampler(device);
    uploadTime += performance.now() - uploadStart;
    
    textures.push({
      texture,
      sampler,
      format: targetFormat,
      width: levels[0].width,
      height: levels[0].height,
      levels: levels.length
    });
  }
  
  const totalTime = performance.now() - startTime;
  
  const quilt: QuiltTextureArray = {
    textures,
    grid: {
      rows: manifest.grid[0],
      cols: manifest.grid[1],
      tiles: manifest.tiles
    },
    totalViews: manifest.views
  };
  
  const metrics: LoadMetrics = {
    totalTime,
    fetchTime,
    transcodeTime,
    uploadTime,
    textureCount: textures.length,
    totalBytes
  };
  
  console.log('[QuiltTexture] Load metrics:', {
    totalTime: `${totalTime.toFixed(2)}ms`,
    fetchTime: `${fetchTime.toFixed(2)}ms`,
    transcodeTime: `${transcodeTime.toFixed(2)}ms`,
    uploadTime: `${uploadTime.toFixed(2)}ms`,
    textureCount: metrics.textureCount,
    totalMB: (totalBytes / (1024 * 1024)).toFixed(2),
    throughputMBps: ((totalBytes / (1024 * 1024)) / (totalTime / 1000)).toFixed(2)
  });
  
  return { quilt, metrics };
}

/**
 * Preload multiple quilts in background
 */
export async function preloadQuilts(
  device: GPUDevice,
  adapter: GPUAdapter,
  manifestUrls: string[]
): Promise<Map<string, QuiltTextureArray>> {
  const quilts = new Map<string, QuiltTextureArray>();
  
  // Initialize transcoder once
  await initBasisWorker();
  
  // Load all quilts in parallel
  const promises = manifestUrls.map(async url => {
    try {
      const quilt = await loadQuiltFromManifest(device, adapter, url);
      quilts.set(url, quilt);
    } catch (error) {
      console.error(`[QuiltTexture] Failed to preload ${url}:`, error);
    }
  });
  
  await Promise.all(promises);
  
  return quilts;
}
