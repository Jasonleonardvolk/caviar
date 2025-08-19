// render_select.ts - The Master Switch
// This is where ALL path decisions are made based on device capabilities

import { getHolographicCapability } from "./webgpu/capabilities";
import { chooseVariant } from "./webgpu/validateDeviceLimits";
import { buildWebGPU } from "./webgpu/engine";
import { buildCPUProjector } from "../hybrid/cpu_projector";
import { buildIOSCompositor } from "./webgpu/pipelines/ios26_compositor";
import type { InitOpts, RendererHandle } from "../types/renderer";

export async function initRenderer(opts: InitOpts): Promise<RendererHandle> {
    // Get device first
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
        // Fallback to CPU if no WebGPU
        console.log("→ No WebGPU, using CPU fallback");
        return await buildCPUPath(opts);
    }
    
    const device = await adapter.requestDevice();
    const cap = getHolographicCapability(device);
    const variant = chooseVariant(device);
    const limits = device.limits;
    
    // Log decision for debugging
    console.log("=== RENDERER PATH DECISION ===");
    console.log("Holographic Support:", cap.supported);
    console.log("Workgroup Memory:", limits.maxComputeWorkgroupStorageSize);
    console.log("Variant:", variant.name);
    
    // THE MASTER SWITCH - This determines EVERYTHING
    
    // PATH 1: Full GPU ASP (Desktop/High-end iPad)
    if (cap.supported && 
        limits.maxComputeWorkgroupStorageSize >= 32768) {
        console.log("→ PATH 1: Full GPU ASP");
        return await buildFullGPUPath({ ...opts, device });
    }
    
    // PATH 2: iOS 26 Metal-Safe (Server does FFT)
    if (cap.supported && 
        limits.maxComputeWorkgroupStorageSize < 32768 &&
        limits.maxComputeWorkgroupStorageSize >= 16384) {
        console.log("→ PATH 2: iOS 26 Server-Assisted");
        return await buildIOS26Path({ ...opts, device });
    }
    
    // PATH 3: Conservative GPU (Limited memory)
    if (device && limits.maxComputeWorkgroupStorageSize >= 8192) {
        console.log("→ PATH 3: Conservative GPU");
        return await buildConservativePath({ ...opts, device });
    }
    
    // PATH 4: CPU Fallback (No adequate GPU)
    console.log("→ PATH 4: CPU Penrose Fallback");
    return await buildCPUPath(opts);
}

// ============================================
// PATH 1: FULL GPU ASP (Desktop)
// ============================================
async function buildFullGPUPath(opts: InitOpts & { device: GPUDevice }) {
    /* FILES EXECUTED:
     * - webgpu/engine.ts
     * - webgpu/pipelines/build_with_specialization.ts
     * - webgpu/shaders/propagation.wgsl (FULL FFT!)
     * - webgpu/shaders/butterflyStage.wgsl
     * - webgpu/shaders/bitReversal.wgsl
     * - webgpu/shaders/normalize.wgsl
     * - webgpu/shaders/multiViewSynthesis.wgsl
     * 
     * SKIPPED:
     * - ALL server API calls
     * - cpu_projector.ts
     * - tile streaming
     */
    
    return await buildWebGPU({
        ...opts,
        device: opts.device,
        config: {
            mode: "full_fft_on_device",
            tileSize: 32,
            workgroupMem: 32768, // Full 32KB
            skipServerManifest: true
        }
    });
}

// ============================================
// PATH 2: iOS 26 SERVER-ASSISTED
// ============================================
async function buildIOS26Path(opts: InitOpts & { device: GPUDevice }) {
    /* FILES EXECUTED:
     * - stream/manifest.ts (GET server manifest)
     * - stream/tile_cache.ts (Stream pre-computed tiles)
     * - webgpu/pipelines/ios26_compositor.ts
     * - webgpu/shaders/compose_tiles.wgsl (TINY!)
     * - webgpu/shaders/blend_views.wgsl
     * 
     * SKIPPED:
     * - propagation.wgsl (Server did it!)
     * - butterflyStage.wgsl (No FFT!)
     * - bitReversal.wgsl (No FFT!)
     * - Large workgroup allocations
     * 
     * SERVER EXECUTES:
     * - pipeline/penrose_precompute.rs
     * - pipeline/fft_asp_engine.py
     * - prajna/api/holo_stream.py
     */
    
    // Tell server our limits
    const manifest = await fetch('/api/holo/caps', {
        method: 'POST',
        body: JSON.stringify({
            maxWorkgroupStorage: opts.device.limits.maxComputeWorkgroupStorageSize,
            maxTextureSize: opts.device.limits.maxTextureDimension2D,
            preferredTileSize: 128 // Small for iOS
        })
    }).then(r => r.json()).catch(() => null);
    
    return await buildIOSCompositor({
        ...opts,
        device: opts.device,
        manifest,
        config: {
            mode: "composition_only",
            tileSize: manifest?.tile_size || 128,
            workgroupMem: 4096, // Only 4KB needed!
            streamTiles: true
        }
    });
}

// ============================================
// PATH 3: CONSERVATIVE GPU
// ============================================
async function buildConservativePath(opts: InitOpts & { device: GPUDevice }) {
    /* FILES EXECUTED:
     * - Same as PATH 1 but with reduced parameters
     * - Uses smaller tiles (16x16)
     * - Reduced workgroup memory (8KB)
     * - More dispatch calls
     */
    
    return await buildWebGPU({
        ...opts,
        device: opts.device,
        config: {
            mode: "reduced_fft_on_device",
            tileSize: 16,
            workgroupMem: 8192,
            skipServerManifest: false // May use server assist
        }
    });
}

// ============================================
// PATH 4: CPU PENROSE FALLBACK
// ============================================
async function buildCPUPath(opts: InitOpts) {
    /* FILES EXECUTED:
     * - hybrid/cpu_projector.ts
     * - concept_mesh/penrose_rs/pkg/*.wasm
     * - hybrid/cpu_kernels/fft.ts (if WASM fails)
     * - hybrid/canvas_renderer.ts
     * 
     * SKIPPED:
     * - ALL WebGPU files
     * - ALL WGSL shaders
     * - GPU device initialization
     */
    
    return await buildCPUProjector({
        ...opts,
        config: {
            mode: "cpu_penrose",
            useWASM: true,
            fallbackToJS: true
        }
    });
}
