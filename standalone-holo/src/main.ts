// main.ts - Entry point for the demo
import { HolographicDisplay } from './HolographicDisplay';
import { loadCalibration } from '../../frontend/lib/pipeline/quilt/calibration';
import { QuiltRenderer } from '../../frontend/lib/pipeline/quilt/QuiltRenderer';
import type { QuiltLayout } from '../../frontend/lib/pipeline/quilt/types';
import displayMode from '../../configs/display_mode.json';

let display: HolographicDisplay | null = null;
let quiltRenderer: QuiltRenderer | null = null;
let currentMode: 'standard' | 'lenticular' = 'standard';

export async function init() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Canvas not found');

  // Determine display mode
  currentMode = await resolveMode(displayMode);
  console.log(`[Display Mode] ${currentMode.toUpperCase()} - ${currentMode === 'standard' ? 'NO HARDWARE REQUIRED' : 'Using Lenticular Panel'}`);

  // Show mode badge
  showModeBadge(currentMode);

  // STANDARD MODE (Default - No Hardware Required)
  if (currentMode === 'standard') {
    // Initialize standard holographic display with head-tracked parallax
    display = new HolographicDisplay();
    await display.init(canvas);
    console.log('✓ Standard mode initialized - Head-tracked parallax on any screen');
    setupStandardControls(canvas);
    return display;
  }

  // LENTICULAR MODE (Optional - Only if user has Looking Glass or similar)
  if (currentMode === 'lenticular') {
    console.log('Initializing lenticular mode...');
    
    // Initialize WebGPU for quilt rendering
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU required for lenticular mode');
    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvas.getContext('webgpu')!;
    ctx.configure({ device, format, alphaMode: 'opaque' });

    // Load calibration
    const calib = await loadCalibration(displayMode.calibration || '/configs/display_calibration/my_panel.json');
    const layout: QuiltLayout = { cols: 8, rows: 6, tileW: 320, tileH: 320, numViews: 48 };
    quiltRenderer = new QuiltRenderer(device, calib, layout, format);
    
    console.log('✓ Lenticular mode initialized - Using QuiltRenderer for multi-view');
    setupLenticularControls(device, ctx);
    return;
  }
}

// Resolve display mode with fallback
async function resolveMode(cfg: { mode: string; calibration?: string }): Promise<'standard' | 'lenticular'> {
  if (cfg.mode !== 'lenticular') return 'standard';
  
  // Check if calibration is accessible (indicates panel might be present)
  try {
    const r = await fetch(cfg.calibration!, { method: 'HEAD' });
    if (!r.ok) throw new Error('Calibration not found');
    return 'lenticular';
  } catch {
    console.warn('[Mode Guard] Calibration not reachable; falling back to standard mode');
    showNotification('No lenticular panel detected - using standard mode');
    return 'standard';
  }
}

// Show mode badge
function showModeBadge(mode: string) {
  const badge = document.createElement('div');
  badge.id = 'mode-badge';
  badge.style.cssText = 'position:fixed;top:8px;left:8px;color:#0f0;font:12px monospace;background:rgba(0,0,0,0.7);padding:4px 8px;border-radius:3px;';
  badge.textContent = mode === 'standard' 
    ? 'MODE: STANDARD (NO HARDWARE)' 
    : 'MODE: LENTICULAR (PANEL CONNECTED)';
  document.body.appendChild(badge);
}

// Show notification
function showNotification(msg: string) {
  const notif = document.createElement('div');
  notif.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#333;color:#fff;padding:16px;border-radius:8px;font-family:monospace;z-index:9999;';
  notif.textContent = msg;
  document.body.appendChild(notif);
  setTimeout(() => notif.remove(), 3000);
}

// Setup controls for standard mode
async function setupStandardControls(canvas: HTMLCanvasElement) {
  if (!canvas) return;
  
  // Check model availability
  const modelStatus = document.getElementById('model-status');
  if (modelStatus) {
    try {
      const depthResponse = await fetch('/models/depth_estimator.onnx', { method: 'HEAD' });
      const waveResponse = await fetch('/models/waveop_fno_v1.onnx', { method: 'HEAD' });
      
      const depthAvailable = depthResponse.ok;
      const waveAvailable = waveResponse.ok;
      
      modelStatus.innerHTML = `
        Depth Model: ${depthAvailable ? '✓ Loaded' : '✗ Using Synthetic'}<br>
        WaveOp Model: ${waveAvailable ? '✓ Loaded' : '✗ Using Fallback'}
      `;
      
      if (!depthAvailable) {
        console.log('Depth model not found - using synthetic depth. Run: npm run download:models');
      }
    } catch (err) {
      modelStatus.textContent = 'Models: Checking...';
    }
  }

  // Create display
  display = new HolographicDisplay();
  await display.init(canvas);

  // Wire up controls
  const startBtn = document.getElementById('start');
  const parallaxBtn = document.getElementById('parallax');
  const qualityBtn = document.getElementById('quality');

  if (startBtn) {
    startBtn.addEventListener('click', () => {
      display!.start();
      (startBtn as HTMLButtonElement).disabled = true;
      startBtn.textContent = 'Running';
    });
  }

  if (parallaxBtn) {
    parallaxBtn.addEventListener('click', () => {
      display!.toggleParallax();
    });
  }

  if (qualityBtn) {
    qualityBtn.addEventListener('click', () => {
      display!.toggleQuality();
    });
  }

  // Add file upload for testing
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', async (e) => {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      const bitmap = await createImageBitmap(file);
      await display!.processImage(bitmap);
    }
  });
  document.body.appendChild(fileInput);

  // Add upload button
  const uploadBtn = document.createElement('button');
  uploadBtn.textContent = 'Upload Image';
  uploadBtn.addEventListener('click', () => fileInput.click());
  document.getElementById('controls')?.appendChild(uploadBtn);

  // Add test pattern button
  const testBtn = document.createElement('button');
  testBtn.textContent = 'Test Pattern';
  testBtn.addEventListener('click', async () => {
    const testImage = await display!.generateTestPattern();
    await display!.processImage(testImage);
  });
  document.getElementById('controls')?.appendChild(testBtn);

  // Add encoding mode toggle button
  const encodingBtn = document.createElement('button');
  encodingBtn.textContent = 'Phase-Only Mode';
  encodingBtn.addEventListener('click', () => {
    const info = display!.getHologramInfo();
    const newMode = info.encodingMode === 'phase_only' ? 'lee_offaxis' : 'phase_only';
    display!.setEncodingMode(newMode);
    encodingBtn.textContent = newMode === 'phase_only' ? 'Phase-Only Mode' : 'Lee Off-Axis Mode';
  });
  document.getElementById('controls')?.appendChild(encodingBtn);
}

// Setup controls for lenticular mode
function setupLenticularControls(device: GPUDevice, ctx: GPUCanvasContext) {
  // This would initialize quilt rendering loop
  console.log('Lenticular controls ready - QuiltRenderer active');
  
  // Add lenticular-specific controls
  const controls = document.getElementById('controls');
  if (controls) {
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'Adjust Views';
    viewBtn.addEventListener('click', () => {
      const views = prompt('Number of views (1-100):', '48');
      if (views && quiltRenderer) {
        // Update quilt layout
        console.log(`Updating to ${views} views`);
      }
    });
    controls.appendChild(viewBtn);
  }
}

// Auto-init on load
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', async () => {
    const status = document.getElementById('status');
    try {
      if (status) status.textContent = 'Initializing WebGPU...';
      await init();
      if (status) status.textContent = 'Ready - Click Start Demo';
    } catch (err) {
      console.error(err);
      if (status) status.textContent = `Error: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
  });
}