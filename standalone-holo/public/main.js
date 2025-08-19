// main.js - Entry point for the demo
import { HolographicDisplay } from '../src/HolographicDisplay.js';

let display = null;

export async function init() {
  const canvas = document.getElementById('canvas');
  if (!canvas) throw new Error('Canvas not found');

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
      display.start();
      startBtn.disabled = true;
      startBtn.textContent = 'Running';
    });
  }

  if (parallaxBtn) {
    parallaxBtn.addEventListener('click', () => {
      display.toggleParallax();
    });
  }

  if (qualityBtn) {
    qualityBtn.addEventListener('click', () => {
      display.toggleQuality();
    });
  }

  // Add file upload for testing
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      const bitmap = await createImageBitmap(file);
      await display.processImage(bitmap);
    }
  });
  document.body.appendChild(fileInput);

  // Add upload button
  const uploadBtn = document.createElement('button');
  uploadBtn.textContent = 'Upload Image';
  uploadBtn.addEventListener('click', () => fileInput.click());
  document.getElementById('controls').appendChild(uploadBtn);

  // Add test pattern button
  const testBtn = document.createElement('button');
  testBtn.textContent = 'Test Pattern';
  testBtn.addEventListener('click', async () => {
    const testImage = await display.generateTestPattern();
    await display.processImage(testImage);
  });
  document.getElementById('controls').appendChild(testBtn);

  return display;
}