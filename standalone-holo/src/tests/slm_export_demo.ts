import { exportPhaseOnly, exportLeeAmplitude } from '../pipelines/slm_export';

async function getDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  return adapter.requestDevice();
}

function synthField(w: number, h: number) {
  const A = new Float32Array(w * h).fill(1);
  const PHI = new Float32Array(w * h);
  for (let y=0; y<h; y++) for (let x=0; x<w; x++) {
    const dx = (x - w/2) / w, dy = (y - h/2) / h;
    PHI[y*w + x] = Math.atan2(dy, dx);
  }
  return { A, PHI };
}

(async () => {
  const dev = await getDevice();
  const { A, PHI } = synthField(256, 256);

  document.getElementById('phase')!.addEventListener('click', () =>
    exportPhaseOnly(dev, A, PHI, 256, 256)
  );

  document.getElementById('lee')!.addEventListener('click', () =>
    exportLeeAmplitude(dev, A, PHI, 256, 256, { fx: 0.25, fy: 0.0, binary: true })
  );
})();