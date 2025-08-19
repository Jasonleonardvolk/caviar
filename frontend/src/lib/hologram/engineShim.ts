import { detectCapabilities, prefersWebGPUHint } from '$lib/device/capabilities';

type Cleanup = () => void;

export async function initHologram(canvas: HTMLCanvasElement): Promise<Cleanup> {
  const caps = await detectCapabilities();

  // If your real engine is already attached to window, prefer it.
  const w = window as any;
  if (w.hologramEngine?.init && typeof w.hologramEngine.init === 'function') {
    // expected to return a cleanup function
    return w.hologramEngine.init(canvas, { caps, preferWebGPU: prefersWebGPUHint(caps) }) as Cleanup;
  }

  // Fallback: pretty CPU demo so we can test the route + recording.
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) return () => {};

  let raf = 0;
  let t = 0;
  const { width, height } = canvas;

  function draw() {
    t += 0.016;
    // smooth background
    const g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, '#0b0f14');
    g.addColorStop(1, '#111a22');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    // "wavefront" rings
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 24; i++) {
      const r = (Math.sin(t * 0.8 + i * 0.35) * 0.5 + 0.5) * Math.min(width, height) * 0.48;
      ctx.beginPath();
      ctx.arc(width * 0.5, height * 0.5, r, 0, Math.PI * 2);
      ctx.lineWidth = 2 + (i % 3);
      ctx.strokeStyle = `rgba(${180 + (i * 3) % 60}, ${160 + (i * 2) % 80}, 255, 0.06)`;
      ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';

    // HUD text
    ctx.fillStyle = '#9ec7ff';
    ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.fillText(`fallback demo  •  ${prefersWebGPUHint(caps) ? 'WebGPU-capable' : 'WebGPU-unavailable'}`, 12, 20);

    raf = requestAnimationFrame(draw);
  }
  draw();

  return () => cancelAnimationFrame(raf);
}