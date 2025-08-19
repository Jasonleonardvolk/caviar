type RecordOpts = {
  sourceCanvas: HTMLCanvasElement;
  fps?: number;
  durationMs: number;
  watermark?: { text: string; font?: string; alpha?: number; x?: number; y?: number };
  includeMic?: boolean;
  mimeCandidates?: string[];
};

const DEFAULT_MIME = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm'
];

function pickMime(candidates: string[] = DEFAULT_MIME): string {
  for (const m of candidates) {
    try { if ((window as any).MediaRecorder?.isTypeSupported?.(m)) return m; } catch {}
  }
  return '';
}

/**
 * Composites source canvas + watermark every RAF into an offscreen "compositor" canvas,
 * records its captureStream, and returns a Blob+URL. Mic track optional.
 */
export async function recordCanvasToVideo(opts: RecordOpts): Promise<{ blob: Blob; url: string }> {
  const fps = opts.fps ?? 30;
  const mime = pickMime(opts.mimeCandidates);
  const compositor = document.createElement('canvas');
  compositor.width = opts.sourceCanvas.width;
  compositor.height = opts.sourceCanvas.height;
  const ctx = compositor.getContext('2d', { alpha: false })!;

  let audioStream: MediaStream | null = null;
  if (opts.includeMic) {
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch {}
  }

  const videoStream = (compositor as HTMLCanvasElement).captureStream(fps);
  const combined = new MediaStream([
    ...videoStream.getVideoTracks(),
    ...(audioStream ? audioStream.getAudioTracks() : [])
  ]);

  const chunks: BlobPart[] = [];
  const rec = new (window as any).MediaRecorder(combined, { mimeType: mime || undefined });
  rec.ondataavailable = (e: BlobEvent) => { if (e.data?.size) chunks.push(e.data); };

  let running = true;
  const start = performance.now();

  function drawFrame() {
    if (!running) return;
    ctx.drawImage(opts.sourceCanvas, 0, 0, compositor.width, compositor.height);

    // Watermark overlay
    if (opts.watermark?.text) {
      ctx.save();
      ctx.globalAlpha = opts.watermark.alpha ?? 0.35;
      ctx.font = opts.watermark.font ?? 'bold 36px system-ui, Arial';
      ctx.fillStyle = '#ffffff';
      const x = opts.watermark.x ?? 24;
      const y = opts.watermark.y ?? (compositor.height - 24);
      // subtle shadow for visibility
      ctx.shadowColor = 'black'; ctx.shadowBlur = 8; ctx.shadowOffsetX = 1; ctx.shadowOffsetY = 1;
      ctx.fillText(opts.watermark.text, x, y);
      ctx.restore();
    }

    if (performance.now() - start < opts.durationMs) {
      requestAnimationFrame(drawFrame);
    } else {
      running = false;
      rec.stop();
      videoStream.getVideoTracks().forEach((t) => t.stop());
      audioStream?.getTracks().forEach((t) => t.stop());
    }
  }

  rec.start();
  requestAnimationFrame(drawFrame);

  const blob: Blob = await new Promise((resolve) => {
    rec.onstop = () => resolve(new Blob(chunks, { type: mime || 'application/octet-stream' }));
  });
  const url = URL.createObjectURL(blob);
  return { blob, url };
}