/**
 * AV1 quilt streaming via WebCodecs. Feeds decoded frames to a <canvas> or OffscreenCanvas.
 * Use when streaming multi-view quilt video instead of static tiles.
 */
export async function startQuiltStream(videoUrl: string, canvas: HTMLCanvasElement) {
  if (!('VideoDecoder' in window)) throw new Error('WebCodecs not supported in this browser.');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D context not available');

  const decoder = new (window as any).VideoDecoder({
    output: (frame: VideoFrame) => {
      try {
        // Draw frame to canvas (2D example; for WebGPU use CopyExternalImageToTexture)
        // @ts-ignore
        ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
      } finally {
        frame.close();
      }
    },
    error: (e: any) => console.error('[WebCodecs] decoder error', e)
  });

  // Configure for AV1; tweak if delivering VP9/AVC
  decoder.configure({ codec: 'av01.0.08M.08' });

  const resp = await fetch(videoUrl);
  if (!resp.ok) throw new Error(`Failed to fetch ${videoUrl}`);
  const reader = resp.body!.getReader();

  // NOTE: This is a minimal example. In production, you'll demux ISO-BMFF/MP4 or IVF to frames.
  // Here we assume Annex B-like chunks pre-split server-side for demo purposes.
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    const chunk = new EncodedVideoChunk({
      timestamp: performance.now() * 1000, // microseconds
      type: 'key',
      data: value
    });
    decoder.decode(chunk);
  }
}
