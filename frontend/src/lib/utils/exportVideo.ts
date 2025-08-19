// exportVideo.ts
// Video export utilities for iRis holographic recordings
// Supports watermarking, resolution adjustment, and format conversion

export interface ExportOptions {
  watermark?: boolean;
  watermarkText?: string;
  resolution?: '1920x1080' | '1080x1920' | '3840x2160';
  format?: 'mp4' | 'webm';
  quality?: number; // 0-1
}

/**
 * Process and export video blob with options
 */
export async function exportVideo(
  inputBlob: Blob,
  options: ExportOptions = {}
): Promise<Blob> {
  const {
    watermark = false,
    watermarkText = 'iRis',
    resolution = '1080x1920',
    format = 'mp4',
    quality = 0.9
  } = options;
  
  // For now, return the input blob with potential watermark overlay
  // In production, this would use WebCodecs or ffmpeg.wasm for full processing
  
  if (watermark) {
    return addWatermark(inputBlob, watermarkText);
  }
  
  return inputBlob;
}

/**
 * Add watermark to video using canvas overlay
 */
async function addWatermark(blob: Blob, text: string): Promise<Blob> {
  // Create video element
  const video = document.createElement('video');
  video.src = URL.createObjectURL(blob);
  video.muted = true;
  
  // Wait for video metadata
  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });
  
  // Create canvas for watermark
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas context');
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  // Create output stream
  const stream = canvas.captureStream(30);
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'video/webm;codecs=h264',
    videoBitsPerSecond: 8_000_000
  });
  
  const chunks: Blob[] = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  
  // Process video frame by frame
  const processFrame = () => {
    if (video.paused || video.ended) return;
    
    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Add watermark
    ctx.font = 'bold 48px SF Pro Display, Arial';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 2;
    
    const textMetrics = ctx.measureText(text);
    const x = canvas.width - textMetrics.width - 40;
    const y = canvas.height - 40;
    
    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);
    
    requestAnimationFrame(processFrame);
  };
  
  // Start processing
  return new Promise((resolve) => {
    mediaRecorder.onstop = () => {
      const outputBlob = new Blob(chunks, { type: 'video/mp4' });
      URL.revokeObjectURL(video.src);
      resolve(outputBlob);
    };
    
    mediaRecorder.start();
    video.play();
    processFrame();
    
    video.onended = () => {
      mediaRecorder.stop();
    };
  });
}

/**
 * Convert video to vertical format (9:16 for TikTok/Snap)
 */
export async function convertToVertical(blob: Blob): Promise<Blob> {
  // This would use ffmpeg.wasm or server-side processing
  // For now, return the original
  console.log('Vertical conversion not yet implemented');
  return blob;
}

/**
 * Extract thumbnail from video at specific timestamp
 */
export async function extractThumbnail(
  blob: Blob,
  timestamp: number = 0
): Promise<Blob> {
  const video = document.createElement('video');
  video.src = URL.createObjectURL(blob);
  video.currentTime = timestamp;
  
  await new Promise((resolve) => {
    video.onseeked = resolve;
  });
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas context');
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      URL.revokeObjectURL(video.src);
      resolve(blob || new Blob());
    }, 'image/jpeg', 0.9);
  });
}

/**
 * Check if MediaRecorder is supported with required codecs
 */
export function checkRecordingSupport(): {
  supported: boolean;
  codecs: string[];
} {
  if (!window.MediaRecorder) {
    return { supported: false, codecs: [] };
  }
  
  const codecs = [
    'video/webm;codecs=h264',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/mp4'
  ];
  
  const supported = codecs.filter(codec => 
    MediaRecorder.isTypeSupported(codec)
  );
  
  return {
    supported: supported.length > 0,
    codecs: supported
  };
}

/**
 * Get optimal recording settings based on device capabilities
 */
export function getOptimalSettings(tier: number = 0): MediaRecorderOptions {
  const support = checkRecordingSupport();
  
  // Prefer H.264 for compatibility
  const mimeType = support.codecs.find(c => c.includes('h264')) 
    || support.codecs[0] 
    || 'video/webm';
  
  // Bitrate based on tier
  const bitrates = {
    0: 4_000_000,  // Free: 4 Mbps
    1: 8_000_000,  // Plus: 8 Mbps
    2: 12_000_000  // Pro: 12 Mbps
  };
  
  return {
    mimeType,
    videoBitsPerSecond: bitrates[tier] || bitrates[0],
    audioBitsPerSecond: 128_000
  };
}
