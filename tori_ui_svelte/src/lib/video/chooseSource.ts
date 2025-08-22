// chooseSource.ts - Smart video source selection based on device capabilities
// Pure capability-based detection - no device model references

export type VidSrc = { type: 'av1'|'hevc'|'h264', url: string, codecs: string };

/**
 * Choose the best video source based on hardware capabilities
 * Priority: AV1 (most efficient) > HEVC (HDR support) > H.264 (universal fallback)
 */
export async function chooseBest(
  srcs: {type: 'av1'|'hevc'|'h264', url: string, codecs: string}[]
): Promise<VidSrc> {
  // Helper: try MediaCapabilities first, then fall back to WebCodecs probe
  const mcTest = async (contentType: string, hdr: boolean) => {
    if (!('mediaCapabilities' in navigator)) return null;
    const video: any = {
      contentType,
      width: 1920, height: 1080, bitrate: 8_000_000, framerate: 60
    };
    if (hdr) (video as any).hdrMetadataType = 'smpte2084'; // HDR10
    try {
      // @ts-ignore
      const info = await (navigator as any).mediaCapabilities.decodingInfo({ type: 'file', video });
      return info?.supported ? { supported: true, efficient: !!info.powerEfficient } : { supported: false, efficient: false };
    } catch {
      return { supported: false, efficient: false };
    }
  };

  const wcTest = async (codec: string) => {
    if (!('VideoDecoder' in window)) return null;
    try {
      // @ts-ignore
      const ok = await (window as any).VideoDecoder.isConfigSupported({
        codec,
        codedWidth: 1920, codedHeight: 1080,
        hardwareAcceleration: 'prefer-hardware'
      });
      return ok?.supported ? { supported: true, efficient: true } : { supported: false, efficient: false };
    } catch { 
      return { supported: false, efficient: false }; 
    }
  };

  // 1) AV1 10-bit (HDR-capable path). Safari 17.5+ exposes AV1 where hardware exists.
  const av1Result = await mcTest('video/mp4; codecs="av01.0.08M.10"', true) || await wcTest('av01.0.08M.10');
  if (av1Result?.supported && av1Result?.efficient) {
    const hit = srcs.find(s => s.type === 'av1'); 
    if (hit) return hit;
  }

  // 2) HEVC 10-bit HDR10 (universal backstop on Apple). Try both brands.
  const hevcHvc1 = await mcTest('video/mp4; codecs="hvc1.2.4.L120.B0"', true);
  const hevcHev1 = await mcTest('video/mp4; codecs="hev1.2.4.L120.B0"', true);
  
  if ((hevcHvc1?.supported && hevcHvc1?.efficient) || (hevcHev1?.supported && hevcHev1?.efficient)) {
    const hit = srcs.find(s => s.type === 'hevc'); 
    if (hit) return hit;
  }

  // 3) SDR fallback (H.264)
  return srcs.find(s => s.type === 'h264') ?? srcs[0];
}

/**
 * Get a readable string describing the chosen codec
 */
export function getCodecLabel(type: VidSrc['type']): string {
  switch(type) {
    case 'av1': return 'AV1 10-bit HDR';
    case 'hevc': return 'HEVC HDR10';
    case 'h264': return 'H.264 SDR';
    default: return 'Unknown';
  }
}
