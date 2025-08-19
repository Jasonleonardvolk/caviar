// Lightweight, deterministic capability probe for iOS 26+ and general WebGPU.
// No UA sniffing beyond coarse iOS hint; we prefer feature detection.
export type Capabilities = {
  webgpu: boolean;
  webgl2: boolean;
  reason?: string;
  iosLike?: boolean;
};

export async function detectCapabilities(): Promise<Capabilities> {
  const iosLike =
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

  if (!('gpu' in navigator)) {
    return { webgpu: false, webgl2: !!document.createElement('canvas').getContext('webgl2'), reason: 'navigator.gpu missing', iosLike };
  }

  try {
    const adapter = await (navigator as any).gpu.requestAdapter?.();
    if (!adapter) {
      return { webgpu: false, webgl2: !!document.createElement('canvas').getContext('webgl2'), reason: 'no GPUAdapter', iosLike };
    }
    return { webgpu: true, webgl2: !!document.createElement('canvas').getContext('webgl2'), iosLike };
  } catch (e: any) {
    return { webgpu: false, webgl2: !!document.createElement('canvas').getContext('webgl2'), reason: String(e?.message ?? e), iosLike };
  }
}

export function prefersWebGPUHint(caps: Capabilities): boolean {
  // Prioritize WebGPU on iOS 26+ and any environment where adapter exists.
  return caps.webgpu === true;
}