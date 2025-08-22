export async function initWebGPU() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    throw new Error('WebGPUUnavailable');
  }
  const adapter = await (navigator as any).gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('NoAdapter');

  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'].filter(f => (adapter.features as any).has?.(f)),
    requiredLimits: { maxStorageBufferBindingSize: 128 * 1024 * 1024 }
  });
  device.lost.then((info) => {
    console.warn('GPU lost:', info);
    // implement your reinitializePipelines() to rebuild bind groups & pipelines
    // reinitializePipelines();
  });
  return { adapter, device };
}