// ${IRIS_ROOT}\frontend\lib\webgpu\caps.ts
// Minimal, safe feature gating for modern WebGPU
export type Caps = {
  subgroups: boolean;
  subgroupsF16: boolean;
  shaderF16: boolean;
  timestampQuery: boolean;
  indirectFirstInstance: boolean;
  // adapter info (for logging/diagnostics)
  subgroupMinSize?: number;
  subgroupMaxSize?: number;
};

export async function requestDeviceWithCaps() {
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPUAdapter');

  const feats = adapter.features;
  const want: GPUFeatureName[] = [];

  const caps: Caps = {
    subgroups: feats.has('subgroups'),
    subgroupsF16: feats.has('subgroups') && feats.has('shader-f16'), // Combined check
    shaderF16: feats.has('shader-f16'),
    timestampQuery: feats.has('timestamp-query'),
    indirectFirstInstance: feats.has('indirect-first-instance'),
    subgroupMinSize: (adapter as any).info?.subgroupMinSize,
    subgroupMaxSize: (adapter as any).info?.subgroupMaxSize,
  };

  if (caps.subgroups) want.push('subgroups');
  if (caps.shaderF16) want.push('shader-f16');
  if (caps.timestampQuery) want.push('timestamp-query');
  if (caps.indirectFirstInstance) want.push('indirect-first-instance');

  const device = await adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits: {}, // keep default portable limits
  });

  // Device lost handler (stability)
  device.lost.then((info) => {
    console.warn('[GPU] device lost:', info);
    // App-level: show a soft banner / offer reload
  });

  return { adapter, device, caps };
}
