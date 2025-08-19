import { probeWebGPULimits, WebGPULimits } from "./capabilities";

let cachedLimits: WebGPULimits | null = null;
let telemetryEmitted = new Set<string>();

/**
 * Hard assertion - throws if workgroup exceeds limits
 */
export async function assertWorkgroupFits(x: number, y: number, z: number) {
  const l = cachedLimits || (cachedLimits = await probeWebGPULimits());
  if (!l) return; // no WebGPU, skip
  
  if ((l.maxComputeWorkgroupSizeX ?? Infinity) < x) 
    throw new Error(`workgroupSizeX ${x} > limit ${l.maxComputeWorkgroupSizeX}`);
  if ((l.maxComputeWorkgroupSizeY ?? Infinity) < y) 
    throw new Error(`workgroupSizeY ${y} > limit ${l.maxComputeWorkgroupSizeY}`);
  if ((l.maxComputeWorkgroupSizeZ ?? Infinity) < z) 
    throw new Error(`workgroupSizeZ ${z} > limit ${l.maxComputeWorkgroupSizeZ}`);
  
  const invocations = x * y * z;
  if ((l.maxComputeInvocationsPerWorkgroup ?? Infinity) < invocations)
    throw new Error(`invocations ${invocations} > limit ${l.maxComputeInvocationsPerWorkgroup}`);
}

/**
 * Soft clamping - adjusts workgroup to fit within limits and logs warning
 */
export async function clampWorkgroupSize(
  requested: { x: number; y: number; z: number },
  options: { 
    warn?: boolean;
    telemetry?: boolean;
    name?: string;
  } = {}
): Promise<{ x: number; y: number; z: number; clamped: boolean }> {
  
  const { warn = true, telemetry = true, name = 'workgroup' } = options;
  const l = cachedLimits || (cachedLimits = await probeWebGPULimits());
  
  if (!l) {
    // No WebGPU or limits unavailable, return as-is
    return { ...requested, clamped: false };
  }
  
  const maxX = l.maxComputeWorkgroupSizeX ?? 256;
  const maxY = l.maxComputeWorkgroupSizeY ?? 256;
  const maxZ = l.maxComputeWorkgroupSizeZ ?? 64;
  const maxInvocations = l.maxComputeInvocationsPerWorkgroup ?? 256;
  
  let { x, y, z } = requested;
  let clamped = false;
  
  // Clamp individual dimensions
  if (x > maxX) {
    if (warn) console.warn(`⚠️ Clamping ${name} X: ${x} -> ${maxX}`);
    x = maxX;
    clamped = true;
  }
  
  if (y > maxY) {
    if (warn) console.warn(`⚠️ Clamping ${name} Y: ${y} -> ${maxY}`);
    y = maxY;
    clamped = true;
  }
  
  if (z > maxZ) {
    if (warn) console.warn(`⚠️ Clamping ${name} Z: ${z} -> ${maxZ}`);
    z = maxZ;
    clamped = true;
  }
  
  // Check total invocations
  let invocations = x * y * z;
  if (invocations > maxInvocations) {
    // Scale down proportionally
    const scale = Math.cbrt(maxInvocations / invocations);
    x = Math.floor(x * scale);
    y = Math.floor(y * scale);
    z = Math.floor(z * scale);
    
    // Ensure at least 1 in each dimension
    x = Math.max(1, x);
    y = Math.max(1, y);
    z = Math.max(1, z);
    
    if (warn) {
      console.warn(`⚠️ Clamping ${name} invocations: ${invocations} -> ${x * y * z}`);
    }
    clamped = true;
  }
  
  // Emit telemetry breadcrumb (once per session per workgroup)
  if (clamped && telemetry && !telemetryEmitted.has(name)) {
    telemetryEmitted.add(name);
    emitTelemetry('workgroup_clamped', {
      name,
      requested: `${requested.x}x${requested.y}x${requested.z}`,
      clamped: `${x}x${y}x${z}`,
      limits: {
        maxX, maxY, maxZ, maxInvocations
      }
    });
  }
  
  return { x, y, z, clamped };
}

/**
 * Check if workgroup storage usage is within limits
 */
export async function checkWorkgroupStorageUsage(
  bytesPerInvocation: number,
  workgroupSize: { x: number; y: number; z: number },
  options: { warn?: boolean; name?: string } = {}
): Promise<{ withinLimits: boolean; usage: number; limit: number; percentage: number }> {
  
  const { warn = true, name = 'workgroup storage' } = options;
  const l = cachedLimits || (cachedLimits = await probeWebGPULimits());
  
  if (!l) {
    return { 
      withinLimits: true, 
      usage: 0, 
      limit: Infinity, 
      percentage: 0 
    };
  }
  
  const limit = l.maxComputeWorkgroupStorageSize ?? 32768;
  const invocations = workgroupSize.x * workgroupSize.y * workgroupSize.z;
  const usage = bytesPerInvocation * invocations;
  const percentage = (usage / limit) * 100;
  const withinLimits = usage <= limit;
  
  if (!withinLimits && warn) {
    console.error(`❌ ${name} exceeds limit: ${usage} bytes > ${limit} bytes`);
  } else if (percentage > 90 && warn) {
    console.warn(`⚠️ ${name} near limit: ${usage} bytes (${percentage.toFixed(1)}% of ${limit})`);
  }
  
  return { withinLimits, usage, limit, percentage };
}

/**
 * Get safe workgroup size suggestions based on limits
 */
export async function getSafeWorkgroupSizes(): Promise<Array<{ x: number; y: number; z: number; invocations: number }>> {
  const l = cachedLimits || (cachedLimits = await probeWebGPULimits());
  
  if (!l) {
    // Return conservative defaults
    return [
      { x: 64, y: 1, z: 1, invocations: 64 },
      { x: 8, y: 8, z: 1, invocations: 64 },
      { x: 4, y: 4, z: 4, invocations: 64 }
    ];
  }
  
  const maxInvocations = l.maxComputeInvocationsPerWorkgroup ?? 256;
  const suggestions = [];
  
  // 1D workgroups
  for (const size of [32, 64, 128, 256]) {
    if (size <= maxInvocations && size <= (l.maxComputeWorkgroupSizeX ?? 256)) {
      suggestions.push({ x: size, y: 1, z: 1, invocations: size });
    }
  }
  
  // 2D workgroups
  for (const size of [8, 16]) {
    const invocations = size * size;
    if (invocations <= maxInvocations && 
        size <= (l.maxComputeWorkgroupSizeX ?? 256) &&
        size <= (l.maxComputeWorkgroupSizeY ?? 256)) {
      suggestions.push({ x: size, y: size, z: 1, invocations });
    }
  }
  
  // 3D workgroups
  for (const size of [4, 8]) {
    const invocations = size * size * size;
    if (invocations <= maxInvocations &&
        size <= (l.maxComputeWorkgroupSizeX ?? 256) &&
        size <= (l.maxComputeWorkgroupSizeY ?? 256) &&
        size <= (l.maxComputeWorkgroupSizeZ ?? 64)) {
      suggestions.push({ x: size, y: size, z: size, invocations });
    }
  }
  
  return suggestions;
}

/**
 * Clear cached limits (useful for testing)
 */
export function clearLimitsCache() {
  cachedLimits = null;
  telemetryEmitted.clear();
}

/**
 * Telemetry emitter (stub - replace with your telemetry system)
 */
function emitTelemetry(event: string, data: any) {
  // In production, send to your telemetry service
  console.log(`📊 Telemetry: ${event}`, data);
  
  // Example: send to analytics
  // if (window.analytics) {
  //   window.analytics.track(event, data);
  // }
}
