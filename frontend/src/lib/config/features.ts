/**
 * Feature flags for Carrollian physics rollout
 * Controls shadow mode testing and A/B experiments
 */

export const FEATURES = {
  // Enable Carrollian physics in shadow mode (compute only, no visual changes)
  ENABLE_CARROLLIAN_SHADOW: false,
  
  // Enable A/B testing (server-controlled rollout)
  ENABLE_CARROLLIAN_AB: false,
  
  // Enable thin-screen curvature phase (server-controlled)
  ENABLE_CURVATURE_PHASE: false,
  
  // Development flags
  DEBUG_TELEMETRY: false,
  LOG_FRAME_METRICS: false,
};

// Device tier detection for targeted rollout
export function detectDeviceTier(): string {
  const gpu = (navigator as any).gpu;
  const userAgent = navigator.userAgent.toLowerCase();
  
  // iOS detection
  if (/iphone/.test(userAgent)) {
    if (/iphone 15/.test(userAgent) || /iphone 16/.test(userAgent)) {
      return 'iphone15.high';
    }
    if (/iphone 14/.test(userAgent)) {
      return 'iphone14.medium';
    }
    return 'iphone.low';
  }
  
  // iPad detection
  if (/ipad/.test(userAgent)) {
    if (/ipad pro/.test(userAgent)) {
      return 'ipadPro.m3.high';
    }
    return 'ipad.medium';
  }
  
  // Desktop GPU detection (simplified)
  if (gpu?.requestAdapter) {
    // This would need actual GPU detection logic
    // For now, basic categorization
    const isHighEnd = /nvidia|radeon rx 7|rtx 40/.test(userAgent);
    if (isHighEnd) {
      return 'desktop.4090.ultra';
    }
  }
  
  return 'desktop.medium';
}

// Check if device is in Tier-A for A/B testing
export function isTierADevice(): boolean {
  const tier = detectDeviceTier();
  const tierADevices = [
    'iphone15.high',
    'ipadPro.m3.high',
    'desktop.4090.ultra'
  ];
  return tierADevices.includes(tier);
}

// Get feature flags from server (for A/B testing)
export async function fetchRemoteFeatures(): Promise<Partial<typeof FEATURES>> {
  try {
    const response = await fetch('/api/features/config', {
      headers: {
        'X-Device-Tier': detectDeviceTier(),
      }
    });
    
    if (response.ok) {
      const remoteFeatures = await response.json();
      return remoteFeatures;
    }
  } catch (error) {
    console.warn('Failed to fetch remote features:', error);
  }
  
  return {};
}

// Apply remote features to local config
export async function initializeFeatures(): Promise<void> {
  const remoteFeatures = await fetchRemoteFeatures();
  Object.assign(FEATURES, remoteFeatures);
  
  if (FEATURES.DEBUG_TELEMETRY) {
    console.log('Features initialized:', FEATURES);
    console.log('Device tier:', detectDeviceTier());
  }
}
