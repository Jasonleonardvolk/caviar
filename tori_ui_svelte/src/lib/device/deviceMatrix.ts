import matrix from '../../../config/device-matrix.json';

export type DeviceCaps = {
  maxN: number;
  zernikeModes: number;
  serverFallback: boolean;
};

export type Tier = 'GOLD' | 'SILVER' | 'UNSUPPORTED';

export function resolveTier(hw: string): Tier {
  // hw is e.g., "iPhone17,1" or "iPad16,3"
  for (const tier of ['GOLD','SILVER'] as const) {
    if (matrix.tiers[tier].models.includes(hw)) return tier;
  }
  return 'UNSUPPORTED';
}

export function isSupported(hw: string): boolean {
  return resolveTier(hw) !== 'UNSUPPORTED';
}

export function resolveCaps(model: string): DeviceCaps {
  const tier = resolveTier(model);
  
  if (tier === 'GOLD') {
    return {
      maxN: matrix.tiers.GOLD.maxN,
      zernikeModes: matrix.tiers.GOLD.zernikeModes,
      serverFallback: matrix.tiers.GOLD.serverFallback
    };
  } else if (tier === 'SILVER') {
    return {
      maxN: matrix.tiers.SILVER.maxN,
      zernikeModes: matrix.tiers.SILVER.zernikeModes,
      serverFallback: matrix.tiers.SILVER.serverFallback
    };
  }
  
  // Unsupported device - minimal capabilities
  return {
    maxN: 128,
    zernikeModes: 8,
    serverFallback: false
  };
}