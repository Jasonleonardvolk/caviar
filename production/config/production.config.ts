// production/config/production.config.ts
// Production configuration for holographic rendering system

export const PRODUCTION_CONFIG = {
  // WebGPU Configuration
  webgpu: {
    enabled: true,
    fallbackToWebGL: true,
    requiredFeatures: [],
    optionalFeatures: ['timestamp-query', 'shader-f16']
  },

  // FFT Configuration
  fft: {
    precomputedSizes: [256, 512, 1024, 2048, 4096],
    defaultSize: 1024,
    precision: 'f32' as const,
    enableBatching: true,
    maxBatchSize: 16,
    cacheStrategy: 'lazy' as const
  },

  // Shader Configuration
  shaders: {
    bundleMode: 'production',
    minify: true,
    sourceMaps: false,
    caching: {
      enabled: true,
      maxAge: 86400 // 24 hours
    }
  },

  // Performance Settings
  performance: {
    targetFPS: 60,
    enableProfiling: false,
    gpuTimeout: 5000,
    memoryLimit: 512 * 1024 * 1024 // 512MB
  },

  // CDN Configuration
  cdn: {
    enabled: true,
    baseUrl: process.env.CDN_URL || 'https://cdn.example.com',
    assets: {
      fft: '/assets/fft/',
      shaders: '/assets/shaders/',
      wasm: '/assets/wasm/'
    }
  },

  // Monitoring
  monitoring: {
    sentry: {
      enabled: true,
      dsn: process.env.SENTRY_DSN,
      environment: 'production',
      sampleRate: 0.1
    },
    analytics: {
      enabled: true,
      id: process.env.GA_ID
    }
  },

  // Security
  security: {
    csp: {
      'default-src': ["'self'"],
      'script-src': ["'self'", "'wasm-unsafe-eval'"],
      'worker-src': ["'self'", 'blob:'],
      'style-src': ["'self'", "'unsafe-inline'"]
    },
    cors: {
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['*'],
      credentials: true
    }
  },

  // Feature Flags
  features: {
    webgpuCompute: true,
    adaptiveQuality: true,
    experimentalShaders: false,
    debugMode: false
  }
};

export type ProductionConfig = typeof PRODUCTION_CONFIG;
