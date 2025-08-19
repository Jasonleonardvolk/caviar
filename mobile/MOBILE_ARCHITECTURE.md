# TORI Hologram Mobile App Architecture

## Size Budget (<100MB Target)

### Asset Breakdown
- **Base app framework**: ~15MB (Capacitor + WebView runtime)
- **WGSL shaders (mobile)**: ~5MB (compressed, f16 variants only)  
- **3D Gaussian Splat decoder**: ~8MB (WASM module)
- **UI assets**: ~3MB (Svelte bundle, minimal CSS)
- **Calibration profiles**: ~1MB (common devices)
- **FFT kernels**: ~2MB (WASM SIMD optimized)
- **Total**: ~34MB base, leaving 66MB for content/updates

### Quality Tiers

```typescript
export enum MobileQualityPreset {
  BATTERY_SAVER = 'battery',    // 10x8 views, 30fps, f16
  BALANCED = 'balanced',         // 15x10 views, 45fps, f16
  PERFORMANCE = 'performance',   // 20x12 views, 60fps, f16/f32 mix
  DESKTOP_STREAM = 'stream'      // Full quality via WebRTC
}
```

### Shader Variants Strategy

```javascript
// Mobile shader loader
const SHADER_VARIANTS = {
  'battery': {
    views: [10, 8],
    propagation: 'wavefieldPropagation_f16_2tap.wgsl',
    synthesis: 'multiViewSynthesis_mobile_low.wgsl',
    maxTextureSize: 512
  },
  'balanced': {
    views: [15, 10],
    propagation: 'wavefieldPropagation_f16_4tap.wgsl',
    synthesis: 'multiViewSynthesis_mobile_med.wgsl',
    maxTextureSize: 768
  },
  'performance': {
    views: [20, 12],
    propagation: 'wavefieldPropagation_mixed.wgsl',
    synthesis: 'multiViewSynthesis_mobile_high.wgsl',
    maxTextureSize: 1024
  }
};
```

## Dickbox Configuration Changes

Since mobile is a separate app, the Dickbox configuration focuses only on the desktop service, but includes mobile support endpoints:

```yaml
# ${IRIS_ROOT}\dickbox\hologram\capsule.yml
name: tori-hologram
version: "1.0.0"
entrypoint: dist/hologram-desktop.js

services:
  - name: tori-hologram
    slice: tori-hologram.slice
    
    resource_limits:
      cpu_quota: 400       # 4 cores for FFT/propagation
      cpu_weight: 200      # High priority
      memory_max: 2G       # System RAM
      memory_high: 1.5G
      tasks_max: 256
      io_weight: 150       # Fast texture I/O
    
    gpu_config:
      enabled: true
      visible_devices: "0"
      mode: exclusive      # No MPS, full GPU access
      vram_reserve_mb: 5734  # 70% of 8GB
    
    environment:
      HOLOGRAM_MODE: desktop
      MOBILE_BRIDGE_PORT: "7691"
      WEBRTC_ENABLED: "true"
      ZEROMQ_ENDPOINT: "ipc:///run/tori/bus.sock"
      METRICS_PORT: "9715"
    
    health_check:
      http:
        path: "/health"
        port: 7690
        interval: 10s

# Mobile bridge configuration
mobile_bridge:
  enabled: true
  auth_mode: jwt_qr
  stream_codecs:
    - h265
    - vp9
  max_clients: 5
  bandwidth_limit_mbps: 50

# ZeroMQ topics
pubsub_topics:
  subscribe:
    - "oscillator.state"
    - "oscillator.psi"
  publish:
    - "hologram.metrics"
    - "hologram.health"
```

## Mobile App Capacitor Structure

```javascript
// capacitor.config.ts
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.tori.hologram',
  appName: 'TORI Hologram',
  webDir: 'dist-mobile',
  bundledWebRuntime: false,
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: "#000000"
    }
  },
  server: {
    // For dev only
    url: 'http://localhost:5173',
    cleartext: true
  }
};

export default config;
```

## Build Pipeline Updates

```bash
# package.json scripts
{
  "scripts": {
    "build:mobile": "vite build --mode mobile --outDir dist-mobile",
    "build:desktop": "vite build --mode desktop --outDir dist",
    "build:capsule": "npm run build:desktop && ./scripts/build_capsule.sh",
    "cap:sync": "cap sync",
    "cap:android": "cap open android",
    "cap:ios": "cap open ios"
  }
}
```

## Mobile-Specific Vite Config

```javascript
// vite.config.mobile.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'webgpu-core': ['./lib/holographicEngine_mobile.ts'],
          'shaders-battery': ['./shaders/mobile/battery/*.wgsl'],
          'shaders-balanced': ['./shaders/mobile/balanced/*.wgsl'],
          'shaders-performance': ['./shaders/mobile/performance/*.wgsl']
        }
      }
    },
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  define: {
    __MOBILE_BUILD__: true,
    __DESKTOP_STREAMING__: true
  }
});
```
