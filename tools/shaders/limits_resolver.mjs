// Shared limits resolver with platform alias support
import fs from 'fs';
import path from 'path';

/**
 * Resolve limits aliases to actual file paths
 * Supports:
 * - Direct paths: tools/shaders/device_limits/iphone15.json
 * - Latest alias: latest -> latest.json pointer
 * - Platform aliases: latest.ios, latest.android, latest.windows, etc.
 * - Auto detection: auto -> detect platform and use appropriate latest
 * - ENV override: SHADER_LIMITS environment variable
 */
export function resolveLimitsPath(limitsArg, options = {}) {
  const { 
    verbose = false,
    autoDetectPlatform = true 
  } = options;
  
  // Priority: ENV > CLI > default
  const env = process.env.SHADER_LIMITS?.trim();
  const cli = limitsArg?.trim();
  const requested = env || cli || 'tools/shaders/device_limits/iphone15.json';
  
  if (verbose && env) {
    console.log(`📋 Using limits from ENV: ${env}`);
  }
  
  // Direct path - not an alias
  if (!requested.startsWith('latest') && requested !== 'auto') {
    const resolved = path.resolve(process.cwd(), requested);
    if (fs.existsSync(resolved)) {
      return requested;
    }
    console.warn(`⚠️ Limits file not found: ${requested}`);
    return getFallback();
  }
  
  // Parse platform from alias (e.g., latest.ios -> ios)
  let platform = null;
  if (requested.includes('.')) {
    const parts = requested.split('.');
    if (parts[0] === 'latest' && parts[1]) {
      platform = parts[1];
    }
  }
  
  // Auto-detect platform
  if (requested === 'auto' && autoDetectPlatform) {
    platform = detectCurrentPlatform();
    if (verbose) {
      console.log(`🔍 Auto-detected platform: ${platform}`);
    }
  }
  
  // Construct pointer filename
  let pointerFile = 'latest.json';
  if (platform) {
    pointerFile = `latest.${platform}.json`;
  }
  
  const pointerPath = path.join(
    process.cwd(),
    'tools',
    'shaders',
    'device_limits',
    pointerFile
  );
  
  // Try to read pointer
  if (fs.existsSync(pointerPath)) {
    try {
      const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf8'));
      
      if (pointer.path) {
        const targetPath = path.resolve(process.cwd(), pointer.path);
        
        if (fs.existsSync(targetPath)) {
          if (verbose) {
            console.log(`📎 Resolved ${requested} -> ${pointer.path}`);
            if (pointer.device) {
              console.log(`   Device: ${pointer.device}`);
            }
            if (pointer.checksum) {
              console.log(`   Checksum: ${pointer.checksum}`);
            }
          }
          return pointer.path;
        }
        
        console.warn(`⚠️ Pointer target not found: ${pointer.path}`);
      }
    } catch (e) {
      console.warn(`⚠️ Could not parse ${pointerFile}: ${e.message}`);
    }
  } else if (verbose) {
    console.log(`📍 No pointer file: ${pointerFile}`);
  }
  
  // Fallback
  const fallback = getFallback(platform);
  if (verbose) {
    console.log(`↩️ Falling back to: ${fallback}`);
  }
  return fallback;
}

/**
 * Get platform-appropriate fallback
 */
function getFallback(platform) {
  const fallbacks = {
    ios: 'tools/shaders/device_limits/iphone15.json',
    android: 'tools/shaders/device_limits/android_baseline.json',
    windows: 'tools/shaders/device_limits/desktop.json',
    mac: 'tools/shaders/device_limits/desktop.json',
    linux: 'tools/shaders/device_limits/desktop_low.json'
  };
  
  const specific = fallbacks[platform];
  if (specific && fs.existsSync(path.resolve(process.cwd(), specific))) {
    return specific;
  }
  
  // Ultimate fallback
  return 'tools/shaders/device_limits/iphone15.json';
}

/**
 * Detect current platform from env/process
 */
function detectCurrentPlatform() {
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform === 'darwin') return 'mac';
  if (platform === 'win32') return 'windows';
  if (platform === 'linux') {
    // Check if Android via env
    if (process.env.ANDROID_ROOT) return 'android';
    return 'linux';
  }
  
  return 'unknown';
}

/**
 * List all available limits profiles
 */
export function listAvailableProfiles() {
  const limitsDir = path.join(
    process.cwd(),
    'tools',
    'shaders',
    'device_limits'
  );
  
  if (!fs.existsSync(limitsDir)) {
    return [];
  }
  
  const files = fs.readdirSync(limitsDir);
  const profiles = [];
  
  for (const file of files) {
    if (file.endsWith('.json') && !file.startsWith('latest')) {
      const fullPath = path.join(limitsDir, file);
      try {
        const content = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
        profiles.push({
          file,
          name: content.label || file.replace('.json', ''),
          platform: content.platform || 'unknown'
        });
      } catch {
        // Skip invalid files
      }
    }
  }
  
  // Check for platform pointers
  const pointers = files.filter(f => f.startsWith('latest.') && f.endsWith('.json'));
  
  return { profiles, pointers };
}

/**
 * Load and validate limits from resolved path
 */
export function loadLimits(limitsPath, options = {}) {
  const { validate = true, verbose = false } = options;
  
  const fullPath = path.resolve(process.cwd(), limitsPath);
  
  if (!fs.existsSync(fullPath)) {
    throw new Error(`Limits file not found: ${limitsPath}`);
  }
  
  try {
    const content = fs.readFileSync(fullPath, 'utf8');
    const limits = JSON.parse(content);
    
    // Handle both capture format and simple format
    const actualLimits = limits.limits || limits;
    
    if (validate) {
      // Basic validation
      const required = [
        'maxComputeInvocationsPerWorkgroup',
        'maxComputeWorkgroupSizeX',
        'maxComputeWorkgroupSizeY', 
        'maxComputeWorkgroupSizeZ'
      ];
      
      for (const field of required) {
        if (actualLimits[field] == null) {
          console.warn(`⚠️ Missing required limit: ${field}`);
        }
      }
    }
    
    if (verbose && limits.version) {
      console.log(`📊 Loaded limits v${limits.version} from ${limits.capturedAt || 'unknown date'}`);
    }
    
    return actualLimits;
    
  } catch (e) {
    throw new Error(`Failed to load limits from ${limitsPath}: ${e.message}`);
  }
}

// For CommonJS compatibility
export default {
  resolveLimitsPath,
  listAvailableProfiles,
  loadLimits
};
