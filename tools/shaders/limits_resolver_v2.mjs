// Shared limits resolver with platform alias support and validation
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { validateLimitsSchema, normalizeLimits } = require('./lib/limits_schema.js');

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
      if (verbose) {
        console.log(`📁 Direct path resolved: ${requested}`);
      }
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
  
  // Try pointer files in order of preference
  const pointerFiles = [];
  
  if (platform) {
    // Platform-specific pointers
    pointerFiles.push(`latest.${platform}.json`);
    pointerFiles.push(`latest.${platform}.good.json`);
  }
  
  // Generic pointer as fallback
  pointerFiles.push('latest.json');
  
  // Try each pointer file
  for (const pointerFile of pointerFiles) {
    const pointerPath = path.join(
      process.cwd(),
      'tools',
      'shaders',
      'device_limits',
      pointerFile
    );
    
    if (fs.existsSync(pointerPath)) {
      try {
        const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf8'));
        
        if (pointer.path) {
          const targetPath = path.resolve(process.cwd(), pointer.path);
          
          if (fs.existsSync(targetPath)) {
            if (verbose) {
              console.log(`📎 Resolved ${requested} -> ${pointer.path}`);
              console.log(`   via pointer: ${pointerFile}`);
              if (pointer.device) {
                console.log(`   Device: ${pointer.device}`);
              }
              if (pointer.checksum) {
                console.log(`   Checksum: ${pointer.checksum}`);
              }
            }
            return pointer.path;
          }
          
          if (verbose) {
            console.warn(`⚠️ Pointer target not found: ${pointer.path}`);
          }
        }
      } catch (e) {
        console.warn(`⚠️ Could not parse ${pointerFile}: ${e.message}`);
      }
    }
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
    return { profiles: [], pointers: [] };
  }
  
  const files = fs.readdirSync(limitsDir);
  const profiles = [];
  const pointers = [];
  
  for (const file of files) {
    if (file.startsWith('latest.') && file.endsWith('.json')) {
      pointers.push(file);
    } else if (file.endsWith('.json') && !file.startsWith('latest')) {
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
  
  return { profiles, pointers };
}

/**
 * Load and validate limits from resolved path
 */
export function loadLimits(limitsPath, options = {}) {
  const { validate = true, verbose = false, platform = 'default' } = options;
  
  const fullPath = path.resolve(process.cwd(), limitsPath);
  
  if (!fs.existsSync(fullPath)) {
    throw new Error(`Limits file not found: ${limitsPath}`);
  }
  
  try {
    const content = fs.readFileSync(fullPath, 'utf8');
    const json = JSON.parse(content);
    
    // Handle both capture format and simple format
    const rawLimits = json.limits || json;
    
    if (validate) {
      const { ok, errors, warnings, normalized } = validateLimitsSchema(rawLimits, { platform });
      
      if (!ok) {
        const msg = `[limits] ${limitsPath} failed validation: ${errors.join(' | ')}`;
        throw new Error(msg);
      }
      
      if (warnings.length && verbose) {
        console.warn(`⚠️ [limits] ${limitsPath} warnings:`);
        warnings.forEach(w => console.warn(`   - ${w}`));
      }
      
      if (verbose && json.version) {
        console.log(`📊 Loaded limits v${json.version} from ${json.capturedAt || 'unknown date'}`);
      }
      
      return normalized;
    }
    
    // If not validating, just normalize
    return normalizeLimits(rawLimits);
    
  } catch (e) {
    if (e.message.includes('failed validation')) {
      throw e; // Re-throw validation errors
    }
    throw new Error(`Failed to load limits from ${limitsPath}: ${e.message}`);
  }
}

/**
 * Load and validate limits with automatic resolution
 */
export function loadAndValidateLimits(limitsArg, options = {}) {
  const { verbose = false } = options;
  
  // Resolve the path
  const resolvedPath = resolveLimitsPath(limitsArg, { verbose });
  
  // Detect platform from resolved path for validation
  let platform = 'default';
  if (resolvedPath.includes('iphone') || resolvedPath.includes('ios')) {
    platform = 'ios';
  } else if (resolvedPath.includes('android')) {
    platform = 'android';
  } else if (resolvedPath.includes('desktop') || resolvedPath.includes('windows')) {
    platform = 'windows';
  } else if (resolvedPath.includes('mac')) {
    platform = 'mac';
  } else if (resolvedPath.includes('linux')) {
    platform = 'linux';
  }
  
  // Load and validate
  const limits = loadLimits(resolvedPath, { validate: true, verbose, platform });
  
  if (verbose) {
    console.log(`✅ Loaded and validated limits from: ${resolvedPath}`);
    console.log(`   Platform hint: ${platform}`);
  }
  
  return {
    limits,
    path: resolvedPath,
    platform
  };
}

// For CommonJS compatibility
export default {
  resolveLimitsPath,
  listAvailableProfiles,
  loadLimits,
  loadAndValidateLimits
};
