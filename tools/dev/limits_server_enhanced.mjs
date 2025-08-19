#!/usr/bin/env node
/**
 * @file limits_server_enhanced.mjs
 * @status INTENTIONAL_ENHANCED_VERSION - DO NOT DELETE
 * @see ENHANCED_FILES_MANIFEST.md for more information
 * 
 * Enhanced limits server with platform aliases, schema validation, and versioning
 * This is a legitimate enhanced version that coexists with the base limits_server.mjs
 * Used by: npm run dev:limits:enhanced
 */
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '10mb' }));

// Only run in development
if (process.env.NODE_ENV === 'production') {
  console.error('❌ Limits server should not run in production!');
  process.exit(1);
}

// CORS for dev only
app.use((req, res, next) => {
  const allowedOrigins = ['http://localhost:3000', 'http://localhost:5173', 'http://localhost:8080'];
  const origin = req.headers.origin;
  
  if (allowedOrigins.includes(origin)) {
    res.header('Access-Control-Allow-Origin', origin);
    res.header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
  }
  
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  
  next();
});

// Schema for limits validation
const LIMITS_SCHEMA = {
  required: [
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX', 
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ'
  ],
  optional: [
    'maxComputeWorkgroupStorageSize',
    'maxWorkgroupStorageSize',
    'maxSampledTexturesPerShaderStage',
    'maxSamplersPerShaderStage',
    'maxStorageBuffersPerShaderStage',
    'maxUniformBuffersPerShaderStage'
  ]
};

// Normalize field names (handle variations)
function normalizeLimits(limits) {
  const normalized = {};
  
  // Handle storage size variations
  normalized.maxComputeWorkgroupStorageSize = 
    limits.maxComputeWorkgroupStorageSize ?? 
    limits.maxWorkgroupStorageSize ?? 
    null;
  
  // Copy standard fields
  for (const field of LIMITS_SCHEMA.required) {
    normalized[field] = limits[field] ?? null;
  }
  
  for (const field of LIMITS_SCHEMA.optional) {
    if (field !== 'maxWorkgroupStorageSize') { // Skip alias
      normalized[field] = limits[field] ?? null;
    }
  }
  
  return normalized;
}

// Validate limits against schema
function validateLimits(limits) {
  const errors = [];
  
  // Check required fields
  for (const field of LIMITS_SCHEMA.required) {
    if (limits[field] == null) {
      errors.push(`Missing required field: ${field}`);
    } else if (typeof limits[field] !== 'number' || limits[field] < 0) {
      errors.push(`Invalid value for ${field}: must be non-negative number`);
    }
  }
  
  // Check optional fields if present
  for (const field of LIMITS_SCHEMA.optional) {
    if (limits[field] != null && (typeof limits[field] !== 'number' || limits[field] < 0)) {
      errors.push(`Invalid value for ${field}: must be non-negative number`);
    }
  }
  
  return errors;
}

// Detect platform from user agent
function detectPlatform(userAgent = '') {
  const ua = userAgent.toLowerCase();
  
  if (ua.includes('iphone') || ua.includes('ipad')) return 'ios';
  if (ua.includes('android')) return 'android';
  if (ua.includes('mac')) return 'mac';
  if (ua.includes('windows')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  
  return 'unknown';
}

// Check if limits are identical to avoid noisy diffs
function areIdenticalLimits(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  
  for (const key of keys) {
    if (key === 'capturedAt' || key === 'version') continue;
    if (a[key] !== b[key]) return false;
  }
  
  return true;
}

// Main endpoint to save GPU limits
app.post('/api/dev/save-gpu-limits', async (req, res) => {
  try {
    const { name, limits, metadata = {} } = req.body;
    
    if (!name || !limits) {
      return res.status(400).json({ 
        ok: false, 
        error: 'Missing name or limits' 
      });
    }
    
    // Normalize the limits
    const normalized = normalizeLimits(limits);
    
    // Validate schema
    const errors = validateLimits(normalized);
    if (errors.length > 0) {
      return res.status(400).json({ 
        ok: false, 
        errors,
        hint: 'Limits must contain valid numeric values for required fields'
      });
    }
    
    // Detect platform
    const userAgent = req.headers['user-agent'] || metadata.userAgent || '';
    const platform = metadata.platform || detectPlatform(userAgent);
    
    // Create directory structure
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const captureDir = path.join(
      process.cwd(), 
      'tools', 
      'shaders', 
      'device_limits',
      'captures',
      platform,
      name
    );
    
    fs.mkdirSync(captureDir, { recursive: true });
    
    // Build complete payload with metadata
    const payload = {
      version: 1,
      name,
      platform,
      capturedAt: new Date().toISOString(),
      userAgent,
      adapterInfo: metadata.adapterInfo || {},
      features: metadata.features || [],
      limits: normalized,
      checksum: crypto.createHash('sha256')
        .update(JSON.stringify(normalized))
        .digest('hex')
        .substring(0, 8)
    };
    
    // Check for identical previous capture (idempotence)
    const latestPointerPath = path.join(
      process.cwd(),
      'tools',
      'shaders', 
      'device_limits',
      `latest.${platform}.json`
    );
    
    if (fs.existsSync(latestPointerPath)) {
      try {
        const pointer = JSON.parse(fs.readFileSync(latestPointerPath, 'utf8'));
        if (pointer.path && fs.existsSync(path.resolve(process.cwd(), pointer.path))) {
          const previous = JSON.parse(
            fs.readFileSync(path.resolve(process.cwd(), pointer.path), 'utf8')
          );
          
          if (areIdenticalLimits(previous.limits, normalized)) {
            console.log(`📦 Skipping identical limits for ${name} on ${platform}`);
            return res.json({
              ok: true,
              skipped: true,
              reason: 'Identical to previous capture',
              path: pointer.path
            });
          }
        }
      } catch (e) {
        console.warn('Could not check previous capture:', e.message);
      }
    }
    
    // Save timestamped capture
    const captureFile = `${timestamp}.json`;
    const capturePath = path.join(captureDir, captureFile);
    fs.writeFileSync(capturePath, JSON.stringify(payload, null, 2));
    
    const relativePath = path.relative(process.cwd(), capturePath).replace(/\\/g, '/');
    
    // Update platform-specific pointer
    const platformPointer = {
      path: relativePath,
      timestamp,
      platform,
      device: name,
      checksum: payload.checksum
    };
    
    fs.writeFileSync(latestPointerPath, JSON.stringify(platformPointer, null, 2));
    
    // Also update generic latest.json for backward compatibility
    const genericPointerPath = path.join(
      process.cwd(),
      'tools',
      'shaders',
      'device_limits', 
      'latest.json'
    );
    
    fs.writeFileSync(genericPointerPath, JSON.stringify(platformPointer, null, 2));
    
    // Save simplified profile for quick access
    const profilePath = path.join(
      process.cwd(),
      'tools',
      'shaders',
      'device_limits',
      `${name}.json`
    );
    
    const profile = {
      ...normalized,
      label: name,
      platform,
      lastUpdated: timestamp
    };
    
    fs.writeFileSync(profilePath, JSON.stringify(profile, null, 2));
    
    console.log(`✅ Saved GPU limits for ${name} on ${platform}`);
    console.log(`   📁 Capture: ${relativePath}`);
    console.log(`   📍 Pointer: latest.${platform}.json`);
    console.log(`   📋 Profile: ${name}.json`);
    
    return res.json({
      ok: true,
      paths: {
        capture: relativePath,
        pointer: `latest.${platform}.json`,
        profile: `${name}.json`
      },
      checksum: payload.checksum
    });
    
  } catch (error) {
    console.error('Error saving limits:', error);
    return res.status(500).json({ 
      ok: false, 
      error: error.message 
    });
  }
});

// Get latest limits for platform
app.get('/api/dev/limits/:platform?', (req, res) => {
  const platform = req.params.platform || 'unknown';
  const pointerPath = path.join(
    process.cwd(),
    'tools',
    'shaders',
    'device_limits',
    `latest.${platform}.json`
  );
  
  if (!fs.existsSync(pointerPath)) {
    return res.status(404).json({ 
      ok: false, 
      error: `No limits captured for platform: ${platform}` 
    });
  }
  
  try {
    const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf8'));
    const limitsPath = path.resolve(process.cwd(), pointer.path);
    
    if (!fs.existsSync(limitsPath)) {
      return res.status(404).json({ 
        ok: false, 
        error: 'Pointed file not found' 
      });
    }
    
    const limits = JSON.parse(fs.readFileSync(limitsPath, 'utf8'));
    return res.json({ ok: true, limits });
    
  } catch (error) {
    return res.status(500).json({ 
      ok: false, 
      error: error.message 
    });
  }
});

const PORT = process.env.LIMITS_PORT || 5178;
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║           GPU LIMITS CAPTURE SERVER (ENHANCED)                 ║
╚════════════════════════════════════════════════════════════════╝

🚀 Server running at: http://localhost:${PORT}

📍 Endpoints:
   POST /api/dev/save-gpu-limits - Save device limits
   GET  /api/dev/limits/:platform - Get latest for platform

🎯 Features:
   ✅ Platform-specific aliases (latest.ios, latest.android, etc.)
   ✅ Timestamped captures with version tracking
   ✅ Schema validation and field normalization
   ✅ Idempotence (skips identical captures)
   ✅ CORS for local dev only

⚠️  Development only - do not use in production!
`);
});
