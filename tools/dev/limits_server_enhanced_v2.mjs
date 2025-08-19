#!/usr/bin/env node
// Enhanced limits server with platform aliases, schema validation, and versioning
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { validateLimitsSchema, normalizeLimits, PLATFORM_BOUNDS } = require('../shaders/lib/limits_schema.js');

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
    if (key === 'capturedAt' || key === 'version' || key === 'timestamp' || key === 'lastUpdated') continue;
    if (a[key] !== b[key]) return false;
  }
  
  return true;
}

// Atomic file write with temp file + rename
function writeJsonAtomic(filePath, obj) {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.tmp.${Date.now()}.${Math.random().toString(36).slice(2)}.json`);
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2));
  fs.renameSync(tmp, filePath);
}

// Sanitize device name to prevent path traversal
function sanitizeName(name) {
  // Allow only alphanumeric, dots, underscores, and hyphens
  return name.replace(/[^A-Za-z0-9._-]/g, '_').substring(0, 64);
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
    
    // Sanitize the device name
    const safeName = sanitizeName(name);
    
    // Detect platform
    const userAgent = req.headers['user-agent'] || metadata.userAgent || '';
    const platform = (metadata.platform || detectPlatform(userAgent)).toLowerCase();
    
    // Validate and normalize using shared library
    const { ok, errors, warnings, normalized } = validateLimitsSchema(limits, { platform });
    
    if (!ok) {
      return res.status(400).json({ 
        ok: false, 
        error: 'Invalid limits', 
        details: { errors, warnings } 
      });
    }
    
    // Log warnings but still accept
    if (warnings.length) {
      console.warn(`[limits] Sanity warnings for ${safeName}:`, warnings);
    }
    
    // Create directory structure
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const captureDir = path.join(
      process.cwd(), 
      'tools', 
      'shaders', 
      'device_limits',
      'captures',
      platform,
      safeName
    );
    
    fs.mkdirSync(captureDir, { recursive: true });
    
    // Build complete payload with metadata
    const payload = {
      version: 1,
      name: safeName,
      platform,
      capturedAt: new Date().toISOString(),
      userAgent,
      adapterInfo: metadata.adapterInfo || {},
      features: metadata.features || [],
      limits: normalized,
      checksum: crypto.createHash('sha256')
        .update(JSON.stringify(normalized))
        .digest('hex')
        .substring(0, 8),
      warnings: warnings.length > 0 ? warnings : undefined
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
            console.log(`📦 Skipping identical limits for ${safeName} on ${platform}`);
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
    writeJsonAtomic(capturePath, payload);
    
    const relativePath = path.relative(process.cwd(), capturePath).replace(/\\/g, '/');
    
    // Update platform-specific pointer
    const platformPointer = {
      path: relativePath,
      timestamp,
      platform,
      device: safeName,
      checksum: payload.checksum
    };
    
    writeJsonAtomic(latestPointerPath, platformPointer);
    
    // Update good pointer only if no warnings
    if (warnings.length === 0) {
      const goodPointerPath = path.join(
        process.cwd(),
        'tools',
        'shaders',
        'device_limits',
        `latest.${platform}.good.json`
      );
      writeJsonAtomic(goodPointerPath, platformPointer);
      console.log(`✅ Updated latest.${platform}.good.json (clean capture)`);
    } else {
      // If there are warnings, create a quarantined pointer
      const quarantinedPath = path.join(
        process.cwd(),
        'tools',
        'shaders',
        'device_limits',
        `latest.${platform}.quarantined.json`
      );
      writeJsonAtomic(quarantinedPath, {
        ...platformPointer,
        warnings
      });
      console.log(`⚠️  Created latest.${platform}.quarantined.json (has warnings)`);
    }
    
    // Also update generic latest.json for backward compatibility
    const genericPointerPath = path.join(
      process.cwd(),
      'tools',
      'shaders',
      'device_limits', 
      'latest.json'
    );
    
    writeJsonAtomic(genericPointerPath, platformPointer);
    
    // Save simplified profile for quick access
    const profilePath = path.join(
      process.cwd(),
      'tools',
      'shaders',
      'device_limits',
      `${safeName}.json`
    );
    
    const profile = {
      ...normalized,
      label: safeName,
      platform,
      lastUpdated: timestamp
    };
    
    writeJsonAtomic(profilePath, profile);
    
    console.log(`✅ Saved GPU limits for ${safeName} on ${platform}`);
    console.log(`   📁 Capture: ${relativePath}`);
    console.log(`   📍 Pointer: latest.${platform}.json`);
    console.log(`   📋 Profile: ${safeName}.json`);
    if (warnings.length > 0) {
      console.log(`   ⚠️  Warnings: ${warnings.length}`);
    }
    
    return res.json({
      ok: true,
      paths: {
        capture: relativePath,
        pointer: `latest.${platform}.json`,
        profile: `${safeName}.json`
      },
      checksum: payload.checksum,
      warnings: warnings.length > 0 ? warnings : undefined
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
  
  // Try different pointer files in order
  const pointerFiles = [
    `latest.${platform}.json`,
    `latest.${platform}.good.json`,
    'latest.json'
  ];
  
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
        const limitsPath = path.resolve(process.cwd(), pointer.path);
        
        if (fs.existsSync(limitsPath)) {
          const limits = JSON.parse(fs.readFileSync(limitsPath, 'utf8'));
          return res.json({ 
            ok: true, 
            limits,
            source: pointerFile
          });
        }
      } catch (error) {
        console.error(`Error reading ${pointerFile}:`, error);
      }
    }
  }
  
  return res.status(404).json({ 
    ok: false, 
    error: `No limits captured for platform: ${platform}` 
  });
});

// List all available profiles
app.get('/api/dev/limits', (req, res) => {
  const limitsDir = path.join(
    process.cwd(),
    'tools',
    'shaders',
    'device_limits'
  );
  
  const profiles = [];
  const pointers = [];
  
  if (fs.existsSync(limitsDir)) {
    const files = fs.readdirSync(limitsDir);
    
    for (const file of files) {
      if (file.startsWith('latest.') && file.endsWith('.json')) {
        pointers.push(file);
      } else if (file.endsWith('.json') && !file.startsWith('latest')) {
        try {
          const content = JSON.parse(fs.readFileSync(path.join(limitsDir, file), 'utf8'));
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
  }
  
  res.json({
    ok: true,
    profiles,
    pointers
  });
});

const PORT = process.env.LIMITS_PORT || 5178;
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║           GPU LIMITS CAPTURE SERVER (ENHANCED v2)              ║
╚════════════════════════════════════════════════════════════════╝

🚀 Server running at: http://localhost:${PORT}

📍 Endpoints:
   POST /api/dev/save-gpu-limits   - Save device limits
   GET  /api/dev/limits/:platform  - Get latest for platform
   GET  /api/dev/limits            - List all profiles

🎯 Features:
   ✅ Shared validation library (limits_schema.js)
   ✅ Platform-specific aliases (latest.ios, latest.android, etc.)
   ✅ Good/quarantined pointer pattern
   ✅ Timestamped captures with version tracking
   ✅ Schema validation with sanity checks
   ✅ Idempotence (skips identical captures)
   ✅ Atomic file writes (temp + rename)
   ✅ CORS for local dev only
   ✅ Name sanitization for security

⚠️  Development only - do not use in production!
`);
});
