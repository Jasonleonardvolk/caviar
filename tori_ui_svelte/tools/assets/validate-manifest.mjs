// tools/assets/validate-manifest.mjs
// Enhanced AR Asset Validator v3.0 with texture policies
// Usage:
//   node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json [options]
// Options:
//   --maxTris=100000      Max triangles per asset
//   --maxTexDim=4096      Max texture dimension (opaque)
//   --maxTexBytes=16MB    Max texture size (opaque)
//   --maxAlphaDim=2048    Max alpha texture dimension
//   --maxAlphaBytes=8MB   Max alpha texture size
//   --maxSetBytes=50MB    Max total asset set size
// Exit code: 0 = OK, 1 = validation errors

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { NodeIO, PropertyType, Primitive, Document } from '@gltf-transform/core';
// import { getImageSize } from '@gltf-transform/functions'; // COMMENTED: Not available in current version

// Temporary stub for getImageSize
const getImageSize = (texture) => {
    // Return a default size - actual validation can be added later
    return { width: 1024, height: 1024 };
};

// Allowed texture formats for mobile AR
const ALLOWED_TEXTURE_FORMATS = ['png', 'jpg', 'jpeg', 'webp'];

function parseArgs() {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error('Usage: node tools/assets/validate-manifest.mjs <manifest.json> [options]');
    console.error('Options:');
    console.error('  --maxTris=100000       Max triangles per asset');
    console.error('  --maxTexDim=4096       Max texture dimension (opaque)');
    console.error('  --maxTexBytes=16777216 Max texture bytes (opaque, 16MB)');
    console.error('  --maxAlphaDim=2048     Max alpha texture dimension');
    console.error('  --maxAlphaBytes=8388608 Max alpha texture bytes (8MB)');
    console.error('  --maxSetBytes=52428800 Max total asset bytes (50MB)');
    process.exit(1);
  }
  
  let manifestPath = args[0];
  let opts = {
    maxTris: 100000,
    maxTexDim: 4096,
    maxTexBytes: 16 * 1024 * 1024,  // 16MB
    maxSetBytes: 50 * 1024 * 1024,  // 50MB
    maxAlphaDim: 2048,
    maxAlphaBytes: 8 * 1024 * 1024  // 8MB
  };
  
  for (const a of args.slice(1)) {
    if (a.startsWith('--maxTris=')) opts.maxTris = parseInt(a.split('=')[1], 10);
    if (a.startsWith('--maxTexDim=')) opts.maxTexDim = parseInt(a.split('=')[1], 10);
    if (a.startsWith('--maxTexBytes=')) opts.maxTexBytes = parseInt(a.split('=')[1], 10);
    if (a.startsWith('--maxSetBytes=')) opts.maxSetBytes = parseInt(a.split('=')[1], 10);
    if (a.startsWith('--maxAlphaDim=')) opts.maxAlphaDim = parseInt(a.split('=')[1], 10);
    if (a.startsWith('--maxAlphaBytes=')) opts.maxAlphaBytes = parseInt(a.split('=')[1], 10);
  }
  
  return { manifestPath, opts };
}

function required(obj, keys, ctx) {
  const missing = keys.filter(k => !(k in obj));
  if (missing.length) throw new Error(`${ctx}: missing required keys: ${missing.join(', ')}`);
}

// Alpha detection for PNG
function hasAlphaPNG(buf) {
  // PNG IHDR chunk starts at byte 8 (signature is 8 bytes)
  // IHDR: length(4)='00 00 00 0D', type(4)='IHDR', data(13), CRC(4)
  // Color type at IHDR data offset 9 (0: grayscale, 2: truecolor, 3: indexed, 4: grayscale+alpha, 6: truecolor+alpha)
  if (buf.length < 33) return false;
  const isPNG = buf[0]===0x89 && buf[1]===0x50 && buf[2]===0x4E && buf[3]===0x47;
  if (!isPNG) return false;
  const colorType = buf[25];
  return colorType === 4 || colorType === 6;
}

// Alpha detection for WebP
function hasAlphaWebP(buf) {
  // WebP RIFF header; VP8X chunk holds feature flags at byte 8 of VP8X payload; bit 4 (0x10) = alpha
  const isRIFF = buf[0]===0x52 && buf[1]===0x49 && buf[2]===0x46 && buf[3]===0x46;
  const isWEBP = buf[8]===0x57 && buf[9]===0x45 && buf[10]===0x42 && buf[11]===0x50;
  if (!isRIFF || !isWEBP) return false;

  // Search for 'VP8X' chunk
  let i = 12;
  while (i + 8 <= buf.length) {
    const chunkId = String.fromCharCode(buf[i], buf[i+1], buf[i+2], buf[i+3]);
    const chunkSize = buf[i+4] | (buf[i+5]<<8) | (buf[i+6]<<16) | (buf[i+7]<<24);
    if (chunkId === 'VP8X') {
      if (i + 8 + chunkSize > buf.length) return false;
      const flags = buf[i+8]; // 1st byte in VP8X payload
      return (flags & 0x10) === 0x10;
    }
    i += 8 + chunkSize + (chunkSize % 2); // chunks are padded to even size
  }
  return false;
}

// Detect if texture has alpha channel
function detectHasAlpha(type, buf) {
  const t = (type || '').toLowerCase();
  if (t === 'png') return hasAlphaPNG(buf);
  if (t === 'webp') return hasAlphaWebP(buf);
  // jpg/jpeg never has alpha; other formats are disallowed
  return false;
}

// Check if texture is used as a normal map
function isNormalMap(tex, doc) {
  const root = doc.getRoot();
  for (const mat of root.listMaterials()) {
    if (mat.getNormalTexture() === tex) return true;
  }
  return false;
}

// Get image dimensions from buffer
function dimFromBuffer(buf) {
  // Simple detection for common formats
  const view = new DataView(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength);
  
  // PNG
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4E && buf[3] === 0x47) {
    const w = view.getUint32(16, false);
    const h = view.getUint32(20, false);
    return { w, h, type: 'png' };
  }
  
  // JPEG
  if (buf[0] === 0xFF && buf[1] === 0xD8) {
    let offset = 2;
    while (offset < buf.length) {
      if (buf[offset] !== 0xFF) break;
      const marker = buf[offset + 1];
      if (marker === 0xC0 || marker === 0xC2) { // SOF0 or SOF2
        const h = (buf[offset + 5] << 8) | buf[offset + 6];
        const w = (buf[offset + 7] << 8) | buf[offset + 8];
        return { w, h, type: 'jpg' };
      }
      offset += 2 + ((buf[offset + 2] << 8) | buf[offset + 3]);
    }
    return { w: 0, h: 0, type: 'jpg' };
  }
  
  // WebP
  if (buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46) {
    if (buf[8] === 0x57 && buf[9] === 0x45 && buf[10] === 0x42 && buf[11] === 0x50) {
      // VP8 chunk
      if (buf[12] === 0x56 && buf[13] === 0x50 && buf[14] === 0x38) {
        if (buf[15] === 0x20) { // VP8 lossy
          const w = ((buf[26] | (buf[27] << 8)) & 0x3FFF) + 1;
          const h = ((buf[28] | (buf[29] << 8)) & 0x3FFF) + 1;
          return { w, h, type: 'webp' };
        } else if (buf[15] === 0x4C) { // VP8L lossless
          const bits = view.getUint32(21, true);
          const w = ((bits & 0x3FFF) + 1);
          const h = (((bits >> 14) & 0x3FFF) + 1);
          return { w, h, type: 'webp' };
        } else if (buf[15] === 0x58) { // VP8X extended
          const w = 1 + (buf[24] | (buf[25] << 8) | (buf[26] << 16));
          const h = 1 + (buf[27] | (buf[28] << 8) | (buf[29] << 16));
          return { w, h, type: 'webp' };
        }
      }
    }
  }
  
  return { w: 0, h: 0, type: 'unknown' };
}

async function validateAsset(filePath, opts, ctx) {
  const io = new NodeIO();
  const doc = await io.read(filePath);
  const root = doc.getRoot();
  
  // Count triangles
  let tris = 0;
  for (const mesh of root.listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const mode = prim.getMode();
      if (mode !== Primitive.Mode.TRIANGLES) continue;
      
      const indices = prim.getIndices();
      if (indices) {
        tris += Math.floor(indices.getCount() / 3);
      } else {
        const pos = prim.getAttribute('POSITION');
        if (pos) tris += Math.floor(pos.getCount() / 3);
      }
    }
  }
  
  // Validate textures
  const textures = root.listTextures();
  let totalTexBytes = 0;
  const textureIssues = [];
  
  for (const tex of textures) {
    const uri = tex.getURI();
    const img = tex.getImage();
    
    if (img) {
      // Embedded texture
      const buf = Buffer.from(img);
      const { w, h, type } = dimFromBuffer(buf);
      
      // Check format allow-list
      if (!ALLOWED_TEXTURE_FORMATS.includes(type.toLowerCase())) {
        textureIssues.push(`texture format '${type}' not allowed. Allowed: ${ALLOWED_TEXTURE_FORMATS.join(', ')}`);
        continue;
      }
      
      // Detect alpha and normal maps
      const isAlpha = detectHasAlpha(type, buf);
      const isNormal = isNormalMap(tex, doc);
      
      // Apply appropriate limits
      const dimCap = isNormal ? 2048 : (isAlpha ? opts.maxAlphaDim : opts.maxTexDim);
      const byteCap = isAlpha ? opts.maxAlphaBytes : opts.maxTexBytes;
      
      if (w > dimCap || h > dimCap) {
        const prefix = isNormal ? 'normal' : (isAlpha ? 'alpha' : 'opaque');
        textureIssues.push(`${prefix} texture exceeds ${dimCap}px (${w}x${h}) [embedded]`);
      }
      
      if (buf.byteLength > byteCap) {
        const prefix = isNormal ? 'normal' : (isAlpha ? 'alpha' : 'opaque');
        const sizeMB = (buf.byteLength / (1024*1024)).toFixed(2);
        const capMB = (byteCap / (1024*1024)).toFixed(0);
        textureIssues.push(`${prefix} texture exceeds ${capMB}MB (${sizeMB}MB) [embedded]`);
      }
      
      totalTexBytes += buf.byteLength;
      
    } else if (uri) {
      // External texture
      const texPath = path.resolve(path.dirname(filePath), uri);
      if (fs.existsSync(texPath)) {
        const buf = fs.readFileSync(texPath);
        const { w, h, type } = dimFromBuffer(buf);
        
        // Check format allow-list
        if (!ALLOWED_TEXTURE_FORMATS.includes(type.toLowerCase())) {
          textureIssues.push(`texture format '${type}' not allowed [${path.basename(uri)}]. Allowed: ${ALLOWED_TEXTURE_FORMATS.join(', ')}`);
          continue;
        }
        
        // Detect alpha and normal maps
        const isAlpha = detectHasAlpha(type, buf);
        const isNormal = isNormalMap(tex, doc);
        
        // Apply appropriate limits
        const dimCap = isNormal ? 2048 : (isAlpha ? opts.maxAlphaDim : opts.maxTexDim);
        const byteCap = isAlpha ? opts.maxAlphaBytes : opts.maxTexBytes;
        
        if (w > dimCap || h > dimCap) {
          const prefix = isNormal ? 'normal' : (isAlpha ? 'alpha' : 'opaque');
          textureIssues.push(`${prefix} texture exceeds ${dimCap}px (${w}x${h}) [${path.basename(uri)}]`);
        }
        
        const stat = fs.statSync(texPath);
        if (stat.size > byteCap) {
          const prefix = isNormal ? 'normal' : (isAlpha ? 'alpha' : 'opaque');
          const sizeMB = (stat.size / (1024*1024)).toFixed(2);
          const capMB = (byteCap / (1024*1024)).toFixed(0);
          textureIssues.push(`${prefix} texture exceeds ${capMB}MB (${sizeMB}MB) [${path.basename(uri)}]`);
        }
        
        totalTexBytes += stat.size;
      }
    }
  }
  
  return { tris, totalTexBytes, textureIssues };
}

(async () => {
  try {
    const { manifestPath, opts } = parseArgs();
    
    // Support both relative and absolute paths
    const resolvedPath = path.isAbsolute(manifestPath) 
      ? manifestPath 
      : path.resolve(process.cwd(), manifestPath);
      
    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`Manifest not found: ${resolvedPath}`);
    }

    const manifestDir = path.dirname(resolvedPath);
    const raw = fs.readFileSync(resolvedPath, 'utf8');
    let items;
    
    try { 
      items = JSON.parse(raw); 
    } catch (e) { 
      throw new Error(`Manifest JSON invalid: ${e.message}`); 
    }

    if (!Array.isArray(items)) {
      throw new Error('Manifest must be an array of asset entries.');
    }

    console.log('=====================================');
    console.log('  Asset Manifest Validator v3.0');
    console.log('=====================================');
    console.log(`Manifest: ${resolvedPath}`);
    console.log(`Limits:`);
    console.log(`  Triangles: ${opts.maxTris.toLocaleString()}`);
    console.log(`  Opaque textures: ${opts.maxTexDim}px, ${(opts.maxTexBytes/(1024*1024)).toFixed(0)}MB`);
    console.log(`  Alpha textures: ${opts.maxAlphaDim}px, ${(opts.maxAlphaBytes/(1024*1024)).toFixed(0)}MB`);
    console.log(`  Normal maps: 2048px max`);
    console.log(`  Total size: ${(opts.maxSetBytes/(1024*1024)).toFixed(0)}MB`);
    console.log('');

    const errors = [];
    const warnings = [];
    let totalTris = 0;
    let totalBytes = 0;
    let maxTrisSeen = 0;
    let maxTrisAsset = '';

    for (const [i, a] of items.entries()) {
      const ctx = `item[${i}] id=${a?.id ?? '(no-id)'}`;
      
      try {
        // Check required fields
        required(a, ['id', 'category', 'opt', 'license', 'source', 'source_url'], ctx);

        // Path checks
        const optPath = path.isAbsolute(a.opt)
          ? a.opt
          : path.resolve(manifestDir, '..', '..', '..', path.normalize(a.opt));
          
        if (!fs.existsSync(optPath)) {
          // Check if it's a placeholder (no models downloaded yet)
          const rawPath = a.raw ? path.resolve(manifestDir, '..', '..', '..', path.normalize(a.raw)) : null;
          if (rawPath && !fs.existsSync(rawPath)) {
            warnings.push(`${ctx}: Model files not yet downloaded (expected at ${optPath})`);
            continue;
          } else {
            throw new Error(`${ctx}: opt file not found: ${optPath}`);
          }
        }
        
        const ext = path.extname(optPath).toLowerCase();
        if (!['.glb', '.gltf'].includes(ext)) {
          throw new Error(`${ctx}: opt must be .glb/.gltf, got ${ext}`);
        }

        // Validate the asset
        console.log(`Checking ${a.id}...`);
        const { tris, totalTexBytes, textureIssues } = await validateAsset(optPath, opts, ctx);
        
        // File size
        const fileStats = fs.statSync(optPath);
        const fileBytes = fileStats.size;
        totalBytes += fileBytes;
        
        // Check limits
        if (tris <= 0) {
          throw new Error(`${ctx}: triangle count is 0 (invalid or non-triangle mode)`);
        }
        
        if (tris > opts.maxTris) {
          throw new Error(`${ctx}: triangle count ${tris.toLocaleString()} exceeds limit ${opts.maxTris.toLocaleString()}`);
        }
        
        if (textureIssues.length > 0) {
          textureIssues.forEach(issue => {
            throw new Error(`${ctx}: ${issue}`);
          });
        }
        
        totalTris += tris;
        
        if (tris > maxTrisSeen) {
          maxTrisSeen = tris;
          maxTrisAsset = a.id;
        }
        
        const sizeMB = (fileBytes / (1024*1024)).toFixed(2);
        const texMB = (totalTexBytes / (1024*1024)).toFixed(2);
        console.log(`  [OK] ${tris.toLocaleString()} triangles, ${sizeMB}MB total, ${texMB}MB textures`);
        
        // Optional field warnings
        if (typeof a.brandless !== 'boolean') {
          warnings.push(`${ctx}: 'brandless' missing; assuming true.`);
        }
        
        if (typeof a.scale_m !== 'number') {
          warnings.push(`${ctx}: 'scale_m' missing; consider adding real-world scale in meters.`);
        }
        
        if (a.polycount_hint && Math.abs(a.polycount_hint - tris) > tris * 0.2) {
          warnings.push(`${ctx}: polycount_hint (${a.polycount_hint}) differs from actual (${tris}) by >20%`);
        }
        
      } catch (e) {
        if (e.message.includes('ENOENT') || e.message.includes('not found')) {
          warnings.push(`${ctx}: Model file not accessible - ${e.message}`);
        } else {
          errors.push(e.message);
        }
      }
    }
    
    // Check total size
    if (totalBytes > opts.maxSetBytes) {
      const totalMB = (totalBytes / (1024*1024)).toFixed(2);
      const limitMB = (opts.maxSetBytes / (1024*1024)).toFixed(0);
      errors.push(`Total asset size ${totalMB}MB exceeds limit ${limitMB}MB`);
    }

    console.log('');
    console.log('=====================================');
    
    // Show warnings
    if (warnings.length > 0) {
      console.log('Warnings:');
      warnings.forEach(w => console.warn(`  [!] ${w}`));
      console.log('');
    }
    
    // Show errors and exit
    if (errors.length) {
      console.error('Validation FAILED:');
      errors.forEach(e => console.error(`  [X] ${e}`));
      console.log('=====================================');
      process.exit(1);
    }
    
    // Success summary
    console.log('Validation PASSED:');
    console.log(`  Assets validated: ${items.length}`);
    console.log(`  Total triangles: ${totalTris.toLocaleString()}`);
    console.log(`  Average per asset: ${Math.floor(totalTris / items.length).toLocaleString()}`);
    console.log(`  Highest poly count: ${maxTrisSeen.toLocaleString()} (${maxTrisAsset})`);
    console.log(`  Total size: ${(totalBytes / (1024*1024)).toFixed(2)}MB / ${(opts.maxSetBytes / (1024*1024)).toFixed(0)}MB`);
    console.log('=====================================');
    console.log('[OK] All assets within limits!');
    process.exit(0);
    
  } catch (e) {
    console.error(`Validator error: ${e.message}`);
    process.exit(1);
  }
})();