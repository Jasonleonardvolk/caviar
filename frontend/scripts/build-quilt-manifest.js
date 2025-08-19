/**
 * Build quilt image manifests by scanning public/assets/quilt/**
 * Emits per-quilt index.json and top-level manifest.json
 * 
 * Usage:
 *   node scripts/build-quilt-manifest.js [--root public/assets/quilt]
 */

import fs from 'fs';
import path from 'path';
import { glob } from 'glob';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Parse command line arguments
const argv = new Map(process.argv.slice(2).map((v, i, a) => {
  if (v.startsWith('--')) {
    return [v.replace(/^--/, ''), a[i+1] && !a[i+1].startsWith('--') ? a[i+1] : '1'];
  }
  return null;
}).filter(Boolean));

const ROOT = argv.get('root') || 'public/assets/quilt';

/**
 * Parse grid dimensions from folder name
 */
function parseGridFromName(name) {
  const match = name.match(/(\d+)x(\d+)/i);
  if (match) {
    return [parseInt(match[1], 10), parseInt(match[2], 10)];
  }
  return null;
}

/**
 * Find factor pairs for a number
 */
function factorPairs(n) {
  const pairs = [];
  for (let r = 1; r * r <= n; r++) {
    if (n % r === 0) {
      pairs.push([r, n / r]);
    }
  }
  return pairs.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
}

/**
 * Numeric sort for tile names
 */
function numericSort(a, b) {
  const aDigits = a.match(/(\d+)/g);
  const bDigits = b.match(/(\d+)/g);
  const aNum = aDigits ? parseInt(aDigits[aDigits.length - 1], 10) : -1;
  const bNum = bDigits ? parseInt(bDigits[bDigits.length - 1], 10) : -1;
  return aNum - bNum || a.localeCompare(b);
}

/**
 * Read KTX2 header to get dimensions
 */
function readKTX2Dims(buffer) {
  // KTX2 header structure:
  // bytes 0-11: identifier
  // bytes 12-15: vkFormat
  // bytes 16-19: typeSize
  // bytes 20-23: pixelWidth
  // bytes 24-27: pixelHeight
  if (buffer.byteLength < 28) return null;
  
  const view = new DataView(buffer.buffer, buffer.byteOffset, 28);
  const signature = Array.from(new Uint8Array(buffer.slice(0, 12)));
  const KTX2_MAGIC = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
  
  if (!KTX2_MAGIC.every((v, i) => v === signature[i])) {
    return null;
  }
  
  const width = view.getUint32(20, true);
  const height = view.getUint32(24, true);
  
  return { width, height };
}

/**
 * Get dimensions of first tile
 */
function getFirstDims(file) {
  try {
    const buffer = fs.readFileSync(file);
    const dims = readKTX2Dims(buffer);
    return dims || null;
  } catch {
    return null;
  }
}

/**
 * Guess grid from tile count
 */
function guessGridFromCount(n) {
  const pairs = factorPairs(n);
  
  // Prefer common light-field grids
  const preferred = [[5, 9], [6, 8], [8, 6], [4, 8], [9, 5]];
  for (const p of preferred) {
    if (p[0] * p[1] === n) return p;
  }
  
  // Choose pair closest to 5x9 aspect ratio
  let best = pairs[0];
  let bestScore = Infinity;
  for (const [r, c] of pairs) {
    const score = Math.abs((c / r) - (9 / 5));
    if (score < bestScore) {
      bestScore = score;
      best = [r, c];
    }
  }
  
  return best;
}

/**
 * Build manifest for a quilt directory
 */
function buildQuilt(dir) {
  const name = path.basename(dir);
  
  // Find all tile files
  const patterns = ['*.ktx2', '*.ktx', '*.png', '*.jpg', '*.jpeg'];
  const tiles = [];
  
  for (const pattern of patterns) {
    const files = glob.sync(path.join(dir, pattern).replace(/\\/g, '/'), { nodir: true });
    tiles.push(...files.map(f => path.basename(f)));
  }
  
  if (tiles.length === 0) {
    return null;
  }
  
  // Sort tiles numerically
  tiles.sort(numericSort);
  
  // Determine grid dimensions
  let grid = parseGridFromName(name);
  if (!grid) {
    grid = guessGridFromCount(tiles.length);
  }
  
  // Get dimensions from first tile
  const firstTilePath = path.join(dir, tiles[0]);
  const dims = getFirstDims(firstTilePath) || { width: 0, height: 0 };
  
  // Get tile format
  const tileFormat = path.extname(tiles[0]).replace('.', '');
  
  return {
    name,
    grid,
    views: tiles.length,
    tileFormat,
    width: dims.width,
    height: dims.height,
    tiles
  };
}

/**
 * Main function
 */
function main() {
  const rootPath = path.resolve(ROOT);
  
  if (!fs.existsSync(rootPath)) {
    console.error(`Root directory not found: ${rootPath}`);
    process.exit(1);
  }
  
  // Find all quilt directories
  const quiltDirs = glob.sync(path.join(rootPath, '**/').replace(/\\/g, '/'))
    .filter(d => d !== rootPath + path.sep)
    .filter(d => fs.statSync(d).isDirectory());
  
  const allQuilts = [];
  
  for (const dir of quiltDirs) {
    const quilt = buildQuilt(dir);
    if (!quilt) continue;
    
    // Write per-quilt index.json
    const relativePath = path.relative(rootPath, dir).replace(/\\/g, '/');
    const indexPath = path.join(rootPath, relativePath, 'index.json');
    
    fs.writeFileSync(indexPath, JSON.stringify(quilt, null, 2));
    console.log(`Generated: ${indexPath}`);
    
    // Add to master list
    allQuilts.push({
      ...quilt,
      path: `/assets/quilt/${relativePath}/index.json`
    });
  }
  
  // Write top-level manifest
  const manifestPath = path.join(rootPath, 'manifest.json');
  const manifest = { quilts: allQuilts };
  
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`Generated: ${manifestPath}`);
  console.log(`Total quilts: ${allQuilts.length}`);
}

// Run if executed directly
if (process.argv[1] === __filename) {
  main();
}

export { buildQuilt, parseGridFromName, guessGridFromCount };
