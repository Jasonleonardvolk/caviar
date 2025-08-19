/**
 * Transcode PNGs to KTX2 using toktx or basisu CLI tools
 * 
 * Usage:
 *   node scripts/transcode-ktx2.js
 * 
 * Environment variables:
 *   SRC_DIR=assets/quilt_src   (source PNG directory)
 *   OUT_DIR=public/assets/quilt (output KTX2 directory)
 *   UASTC=1                     (use UASTC encoding, default on)
 *   MIPMAPS=1                   (generate mipmaps, default on)
 */

import { execFileSync } from 'child_process';
import { glob } from 'glob';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration from environment
const SRC_DIR = process.env.SRC_DIR || 'assets/quilt_src';
const OUT_DIR = process.env.OUT_DIR || 'public/assets/quilt';
const USE_UASTC = process.env.UASTC === '0' ? false : true;
const GEN_MIPS = process.env.MIPMAPS === '0' ? false : true;

/**
 * Check if a command exists
 */
function hasCommand(cmd) {
  try {
    execFileSync(cmd, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

// Check for available tools
const hasToktx = hasCommand('toktx');
const hasBasisu = hasCommand('basisu');

if (!hasToktx && !hasBasisu) {
  console.error('❌ Neither toktx nor basisu found on PATH');
  console.error('Install KTX-Software or Basis Universal:');
  console.error('  Ubuntu: sudo apt-get install ktx-tools');
  console.error('  macOS: brew install ktx');
  console.error('  Windows: Download from https://github.com/KhronosGroup/KTX-Software/releases');
  process.exit(2);
}

console.log('✅ Using tool:', hasToktx ? 'toktx' : 'basisu');

/**
 * Find all PNG files in source directory
 */
function findPNGs() {
  const pattern = path.join(SRC_DIR, '**/*.png').replace(/\\/g, '/');
  return glob.sync(pattern, { nodir: true });
}

/**
 * Transcode a single PNG to KTX2
 */
function transcodePNG(inputPath, outputPath) {
  // Create output directory if needed
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  try {
    if (hasToktx) {
      // Use toktx (KTX-Software)
      const args = [
        '--t2',                           // Output KTX2 format
        '--uastc', USE_UASTC ? '4' : '0', // UASTC quality level
        '--zcmp', '18',                   // Zstd compression level
        ...(GEN_MIPS ? ['--genmipmap'] : []),
        '--assign-oetf', 'srgb',          // sRGB color space
        outputPath,
        inputPath
      ];
      
      execFileSync('toktx', args, { stdio: 'inherit' });
    } else {
      // Use basisu (Binomial)
      const args = [
        '-ktx2',
        USE_UASTC ? '-uastc' : '',
        ...(GEN_MIPS ? ['-mipmap'] : []),
        '-q', '255',                      // Maximum quality
        '-output_file', outputPath,
        inputPath
      ].filter(Boolean);
      
      execFileSync('basisu', args, { stdio: 'inherit' });
    }
    
    return true;
  } catch (error) {
    console.error(`❌ Failed to transcode ${inputPath}:`, error.message);
    return false;
  }
}

/**
 * Main function
 */
function main() {
  const pngs = findPNGs();
  
  if (pngs.length === 0) {
    console.log(`ℹ️ No PNG files found in ${SRC_DIR}`);
    process.exit(0);
  }
  
  console.log(`Found ${pngs.length} PNG files to transcode`);
  
  let successCount = 0;
  let failCount = 0;
  
  for (const png of pngs) {
    // Calculate output path
    const relativePath = path.relative(SRC_DIR, png);
    const outputPath = path.join(OUT_DIR, relativePath).replace(/\.png$/i, '.ktx2');
    
    // Check if output already exists and is newer
    if (fs.existsSync(outputPath)) {
      const inputStats = fs.statSync(png);
      const outputStats = fs.statSync(outputPath);
      
      if (outputStats.mtime > inputStats.mtime) {
        console.log(`⏩ Skipping ${relativePath} (up to date)`);
        successCount++;
        continue;
      }
    }
    
    console.log(`🔄 Transcoding ${relativePath}...`);
    
    if (transcodePNG(png, outputPath)) {
      console.log(`✅ Created ${outputPath}`);
      successCount++;
    } else {
      failCount++;
    }
  }
  
  // Summary
  console.log('\n📊 Transcode Summary:');
  console.log(`  ✅ Success: ${successCount}`);
  console.log(`  ❌ Failed: ${failCount}`);
  console.log(`  📁 Output: ${OUT_DIR}`);
  
  if (failCount > 0) {
    process.exitCode = 1;
  }
}

// Run if executed directly
if (process.argv[1] === __filename) {
  main();
}

export { transcodePNG, findPNGs };
