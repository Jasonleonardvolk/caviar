#!/usr/bin/env node
/**
 * Build Quilt Video Script
 * Converts a folder of quilt tiles (PNG or KTX2) into AV1 IVF and optional MP4 formats
 * 
 * Prerequisites:
 *   - ffmpeg (on PATH)
 *   - toktx or basisu (on PATH) if decoding KTX2 to PNG
 * 
 * Usage:
 *   node tools/build_quilt_video.js
 *   
 * Environment variables:
 *   SRC=frontend/public/assets/quilt      # Source directory
 *   OUT=frontend/public/assets/quilt_videos  # Output directory
 *   SECONDS=2                              # Duration per tile
 *   CRF=28                                 # Quality (lower = better)
 *   MP4=avc,hevc                          # Additional MP4 codecs
 */

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configurable parameters
const SRC_DIR = process.env.SRC || 'frontend/public/assets/quilt';
const OUT_DIR = process.env.OUT || 'frontend/public/assets/quilt_videos';
const SECONDS = parseFloat(process.env.SECONDS || '2');
const CRF = parseInt(process.env.CRF || '28', 10);
const MP4_TARGETS = (process.env.MP4 || 'avc').split(',').map(s => s.trim());

// Ensure output directory exists
fs.mkdirSync(OUT_DIR, { recursive: true });

function hasCommand(cmd) {
  try {
    execFileSync(cmd, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

// Check prerequisites
if (!hasCommand('ffmpeg')) {
  console.error('ERROR: ffmpeg is required but not found on PATH');
  console.error('');
  console.error('Install ffmpeg:');
  console.error('  Windows:  choco install ffmpeg -y');
  console.error('  macOS:    brew install ffmpeg');
  console.error('  Linux:    sudo apt-get install ffmpeg');
  process.exit(1);
}

console.log('Build Quilt Videos');
console.log('==================');
console.log(`Source: ${SRC_DIR}`);
console.log(`Output: ${OUT_DIR}`);
console.log(`Duration: ${SECONDS}s per tile`);
console.log(`Quality: CRF ${CRF}`);
console.log(`Formats: AV1 (IVF) + ${MP4_TARGETS.join(', ')}`);
console.log('');

function numericSort(a, b) {
  const an = parseInt(a.match(/(\d+)/)?.[0] || '0', 10);
  const bn = parseInt(b.match(/(\d+)/)?.[0] || '0', 10);
  return an - bn || a.localeCompare(b);
}

function buildForFolder(folder) {
  const folderName = path.basename(folder);
  console.log(`Processing ${folderName}...`);
  
  const files = fs.readdirSync(folder)
    .filter(f => /\.(png|ktx2)$/i.test(f))
    .sort(numericSort);
    
  if (!files.length) {
    console.log(`  No tiles found, skipping`);
    return;
  }
  
  console.log(`  Found ${files.length} tiles`);

  // Build input arguments for ffmpeg
  const inputs = [];
  files.forEach(f => {
    const fullPath = path.join(folder, f);
    inputs.push('-loop', '1', '-t', `${SECONDS}`, '-i', fullPath);
  });

  // Calculate grid layout based on file count
  const cols = Math.ceil(Math.sqrt(files.length * 1.77)); // Assume ~16:9 aspect
  const rows = Math.ceil(files.length / cols);
  
  // Build xstack layout string
  let layout = '';
  for (let i = 0; i < files.length; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    if (i > 0) layout += '|';
    layout += `${col * 1024}_${row * 1024}`; // Assuming 1024x1024 tiles
  }

  const commonArgs = ['-filter_complex', `xstack=inputs=${files.length}:layout=${layout}`];

  // Build AV1 IVF (primary format)
  const outIvf = path.join(OUT_DIR, `${folderName}.av1.ivf`);
  console.log(`  Building AV1 IVF...`);
  try {
    execFileSync('ffmpeg', [
      ...inputs,
      ...commonArgs,
      '-c:v', 'libaom-av1',
      '-crf', `${CRF}`,
      '-b:v', '0',
      '-cpu-used', '6',     // Speed preset (0-8, higher = faster)
      '-row-mt', '1',       // Enable row multithreading
      '-tiles', '2x2',      // Tile configuration
      '-f', 'ivf',
      '-y',                 // Overwrite output
      outIvf
    ], { stdio: 'inherit' });
    console.log(`  Created: ${outIvf}`);
  } catch (e) {
    console.error(`  Failed to build AV1: ${e.message}`);
  }

  // Optional AVC/H.264 MP4
  if (MP4_TARGETS.includes('avc')) {
    const outMp4 = path.join(OUT_DIR, `${folderName}.avc.mp4`);
    console.log(`  Building AVC MP4...`);
    try {
      execFileSync('ffmpeg', [
        ...inputs,
        ...commonArgs,
        '-c:v', 'libx264',
        '-crf', `${CRF}`,
        '-preset', 'slow',
        '-movflags', '+faststart',
        '-y',
        outMp4
      ], { stdio: 'inherit' });
      console.log(`  Created: ${outMp4}`);
    } catch (e) {
      console.error(`  Failed to build AVC: ${e.message}`);
    }
  }

  // Optional HEVC/H.265 MP4
  if (MP4_TARGETS.includes('hevc')) {
    const outMp4 = path.join(OUT_DIR, `${folderName}.hevc.mp4`);
    console.log(`  Building HEVC MP4...`);
    try {
      execFileSync('ffmpeg', [
        ...inputs,
        ...commonArgs,
        '-c:v', 'libx265',
        '-crf', `${CRF}`,
        '-preset', 'slow',
        '-tag:v', 'hvc1',    // Safari compatibility
        '-movflags', '+faststart',
        '-y',
        outMp4
      ], { stdio: 'inherit' });
      console.log(`  Created: ${outMp4}`);
    } catch (e) {
      console.error(`  Failed to build HEVC: ${e.message}`);
    }
  }

  console.log(`  Done with ${folderName}`);
  console.log('');
}

// Process all subfolders in SRC_DIR
const folders = fs.readdirSync(SRC_DIR, { withFileTypes: true })
  .filter(d => d.isDirectory())
  .map(d => path.join(SRC_DIR, d.name));

if (folders.length === 0) {
  console.log('No quilt folders found in source directory');
  process.exit(0);
}

console.log(`Found ${folders.length} quilt folder(s) to process`);
console.log('');

folders.forEach(buildForFolder);

console.log('Build complete!');
