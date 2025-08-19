// copy_canonical_to_public.mjs
// Copies canonical shader files from lib/webgpu/shaders to public/hybrid/wgsl
// Maintains single source of truth for all shaders

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CANONICAL_DIR = path.join(__dirname, '../../frontend/lib/webgpu/shaders');
const PUBLIC_DIR = path.join(__dirname, '../../frontend/public/hybrid/wgsl');

async function ensureDir(dir) {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (err) {
    // Directory exists
  }
}

async function copyShaders() {
  console.log('[copy] Starting canonical → public sync...');
  
  await ensureDir(PUBLIC_DIR);
  
  const files = await fs.readdir(CANONICAL_DIR);
  const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
  
  for (const file of wgslFiles) {
    const src = path.join(CANONICAL_DIR, file);
    const dest = path.join(PUBLIC_DIR, file);
    
    await fs.copyFile(src, dest);
    console.log(`[copy] ${file}`);
  }
  
  console.log(`[copy] Done → ${PUBLIC_DIR}`);
  return wgslFiles.length;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  copyShaders().catch(console.error);
}

export { copyShaders };
