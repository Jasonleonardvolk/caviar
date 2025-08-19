// copy_shaders_to_public.mjs
// Copies all fixed shaders from lib to public directory

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sourceDir = path.join(__dirname, '..', 'frontend', 'lib', 'webgpu', 'shaders');
const targetDir = path.join(__dirname, '..', 'frontend', 'public', 'hybrid', 'wgsl');

async function copyShaders() {
  try {
    console.log('Copying fixed shaders from lib to public...');
    
    const files = await fs.readdir(sourceDir);
    const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
    
    for (const file of wgslFiles) {
      const sourcePath = path.join(sourceDir, file);
      const targetPath = path.join(targetDir, file);
      
      await fs.copyFile(sourcePath, targetPath);
      console.log(`  Copied: ${file}`);
    }
    
    console.log(`\nSuccessfully copied ${wgslFiles.length} shader files!`);
    console.log('All shaders are now synchronized between lib and public directories.');
    
  } catch (error) {
    console.error('Error copying shaders:', error);
    process.exit(1);
  }
}

copyShaders();
