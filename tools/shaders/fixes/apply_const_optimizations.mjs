// apply_const_optimizations.mjs
// Changes immutable 'let' to 'const' for better compiler optimization

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

const fixes = [
  // File-specific const optimizations from your notes
  {
    file: 'lenticularInterlace.wgsl',
    replacements: [
      { from: 'let subpixel_width = 1.0 / 3.0;', to: 'const subpixel_width = 1.0 / 3.0;' }
    ]
  },
  {
    file: 'propagation.wgsl', 
    replacements: [
      { from: '// const view_angle = 0.1; // radians', to: 'const view_angle = 0.1; // radians' }
    ]
  },
  {
    file: 'velocityField.wgsl',
    replacements: [
      { from: 'const momentum = 0.85;', to: 'const momentum = 0.85;' }, // Already const
      { from: 'const value = 1.0;', to: 'const value = 1.0;' } // Already const
    ]
  }
];

async function applyFixes() {
  console.log('🔧 Applying const optimizations...\n');
  
  for (const fix of fixes) {
    const filePath = path.join(shadersDir, fix.file);
    
    try {
      let content = await fs.readFile(filePath, 'utf8');
      let modified = false;
      
      for (const replacement of fix.replacements) {
        if (content.includes(replacement.from)) {
          content = content.replace(replacement.from, replacement.to);
          console.log(`  ✅ Fixed in ${fix.file}: const optimization`);
          modified = true;
        }
      }
      
      if (modified) {
        await fs.writeFile(filePath, content);
      }
    } catch (err) {
      console.error(`  ❌ Error processing ${fix.file}:`, err.message);
    }
  }
  
  console.log('\n✅ Const optimizations complete!');
}

applyFixes();
