/**
 * Generate shader manifest for mobile app
 * Creates a mapping of quality presets to shader bundles
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { gzip } = require('zlib');
const { promisify } = require('util');

const gzipAsync = promisify(gzip);

const SHADER_DIR = path.join(__dirname, '../dist-mobile/shaders');
const MANIFEST_PATH = path.join(__dirname, '../dist-mobile/shader-manifest.json');

const QUALITY_PRESETS = ['battery', 'balanced', 'performance'];

async function generateManifest() {
  const manifest = {
    version: '1.0.0',
    generated: new Date().toISOString(),
    presets: {}
  };

  for (const preset of QUALITY_PRESETS) {
    const presetDir = path.join(SHADER_DIR, 'mobile', preset);
    
    if (!fs.existsSync(presetDir)) {
      console.warn(`Shader directory not found: ${presetDir}`);
      continue;
    }

    const shaders = {};
    const files = fs.readdirSync(presetDir);

    for (const file of files) {
      if (!file.endsWith('.wgsl')) continue;

      const filePath = path.join(presetDir, file);
      const content = fs.readFileSync(filePath, 'utf8');
      
      // Calculate hash
      const hash = crypto.createHash('sha256').update(content).digest('hex');
      
      // Get file size
      const stats = fs.statSync(filePath);
      const originalSize = stats.size;
      
      // Check compressed size
      const compressed = await gzipAsync(content);
      const compressedSize = compressed.length;
      
      // Store metadata
      shaders[path.basename(file, '.wgsl')] = {
        file: `mobile/${preset}/${file}`,
        hash: hash.substring(0, 8),
        size: originalSize,
        compressedSize,
        compression: ((1 - compressedSize / originalSize) * 100).toFixed(1) + '%'
      };
    }

    manifest.presets[preset] = {
      shaderCount: Object.keys(shaders).length,
      totalSize: Object.values(shaders).reduce((sum, s) => sum + s.size, 0),
      totalCompressedSize: Object.values(shaders).reduce((sum, s) => sum + s.compressedSize, 0),
      shaders
    };
  }

  // Add size summary
  const totalSize = Object.values(manifest.presets).reduce(
    (sum, preset) => sum + preset.totalSize, 0
  );
  const totalCompressedSize = Object.values(manifest.presets).reduce(
    (sum, preset) => sum + preset.totalCompressedSize, 0
  );

  manifest.summary = {
    totalShaders: Object.values(manifest.presets).reduce(
      (sum, preset) => sum + preset.shaderCount, 0
    ),
    totalSize,
    totalCompressedSize,
    totalSizeMB: (totalSize / 1024 / 1024).toFixed(2) + ' MB',
    totalCompressedSizeMB: (totalCompressedSize / 1024 / 1024).toFixed(2) + ' MB',
    compressionRatio: ((1 - totalCompressedSize / totalSize) * 100).toFixed(1) + '%'
  };

  // Write manifest
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
  
  console.log('Shader manifest generated:');
  console.log(`- Total shaders: ${manifest.summary.totalShaders}`);
  console.log(`- Original size: ${manifest.summary.totalSizeMB}`);
  console.log(`- Compressed size: ${manifest.summary.totalCompressedSizeMB}`);
  console.log(`- Compression ratio: ${manifest.summary.compressionRatio}`);
  
  // Verify we're under budget for mobile
  const SHADER_BUDGET_MB = 5; // 5MB budget for shaders
  if (totalCompressedSize / 1024 / 1024 > SHADER_BUDGET_MB) {
    console.error(`WARNING: Shaders exceed ${SHADER_BUDGET_MB}MB budget!`);
    process.exit(1);
  }
}

// Generate shader variants if they don't exist
async function generateShaderVariants() {
  const sourceShaderDir = path.join(__dirname, '../../frontend/shaders');
  
  for (const preset of QUALITY_PRESETS) {
    const targetDir = path.join(SHADER_DIR, 'mobile', preset);
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }
    
    // Generate variants for each shader
    const shaderFiles = [
      'wavefieldEncoder.wgsl',
      'propagation.wgsl',
      'multiViewSynthesis.wgsl',
      'lenticularRender.wgsl'
    ];
    
    for (const shaderFile of shaderFiles) {
      const sourcePath = path.join(sourceShaderDir, shaderFile);
      
      if (!fs.existsSync(sourcePath)) {
        console.warn(`Source shader not found: ${sourcePath}`);
        continue;
      }
      
      const content = fs.readFileSync(sourcePath, 'utf8');
      const variant = generateVariant(content, preset);
      
      const targetPath = path.join(targetDir, shaderFile);
      fs.writeFileSync(targetPath, variant);
    }
  }
}

function generateVariant(shaderCode, preset) {
  let code = shaderCode;
  
  // Preset-specific configurations
  const configs = {
    battery: {
      precision: 'f16',
      viewCols: 10,
      viewRows: 8,
      maxTextureSize: 512,
      workgroupSize: 4,
      fftTaps: 2
    },
    balanced: {
      precision: 'f16',
      viewCols: 15,
      viewRows: 10,
      maxTextureSize: 768,
      workgroupSize: 4,
      fftTaps: 4
    },
    performance: {
      precision: 'f32',
      viewCols: 20,
      viewRows: 12,
      maxTextureSize: 1024,
      workgroupSize: 8,
      fftTaps: 4
    }
  };
  
  const config = configs[preset];
  
  // Replace placeholders
  code = code.replace(/override PRECISION_TYPE[^;]*;/g, 
    `alias PrecisionType = ${config.precision};`);
  code = code.replace(/override VIEW_COLS[^;]*;/g, 
    `const VIEW_COLS: u32 = ${config.viewCols}u;`);
  code = code.replace(/override VIEW_ROWS[^;]*;/g, 
    `const VIEW_ROWS: u32 = ${config.viewRows}u;`);
  code = code.replace(/override MAX_TEXTURE_SIZE[^;]*;/g, 
    `const MAX_TEXTURE_SIZE: u32 = ${config.maxTextureSize}u;`);
  code = code.replace(/@workgroup_size\(\d+, \d+\)/g, 
    `@workgroup_size(${config.workgroupSize}, ${config.workgroupSize})`);
  
  // Mobile-specific optimizations
  if (preset === 'battery') {
    // Remove complex calculations for battery saver
    code = code.replace(/\/\/ MOBILE_OPTIMIZE_START[\s\S]*?\/\/ MOBILE_OPTIMIZE_END/g, 
      '// Optimized for battery saver');
  }
  
  // Add mobile markers
  code = `// Auto-generated mobile variant: ${preset}\n// Generated: ${new Date().toISOString()}\n\n` + code;
  
  return code;
}

// Main execution
async function main() {
  console.log('Generating mobile shader variants...');
  await generateShaderVariants();
  
  console.log('Generating shader manifest...');
  await generateManifest();
  
  console.log('Done!');
}

main().catch(console.error);
