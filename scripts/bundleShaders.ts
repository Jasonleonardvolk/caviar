// scripts/bundleShaders.ts
// Enhanced shader bundler with validation - FIXED PATH RESOLUTION
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

// Get directory of this script for proper path resolution
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface ShaderValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface ShaderInfo {
  name: string;
  path: string;
  content: string;
  size: number;
  valid: boolean;
  errors?: string[];
}

// Configuration - FIXED to use proper relative paths
const SHADER_DIR = path.resolve(__dirname, '../frontend/lib/webgpu/shaders');
const OUTPUT_FILE = path.resolve(__dirname, '../frontend/lib/webgpu/generated/shaderSources.ts');
const VALIDATE_WITH_NAGA = process.env.VALIDATE_SHADERS === 'true';

console.log('📁 Shader directory:', SHADER_DIR);
console.log('📄 Output file:', OUTPUT_FILE);

// Shader validation
function validateShader(content: string, filename: string): ShaderValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check for JavaScript object syntax (common error)
  if (content.includes('path:') && content.includes('content:')) {
    errors.push('Contains JavaScript object syntax instead of WGSL');
  }

  // Check for shader entry point
  if (!content.match(/@(compute|vertex|fragment)/)) {
    errors.push('Missing shader entry point (@compute, @vertex, or @fragment)');
  }

  // Check for minimum content
  if (content.trim().length < 50) {
    errors.push('Shader content too small to be valid');
  }

  // Check for struct definitions
  if (!content.includes('struct')) {
    warnings.push('No struct definitions found');
  }

  // Check for proper WGSL syntax patterns
  const wgslPatterns = {
    bindings: /@(group|binding)\s*\(\s*\d+\s*\)/,
    functions: /fn\s+\w+\s*\(/,
    types: /:\s*(f32|u32|i32|vec[234]<[fui]32>|mat[234]x[234]<f32>)/
  };

  if (!wgslPatterns.bindings.test(content)) {
    warnings.push('No resource bindings found');
  }

  if (!wgslPatterns.functions.test(content)) {
    errors.push('No functions defined');
  }

  if (!wgslPatterns.types.test(content)) {
    warnings.push('No typed declarations found');
  }

  // Check for common WGSL errors
  const commonErrors = [
    { pattern: /override\s+\w+\s*=/, message: 'Override without type specification' },
    { pattern: /var<\s*>/, message: 'Empty var declaration' },
    { pattern: /array<[^>]*,\s*0\s*>/, message: 'Zero-sized array' }
  ];

  for (const { pattern, message } of commonErrors) {
    if (pattern.test(content)) {
      errors.push(message);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

// Validate with external tool if available
async function validateWithNaga(filepath: string): Promise<string[]> {
  if (!VALIDATE_WITH_NAGA) return [];

  try {
    execSync(`naga ${filepath}`, { encoding: 'utf8' });
    return [];
  } catch (error: any) {
    const output = error.stdout || error.message;
    return output.split('\n').filter((line: string) => line.trim());
  }
}

// Main bundling function
async function bundleShaders() {
  console.log('🔧 Shader Bundler with Validation\n');

  // Check if shader directory exists
  if (!fs.existsSync(SHADER_DIR)) {
    console.error(`❌ Shader directory not found: ${SHADER_DIR}`);
    console.error('   Make sure you are in the correct project directory (kha)');
    console.error('   Or run: .\\Setup-WebGPU-Shaders.ps1 first');
    process.exit(1);
  }

  // Ensure output directory exists
  const outputDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`📁 Created output directory: ${outputDir}`);
  }

  // Find all shader files
  const shaderFiles = fs.readdirSync(SHADER_DIR)
    .filter(file => file.endsWith('.wgsl'))
    .sort();

  if (shaderFiles.length === 0) {
    console.error('❌ No shader files found in', SHADER_DIR);
    process.exit(1);
  }

  console.log(`Found ${shaderFiles.length} shader files in ${SHADER_DIR}\n`);

  // Process each shader
  const shaders: ShaderInfo[] = [];
  let hasErrors = false;

  for (const file of shaderFiles) {
    const filepath = path.join(SHADER_DIR, file);
    const content = fs.readFileSync(filepath, 'utf8');
    const validation = validateShader(content, file);
    
    // External validation if enabled
    const externalErrors = await validateWithNaga(filepath);
    if (externalErrors.length > 0) {
      validation.errors.push(...externalErrors);
      validation.valid = false;
    }

    const shaderInfo: ShaderInfo = {
      name: file.replace(/[.-]/g, '_').replace(/\.wgsl$/, ''),
      path: filepath,
      content,
      size: content.length,
      valid: validation.valid,
      errors: validation.errors.length > 0 ? validation.errors : undefined
    };

    shaders.push(shaderInfo);

    // Report status
    const status = validation.valid ? '✅' : '❌';
    console.log(`${status} ${file} (${shaderInfo.size} bytes)`);
    
    if (validation.errors.length > 0) {
      hasErrors = true;
      validation.errors.forEach(err => console.log(`   ❌ ${err}`));
    }
    
    if (validation.warnings.length > 0) {
      validation.warnings.forEach(warn => console.log(`   ⚠️  ${warn}`));
    }
  }

  // Decide whether to continue
  if (hasErrors && process.env.STRICT_VALIDATION === 'true') {
    console.error('\n❌ Validation errors found. Aborting bundle generation.');
    process.exit(1);
  } else if (hasErrors) {
    console.warn('\n⚠️  Validation errors found. Generating bundle anyway...');
  }

  // Generate TypeScript output
  const timestamp = new Date().toISOString();
  const validShaders = shaders.filter(s => s.valid);
  const invalidShaders = shaders.filter(s => !s.valid);

  let output = `// Auto-generated by bundleShaders.ts
// Generated: ${timestamp}
// Valid shaders: ${validShaders.length}/${shaders.length}

`;

  // Add validation report as comment
  if (invalidShaders.length > 0) {
    output += `/* ⚠️ VALIDATION ERRORS:
${invalidShaders.map(s => `${s.name}: ${s.errors?.join(', ')}`).join('\n')}
*/

`;
  }

  // Export individual shaders
  for (const shader of shaders) {
    if (shader.valid) {
      output += `export const ${shader.name} = ${JSON.stringify(shader.content)};\n\n`;
    } else {
      output += `// ❌ ${shader.name} - INVALID: ${shader.errors?.join(', ')}\n`;
      output += `export const ${shader.name} = "";\n\n`;
    }
  }

  // Export as object
  output += `// Shader map object
export const shaderSources = {
${validShaders.map(s => `  ${s.name}`).join(',\n')}
};

// Export metadata
export const shaderMetadata = {
  generated: "${timestamp}",
  totalShaders: ${shaders.length},
  validShaders: ${validShaders.length},
  shaderDir: "${SHADER_DIR.replace(/\\/g, '\\\\')}",
  shaders: {
${shaders.map(s => `    "${s.name}": {
      valid: ${s.valid},
      size: ${s.size}${s.errors ? `,
      errors: ${JSON.stringify(s.errors)}` : ''}
    }`).join(',\n')}
  }
};

// Type exports for better TypeScript support
export type ShaderName = keyof typeof shaderSources;
export type ShaderMap = typeof shaderSources;

// Helper function to get shader with validation
export function getShader(name: ShaderName): string {
  const shader = shaderSources[name];
  if (!shader) {
    throw new Error(\`Shader "\${name}" not found in bundle\`);
  }
  if (shader === "") {
    throw new Error(\`Shader "\${name}" failed validation and is not available\`);
  }
  return shader;
}

export default shaderSources;
`;

  // Write output file
  fs.writeFileSync(OUTPUT_FILE, output);

  // Final report
  console.log(`\n📦 Bundle generated: ${OUTPUT_FILE}`);
  console.log(`   Total size: ${Math.round(output.length / 1024)} KB`);
  console.log(`   Valid shaders: ${validShaders.length}`);
  if (invalidShaders.length > 0) {
    console.log(`   Invalid shaders: ${invalidShaders.length}`);
  }

  // Create validation report file if there were errors
  if (invalidShaders.length > 0) {
    const reportPath = path.join(outputDir, 'shader-validation-report.json');
    const report = {
      timestamp,
      totalShaders: shaders.length,
      validShaders: validShaders.length,
      invalidShaders: invalidShaders.length,
      errors: invalidShaders.map(s => ({
        shader: s.name,
        path: s.path,
        errors: s.errors
      }))
    };
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Validation report: ${reportPath}`);
  }

  // Exit with appropriate code
  process.exit(hasErrors ? 1 : 0);
}

// Run bundler
bundleShaders().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
