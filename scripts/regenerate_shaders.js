// Quick script to update the shader bundle with normalization_mode constants
const fs = require('fs');
const path = require('path');

const shaderPath = path.join(__dirname, '../frontend/lib/webgpu/generated/shaderSources.ts');
let content = fs.readFileSync(shaderPath, 'utf8');

// Update bitReversal shader
content = content.replace(
  /\\"\/\/ Specialization constant for workgroup size\\\\n.*?override workgroup_size_x: u32 = 256u;/g,
  '\\"// Specialization constants\\\\noverride workgroup_size_x: u32 = 256u;\\\\noverride normalization_mode: u32 = 0u;  // 0=dynamic, 1=1/N, 2=1/sqrt(N), 3=none'
);

// Update butterflyStage shader
content = content.replace(
  /butterflyStage\.wgsl.*?\\"\/\/ Specialization constant for workgroup size\\\\n.*?override workgroup_size_x: u32 = 256u;/g,
  function(match) {
    return match.replace(
      '\\"// Specialization constant for workgroup size\\\\noverride workgroup_size_x: u32 = 256u;',
      '\\"// Specialization constants\\\\noverride workgroup_size_x: u32 = 256u;\\\\noverride normalization_mode: u32 = 0u;  // 0=dynamic, 1=1/N, 2=1/sqrt(N), 3=none'
    );
  }
);

// Update fftShift shader
content = content.replace(
  /fftShift\.wgsl.*?\\"\/\/ Specialization constant for workgroup size\\\\n.*?override workgroup_size_x: u32 = 256u;/g,
  function(match) {
    return match.replace(
      '\\"// Specialization constant for workgroup size\\\\noverride workgroup_size_x: u32 = 256u;',
      '\\"// Specialization constants\\\\noverride workgroup_size_x: u32 = 256u;\\\\noverride normalization_mode: u32 = 0u;  // 0=dynamic, 1=1/N, 2=1/sqrt(N), 3=none'
    );
  }
);

fs.writeFileSync(shaderPath, content);
console.log('Shader bundle updated with normalization_mode constants');
