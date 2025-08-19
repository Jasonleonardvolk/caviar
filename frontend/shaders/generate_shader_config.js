// generate_shader_config.js
// Automation script to generate WGSL structs and TypeScript interfaces from config

const fs = require('fs');
const path = require('path');

// Display presets configuration
const DISPLAY_PRESETS = {
    standard: {
        name: "Looking Glass 15.6\"",
        pitch: 49.55,
        tilt: -0.13509,
        center: 0.0,
        ri: 1.5,
        bi: 2.5,
        dpi: 218,
        numViews: 45,
        viewCone: Math.PI / 6
    },
    portrait: {
        name: "Looking Glass Portrait",
        pitch: 85.3,
        tilt: -0.1859,
        center: 0.0,
        ri: 1.48,
        bi: 3.0,
        dpi: 324,
        numViews: 48,
        viewCone: Math.PI / 4
    },
    gen2_8k: {
        name: "Looking Glass 32\" Gen2",
        pitch: 52.27,
        tilt: -0.11978,
        center: 0.0,
        ri: 1.52,
        bi: 4.0,
        dpi: 280,
        numViews: 45,
        viewCone: Math.PI / 5
    },
    gen2_portrait: {
        name: "Looking Glass Portrait Gen2",
        pitch: 82.5,
        tilt: -0.1765,
        center: 0.0,
        ri: 1.49,
        bi: 3.2,
        dpi: 330,
        numViews: 48,
        viewCone: Math.PI / 4
    }
};

// Uniform structure definitions
const UNIFORM_STRUCTS = {
    CoreUniforms: {
        fields: [
            // Display parameters
            { name: 'screen_width', type: 'f32', precompute: null },
            { name: 'screen_height', type: 'f32', precompute: null },
            { name: 'inv_screen_width', type: 'f32', precompute: '1.0 / screen_width' },
            { name: 'inv_screen_height', type: 'f32', precompute: '1.0 / screen_height' },
            
            // Lens parameters
            { name: 'pitch', type: 'f32', precompute: null },
            { name: 'inv_pitch', type: 'f32', precompute: '1.0 / pitch' },
            { name: 'cos_tilt', type: 'f32', precompute: 'Math.cos(tilt)' },
            { name: 'sin_tilt', type: 'f32', precompute: 'Math.sin(tilt)' },
            { name: 'center', type: 'f32', precompute: null },
            { name: 'subp', type: 'f32', precompute: null },
            { name: 'ri', type: 'f32', precompute: null },
            { name: 'bi', type: 'f32', precompute: null },
            
            // View parameters
            { name: 'num_views', type: 'f32', precompute: null },
            { name: 'inv_num_views', type: 'f32', precompute: '1.0 / num_views' },
            { name: 'view_cone', type: 'f32', precompute: null },
            { name: 'lens_curve', type: 'f32', precompute: null },
            
            // Calibration
            { name: 'gamma', type: 'f32', default: 2.2 },
            { name: 'exposure', type: 'f32', default: 1.0 },
            { name: 'black_level', type: 'f32', default: 0.0 },
            { name: 'white_point', type: 'f32', default: 1.0 },
            
            // Flags
            { name: 'flip_x', type: 'f32', default: 0.0 },
            { name: 'flip_y', type: 'f32', default: 0.0 },
            { name: 'aa_samples', type: 'f32', default: 4.0 },
            { name: 'interpolation_mode', type: 'f32', default: 1.0 }
        ],
        size: 256 // Target size in bytes
    },
    
    QuiltUniforms: {
        fields: [
            { name: 'cols', type: 'f32', precompute: null },
            { name: 'rows', type: 'f32', precompute: null },
            { name: 'inv_cols', type: 'f32', precompute: '1.0 / cols' },
            { name: 'inv_rows', type: 'f32', precompute: '1.0 / rows' },
            { name: 'tile_width', type: 'f32', precompute: null },
            { name: 'tile_height', type: 'f32', precompute: null },
            { name: 'inv_tile_width', type: 'f32', precompute: '1.0 / tile_width' },
            { name: 'inv_tile_height', type: 'f32', precompute: '1.0 / tile_height' },
            { name: 'quilt_width', type: 'f32', precompute: null },
            { name: 'quilt_height', type: 'f32', precompute: null },
            { name: 'inv_quilt_width', type: 'f32', precompute: '1.0 / quilt_width' },
            { name: 'inv_quilt_height', type: 'f32', precompute: '1.0 / quilt_height' },
            { name: 'total_views', type: 'f32', precompute: null },
            { name: 'edge_enhancement', type: 'f32', default: 0.05 },
            { name: 'temporal_blend', type: 'f32', default: 0.1 },
            { name: 'display_type', type: 'f32', default: 0.0 }
        ],
        size: 128
    }
};

// Generate WGSL struct definitions
function generateWGSLStructs() {
    let wgsl = `// Auto-generated shader structures
// Generated on: ${new Date().toISOString()}

`;
    
    for (const [structName, structDef] of Object.entries(UNIFORM_STRUCTS)) {
        wgsl += `struct ${structName} {\n`;
        
        let currentSize = 0;
        for (const field of structDef.fields) {
            wgsl += `    ${field.name}: ${field.type},\n`;
            currentSize += 4; // f32 = 4 bytes
        }
        
        // Add padding if needed
        const targetSize = structDef.size;
        if (targetSize && currentSize < targetSize) {
            const paddingFloats = (targetSize - currentSize) / 4;
            wgsl += `    _padding: array<f32, ${paddingFloats}>\n`;
        }
        
        wgsl += `}\n\n`;
    }
    
    // Add display preset switch function
    wgsl += `fn get_display_preset(display_type: u32) -> CoreUniforms {
    var params: CoreUniforms;
    
    switch (display_type) {\n`;
    
    let presetIndex = 0;
    for (const [key, preset] of Object.entries(DISPLAY_PRESETS)) {
        wgsl += `        case ${presetIndex}u: { // ${preset.name}
            params.pitch = ${preset.pitch};
            params.cos_tilt = ${Math.cos(preset.tilt)};
            params.sin_tilt = ${Math.sin(preset.tilt)};
            params.center = ${preset.center};
            params.ri = ${preset.ri};
            params.bi = ${preset.bi};
            params.num_views = ${preset.numViews}.0;
            params.inv_num_views = ${1.0 / preset.numViews};
            params.view_cone = ${preset.viewCone};
        }\n`;
        presetIndex++;
    }
    
    wgsl += `        default: { /* Use passed uniforms */ }
    }
    
    return params;
}\n`;
    
    return wgsl;
}

// Generate TypeScript interfaces
function generateTypeScriptInterfaces() {
    let ts = `// Auto-generated TypeScript interfaces for shader uniforms
// Generated on: ${new Date().toISOString()}

export interface DisplayPreset {
    name: string;
    pitch: number;
    tilt: number;
    center: number;
    ri: number;
    bi: number;
    dpi: number;
    numViews: number;
    viewCone: number;
}

export const DISPLAY_PRESETS: Record<string, DisplayPreset> = ${JSON.stringify(DISPLAY_PRESETS, null, 4)};

`;
    
    // Generate interfaces for uniform structs
    for (const [structName, structDef] of Object.entries(UNIFORM_STRUCTS)) {
        ts += `export interface ${structName} {\n`;
        
        for (const field of structDef.fields) {
            const tsType = field.type === 'f32' ? 'number' : field.type;
            const optional = field.default !== undefined ? '?' : '';
            ts += `    ${field.name}${optional}: ${tsType};\n`;
        }
        
        ts += `}\n\n`;
    }
    
    // Generate uniform builder functions
    ts += `// Uniform builder functions with precomputation\n\n`;
    
    for (const [structName, structDef] of Object.entries(UNIFORM_STRUCTS)) {
        ts += `export function build${structName}(input: Partial<${structName}>): ArrayBuffer {
    const buffer = new ArrayBuffer(${structDef.size || 256});
    const view = new Float32Array(buffer);
    let offset = 0;
    
`;
        
        for (const field of structDef.fields) {
            if (field.precompute) {
                // Generate precomputation code
                const deps = field.precompute.match(/\b\w+\b/g)
                    .filter(w => !['Math', '1.0'].includes(w));
                ts += `    // Precompute ${field.name}\n`;
                ts += `    const ${field.name} = ${field.precompute.replace(/\b(\w+)\b/g, (match) => {
                    return ['Math', '1.0', field.name].includes(match) ? match : `input.${match}`;
                })};\n`;
                ts += `    view[offset++] = ${field.name};\n\n`;
            } else {
                const defaultValue = field.default !== undefined ? field.default : '0';
                ts += `    view[offset++] = input.${field.name} ?? ${defaultValue};\n`;
            }
        }
        
        ts += `    return buffer;
}\n\n`;
    }
    
    return ts;
}

// Generate validation tests
function generateValidationTests() {
    let test = `// Auto-generated validation tests
import { expect, test } from '@jest/globals';
import { build${Object.keys(UNIFORM_STRUCTS).join(', build')} } from './shaderUniforms';

`;
    
    for (const [structName, structDef] of Object.entries(UNIFORM_STRUCTS)) {
        test += `test('${structName} buffer size', () => {
    const buffer = build${structName}({});
    expect(buffer.byteLength).toBe(${structDef.size || 256});
});

test('${structName} precomputation', () => {
    const input = {
`;
        
        // Add test values
        for (const field of structDef.fields) {
            if (!field.precompute && field.name !== '_padding') {
                test += `        ${field.name}: ${field.name.includes('inv') ? '2.0' : '1.0'},\n`;
            }
        }
        
        test += `    };
    const buffer = build${structName}(input);
    const view = new Float32Array(buffer);
    
`;
        
        // Add assertions for precomputed values
        let offset = 0;
        for (const field of structDef.fields) {
            if (field.precompute) {
                test += `    // Check ${field.name}\n`;
                test += `    expect(view[${offset}]).toBeCloseTo(${field.precompute.includes('1.0 /') ? '0.5' : '1.0'});\n`;
            }
            offset++;
        }
        
        test += `});\n\n`;
    }
    
    return test;
}

// Main generation function
function generateAll() {
    const outputDir = path.join(__dirname, 'generated');
    
    // Create output directory
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Generate and write files
    const wgslStructs = generateWGSLStructs();
    fs.writeFileSync(path.join(outputDir, 'shaderStructs.wgsl'), wgslStructs);
    console.log('Generated shaderStructs.wgsl');
    
    const tsInterfaces = generateTypeScriptInterfaces();
    fs.writeFileSync(path.join(outputDir, 'shaderUniforms.ts'), tsInterfaces);
    console.log('Generated shaderUniforms.ts');
    
    const tests = generateValidationTests();
    fs.writeFileSync(path.join(outputDir, 'shaderUniforms.test.ts'), tests);
    console.log('Generated shaderUniforms.test.ts');
    
    // Generate include snippet for WGSL
    const includeSnippet = `// Add this to your main shader file:
// #include "generated/shaderStructs.wgsl"`;
    fs.writeFileSync(path.join(outputDir, 'README.md'), includeSnippet);
}

// Run generation
if (require.main === module) {
    generateAll();
}

module.exports = { generateAll, DISPLAY_PRESETS };
