// ShaderConstantManager.ts
// Centralized shader constant management with per-shader validation

export interface ShaderConstantConfig {
    workgroupSizeX: number;
    normalizationMode: number;
    enableDebug: boolean;
    enableTemporal: boolean;
    maxAASamples: number;
    hologramSize: number;
}

/**
 * Maps shader names to their exact @id override requirements
 * Based on complete shader audit - DO NOT MODIFY without updating shaders
 */
const SHADER_CONSTANT_MAPPINGS: Record<string, Record<number, string>> = {
    // FFT shaders
    'bitReversal': {
        0: 'workgroupSizeX'     // @id(0) override workgroup_size_x: u32 = 256u;
        // Note: bitReversal.wgsl only has @id(0), no @id(1)
    },
    
    'normalize': {
        0: 'workgroupSizeX'     // @id(0) override workgroup_size_x: u32 = 256u;
        // Note: No @id(1) - only has workgroup_size_x override
    },
    
    // Lenticular display shader
    'lenticularInterlace': {
        0: 'enableDebug',       // @id(0) override ENABLE_DEBUG: bool = false;
        1: 'enableTemporal',    // @id(1) override ENABLE_TEMPORAL: bool = true;
        2: 'maxAASamples'       // @id(2) override MAX_AA_SAMPLES: u32 = 4u;
    },
    
    // Wavefield encoder
    'wavefieldEncoder': {
        0: 'hologramSize'       // @id(0) override HOLOGRAM_SIZE: u32 = 1024u;
    },
    'wavefieldEncoder_optimized': {
        0: 'hologramSize'       // override HOLOGRAM_SIZE: u32 = 1024u;
    },
    
    // Shaders with no override constants (use const or no constants)
    'butterflyStage': {},         // const workgroup_size_x: u32 = 256u;
    'fftShift': {},              // const workgroup_size_x: u32 = 256u;
    'transpose': {},             // const TILE_SIZE: u32 = 16u;
    'multiViewSynthesis': {},    // const WORKGROUP_SIZE: u32 = 16u;  
    'propagation': {},           // const WORKGROUP_SIZE: u32 = 16u;
    'velocityField': {},         // No workgroup size constants
    'avatarShader': {}           // No constants found
};

/**
 * Generate exact constant map for a specific shader
 * Only includes constants that the shader actually declares
 */
export function getConstantsForShader(
    shaderName: string, 
    config: ShaderConstantConfig
): Record<string, number> {
    const mapping = SHADER_CONSTANT_MAPPINGS[shaderName];
    
    if (!mapping) {
        console.warn(`Warning: Unknown shader "${shaderName}" - no constants applied`);
        return {};
    }
    
    const constants: Record<string, number> = {};
    
    for (const [id, configKey] of Object.entries(mapping)) {
        const numericId = parseInt(id);
        
        switch (configKey) {
            case 'workgroupSizeX':
                constants[numericId] = config.workgroupSizeX;
                break;
            case 'normalizationMode':
                constants[numericId] = config.normalizationMode;
                break;
            case 'enableDebug':
                constants[numericId] = config.enableDebug ? 1 : 0;
                break;
            case 'enableTemporal':
                constants[numericId] = config.enableTemporal ? 1 : 0;
                break;
            case 'maxAASamples':
                constants[numericId] = config.maxAASamples;
                break;
            case 'hologramSize':
                constants[numericId] = config.hologramSize;
                break;
            default:
                console.error(`Unknown config key: ${configKey} for shader ${shaderName}`);
        }
    }
    
    // Validation: ensure we're not supplying more constants than declared
    const declaredCount = Object.keys(mapping).length;
    const suppliedCount = Object.keys(constants).length;
    
    if (suppliedCount !== declaredCount) {
        console.error(`Constant count mismatch for ${shaderName}:`);
        console.error(`   Declared: ${declaredCount}, Supplied: ${suppliedCount}`);
        console.error(`   Mapping:`, mapping);
        console.error(`   Constants:`, constants);
    }
    
    if (Object.keys(constants).length > 0) {
        console.log(`Generated constants for ${shaderName}:`, constants);
    }
    
    return constants;
}

/**
 * Validate that a shader name is known and has correct mapping
 */
export function validateShaderMapping(shaderName: string): boolean {
    return shaderName in SHADER_CONSTANT_MAPPINGS;
}

/**
 * Get all supported shader names
 */
export function getSupportedShaders(): string[] {
    return Object.keys(SHADER_CONSTANT_MAPPINGS);
}

/**
 * Development helper: log all shader mappings
 */
export function logShaderMappings(): void {
    console.group('Shader Constant Mappings');
    for (const [shader, mapping] of Object.entries(SHADER_CONSTANT_MAPPINGS)) {
        if (Object.keys(mapping).length > 0) {
            console.log(`${shader}:`, mapping);
        } else {
            console.log(`${shader}: (no override constants)`);
        }
    }
    console.groupEnd();
}

// Type-safe configuration presets
export const SHADER_CONSTANT_PRESETS = {
    development: {
        workgroupSizeX: 256,
        normalizationMode: 0,  // Dynamic normalization
        enableDebug: true,
        enableTemporal: true,
        maxAASamples: 4,
        hologramSize: 1024
    } as ShaderConstantConfig,
    
    production: {
        workgroupSizeX: 256,
        normalizationMode: 1,  // 1/N normalization
        enableDebug: false,
        enableTemporal: true,
        maxAASamples: 4,
        hologramSize: 1024
    } as ShaderConstantConfig,
    
    performance: {
        workgroupSizeX: 128,
        normalizationMode: 3,  // No normalization
        enableDebug: false,
        enableTemporal: false,
        maxAASamples: 1,
        hologramSize: 512
    } as ShaderConstantConfig
};
