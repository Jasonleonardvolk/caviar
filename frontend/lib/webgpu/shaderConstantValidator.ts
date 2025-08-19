// shaderConstantValidator.ts
// Comprehensive validation utility for shader constant mappings

import { getConstantsForShader, validateShaderMapping, getSupportedShaders, SHADER_CONSTANT_PRESETS } from './shaderConstantManager';

interface ValidationResult {
    shaderName: string;
    success: boolean;
    constants: Record<string, number>;
    errors: string[];
    warnings: string[];
}

/**
 * Validate all shader constant mappings against a test configuration
 */
export async function validateAllShaderConstants(): Promise<ValidationResult[]> {
    const results: ValidationResult[] = [];
    const testConfig = SHADER_CONSTANT_PRESETS.development;
    
    console.group('🔍 Validating Shader Constants');
    
    for (const shaderName of getSupportedShaders()) {
        const result: ValidationResult = {
            shaderName,
            success: true,
            constants: {},
            errors: [],
            warnings: []
        };
        
        try {
            // Validate shader mapping exists
            if (!validateShaderMapping(shaderName)) {
                result.errors.push(`Shader mapping not found for: ${shaderName}`);
                result.success = false;
            }
            
            // Generate constants
            result.constants = getConstantsForShader(shaderName, testConfig);
            
            // Validate constant structure
            for (const [key, value] of Object.entries(result.constants)) {
                const id = parseInt(key);
                
                if (isNaN(id)) {
                    result.errors.push(`Invalid constant ID: ${key} (must be numeric)`);
                    result.success = false;
                }
                
                if (typeof value !== 'number') {
                    result.errors.push(`Invalid constant value type for ID ${id}: ${typeof value}`);
                    result.success = false;
                }
                
                if (id < 0 || id > 15) {
                    result.warnings.push(`Unusual constant ID: ${id} (typically 0-15)`);
                }
            }
            
            // Shader-specific validations
            switch (shaderName) {
                case 'bitReversal':
                    if (Object.keys(result.constants).length !== 2) {
                        result.errors.push(`bitReversal should have exactly 2 constants, got ${Object.keys(result.constants).length}`);
                        result.success = false;
                    }
                    break;
                    
                case 'normalize':
                    if (Object.keys(result.constants).length !== 1) {
                        result.errors.push(`normalize should have exactly 1 constant, got ${Object.keys(result.constants).length}`);
                        result.success = false;
                    }
                    break;
                    
                case 'lenticularInterlace':
                    if (Object.keys(result.constants).length !== 3) {
                        result.errors.push(`lenticularInterlace should have exactly 3 constants, got ${Object.keys(result.constants).length}`);
                        result.success = false;
                    }
                    break;
                    
                case 'butterflyStage':
                case 'fftShift':
                case 'transpose':
                    if (Object.keys(result.constants).length !== 0) {
                        result.warnings.push(`${shaderName} uses hardcoded constants but got ${Object.keys(result.constants).length} overrides`);
                    }
                    break;
            }
            
        } catch (error) {
            result.errors.push(`Exception during validation: ${error}`);
            result.success = false;
        }
        
        results.push(result);
        
        // Log result
        const status = result.success ? '✅' : '❌';
        const constCount = Object.keys(result.constants).length;
        console.log(`${status} ${shaderName}: ${constCount} constants, ${result.errors.length} errors, ${result.warnings.length} warnings`);
        
        if (result.errors.length > 0) {
            console.error(`   Errors:`, result.errors);
        }
        
        if (result.warnings.length > 0) {
            console.warn(`   Warnings:`, result.warnings);
        }
    }
    
    console.groupEnd();
    
    const successCount = results.filter(r => r.success).length;
    const totalCount = results.length;
    
    console.log(`📊 Validation Summary: ${successCount}/${totalCount} shaders passed`);
    
    return results;
}

/**
 * Create a mock WebGPU pipeline to test shader constant application
 */
export async function testPipelineCreation(device: GPUDevice, shaderSource: string, shaderName: string): Promise<boolean> {
    try {
        const testConfig = SHADER_CONSTANT_PRESETS.development;
        const constants = getConstantsForShader(shaderName, testConfig);
        
        console.log(`🧪 Testing pipeline creation for ${shaderName}:`, constants);
        
        const shaderModule = device.createShaderModule({
            label: `Test_${shaderName}`,
            code: shaderSource
        });
        
        // Test compilation info
        const compilationInfo = await shaderModule.getCompilationInfo();
        for (const message of compilationInfo.messages) {
            if (message.type === 'error') {
                console.error(`Shader compilation error in ${shaderName}:`, message.message);
                return false;
            }
        }
        
        // Try to create a compute pipeline
        const pipeline = device.createComputePipeline({
            label: `TestPipeline_${shaderName}`,
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
                constants: Object.keys(constants).length > 0 ? constants : undefined
            }
        });
        
        console.log(`✅ Pipeline created successfully for ${shaderName}`);
        return true;
        
    } catch (error) {
        console.error(`❌ Pipeline creation failed for ${shaderName}:`, error);
        return false;
    }
}

/**
 * Development utility: Run complete validation suite
 */
export async function runValidationSuite(device?: GPUDevice): Promise<void> {
    console.log('🚀 Starting Shader Constant Validation Suite');
    
    // Step 1: Validate mappings
    const validationResults = await validateAllShaderConstants();
    
    const failedShaders = validationResults.filter(r => !r.success);
    if (failedShaders.length > 0) {
        console.error('❌ Validation failed for shaders:', failedShaders.map(s => s.shaderName));
        return;
    }
    
    // Step 2: Test pipeline creation if device available
    if (device) {
        console.log('🔧 Testing WebGPU pipeline creation...');
        
        // Create minimal test shader for each type
        const testShaders = {
            compute: `
                @id(0) override workgroup_size_x: u32 = 256u;
                @id(1) override normalization_mode: u32 = 0u;
                @compute @workgroup_size(workgroup_size_x, 1, 1)
                fn main() {}
            `,
            simpleCompute: `
                @compute @workgroup_size(256, 1, 1)
                fn main() {}
            `
        };
        
        const pipelineTests = [
            { name: 'bitReversal', shader: testShaders.compute },
            { name: 'normalize', shader: testShaders.compute.replace('@id(1) override normalization_mode: u32 = 0u;', '') },
            { name: 'butterflyStage', shader: testShaders.simpleCompute }
        ];
        
        for (const test of pipelineTests) {
            await testPipelineCreation(device, test.shader, test.name);
        }
    }
    
    console.log('✅ Validation suite completed');
}

/**
 * Browser-friendly validation that doesn't require WebGPU device
 */
export function validateInBrowser(): Promise<void> {
    return runValidationSuite();
}
