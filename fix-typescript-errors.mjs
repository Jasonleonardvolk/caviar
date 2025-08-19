// Fix script for all remaining TypeScript errors
// Run this to apply all fixes at once

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const ROOT = 'D:\\Dev\\kha';

// Fix 1: Remove all IOBinding references from onnxWaveOpRunner.ts
function fixOnnxRunner() {
    const filePath = join(ROOT, 'frontend\\lib\\webgpu\\kernels\\onnxWaveOpRunner.ts');
    let content = readFileSync(filePath, 'utf-8');
    
    // Remove IOBinding from interface
    content = content.replace(/ioBinding\?: ort\.IOBinding;/g, '// ioBinding removed - not available in web');
    content = content.replace(/ioBinding\?: IOBinding;/g, '// ioBinding removed - not available in web');
    
    // Change ort.InferenceSession to InferenceSession
    content = content.replace(/ort\.InferenceSession/g, 'InferenceSession');
    content = content.replace(/ort\.Tensor/g, 'Tensor');
    
    // Fix the session.run pattern (remove createIoBinding calls)
    content = content.replace(/const binding = await session\.createIoBinding\(\);[\s\S]*?await session\.runWithBinding\(binding\);/g, 
        'const results = await session.run(feeds);');
    
    writeFileSync(filePath, content);
    console.log('✓ Fixed onnxWaveOpRunner.ts');
}

// Fix 2: Add definite assignment assertions to splitStepOrchestrator.ts
function fixSplitStepOrchestrator() {
    const filePath = join(ROOT, 'frontend\\lib\\webgpu\\kernels\\splitStepOrchestrator.ts');
    let content = readFileSync(filePath, 'utf-8');
    
    // Add ! to all module properties
    content = content.replace(/private fftModule: GPUShaderModule;/g, 'private fftModule!: GPUShaderModule;');
    content = content.replace(/private transposeModule: GPUShaderModule;/g, 'private transposeModule!: GPUShaderModule;');
    content = content.replace(/private phaseModule: GPUShaderModule;/g, 'private phaseModule!: GPUShaderModule;');
    content = content.replace(/private kspaceModule: GPUShaderModule;/g, 'private kspaceModule!: GPUShaderModule;');
    content = content.replace(/private normalizeModule: GPUShaderModule;/g, 'private normalizeModule!: GPUShaderModule;');
    
    // Add ! to pipeline properties
    content = content.replace(/private fftPipeline: GPUComputePipeline;/g, 'private fftPipeline!: GPUComputePipeline;');
    content = content.replace(/private transposePipeline: GPUComputePipeline;/g, 'private transposePipeline!: GPUComputePipeline;');
    content = content.replace(/private phasePipeline: GPUComputePipeline;/g, 'private phasePipeline!: GPUComputePipeline;');
    content = content.replace(/private kspacePipeline: GPUComputePipeline;/g, 'private kspacePipeline!: GPUComputePipeline;');
    content = content.replace(/private normalizePipeline: GPUComputePipeline;/g, 'private normalizePipeline!: GPUComputePipeline;');
    
    // Add ! to buffer properties
    content = content.replace(/private bufferA: GPUBuffer;/g, 'private bufferA!: GPUBuffer;');
    content = content.replace(/private bufferB: GPUBuffer;/g, 'private bufferB!: GPUBuffer;');
    content = content.replace(/private uniformBuffer: GPUBuffer;/g, 'private uniformBuffer!: GPUBuffer;');
    
    // Fix encoder.writeTimestamp to use pass.writeTimestamp
    content = content.replace(/encoder\.writeTimestamp\(/g, 'pass.writeTimestamp(');
    
    writeFileSync(filePath, content);
    console.log('✓ Fixed splitStepOrchestrator.ts');
}

// Fix 3: Fix all buffer write calls
function fixBufferWrites() {
    const files = [
        'frontend\\lib\\holographicEngine.ts',
        'frontend\\lib\\webgpu\\fftCompute.ts',
        'frontend\\lib\\webgpu\\indirect.ts',
        'frontend\\lib\\webgpu\\pipelines\\phaseLUT.ts',
        'frontend\\lib\\webgpu\\quilt\\WebGPUQuiltGenerator.ts',
        'frontend\\lib\\webgpu\\kernels\\schrodingerBenchmark.ts',
        'frontend\\lib\\webgpu\\kernels\\schrodingerEvolution.ts',
        'frontend\\lib\\webgpu\\utils\\bufferHelpers.ts',
        'tori_ui_svelte\\src\\lib\\webgpu\\photoMorphPipeline.ts'
    ];
    
    for (const file of files) {
        const filePath = join(ROOT, file);
        try {
            let content = readFileSync(filePath, 'utf-8');
            
            // Fix Float32Array writeBuffer calls
            content = content.replace(/device\.queue\.writeBuffer\(([^,]+),\s*([^,]+),\s*([a-zA-Z]+Data)\)/g, 
                'device.queue.writeBuffer($1, $2, $3.buffer)');
            
            // Fix Uint8Array writeBuffer calls  
            content = content.replace(/device\.queue\.writeBuffer\(([^,]+),\s*([^,]+),\s*new Uint8Array\(([^)]+)\)\)/g,
                'device.queue.writeBuffer($1, $2, $3.buffer)');
                
            writeFileSync(filePath, content);
            console.log(`✓ Fixed buffer writes in ${file}`);
        } catch (e) {
            console.log(`⚠ Skipped ${file}: ${e.message}`);
        }
    }
}

// Fix 4: Fix compilation info access
function fixCompilationInfo() {
    const filePath = join(ROOT, 'frontend\\lib\\webgpu\\fftCompute.ts');
    let content = readFileSync(filePath, 'utf-8');
    
    // Fix shaderModule.compilationInfo access
    content = content.replace(/const compilationInfo = await shaderModule\.getCompilationInfo\(\);/g,
        'const compilationInfo = \'getCompilationInfo\' in shaderModule ? await (shaderModule as any).getCompilationInfo() : { messages: [] };');
    
    // Fix msg type annotation
    content = content.replace(/compilationInfo\.messages\.filter\(m => m\.type === 'error'\)/g,
        'compilationInfo.messages.filter((m: any) => m.type === \'error\')');
    
    writeFileSync(filePath, content);
    console.log('✓ Fixed compilation info in fftCompute.ts');
}

// Run all fixes
console.log('Starting TypeScript fixes...\n');
fixOnnxRunner();
fixSplitStepOrchestrator();
fixBufferWrites();
fixCompilationInfo();
console.log('\n✅ All fixes applied!');
