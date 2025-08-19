/**
 * Emit Device Limits
 * Queries WebGPU adapter limits at runtime and writes them to a JSON file
 * 
 * Usage: Run this in a browser console or as part of your test suite
 */

export async function emitDeviceLimits(deviceName = 'current') {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No WebGPU adapter found');
  }
  
  const info = await adapter.requestAdapterInfo();
  const limits = adapter.limits;
  
  // Extract relevant limits for shader validation
  const deviceLimits = {
    // Device info
    vendor: info.vendor || 'unknown',
    architecture: info.architecture || 'unknown',
    device: info.device || 'unknown',
    description: info.description || 'unknown',
    
    // Compute limits
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    
    // Memory limits
    maxWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize || 16384,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
    
    // Texture limits
    maxTextureDimension1D: limits.maxTextureDimension1D,
    maxTextureDimension2D: limits.maxTextureDimension2D,
    maxTextureDimension3D: limits.maxTextureDimension3D,
    maxTextureArrayLayers: limits.maxTextureArrayLayers,
    
    // Binding limits
    maxBindGroups: limits.maxBindGroups,
    maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
    maxDynamicUniformBuffersPerPipelineLayout: limits.maxDynamicUniformBuffersPerPipelineLayout,
    maxDynamicStorageBuffersPerPipelineLayout: limits.maxDynamicStorageBuffersPerPipelineLayout,
    maxSampledTexturesPerShaderStage: limits.maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: limits.maxSamplersPerShaderStage,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
    maxUniformBuffersPerShaderStage: limits.maxUniformBuffersPerShaderStage,
    
    // Vertex limits
    maxVertexAttributes: limits.maxVertexAttributes,
    maxVertexBufferArrayStride: limits.maxVertexBufferArrayStride,
    maxInterStageShaderComponents: limits.maxInterStageShaderComponents,
    maxInterStageShaderVariables: limits.maxInterStageShaderVariables,
    
    // Features (optional, for reference)
    features: Array.from(adapter.features || [])
  };
  
  // Generate filename
  const sanitizedName = deviceName.toLowerCase().replace(/[^a-z0-9]/g, '-');
  const filename = `device_limits.${sanitizedName}.json`;
  
  // Create download link (browser environment)
  if (typeof document !== 'undefined') {
    const blob = new Blob([JSON.stringify(deviceLimits, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log(`✅ Device limits saved to ${filename}`);
    console.log('📋 Copy this file to tools/shaders/ directory');
  }
  
  return deviceLimits;
}

// Node.js script version
if (typeof module !== 'undefined' && module.exports) {
  const fs = require('fs');
  const path = require('path');
  const puppeteer = require('puppeteer');
  
  async function emitDeviceLimitsNode(deviceName = 'current') {
    const browser = await puppeteer.launch({
      headless: false,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=vulkan'
      ]
    });
    
    const page = await browser.newPage();
    
    // Inject the function and run it
    const limits = await page.evaluate(async () => {
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
      }
      
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No adapter found');
      }
      
      const info = await adapter.requestAdapterInfo();
      return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        limits: Object.fromEntries(
          Object.entries(adapter.limits).map(([key, value]) => [key, value])
        ),
        features: Array.from(adapter.features)
      };
    });
    
    await browser.close();
    
    // Format for our validator
    const deviceLimits = {
      vendor: limits.vendor,
      architecture: limits.architecture,
      device: limits.device,
      description: limits.description,
      ...limits.limits,
      features: limits.features
    };
    
    // Save to file
    const sanitizedName = deviceName.toLowerCase().replace(/[^a-z0-9]/g, '-');
    const filename = `device_limits.${sanitizedName}.json`;
    const filepath = path.join(__dirname, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(deviceLimits, null, 2));
    console.log(`✅ Device limits saved to ${filepath}`);
    
    return deviceLimits;
  }
  
  // CLI execution
  if (require.main === module) {
    const deviceName = process.argv[2] || 'current';
    emitDeviceLimitsNode(deviceName).catch(console.error);
  }
  
  module.exports = { emitDeviceLimitsNode };
}
