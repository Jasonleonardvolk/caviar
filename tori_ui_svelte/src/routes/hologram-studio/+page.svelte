<script lang="ts">
  import { onMount } from 'svelte';
  import HologramRecorder from '$lib/components/HologramRecorder.svelte';
  
  let canvas: HTMLCanvasElement;
  let context: GPUCanvasContext | null = null;
  let device: GPUDevice | null = null;
  
  onMount(async () => {
    // Initialize WebGPU hologram rendering
    if (!navigator.gpu) {
      console.error('WebGPU not supported');
      return;
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.error('No GPU adapter found');
      return;
    }
    
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    
    if (!context) {
      console.error('Failed to get WebGPU context');
      return;
    }
    
    // Configure the canvas context
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });
    
    // Simple render loop for demo (replace with actual hologram rendering)
    function render() {
      if (!context || !device) return;
      
      const commandEncoder = device.createCommandEncoder();
      const textureView = context.getCurrentTexture().createView();
      
      const renderPassDescriptor: GPURenderPassDescriptor = {
        colorAttachments: [
          {
            view: textureView,
            clearValue: { r: 0.1, g: 0.1, b: 0.2, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      };
      
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      // Add your hologram rendering commands here
      passEncoder.end();
      
      device.queue.submit([commandEncoder.finish()]);
      requestAnimationFrame(render);
    }
    
    render();
  });
</script>

<style>
  .studio-container {
    position: relative;
    width: 100vw;
    height: 100vh;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  canvas {
    max-width: 100%;
    max-height: 100%;
    display: block;
  }
  
  .title {
    position: absolute;
    top: 20px;
    left: 20px;
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 600;
  }
</style>

<div class="studio-container">
  <div class="title">iRis Hologram Studio</div>
  <canvas bind:this={canvas} id="holo-canvas" width="1080" height="1920"></canvas>
  <HologramRecorder 
    sourceCanvasId="holo-canvas" 
    hudTheme="auto" 
    hudPos="bc" 
  />
</div>