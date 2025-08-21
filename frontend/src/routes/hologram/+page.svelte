<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import HologramRecorder from '$lib/components/HologramRecorder.svelte';
  import HologramSourceSelector from '$lib/components/HologramSourceSelector.svelte';
  import { detectCapabilities, prefersWebGPUHint } from '$lib/device/capabilities';

  let caps: Awaited<ReturnType<typeof detectCapabilities>> | null = null;
  let status = 'init';
  let videoEl: HTMLVideoElement;
  let animationId: number;
  let gl: WebGL2RenderingContext | null = null;
  let resizeHandler: () => void;
  let isFullscreen = false;
  let hudVisible = true;

  function toggleFullscreen() {
    const wrapper = document.getElementById('hologram-wrapper');
    if (!wrapper) return;
    
    if (!document.fullscreenElement) {
      wrapper.requestFullscreen().catch((err) => {
        console.error('Fullscreen error:', err);
      });
    } else {
      document.exitFullscreen();
    }
  }

  function handleFullscreenChange() {
    isFullscreen = !!document.fullscreenElement;
    hudVisible = !isFullscreen; // Auto-hide HUD when entering fullscreen
  }

  function toggleHud() {
    if (isFullscreen) {
      hudVisible = !hudVisible;
    }
  }

  onMount(async () => {
    status = 'probing';
    caps = await detectCapabilities();
    status = prefersWebGPUHint(caps) ? 'webgpu' : (caps.webgl2 ? 'webgl2' : 'cpu');

    // Initialize WebGL2 holographic renderer
    const canvas = document.getElementById('hologram-canvas') as HTMLCanvasElement;
    if (!canvas) return;
    
    gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: true,
      powerPreference: 'high-performance'
    });
    
    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    // Responsive canvas sizing
    function resizeCanvas() {
      const container = canvas.parentElement;
      if (!container || !gl) return;
      
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      
      // Set canvas size with device pixel ratio for sharp rendering
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      
      // Ensure canvas CSS size matches container
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      
      // Update WebGL viewport
      gl.viewport(0, 0, canvas.width, canvas.height);
    }

    // Vertex shader
    const vsSource = `#version 300 es
      in vec2 aPos;
      in vec2 aTex;
      out vec2 vTex;
      void main(void) {
        vTex = aTex;
        gl_Position = vec4(aPos, 0.0, 1.0);
      }
    `;

    // Fragment shader with holographic effects
    const fsSource = `#version 300 es
      precision highp float;
      in vec2 vTex;
      out vec4 fragColor;
      uniform sampler2D uSampler;
      uniform float uTime;
      uniform vec2 uResolution;
      
      void main(void) {
        // Dynamic wave distortion for holographic effect
        float wave = sin(vTex.y * 30.0 + uTime * 2.0) * 0.008;
        float wave2 = cos(vTex.x * 25.0 + uTime * 1.5) * 0.005;
        
        // RGB channel separation for diffraction effect
        vec2 uvR = vec2(vTex.x + wave * 1.2 + wave2, vTex.y + sin(uTime * 0.8) * 0.004);
        vec2 uvG = vec2(vTex.x + wave * 0.7, vTex.y);
        vec2 uvB = vec2(vTex.x + wave * 1.5 - wave2, vTex.y - cos(uTime * 0.6) * 0.004);
        
        // Sample each color channel separately
        float r = texture(uSampler, uvR).r;
        float g = texture(uSampler, uvG).g;
        float b = texture(uSampler, uvB).b;
        
        // Add holographic shimmer
        float shimmer = 0.95 + 0.05 * sin(uTime * 3.0 + vTex.x * 10.0);
        
        // Edge glow effect with resolution awareness
        vec2 center = vTex - 0.5;
        float dist = length(center);
        float edgeGlow = 1.0 - smoothstep(0.3, 0.5, dist);
        vec3 glowColor = vec3(0.4, 0.7, 1.0) * edgeGlow * 0.2;
        
        // Interference pattern
        float interference = sin(dist * 50.0 - uTime * 2.0) * 0.05 + 1.0;
        
        // Combine effects
        vec3 color = vec3(r, g, b) * shimmer * interference + glowColor;
        
        // Output with full opacity
        fragColor = vec4(color, 1.0);
      }
    `;

    // Compile shaders
    function compileShader(source: string, type: number): WebGLShader {
      const shader = gl!.createShader(type)!;
      gl!.shaderSource(shader, source);
      gl!.compileShader(shader);
      
      if (!gl!.getShaderParameter(shader, gl!.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl!.getShaderInfoLog(shader));
      }
      return shader;
    }

    const vertexShader = compileShader(vsSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(fsSource, gl.FRAGMENT_SHADER);

    // Create program
    const program = gl.createProgram()!;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
    }
    
    gl.useProgram(program);

    // Create fullscreen quad
    const vertices = new Float32Array([
      -1, -1, 0, 1,  // bottom left
       1, -1, 1, 1,  // bottom right
      -1,  1, 0, 0,  // top left
       1,  1, 1, 0   // top right
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // Set up attributes
    const aPos = gl.getAttribLocation(program, 'aPos');
    const aTex = gl.getAttribLocation(program, 'aTex');
    
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    
    gl.enableVertexAttribArray(aTex);
    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);

    // Get uniform locations
    const uSampler = gl.getUniformLocation(program, 'uSampler');
    const uTime = gl.getUniformLocation(program, 'uTime');
    const uResolution = gl.getUniformLocation(program, 'uResolution');

    // Create texture for video
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Initial resize
    resizeCanvas();
    
    // Set up resize handler
    resizeHandler = resizeCanvas;
    window.addEventListener('resize', resizeHandler);

    // Render loop
    function render(time: number) {
      if (!gl) return;
      
      // Update video texture if video is ready
      if (videoEl && videoEl.readyState >= 2) {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
      }
      
      // Set uniforms
      gl.uniform1i(uSampler, 0);
      gl.uniform1f(uTime, time * 0.001);
      gl.uniform2f(uResolution, canvas.width, canvas.height);
      
      // Clear and draw
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      
      animationId = requestAnimationFrame(render);
    }
    
    render(0);
    
    // Fullscreen event listeners
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
  });

  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    if (resizeHandler) {
      window.removeEventListener('resize', resizeHandler);
    }
    document.removeEventListener('fullscreenchange', handleFullscreenChange);
    document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
  });
</script>

<style>
  .hologram-wrapper {
    position: relative;
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: radial-gradient(ellipse at center, #0a0a1a 0%, #000000 100%);
    overflow: hidden;
  }
  
  .canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
    max-width: 1920px;
    max-height: 1080px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  #hologram-canvas {
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 16px;
    box-shadow: 
      0 0 60px rgba(74, 199, 255, 0.4),
      0 0 100px rgba(142, 197, 255, 0.2),
      inset 0 0 20px rgba(0, 255, 200, 0.1);
  }
  
  .hud-container {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 100;
  }
  
  .hud-container > :global(*) {
    pointer-events: auto;
  }
  
  .fullscreen-toggle {
    position: absolute;
    top: 1rem;
    left: 1rem;
    padding: 0.5rem 1rem;
    background: rgba(13, 13, 13, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(142, 197, 255, 0.3);
    border-radius: 999px;
    color: #fff;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.3s;
    z-index: 101;
  }
  
  .fullscreen-toggle:hover {
    background: rgba(142, 197, 255, 0.2);
    transform: scale(1.05);
  }
  
  .tap-hint {
    position: absolute;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.5rem 1rem;
    background: rgba(0, 0, 0, 0.7);
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.75rem;
    border-radius: 999px;
    animation: fadeInOut 3s ease-in-out;
    pointer-events: none;
  }
  
  @keyframes fadeInOut {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
  }
  
  /* Fullscreen mode adjustments */
  :global(.hologram-wrapper:fullscreen) {
    background: #000;
  }
  
  :global(.hologram-wrapper:fullscreen) .canvas-container {
    max-width: 100%;
    max-height: 100%;
  }
  
  :global(.hologram-wrapper:fullscreen) #hologram-canvas {
    border-radius: 0;
  }
  
  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .canvas-container {
      max-width: 100%;
      max-height: 100%;
    }
    
    #hologram-canvas {
      border-radius: 0;
    }
  }
  
  /* iPad specific */
  @media only screen and (min-device-width: 768px) and (max-device-width: 1024px) {
    .hologram-wrapper {
      height: 100vh;
      height: -webkit-fill-available;
    }
  }
</style>

<div id="hologram-wrapper" class="hologram-wrapper" on:click={toggleHud}>
  <div class="canvas-container">
    <!-- Main hologram canvas with shader effects -->
    <canvas id="hologram-canvas"></canvas>
    
    <!-- Hidden video element that feeds the canvas -->
    <video bind:this={videoEl} autoplay muted loop playsinline style="display:none"></video>
  </div>
  
  <!-- HUD overlay container -->
  {#if hudVisible}
    <div class="hud-container" transition:fade={{ duration: 300 }}>
      <!-- Source selector HUD (top right) -->
      <HologramSourceSelector {videoEl} />
      
      <!-- Fullscreen toggle (top left) -->
      <button class="fullscreen-toggle" on:click|stopPropagation={toggleFullscreen}>
        {isFullscreen ? '◱ Exit' : '⛶ Fullscreen'}
      </button>
    </div>
  {/if}
  
  <!-- Show hint when in fullscreen and HUD is hidden -->
  {#if isFullscreen && !hudVisible}
    <div class="tap-hint">Tap to show controls</div>
  {/if}
  
  <!-- Recorder stays at bottom -->
  <div style="position: absolute; bottom: 1rem; left: 50%; transform: translateX(-50%); z-index: 90;">
    <HologramRecorder hologramCanvasSelector="#hologram-canvas" />
  </div>
</div>