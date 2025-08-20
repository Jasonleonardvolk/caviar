<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import HologramRecorder from '$lib/components/HologramRecorder.svelte';
  import HologramSourceSelector from '$lib/components/HologramSourceSelector.svelte';
  import { detectCapabilities, prefersWebGPUHint } from '$lib/device/capabilities';

  let caps: Awaited<ReturnType<typeof detectCapabilities>> | null = null;
  let status = 'init';
  let videoEl: HTMLVideoElement;
  let animationId: number;
  let gl: WebGL2RenderingContext | null = null;

  onMount(async () => {
    status = 'probing';
    caps = await detectCapabilities();
    status = prefersWebGPUHint(caps) ? 'webgpu' : (caps.webgl2 ? 'webgl2' : 'cpu');

    // Initialize WebGL2 holographic renderer
    const canvas = document.getElementById('hologram-canvas') as HTMLCanvasElement;
    if (!canvas) return;
    
    gl = canvas.getContext('webgl2');
    if (!gl) {
      console.error('WebGL2 not supported');
      return;
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
      
      void main(void) {
        // Wave distortion for holographic effect
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
        
        // Edge glow effect
        float edgeGlow = 1.0 - smoothstep(0.0, 0.1, min(vTex.x, min(vTex.y, min(1.0 - vTex.x, 1.0 - vTex.y))));
        vec3 glowColor = vec3(0.4, 0.7, 1.0) * edgeGlow * 0.3;
        
        // Combine effects
        vec3 color = vec3(r, g, b) * shimmer + glowColor;
        
        // Output with slight transparency for depth
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

    // Create texture for video
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

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
      
      // Clear and draw
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      
      animationId = requestAnimationFrame(render);
    }
    
    render(0);
  });

  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  });
</script>

<style>
  .wrap { 
    display: grid; 
    gap: 1rem; 
    grid-template-columns: 1fr; 
    position: relative;
  }
  .bar { 
    display: flex; 
    gap: 0.5rem; 
    align-items: center; 
    border: 1px solid #2b2b2b; 
    border-radius: 12px; 
    padding: 0.5rem 0.75rem; 
    background: #0d0d0d; 
    color: #fff; 
  }
  .pill { 
    padding: 0.2rem 0.5rem; 
    border-radius: 999px; 
    border: 1px solid #333; 
    background: #111; 
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; 
  }
  .canvas-container {
    position: relative;
    width: 100%;
    max-width: 540px;
    margin: 0 auto;
  }
  canvas { 
    width: 100%; 
    height: auto; 
    aspect-ratio: 9 / 16; 
    border-radius: 16px; 
    display: block; 
    background: #000;
    box-shadow: 0 0 40px rgba(142, 197, 255, 0.3);
  }
  .grid { 
    display: flex; 
    gap: 0.75rem; 
    flex-wrap: wrap; 
  }
  a.btn { 
    border: 1px solid #444; 
    border-radius: 10px; 
    padding: 0.5rem 0.9rem; 
    background: #111; 
    color: #fff; 
    text-decoration: none; 
  }
  a.btn:hover {
    background: #222;
    border-color: #666;
  }
</style>

<div class="wrap">
  <div class="bar">
    <div class="pill">/hologram</div>
    {#if caps}
      <div class="pill">{caps.iosLike ? 'iOS' : 'Desktop'}</div>
      <div class="pill">{status}</div>
      {#if caps.reason}<div class="pill" title={caps.reason}>note</div>{/if}
    {:else}
      <div class="pill">initializing...</div>
    {/if}
    <div class="grid" style="margin-left:auto">
      <a class="btn" href="/pricing">Pricing</a>
      <a class="btn" href="/templates">Templates</a>
      <a class="btn" href="/dashboard">Dashboard</a>
    </div>
  </div>

  <div class="canvas-container">
    <!-- Main hologram canvas with shader effects -->
    <canvas id="hologram-canvas" width={1080} height={1920}></canvas>
    
    <!-- Hidden video element that feeds the canvas -->
    <video bind:this={videoEl} autoplay muted loop playsinline style="display:none"></video>
    
    <!-- Source selector HUD overlay -->
    <HologramSourceSelector {videoEl} />
  </div>

  <!-- Recorder with plan-gated limits and watermarking -->
  <HologramRecorder hologramCanvasSelector="#hologram-canvas" />
</div>