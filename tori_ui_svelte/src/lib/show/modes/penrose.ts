// Penrose Visualizer with Live Solve and Iso-lines
// WebGL2-based field visualization with iso-line rendering

let alive = false;
let raf = 0;
let gl: WebGL2RenderingContext;
let prog: WebGLProgram;
let fieldTex: WebGLTexture | null = null;
let W = 256;
let H = 256;

// Fetch field data from the Penrose solver API
async function fetchField(): Promise<Float32Array> {
  try {
    const response = await fetch('/api/penrose/solve', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scene: 'demo', N: W })
    });

    // Check if we got actual binary data
    if (response.ok && response.headers.get('content-type')?.includes('application/octet-stream')) {
      // Direct binary field data from server
      const buffer = await response.arrayBuffer();
      return new Float32Array(buffer);
    }

    // Fallback: Generate synthetic demo field for testing
    const f = new Float32Array(W * H);
    const t = performance.now() * 0.001;
    
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const nx = (x / W - 0.5) * 2.0;
        const ny = (y / H - 0.5) * 2.0;
        
        // Create interesting field pattern
        const r = Math.sqrt(nx * nx + ny * ny);
        const theta = Math.atan2(ny, nx);
        
        // Combine radial and angular components with time animation
        f[y * W + x] = 
          Math.sin(10.0 * r - t * 2.0) * 0.5 +
          Math.cos(5.0 * theta + t) * 0.3 +
          Math.sin(15.0 * nx + t) * Math.cos(15.0 * ny - t) * 0.2;
      }
    }
    
    return f;
  } catch (error) {
    console.warn('Failed to fetch field data, using synthetic field:', error);
    
    // Generate synthetic field on error
    const f = new Float32Array(W * H);
    const t = performance.now() * 0.001;
    
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const nx = (x / W - 0.5);
        const ny = (y / H - 0.5);
        f[y * W + x] = Math.sin(10.0 * nx + t) * Math.cos(10.0 * ny - t);
      }
    }
    
    return f;
  }
}

// Vertex shader - creates full-screen triangle
const vertexShaderSource = `#version 300 es
precision highp float;

out vec2 uv;

void main() {
  // Generate full-screen triangle
  vec2 position = vec2(
    float((gl_VertexID << 1) & 2) - 1.0,
    float((gl_VertexID & 2)) - 1.0
  );
  
  uv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}`;

// Fragment shader - renders iso-lines with color mapping
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_field;
uniform float u_time;
uniform float u_isolines;
uniform float u_lineWidth;

// Color palette for visualization
vec3 getColor(float value) {
  // Normalize value to [0, 1]
  float v = value * 0.5 + 0.5;
  
  // Multi-stop gradient
  vec3 c1 = vec3(0.02, 0.2, 0.35);  // Deep blue
  vec3 c2 = vec3(0.1, 0.5, 0.7);    // Medium blue
  vec3 c3 = vec3(0.2, 0.8, 0.9);    // Light cyan
  vec3 c4 = vec3(1.0, 0.9, 0.1);    // Yellow
  vec3 c5 = vec3(1.0, 0.3, 0.1);    // Orange-red
  
  vec3 color;
  if (v < 0.25) {
    color = mix(c1, c2, v * 4.0);
  } else if (v < 0.5) {
    color = mix(c2, c3, (v - 0.25) * 4.0);
  } else if (v < 0.75) {
    color = mix(c3, c4, (v - 0.5) * 4.0);
  } else {
    color = mix(c4, c5, (v - 0.75) * 4.0);
  }
  
  return color;
}

void main() {
  // Sample field value
  float fieldValue = texture(u_field, uv).r;
  
  // Calculate iso-lines
  float scaledValue = fieldValue * u_isolines;
  float lineIntensity = abs(fract(scaledValue) - 0.5) * 2.0;
  
  // Sharp iso-lines with anti-aliasing
  float lineAlpha = 1.0 - smoothstep(1.0 - u_lineWidth, 1.0, lineIntensity);
  
  // Get base color from field value
  vec3 baseColor = getColor(fieldValue);
  
  // Mix with iso-line color
  vec3 lineColor = vec3(0.9, 0.95, 1.0);
  vec3 finalColor = mix(baseColor, lineColor, lineAlpha * 0.7);
  
  // Add subtle gradient overlay
  float gradient = dot(uv - 0.5, uv - 0.5) * 0.5;
  finalColor *= 1.0 - gradient * 0.2;
  
  fragColor = vec4(finalColor, 1.0);
}`;

// Compile and link shader program
function createShaderProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram {
  const program = gl.createProgram()!;
  
  // Compile vertex shader
  const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vertexShader, vertexSource);
  gl.compileShader(vertexShader);
  
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(vertexShader);
    throw new Error(`Vertex shader compilation failed: ${error}`);
  }
  
  // Compile fragment shader
  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fragmentShader, fragmentSource);
  gl.compileShader(fragmentShader);
  
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(fragmentShader);
    throw new Error(`Fragment shader compilation failed: ${error}`);
  }
  
  // Link program
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program);
    throw new Error(`Shader program linking failed: ${error}`);
  }
  
  // Clean up shaders (they're linked to the program now)
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  
  return program;
}

// Uniform locations
interface Uniforms {
  field: WebGLUniformLocation | null;
  time: WebGLUniformLocation | null;
  isolines: WebGLUniformLocation | null;
  lineWidth: WebGLUniformLocation | null;
}

let uniforms: Uniforms;
let startTime = 0;
let updateInterval: number | null = null;
let lastFieldUpdate = 0;
const FIELD_UPDATE_INTERVAL = 100; // Update field every 100ms

// Main render loop
async function render(canvas: HTMLCanvasElement, timestamp: number) {
  if (!alive) return;
  
  const elapsed = timestamp - startTime;
  
  // Update field periodically
  if (timestamp - lastFieldUpdate > FIELD_UPDATE_INTERVAL) {
    lastFieldUpdate = timestamp;
    
    try {
      const fieldData = await fetchField();
      
      // Update texture with new field data
      gl.bindTexture(gl.TEXTURE_2D, fieldTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R32F,
        W,
        H,
        0,
        gl.RED,
        gl.FLOAT,
        fieldData
      );
    } catch (error) {
      console.error('Failed to update field:', error);
    }
  }
  
  // Resize canvas if needed
  const displayWidth = canvas.clientWidth;
  const displayHeight = canvas.clientHeight;
  
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    gl.viewport(0, 0, displayWidth, displayHeight);
  }
  
  // Clear and render
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  
  // Use shader program
  gl.useProgram(prog);
  
  // Set uniforms
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fieldTex);
  gl.uniform1i(uniforms.field, 0);
  
  gl.uniform1f(uniforms.time, elapsed * 0.001);
  gl.uniform1f(uniforms.isolines, 15.0); // Number of iso-lines
  gl.uniform1f(uniforms.lineWidth, 0.05); // Line width
  
  // Draw full-screen triangle
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  
  // Schedule next frame
  raf = requestAnimationFrame((t) => render(canvas, t));
}

// Export ShowAPI implementation
export default {
  async start(canvas: HTMLCanvasElement) {
    alive = true;
    startTime = performance.now();
    lastFieldUpdate = 0;
    
    // Initialize WebGL2 context
    gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: true
    })!;
    
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }
    
    // Create shader program
    prog = createShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
    
    // Get uniform locations
    uniforms = {
      field: gl.getUniformLocation(prog, 'u_field'),
      time: gl.getUniformLocation(prog, 'u_time'),
      isolines: gl.getUniformLocation(prog, 'u_isolines'),
      lineWidth: gl.getUniformLocation(prog, 'u_lineWidth')
    };
    
    // Create and configure field texture
    fieldTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, fieldTex);
    
    // Configure texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    // Initialize with first field data
    const initialField = await fetchField();
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      W,
      H,
      0,
      gl.RED,
      gl.FLOAT,
      initialField
    );
    
    // Start render loop
    raf = requestAnimationFrame((t) => render(canvas, t));
  },
  
  stop() {
    alive = false;
    
    if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
    
    if (updateInterval) {
      clearInterval(updateInterval);
      updateInterval = null;
    }
    
    // Clean up WebGL resources
    if (gl && fieldTex) {
      gl.deleteTexture(fieldTex);
      fieldTex = null;
    }
    
    if (gl && prog) {
      gl.deleteProgram(prog);
    }
  },
  
  onResize() {
    // Resize handling is done in the render loop
  },
  
  onBoost(on: boolean) {
    // Adjust update frequency based on boost mode
    if (on) {
      // Increase field update rate in boost mode
      // Could also increase resolution or iso-line count
      console.log('Boost mode:', on);
    }
  }
} satisfies import('../ShowController').ShowAPI;