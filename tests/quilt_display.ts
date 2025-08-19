import { loadCalibration } from '../frontend/lib/pipeline/quilt/calibration';
import { QuiltRenderer } from '../frontend/lib/pipeline/quilt/QuiltRenderer';
import type { QuiltLayout } from '../frontend/lib/pipeline/quilt/types';

// Default layout for Looking Glass Portrait (adjust for your display)
const layout: QuiltLayout = { 
  cols: 8, 
  rows: 6, 
  tileW: 320, 
  tileH: 320, 
  numViews: 48 
};

const calibUrl = '/configs/display_calibration/my_panel.json';

async function main() {
  const canvas = document.getElementById('c') as HTMLCanvasElement;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter not found');
  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();
  const ctx = canvas.getContext('webgpu')!;
  ctx.configure({ device, format, alphaMode: 'opaque' });

  // Load calibration (or use defaults)
  let calib;
  try {
    calib = await loadCalibration(calibUrl);
    console.log('Loaded calibration:', calib);
  } catch (err) {
    console.warn('Using default calibration');
    calib = {
      pitch: 49.8,
      tilt: 0.12,
      center: 0.0,
      subp: 0,
      panelW: canvas.width,
      panelH: canvas.height
    };
  }

  const quilt = new QuiltRenderer(device, calib, layout, format);
  console.log('Quilt info:', quilt.getQuiltInfo());

  // Create a more interesting scene shader - rotating cube with depth
  const sceneShader = `
    struct Uniforms {
      viewIndex: f32,
      time: f32,
      numViews: f32,
      _pad: f32,
    }
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VSOut {
      @builtin(position) pos: vec4<f32>,
      @location(0) color: vec3<f32>,
      @location(1) depth: f32,
    }

    @vertex
    fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
      // Create a 3D cube
      var positions = array<vec3<f32>, 36>(
        // Front face
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5),
        // Back face
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>( 0.5, -0.5, -0.5),
        // Top face
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>(-0.5,  0.5,  0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>( 0.5,  0.5, -0.5),
        // Bottom face
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5,  0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>( 0.5, -0.5,  0.5), vec3<f32>(-0.5, -0.5,  0.5),
        // Right face
        vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>( 0.5, -0.5, -0.5), vec3<f32>( 0.5,  0.5,  0.5), vec3<f32>( 0.5, -0.5,  0.5),
        // Left face
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5, -0.5,  0.5), vec3<f32>(-0.5,  0.5,  0.5),
        vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5,  0.5,  0.5), vec3<f32>(-0.5,  0.5, -0.5)
      );

      var colors = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.0, 0.0), // Front - Red
        vec3<f32>(0.0, 1.0, 0.0), // Back - Green
        vec3<f32>(0.0, 0.0, 1.0), // Top - Blue
        vec3<f32>(1.0, 1.0, 0.0), // Bottom - Yellow
        vec3<f32>(1.0, 0.0, 1.0), // Right - Magenta
        vec3<f32>(0.0, 1.0, 1.0)  // Left - Cyan
      );

      let pos = positions[vid];
      let faceIndex = vid / 6u;
      let color = colors[faceIndex];

      // Apply rotation based on time
      let angle = u.time * 0.5;
      let c = cos(angle);
      let s = sin(angle);
      
      // Rotate around Y axis
      let rotatedPos = vec3<f32>(
        pos.x * c - pos.z * s,
        pos.y,
        pos.x * s + pos.z * c
      );

      // Apply view-dependent camera position (for parallax)
      let viewOffset = (u.viewIndex - u.numViews * 0.5) / u.numViews;
      let cameraX = viewOffset * 0.3; // Horizontal parallax
      
      // Simple perspective projection
      let viewPos = rotatedPos - vec3<f32>(cameraX, 0.0, 2.0);
      let projected = vec4<f32>(
        viewPos.x / (2.0 - viewPos.z),
        viewPos.y / (2.0 - viewPos.z),
        -viewPos.z / 4.0,
        1.0
      );

      var out: VSOut;
      out.pos = projected;
      out.color = color;
      out.depth = -viewPos.z;
      return out;
    }

    @fragment
    fn fs_main(inp: VSOut) -> @location(0) vec4<f32> {
      // Add some depth-based shading
      let depthFactor = clamp(1.0 - inp.depth * 0.3, 0.3, 1.0);
      return vec4<f32>(inp.color * depthFactor, 1.0);
    }
  `;

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: sceneShader }),
      entryPoint: 'vs_main'
    },
    fragment: {
      module: device.createShaderModule({ code: sceneShader }),
      entryPoint: 'fs_main',
      targets: [{ format: 'rgba16float' }]
    },
    primitive: { 
      topology: 'triangle-list',
      cullMode: 'back'
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  });

  // Create uniform buffer for per-view data
  const uniformBuffer = device.createBuffer({ 
    size: 16, // 4 floats
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
  });

  // Create depth texture for quilt
  const depthTexture = device.createTexture({
    size: [layout.cols * layout.tileW, layout.rows * layout.tileH],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  });

  let animating = true;
  let frameCount = 0;
  let lastTime = performance.now();

  function drawScene(enc: GPUCommandEncoder, time: number) {
    const pass = quilt.beginQuiltPass(enc);
    
    // Add depth attachment
    (pass as any).depthStencilAttachment = {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store'
    };

    pass.setPipeline(pipeline);
    
    for (let v = 0; v < layout.numViews; v++) {
      quilt.setTileViewport(pass, v);
      
      // Update uniforms for this view
      const uniforms = new Float32Array([
        v,                  // viewIndex
        animating ? time : 0, // time
        layout.numViews,    // numViews
        0                   // padding
      ]);
      device.queue.writeBuffer(uniformBuffer, 0, uniforms);
      
      const bind = device.createBindGroup({ 
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }] 
      });
      
      pass.setBindGroup(0, bind);
      pass.draw(36, 1, 0, 0); // Draw cube (36 vertices)
    }
    
    quilt.endQuiltPass(pass);
  }

  function frame() {
    const now = performance.now();
    const time = now * 0.001; // Convert to seconds
    
    // Update FPS counter
    frameCount++;
    if (now - lastTime > 1000) {
      const fps = Math.round(frameCount * 1000 / (now - lastTime));
      document.getElementById('fps')!.textContent = fps.toString();
      frameCount = 0;
      lastTime = now;
    }

    const enc = device.createCommandEncoder();
    drawScene(enc, time);
    const back = ctx.getCurrentTexture().createView();
    quilt.drawFullscreenInterleaver(enc, back);
    device.queue.submit([enc.finish()]);
    
    requestAnimationFrame(frame);
  }

  // Wire up controls
  document.getElementById('toggleAnimation')!.addEventListener('click', () => {
    animating = !animating;
  });

  document.getElementById('changeViews')!.addEventListener('click', () => {
    const newViews = prompt('Number of views (1-100):', layout.numViews.toString());
    if (newViews) {
      const num = Math.max(1, Math.min(100, parseInt(newViews)));
      layout.numViews = num;
      // Recalculate grid
      layout.cols = Math.ceil(Math.sqrt(num * 1.5));
      layout.rows = Math.ceil(num / layout.cols);
      quilt.updateLayout(layout);
      document.getElementById('views')!.textContent = num.toString();
    }
  });

  frame();
}

main().catch(console.error);