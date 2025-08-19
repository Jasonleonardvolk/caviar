import type { DisplayCalib, QuiltLayout } from './types';

// Import shader as string - we'll wire this up properly with the bundler later
const lenticular_interleave_wgsl = `
struct Calib {
  pitch   : f32,
  tilt    : f32,
  center  : f32,
  subp    : f32,
  cols    : f32,
  rows    : f32,
  tileW   : f32,
  tileH   : f32,
  quiltW  : f32,
  quiltH  : f32,
  numViews: f32,
  panelW  : f32,
  panelH  : f32,
  gamma   : f32,
  overscan: f32,
}

@group(0) @binding(0) var quiltTex : texture_2d<f32>;
@group(0) @binding(1) var quiltSamp: sampler;
@group(0) @binding(2) var<uniform> C : Calib;

struct VSOut { 
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0), 
    vec2<f32>(-1.0,  1.0), 
    vec2<f32>( 3.0,  1.0)
  );
  var o: VSOut;
  let xy = p[vid];
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  o.uv  = (xy * 0.5) + vec2<f32>(0.5);
  return o;
}

fn view_index(screen_px: vec2<f32>, subpixel: f32) -> f32 {
  let v = ((screen_px.x - C.center) / C.pitch) - (screen_px.y * C.tilt) + (subpixel * 0.5);
  return clamp(floor(v + 0.5), 0.0, C.numViews - 1.0);
}

fn tile_uv_from_view(v: f32, uv: vec2<f32>) -> vec2<f32> {
  let cols = C.cols;
  let rows = C.rows;
  let col  = floor(v % cols);
  let row  = floor(v / cols);

  let osc = clamp(C.overscan, 0.0, 0.15);
  let inner = vec2<f32>(1.0 - osc * 2.0);
  let offs  = vec2<f32>(osc);

  let tileUV = offs + uv * inner;

  let u = (col + tileUV.x) / cols;
  let vq = (row + tileUV.y) / rows;
  return vec2<f32>(u, vq);
}

fn gamma_correct(c: vec3<f32>, g: f32) -> vec3<f32> {
  if (g <= 0.0) { return c; }
  return pow(c, vec3<f32>(1.0 / g));
}

@fragment
fn fs_main(inp: VSOut) -> @location(0) vec4<f32> {
  let screen_px = vec2<f32>(inp.uv.x * C.panelW, (1.0 - inp.uv.y) * C.panelH);

  let vR = view_index(screen_px, 0.0);
  let vG = view_index(screen_px, 1.0);
  let vB = view_index(screen_px, 2.0);

  let tileUV_R = tile_uv_from_view(vR, inp.uv);
  let tileUV_G = tile_uv_from_view(vG, inp.uv);
  let tileUV_B = tile_uv_from_view(vB, inp.uv);

  let r = textureSample(quiltTex, quiltSamp, tileUV_R).x;
  let g = textureSample(quiltTex, quiltSamp, tileUV_G).y;
  let b = textureSample(quiltTex, quiltSamp, tileUV_B).z;

  let rgb = gamma_correct(vec3<f32>(r, g, b), C.gamma);
  return vec4<f32>(rgb, 1.0);
}
`;

export class QuiltRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private layout: QuiltLayout;
  private calib: DisplayCalib;

  quiltTexture!: GPUTexture; // rgba16float
  private quiltView!: GPUTextureView;
  private quiltSampler!: GPUSampler;

  private interleavePipeline!: GPURenderPipeline;
  private interleaveBGLayout!: GPUBindGroupLayout;
  private interleaveBind!: GPUBindGroup;
  private calibUBO!: GPUBuffer;

  // Future hook for holographic pipeline
  private cachedField: { amp: Float32Array; phi: Float32Array } | null = null;

  constructor(device: GPUDevice, calib: DisplayCalib, layout: QuiltLayout, format?: GPUTextureFormat) {
    this.device = device;
    this.format = format ?? navigator.gpu.getPreferredCanvasFormat();
    this.layout = layout;
    this.calib = calib;

    this.initQuilt();
    this.initInterleaver();
  }

  updateCalib(calib: DisplayCalib) {
    this.calib = calib;
    this.writeCalib();
  }

  updateLayout(layout: QuiltLayout) {
    this.layout = layout;
    this.initQuilt();      // reallocate quilt
    this.writeCalib();     // update dims
  }

  private initQuilt() {
    const { cols, rows, tileW, tileH } = this.layout;
    const w = cols * tileW;
    const h = rows * tileH;

    this.quiltTexture?.destroy?.();
    this.quiltTexture = this.device.createTexture({
      size: { width: w, height: h },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    this.quiltView = this.quiltTexture.createView();
    this.quiltSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    if (this.calibUBO) this.writeCalib(); // refresh dims if UBO already exists
  }

  private initInterleaver() {
    const code = lenticular_interleave_wgsl;

    this.interleavePipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { 
        module: this.device.createShaderModule({ code }), 
        entryPoint: 'vs_main' 
      },
      fragment: { 
        module: this.device.createShaderModule({ code }), 
        entryPoint: 'fs_main',
        targets: [{ format: this.format }] 
      },
      primitive: { topology: 'triangle-list' }
    });

    this.interleaveBGLayout = this.interleavePipeline.getBindGroupLayout(0);
    this.calibUBO = this.device.createBuffer({ 
      size: 16 * 4, 
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
    });
    this.writeCalib();

    this.interleaveBind = this.device.createBindGroup({
      layout: this.interleaveBGLayout,
      entries: [
        { binding: 0, resource: this.quiltView },
        { binding: 1, resource: this.quiltSampler },
        { binding: 2, resource: { buffer: this.calibUBO } }
      ]
    });
  }

  private writeCalib() {
    const { pitch, tilt, center, subp, panelW, panelH } = this.calib;
    const { cols, rows, tileW, tileH, numViews } = this.layout;
    const quiltW = cols * tileW, quiltH = rows * tileH;
    const gamma = 2.2, overscan = 0.02;

    const u8 = new Uint8Array(16 * 4); 
    const dv = new DataView(u8.buffer);
    dv.setFloat32( 0, pitch,  true);
    dv.setFloat32( 4, tilt,   true);
    dv.setFloat32( 8, center, true);
    dv.setFloat32(12, subp,   true);
    dv.setFloat32(16, cols,   true);
    dv.setFloat32(20, rows,   true);
    dv.setFloat32(24, tileW,  true);
    dv.setFloat32(28, tileH,  true);
    dv.setFloat32(32, quiltW, true);
    dv.setFloat32(36, quiltH, true);
    dv.setFloat32(40, numViews, true);
    dv.setFloat32(44, panelW, true);
    dv.setFloat32(48, panelH, true);
    dv.setFloat32(52, gamma,  true);
    dv.setFloat32(56, overscan, true);
    
    this.device.queue.writeBuffer(this.calibUBO, 0, u8);
  }

  // --- Quilt render pass control ---

  beginQuiltPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      colorAttachments: [{
        view: this.quiltView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', 
        storeOp: 'store'
      }]
    });
  }

  setTileViewport(pass: GPURenderPassEncoder, viewIndex: number) {
    const { cols, tileW, tileH } = this.layout;
    const col = viewIndex % cols;
    const row = Math.floor(viewIndex / cols);
    const x = col * tileW;
    const y = row * tileH;
    pass.setViewport(x, y, tileW, tileH, 0, 1);
    pass.setScissorRect(x, y, tileW, tileH);
  }

  endQuiltPass(pass: GPURenderPassEncoder) { 
    pass.end(); 
  }

  drawFullscreenInterleaver(encoder: GPUCommandEncoder, dst: GPUTextureView) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ 
        view: dst, 
        loadOp: 'clear', 
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 } 
      }]
    });
    pass.setPipeline(this.interleavePipeline);
    pass.setBindGroup(0, this.interleaveBind);
    pass.draw(3, 1, 0, 0);
    pass.end();
  }

  // Future hook for holographic pipeline integration
  async setPhaseField(amp: Float32Array, phi: Float32Array) { 
    // Store for later use when we integrate with phase_field_to_views compute shader
    this.cachedField = { amp, phi };
    console.log('Phase field cached for future view synthesis');
  }

  // Get current quilt info
  getQuiltInfo() {
    return {
      layout: this.layout,
      calib: this.calib,
      textureSize: {
        width: this.layout.cols * this.layout.tileW,
        height: this.layout.rows * this.layout.tileH
      }
    };
  }
}