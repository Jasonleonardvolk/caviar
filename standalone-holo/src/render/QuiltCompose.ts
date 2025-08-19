// QuiltCompose.ts - minimal blit with UV offset. Stub for multi-view quilt later.
export class QuiltCompose {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;

  private sampler!: GPUSampler;
  private pipeline!: GPURenderPipeline;
  private bindLayout!: GPUBindGroupLayout;

  private uvBuf!: GPUBuffer;

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device; this.context = context; this.format = format;
    this.init();
  }

  private init() {
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    const vs = /* wgsl */`
      @vertex
      fn main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
        // Fullscreen triangle
        var pos = array<vec2<f32>, 3>(
          vec2<f32>(-1.0, -3.0),
          vec2<f32>(-1.0,  1.0),
          vec2<f32>( 3.0,  1.0)
        );
        let p = pos[vid];
        return vec4<f32>(p, 0.0, 1.0);
      }
    `;
    const fs = /* wgsl */`
      struct UBO { uvOffset: vec2<f32> };
      @group(0) @binding(0) var samp: sampler;
      @group(0) @binding(1) var img : texture_2d<f32>;
      @group(0) @binding(2) var<uniform> U : UBO;

      @fragment
      fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
        let uv = (p.xy / vec2<f32>(f32(textureDimensions(img).x), f32(textureDimensions(img).y))) + U.uvOffset;
        // Map screen coords back to 0..1
        let u = (p.x / f32(textureDimensions(img).x));
        let v = (p.y / f32(textureDimensions(img).y));
        let st = vec2<f32>(u, v) + U.uvOffset;
        return textureSampleLevel(img, samp, clamp(st, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
      }
    `;

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: this.device.createShaderModule({ code: vs }), entryPoint: 'main' },
      fragment: {
        module: this.device.createShaderModule({ code: fs }), entryPoint: 'main',
        targets: [{ format: this.format }]
      },
      primitive: { topology: 'triangle-list' }
    });

    this.bindLayout = this.pipeline.getBindGroupLayout(0);
    this.uvBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  draw(source: GPUTexture, uvOffset: { x: number, y: number }) {
    // Write uvOffset
    const u8 = new Uint8Array(8); const dv = new DataView(u8.buffer);
    dv.setFloat32(0, uvOffset.x, true); dv.setFloat32(4, uvOffset.y, true);
    this.device.queue.writeBuffer(this.uvBuf, 0, u8);

    const view = this.context.getCurrentTexture().createView();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
    });

    const bind = this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: source.createView() },
        { binding: 2, resource: { buffer: this.uvBuf } }
      ]
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3, 1, 0, 0);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}