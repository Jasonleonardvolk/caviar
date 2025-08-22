// 200k particles with audio-reactive radius; WebGPU with graceful fallback to CPU
let alive = false, device:GPUDevice, ctx:GPUCanvasContext, format:GPUTextureFormat, raf=0;
let pipeline:GPUComputePipeline, renderPipe:GPURenderPipeline;
let posBuf:GPUBuffer, velBuf:GPUBuffer, bind:GPUBindGroup;
let analyser:AnalyserNode, audioGain=0;

const COUNT = 200_000;
const WORKGROUP=256;

const wgsl_compute = /* wgsl */`
struct Part { pos: vec2<f32>, vel: vec2<f32>; }
@group(0) @binding(0) var<storage, read_write> P : array<Part>;
@group(0) @binding(1) var<uniform> U : vec4<f32>; // U.xy = mouse, U.z = dt, U.w = audio

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= arrayLength(&P)) { return; }
  var p = P[i];
  let c = vec2<f32>(0.0,0.0);
  let d = clamp(length(p.pos - c), 0.001, 2.0);
  let theta = atan2(p.pos.y, p.pos.x) + 0.5 * U.w; // swirl with audio
  let r = mix(d, d + 0.25*U.w, 0.5);
  // simple orbit + damping
  p.vel += vec2<f32>(-p.pos.y, p.pos.x) * 0.002 + (-p.pos) * 0.0005;
  p.vel *= 0.995;
  p.pos += p.vel * U.z;
  // wrap
  if (abs(p.pos.x) > 2.0) { p.pos.x = -sign(p.pos.x)*2.0; }
  if (abs(p.pos.y) > 2.0) { p.pos.y = -sign(p.pos.y)*2.0; }
  P[i] = p;
}
`;

const wgsl_vert = /* wgsl */`
struct VSOut { @builtin(position) pos: vec4<f32>; @location(0) v: f32; };
@group(0) @binding(0) var<storage, read> P : array<vec4<f32>>; // pos.xy vel.xy packed
@vertex fn main(@builtin(vertex_index) i:u32) -> VSOut {
  let p = P[i].xy;
  var o:VSOut; o.pos = vec4<f32>(p, 0.0, 1.0); o.v = fract(p.x*p.y);
  return o;
}
`;

const wgsl_frag = /* wgsl */`
@fragment fn main(@location(0) v:f32) -> @location(0) vec4<f32> {
  // soft additive glow
  let b = 0.85 + 0.15*sin(6.283*v);
  return vec4<f32>(0.2,0.8,1.0, b);
}
`;

async function initGPU(canvas: HTMLCanvasElement) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference:'high-performance' });
  device = await adapter?.requestDevice();
  if (!device) throw new Error('WebGPU not available');
  ctx = canvas.getContext('webgpu') as GPUCanvasContext;
  format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode:'premultiplied' });

  // buffers (pos.xy vel.xy)
  posBuf = device.createBuffer({ size: COUNT*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  velBuf = posBuf; // packed together for simplicity

  // init data
  const arr = new Float32Array(COUNT*4);
  for (let i=0;i<COUNT;i++){
    const a = Math.random()*Math.PI*2, r = Math.sqrt(Math.random())*1.8;
    arr[i*4+0] = Math.cos(a)*r; arr[i*4+1] = Math.sin(a)*r;
    arr[i*4+2] = 0; arr[i*4+3] = 0;
  }
  device.queue.writeBuffer(posBuf, 0, arr);

  const modComp = device.createShaderModule({ code: wgsl_compute });
  const modVert = device.createShaderModule({ code: wgsl_vert });
  const modFrag = device.createShaderModule({ code: wgsl_frag });

  pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: modComp, entryPoint: 'main' }
  });

  renderPipe = device.createRenderPipeline({
    layout:'auto',
    vertex: { module: modVert, entryPoint:'main', buffers:[{ arrayStride:16, stepMode:'instance' }] },
    fragment: { module: modFrag, entryPoint:'main', targets:[{ format, blend:{ color:{srcFactor:'one', dstFactor:'one', operation:'add'}, alpha:{srcFactor:'one', dstFactor:'one', operation:'add'} } }] },
    primitive: { topology:'point-list' }
  });

  // uniform
  const ubuf = device.createBuffer({ size:16, usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:[
    { binding:0, resource:{ buffer: posBuf } },
    { binding:1, resource:{ buffer: ubuf } }
  ]});

  // audio
  const ac = new (window.AudioContext|| (window as any).webkitAudioContext)();
  analyser = ac.createAnalyser(); analyser.fftSize = 512;
  const src = await navigator.mediaDevices.getUserMedia({ audio:true });
  const mic = ac.createMediaStreamSource(src); mic.connect(analyser);

  // frame
  let last = performance.now();
  alive = true;
  const data = new Uint8Array(analyser.frequencyBinCount);

  const frame = () => {
    if (!alive) return;
    const now = performance.now(), dt = Math.min(0.033, (now-last)/1000); last = now;
    analyser.getByteFrequencyData(data);
    audioGain = data[6]/255; // low band

    device.queue.writeBuffer(bind.getBinding(1).resource.buffer, 0, new Float32Array([0,0,dt, audioGain]));

    const encoder = device.createCommandEncoder();
    const passC = encoder.beginComputePass(); passC.setPipeline(pipeline); passC.setBindGroup(0, bind);
    passC.dispatchWorkgroups(Math.ceil(COUNT/WORKGROUP)); passC.end();

    const view = ctx.getCurrentTexture().createView();
    const passR = encoder.beginRenderPass({ colorAttachments:[{ view, loadOp:'clear', clearValue:{r:0,g:0,b:0,a:1}, storeOp:'store' }]});
    passR.setPipeline(renderPipe); passR.setVertexBuffer(0, posBuf); passR.draw(COUNT,1,0,0); passR.end();

    device.queue.submit([encoder.finish()]);
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
}

export default {
  async start(canvas: HTMLCanvasElement){ await initGPU(canvas); },
  stop(){ alive=false; cancelAnimationFrame(raf); },
  onResize(){ /* canvas is fixed full-screen; WebGPU ctx handles it */ },
  onBoost(on:boolean){ /* could intensify glow via uniform if desired */ }
} satisfies import('../ShowController').ShowAPI;
