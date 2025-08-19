// ${IRIS_ROOT}\frontend\lib\webgpu\shaders\post\applyPhaseLUT.wgsl
// Apply a prebaked Dphi LUT with texture sampling for better performance
// Updated to use texture2d + sampler for the LUT instead of storage buffer

struct Params {
  width: u32,           // render W
  height: u32,          // render H
  lutW: u32,            // LUT W
  lutH: u32,            // LUT H
  gain: f32,            // scale Dphi (default 1.0)
  max_correction: f32,  // clamp |Dphi|
  scaleX: f32,          // mapping: u = x * scaleX + offsetX
  scaleY: f32,          // mapping: v = y * scaleY + offsetY
  offsetX: f32,         // see above
  offsetY: f32,
  _pad0: u32, 
  _pad1: u32, 
  _pad2: u32, 
  _pad3: u32, // pad to 64B
}

// Fixed bindings - each resource has unique slot
@group(0) @binding(0) var<storage, read_write> reBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> imBuf: array<f32>;
@group(0) @binding(2) var<uniform>             P: Params;
@group(0) @binding(3) var lutTex: texture_2d<f32>;
@group(0) @binding(4) var lutSampler: sampler;

fn idx(x:u32, y:u32, w:u32) -> u32 { 
  return y*w + x; 
}

fn sampleLUT(u: f32, v: f32) -> f32 {
  // Normalize to [0,1] for texture sampling
  let uNorm = u / f32(P.lutW - 1u);
  let vNorm = v / f32(P.lutH - 1u);
  
  // Sample the texture with built-in bilinear filtering
  // Must use textureSampleLevel in compute shaders (not textureSample)
  let dphi = textureSampleLevel(lutTex, lutSampler, vec2<f32>(uNorm, vNorm), 0.0).r;
  return dphi;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let i = idx(gid.x, gid.y, P.width);

  var re = reBuf[i];
  var im = imBuf[i];

  // Map render pixel (x,y) -> LUT pixel (u,v)
  // Choose scale to align edges exactly: scaleX=(lutW-1)/(W-1), offset=0
  let u = f32(gid.x) * P.scaleX + P.offsetX;
  let v = f32(gid.y) * P.scaleY + P.offsetY;

  var dphi = sampleLUT(u, v) * P.gain;
  dphi = clamp(dphi, -P.max_correction, P.max_correction);

  let c = cos(dphi);
  let s = sin(dphi);
  let reNew = re * c - im * s;
  let imNew = re * s + im * c;

  reBuf[i] = reNew;
  imBuf[i] = imNew;
}
