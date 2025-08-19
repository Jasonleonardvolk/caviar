// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\jointBilateralUpsample.wgsl
struct Params {
  hiW:u32, hiH:u32,
  loW:u32, loH:u32,
  sigmaSpatial:f32,   // e.g., 1.0
  sigmaRange:f32,     // e.g., 0.1..0.2 in luma units
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var guideHi : texture_2d<f32>;  // full-res color guidance
@group(0) @binding(2) var depthLo : texture_2d<f32>;  // low-res single-channel depth
@group(0) @binding(3) var<storage, read_write> outHi : array<f32>;
@group(0) @binding(4) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }
fn luma(c:vec3f)->f32 { return dot(c, vec3f(0.299,0.587,0.114)); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x>=params.hiW || gid.y>=params.hiH) { return; }

  let uvHi = (vec2f(gid.xy)+vec2f(0.5))/vec2f(f32(params.hiW), f32(params.hiH));
  let uvLo = vec2f(f32(gid.x) * f32(params.loW) / f32(params.hiW),
                   f32(gid.y) * f32(params.loH) / f32(params.hiH)) / vec2f(f32(params.loW), f32(params.loH));

  let gC = textureSample(guideHi, samp, uvHi).rgb;
  let gL = luma(gC);

  let offHi = vec2f(1.0/f32(params.hiW), 1.0/f32(params.hiH));
  let offLo = vec2f(1.0/f32(params.loW), 1.0/f32(params.loH));

  var acc = 0.0;
  var wsum = 0.0;

  for (var j:i32=-1; j<=1; j++) {
    for (var i:i32=-1; i<=1; i++) {
      let uvNHi = uvHi + vec2f(f32(i), f32(j)) * offHi;
      let uvNLo = uvLo + vec2f(f32(i), f32(j)) * offLo;

      let d = textureSample(depthLo, samp, uvNLo).r;
      let gN = luma(textureSample(guideHi, samp, uvNHi).rgb);

      let ds = length(vec2f(f32(i),f32(j)));
      let wSpatial = exp(- (ds*ds) / (2.0 * params.sigmaSpatial * params.sigmaSpatial));
      let wRange   = exp(- pow(abs(gN - gL), 2.0) / (2.0 * params.sigmaRange * params.sigmaRange));

      let w = wSpatial * wRange;
      acc += d * w;
      wsum += w;
    }
  }

  let out = select(textureSample(depthLo, samp, uvLo).r, acc / wsum, wsum > 1e-6);
  outHi[idx(gid.x, gid.y, params.hiW)] = out;
}
