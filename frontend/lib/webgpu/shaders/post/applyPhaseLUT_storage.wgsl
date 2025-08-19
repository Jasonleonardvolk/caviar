// Alternative version using storage buffer for LUT (if texture version doesn't work)
// This keeps the original storage buffer approach but ensures unique bindings

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

// Ensure each binding is unique - no duplicates
@group(0) @binding(0) var<storage, read_write> reBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> imBuf: array<f32>;
@group(0) @binding(2) var<storage, read>       dphiBuf: array<f32>;
@group(0) @binding(3) var<uniform>             P: Params;

fn idx(x:u32, y:u32, w:u32) -> u32 { 
  return y*w + x; 
}

fn clampu(v:i32, lo:i32, hi:i32) -> u32 {
  return u32(max(lo, min(v, hi)));
}

fn sampleLUT(u: f32, v: f32) -> f32 {
  // u,v in pixel space [0..lutW-1], [0..lutH-1]
  let w = i32(P.lutW);
  let h = i32(P.lutH);
  let uf = clamp(u, 0.0, f32(P.lutW - 1u));
  let vf = clamp(v, 0.0, f32(P.lutH - 1u));
  let x0 = i32(floor(uf));
  let y0 = i32(floor(vf));
  let x1 = min(x0 + 1, w - 1);
  let y1 = min(y0 + 1, h - 1);
  let fx = uf - f32(x0);
  let fy = vf - f32(y0);

  let i00 = idx(clampu(x0, 0, w-1), clampu(y0, 0, h-1), u32(w));
  let i10 = idx(clampu(x1, 0, w-1), clampu(y0, 0, h-1), u32(w));
  let i01 = idx(clampu(x0, 0, w-1), clampu(y1, 0, h-1), u32(w));
  let i11 = idx(clampu(x1, 0, w-1), clampu(y1, 0, h-1), u32(w));

  // Add bounds checking to reduce warnings
  let n = arrayLength(&dphiBuf);
  let d00 = select(0.0, dphiBuf[i00], i00 < n);
  let d10 = select(0.0, dphiBuf[i10], i10 < n);
  let d01 = select(0.0, dphiBuf[i01], i01 < n);
  let d11 = select(0.0, dphiBuf[i11], i11 < n);

  let d0 = mix(d00, d10, fx);
  let d1 = mix(d01, d11, fx);
  return mix(d0, d1, fy);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let i = idx(gid.x, gid.y, P.width);

  // Add bounds checking
  let nRe = arrayLength(&reBuf);
  let nIm = arrayLength(&imBuf);
  if (i >= nRe || i >= nIm) { return; }

  var re = reBuf[i];
  var im = imBuf[i];

  // Map render pixel (x,y) -> LUT pixel (u,v)
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
