// ${IRIS_ROOT}\frontend\lib\webgpu\shaders\post\phasePolisher.wgsl
// Stateless phase polish: phi' = phi + tau*Laplacian(phi) with seam-aware attenuation.
// Then Dphi = clamp(phi' - phi, +-maxCorrection); apply e^{iDphi} to (Re,Im).

struct Params {
  width: u32,
  height: u32,
  tv_lambda: f32,       // e.g., 0.08
  max_correction: f32,  // e.g., 0.25 rad
  use_mask: u32,        // 0/1
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> reBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> imBuf: array<f32>;
@group(0) @binding(2) var<storage, read>      maskBuf: array<f32>; // 0..1 (1=keep detail / reduce smoothing)
@group(0) @binding(3) var<uniform>            P: Params;

fn idx(x:u32, y:u32, w:u32) -> u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = P.width;
  let h = P.height;
  if (gid.x >= w || gid.y >= h) { return; }

  let i = idx(gid.x, gid.y, w);
  let re = reBuf[i];
  let im = imBuf[i];
  // phase
  var phi = atan2(im, re);

  // Neighbor sampling with clamped borders
  let xm = select(0u, gid.x - 1u, gid.x > 0u);
  let xp = select(w - 1u, gid.x + 1u, gid.x + 1u < w);
  let ym = select(0u, gid.y - 1u, gid.y > 0u);
  let yp = select(h - 1u, gid.y + 1u, gid.y + 1u < h);

  // Neighbor phases
  let iL = idx(xm, gid.y, w);
  let iR = idx(xp, gid.y, w);
  let iU = idx(gid.x, ym, w);
  let iD = idx(gid.x, yp, w);

  let phiL = atan2(imBuf[iL], reBuf[iL]);
  let phiR = atan2(imBuf[iR], reBuf[iR]);
  let phiU = atan2(imBuf[iU], reBuf[iU]);
  let phiD = atan2(imBuf[iD], reBuf[iD]);

  // Simple 5-point Laplacian on phase (unwrap-free small-step)
  let lap = (phiL + phiR + phiU + phiD - 4.0 * phi);

  // Seam-aware attenuation: reduce smoothing where mask~1 (edges/important ROIs)
  var atten = 1.0;
  if (P.use_mask == 1u) {
    let m = maskBuf[i];           // 0..1
    atten = 1.0 - clamp(m, 0.0, 1.0); // keep details where m high
  }

  // Small update
  let phiPrime = phi + P.tv_lambda * atten * lap;

  // Dphi clamp and apply phasor
  var dphi = phiPrime - phi;
  let maxc = P.max_correction;
  dphi = clamp(dphi, -maxc, maxc);

  let c = cos(dphi);
  let s = sin(dphi);
  let reNew = re * c - im * s;
  let imNew = re * s + im * c;

  reBuf[i] = reNew;
  imBuf[i] = imNew;
}
