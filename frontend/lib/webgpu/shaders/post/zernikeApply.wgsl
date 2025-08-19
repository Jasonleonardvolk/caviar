// ${IRIS_ROOT}\frontend\lib\webgpu\shaders\post\zernikeApply.wgsl
// Zernike-based micro-phase corrector: Dphi(x,y) = Sum c_k Z_k(r,theta), k in {8 basis below}
// Basis order (coeff[0..7]):
//  0 tipX       :  r * cos(theta)
//  1 tiltY      :  r * sin(theta)
//  2 defocus    :  2 r^2 - 1
//  3 astig0     :  r^2 * cos(2theta)
//  4 astig45    :  r^2 * sin(2theta)
//  5 comaX      :  (3 r^3 - 2 r) * cos(theta)
//  6 comaY      :  (3 r^3 - 2 r) * sin(theta)
//  7 spherical  :  6 r^4 - 6 r^2 + 1
//
// Coordinates: pixel -> normalized aperture using ellipse centered at (cx,cy) with
// semi-axes (ax,ay) in pixels.
// Outside-behavior:
//   0 = zero      (no correction outside unit disk)
//   1 = attenuate (smooth ramp over 'softness' band)
//   2 = hold      (apply everywhere, no attenuation)

struct Params {
  width: u32,
  height: u32,
  cx: f32,          // aperture center (px)
  cy: f32,
  ax: f32,          // aperture semi-axis (px) X
  ay: f32,          // aperture semi-axis (px) Y
  max_correction: f32, // |Dphi| clamp (radians)
  softness: f32,       // ramp width (0..0.5, fraction of radius)
  outside_behavior: u32, // 0 zero, 1 attenuate, 2 hold
  _pad0: u32, 
  _pad1: u32, 
  _pad2: u32, // keep 16B alignment-friendly
}

struct Coeffs {
  c: array<f32>, // expect length >= 8
}

@group(0) @binding(0) var<storage, read_write> reBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> imBuf: array<f32>;
@group(0) @binding(2) var<storage, read> coeffs: Coeffs;
@group(0) @binding(3) var<uniform> P: Params;

fn idx(x:u32, y:u32, w:u32) -> u32 { return y*w + x; }

fn saturate(x:f32) -> f32 { return clamp(x, 0.0, 1.0); }

// Smoothstep (Hermite) from 1.0 at r<=1.0 to 0.0 at r>=1.0+soft
fn edge_attenuation(r:f32, soft:f32) -> f32 {
  let s = max(soft, 1e-4);
  let t = saturate((1.0 + s - r) / s);
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = P.width;
  let h = P.height;
  if (gid.x >= w || gid.y >= h) { return; }

  let i = idx(gid.x, gid.y, w);
  let re = reBuf[i];
  let im = imBuf[i];

  // Map pixel -> normalized aperture coords
  let xpx = f32(gid.x) + 0.5;
  let ypx = f32(gid.y) + 0.5;
  let nx = (xpx - P.cx) / P.ax; // [-1,1] on ellipse
  let ny = (ypx - P.cy) / P.ay;
  let r  = sqrt(nx*nx + ny*ny);
  let th = atan2(ny, nx);

  // Outside handling
  var atten: f32 = 1.0;
  if (P.outside_behavior == 0u) {
    if (r > 1.0) {
      // zero correction outside
      // write-through without change
      return;
    }
  } else if (P.outside_behavior == 1u) {
    atten = edge_attenuation(r, P.softness);
  } else {
    // hold: apply everywhere (atten=1)
    atten = 1.0;
  }

  // Zernike basis (unnormalized conventional forms)
  let r2 = r * r;
  let cos_t = cos(th);
  let sin_t = sin(th);
  let cos2t = cos(2.0 * th);
  let sin2t = sin(2.0 * th);

  // 0..7 as documented above
  let b0 = r * cos_t;                 // tipX
  let b1 = r * sin_t;                 // tiltY
  let b2 = 2.0 * r2 - 1.0;            // defocus
  let b3 = r2 * cos2t;                // astig0
  let b4 = r2 * sin2t;                // astig45
  let b5 = (3.0 * r2 * r - 2.0 * r) * cos_t; // comaX
  let b6 = (3.0 * r2 * r - 2.0 * r) * sin_t; // comaY
  let b7 = 6.0 * r2 * r2 - 6.0 * r2 + 1.0;   // spherical

  // Combine with coefficients
  var dphi = atten * (
    coeffs.c[0] * b0 +
    coeffs.c[1] * b1 +
    coeffs.c[2] * b2 +
    coeffs.c[3] * b3 +
    coeffs.c[4] * b4 +
    coeffs.c[5] * b5 +
    coeffs.c[6] * b6 +
    coeffs.c[7] * b7
  );

  // Clamp Dphi and apply phasor
  let maxc = P.max_correction;
  dphi = clamp(dphi, -maxc, maxc);

  let c = cos(dphi);
  let s = sin(dphi);
  let reNew = re * c - im * s;
  let imNew = re * s + im * c;

  reBuf[i] = reNew;
  imBuf[i] = imNew;
}
