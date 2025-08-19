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
  overscan: f32, // 0..1, expands quilt sampling a touch to hide seams
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
  // Core formula: ((x - center)/pitch) - y*tilt + subp*0.5
  let v = ((screen_px.x - C.center) / C.pitch) - (screen_px.y * C.tilt) + (subpixel * 0.5);
  return clamp(floor(v + 0.5), 0.0, C.numViews - 1.0);
}

fn tile_uv_from_view(v: f32, uv: vec2<f32>) -> vec2<f32> {
  // Map final panel uv to quilt tile uv
  let cols = C.cols;
  let rows = C.rows;
  let col  = floor(v % cols);
  let row  = floor(v / cols);

  // Optional overscan to hide tile edges
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
  // Convert normalized uv to panel pixel coords
  let screen_px = vec2<f32>(inp.uv.x * C.panelW, (1.0 - inp.uv.y) * C.panelH);

  // Sample per RGB subpixel (simple subp routing)
  // NOTE: If your panel uses different subpixel order, adjust mapping.
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