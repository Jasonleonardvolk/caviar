// phase_field_to_views.wgsl - Compute shader to generate quilt views from complex field
// This is the hook point for the holographic pipeline integration

struct ViewParams {
  numViews: u32,
  cols: u32,
  rows: u32,
  tileW: u32,
  tileH: u32,
  propagationDistance: f32,
  wavelength: f32,
  viewAngleStep: f32,
}

@group(0) @binding(0) var<uniform> params: ViewParams;
@group(0) @binding(1) var<storage, read> fieldReal: array<f32>;
@group(0) @binding(2) var<storage, read> fieldImag: array<f32>;
@group(0) @binding(3) var outputQuilt: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const WORKGROUP_SIZE: u32 = 8u;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let pixelCoord = vec2<u32>(global_id.xy);
  
  // Determine which view and which pixel within that view
  let quiltW = params.cols * params.tileW;
  let quiltH = params.rows * params.tileH;
  
  if (pixelCoord.x >= quiltW || pixelCoord.y >= quiltH) {
    return;
  }
  
  // Figure out which tile we're in
  let tileX = pixelCoord.x / params.tileW;
  let tileY = pixelCoord.y / params.tileH;
  let viewIndex = tileY * params.cols + tileX;
  
  if (viewIndex >= params.numViews) {
    return;
  }
  
  // Local coordinates within tile
  let localX = pixelCoord.x % params.tileW;
  let localY = pixelCoord.y % params.tileH;
  
  // Apply per-view phase shift (simulating different viewing angles)
  let viewAngle = f32(viewIndex - params.numViews / 2u) * params.viewAngleStep;
  let k = 2.0 * PI / params.wavelength;
  let phaseShift = k * sin(viewAngle) * f32(localX) * 0.001; // Convert to appropriate units
  
  // Sample the complex field at this position
  let fieldIdx = localY * params.tileW + localX;
  if (fieldIdx < arrayLength(&fieldReal)) {
    let real = fieldReal[fieldIdx];
    let imag = fieldImag[fieldIdx];
    
    // Apply view-specific phase shift
    let cosPhase = cos(phaseShift);
    let sinPhase = sin(phaseShift);
    let newReal = real * cosPhase - imag * sinPhase;
    let newImag = real * sinPhase + imag * cosPhase;
    
    // Convert complex field to intensity (for display)
    let intensity = sqrt(newReal * newReal + newImag * newImag);
    
    // Write to quilt texture
    textureStore(outputQuilt, pixelCoord, vec4<f32>(intensity, intensity, intensity, 1.0));
  }
}