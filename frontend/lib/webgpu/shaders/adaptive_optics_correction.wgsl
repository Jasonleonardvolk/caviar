// adaptive_optics_correction.wgsl
// Zernike-based aberration correction with user controls
// Supports defocus + astigmatism + coma; controlled from UI sliders (no EEG, no external tracker)

struct Params {
    width: u32,
    height: u32,
    pupil_radius_px: f32,   // map screen coords to pupil
    // Zernike coefficients (user-controlled via sliders)
    c_defocus: f32,         // Z4: Defocus
    c_astig0: f32,          // Z5: Astigmatism 0°
    c_astig45: f32,         // Z6: Astigmatism 45°
    c_coma_x: f32,          // Z7: Coma X
    c_coma_y: f32,          // Z8: Coma Y
    c_spherical: f32,       // Z11: Spherical aberration
    c_trefoil_x: f32,       // Z9: Trefoil X
    c_trefoil_y: f32,       // Z10: Trefoil Y
    phase_scale: f32,       // 2π/λ
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> field: array<vec2<f32>>;

// Complex multiplication
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// Complex from phase
fn c_from_phase(phi: f32) -> vec2<f32> { 
    return vec2<f32>(cos(phi), sin(phi)); 
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let idx = gid.y * p.width + gid.x;
    
    // Convert to normalized pupil coordinates
    let cx = f32(p.width) * 0.5;
    let cy = f32(p.height) * 0.5;
    let x = (f32(gid.x) - cx) / p.pupil_radius_px;
    let y = (f32(gid.y) - cy) / p.pupil_radius_px;
    let r = sqrt(x*x + y*y);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    
    // Outside pupil → unchanged
    if (r > 1.0) { 
        return; 
    }
    
    let theta = atan2(y, x);
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    let cos_2theta = cos(2.0 * theta);
    let sin_2theta = sin(2.0 * theta);
    let cos_3theta = cos(3.0 * theta);
    let sin_3theta = sin(3.0 * theta);
    
    // Zernike polynomials (ANSI standard ordering)
    var phase = 0.0;
    
    // Z4: Defocus (2r² - 1)
    phase += p.c_defocus * (2.0 * r2 - 1.0);
    
    // Z5-Z6: Astigmatism
    phase += p.c_astig0 * r2 * cos_2theta;  // 0° astigmatism
    phase += p.c_astig45 * r2 * sin_2theta; // 45° astigmatism
    
    // Z7-Z8: Coma
    phase += p.c_coma_x * (3.0*r3 - 2.0*r) * cos_theta;
    phase += p.c_coma_y * (3.0*r3 - 2.0*r) * sin_theta;
    
    // Z9-Z10: Trefoil
    phase += p.c_trefoil_x * r3 * cos_3theta;
    phase += p.c_trefoil_y * r3 * sin_3theta;
    
    // Z11: Spherical aberration
    phase += p.c_spherical * (6.0*r4 - 6.0*r2 + 1.0);
    
    // Apply phase correction (negative to pre-correct)
    let phi = -p.phase_scale * phase;
    
    // Apply to field
    if (idx < arrayLength(&field)) {
        field[idx] = c_mul(field[idx], c_from_phase(phi));
    }
}

// Advanced: Shack-Hartmann-inspired wavefront sensing
@compute @workgroup_size(8, 8, 1)
fn measure_aberrations(@builtin(global_invocation_id) gid: vec3<u32>) {
    // This would measure the actual aberrations from the wavefront
    // For now, it's a placeholder for future auto-correction
    // Could use spot patterns or phase diversity to estimate Zernike coeffs
}