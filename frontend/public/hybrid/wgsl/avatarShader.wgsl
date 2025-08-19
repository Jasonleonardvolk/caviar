// avatarShader.wgsl
// Avatar rendering shader with animation support

struct AvatarParams {
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    time: f32,
    jawOpen: f32,
    glowIntensity: f32,
    _padding: f32,  // For 16-byte alignment
}

@group(0) @binding(0) var<uniform> params: AvatarParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPosition: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Apply jaw animation
    var animatedPosition = input.position;
    let jawMovement = sin(params.time * 15.0) * 0.03 * params.jawOpen;
    animatedPosition.y += jawMovement;
    
    // Apply breathing animation
    let breathingScale = 1.0 + sin(params.time * 1.5) * 0.002;
    animatedPosition *= breathingScale;
    
    // Transform position
    let worldPos = params.modelMatrix * vec4<f32>(animatedPosition, 1.0);
    let viewPos = params.viewMatrix * worldPos;
    output.position = params.projectionMatrix * viewPos;
    
    output.worldPosition = worldPos.xyz;
    output.normal = (params.modelMatrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate glow with pulsing effect
    let glowPulse = 0.5 + 0.5 * sin(params.time * 2.0);
    let talkingBoost = 1.0 + params.jawOpen * 0.5;
    let baseIntensity = params.glowIntensity * glowPulse * talkingBoost;
    
    // Color calculation
    let redChannel = baseIntensity * (1.0 + params.jawOpen * 0.2);
    let greenChannel = 0.2 + params.jawOpen * 0.1;
    let blueChannel = 0.8 - params.jawOpen * 0.1;
    
    var color = vec3<f32>(redChannel, greenChannel, blueChannel);
    
    // Apply simple lighting
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let ndotl = max(dot(normalize(input.normal), lightDir), 0.0);
    color *= 0.5 + 0.5 * ndotl;
    
    return vec4<f32>(color, 1.0);
}

// Compute shader variant for avatar processing
@compute @workgroup_size(8, 8, 1)
fn compute_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Placeholder compute shader functionality
    // Could be used for avatar mesh deformation, particle effects, etc.
    let index = gid.x + gid.y * 8u;
}
