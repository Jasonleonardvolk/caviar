/**
 * WebGPU Quilt Generator
 * =======================
 * WGSL-based multi-view quilt generation for holographic displays.
 * Generates 45-view quilts with mesh-driven phase parameters.
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface QuiltConfig {
    views: number;
    rows: number;
    cols: number;
    width: number;
    height: number;
    viewCone: number;
    depthScale: number;
    meshIntegration: boolean;
}

const DEFAULT_CONFIG: QuiltConfig = {
    views: 45,
    rows: 9,
    cols: 5,
    width: 3840,
    height: 2160,
    viewCone: 40,
    depthScale: 1.0,
    meshIntegration: true
};

// ============================================================================
// WGSL SHADERS
// ============================================================================

const QUILT_VERTEX_SHADER = `
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) viewIndex: f32,
};

struct Uniforms {
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    modelMatrix: mat4x4<f32>,
    viewIndex: f32,
    time: f32,
    coherence: f32,
    meshPhase: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Apply mesh-driven deformation
    var pos = input.position;
    let meshWave = sin(pos.x * 10.0 + uniforms.time) * uniforms.meshPhase * 0.1;
    pos.y += meshWave;
    
    // Transform to world space
    let worldPos = (uniforms.modelMatrix * vec4<f32>(pos, 1.0)).xyz;
    
    // Transform to clip space
    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * vec4<f32>(worldPos, 1.0);
    output.worldPos = worldPos;
    output.normal = normalize((uniforms.modelMatrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    output.viewIndex = uniforms.viewIndex;
    
    return output;
}
`;

const QUILT_FRAGMENT_SHADER = `
struct FragmentInput {
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) viewIndex: f32,
};

struct Uniforms {
    cameraPosition: vec3<f32>,
    time: f32,
    coherence: f32,
    meshPhase: f32,
    baseColor: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var meshTexture: texture_2d<f32>;
@group(0) @binding(3) var meshSampler: sampler;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Calculate view direction
    let viewDir = normalize(uniforms.cameraPosition - input.worldPos);
    
    // Fresnel effect
    let fresnel = pow(1.0 - dot(viewDir, input.normal), 2.0);
    
    // Sample mesh texture
    let meshColor = textureSample(meshTexture, meshSampler, input.uv);
    
    // Holographic effect
    var color = uniforms.baseColor * (0.5 + 0.5 * uniforms.coherence);
    color = mix(color, meshColor.rgb, meshColor.a * 0.5);
    
    // Add view-dependent color shift
    let viewShift = sin(input.viewIndex * 0.2) * 0.1;
    color.r += viewShift;
    color.b -= viewShift;
    
    // Scanline effect
    let scanline = sin(input.uv.y * 300.0 + uniforms.time * 10.0) * 0.04;
    color += vec3<f32>(scanline);
    
    // Fresnel glow
    color += vec3<f32>(0.0, 1.0, 1.0) * fresnel * uniforms.coherence;
    
    // Output with transparency
    return vec4<f32>(color, 0.8 + fresnel * 0.2);
}
`;

// ============================================================================
// QUILT GENERATOR
// ============================================================================

export class WebGPUQuiltGenerator {
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    private pipeline: GPURenderPipeline | null = null;
    private canvas: HTMLCanvasElement;
    private config: QuiltConfig;
    
    // Buffers
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    
    // Textures
    private quiltTexture: GPUTexture | null = null;
    private meshTexture: GPUTexture | null = null;
    
    // State
    private state = {
        initialized: false,
        rendering: false,
        currentView: 0,
        time: 0,
        coherence: 1.0,
        meshPhase: 0,
        meshContext: null as any
    };
    
    constructor(canvas: HTMLCanvasElement, config?: Partial<QuiltConfig>) {
        this.canvas = canvas;
        this.config = { ...DEFAULT_CONFIG, ...config };
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    async init(): Promise<boolean> {
        // Check WebGPU support
        if (!navigator.gpu) {
            console.error('[WebGPUQuiltGenerator] WebGPU not supported');
            return false;
        }
        
        // Request adapter and device
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error('[WebGPUQuiltGenerator] No GPU adapter found');
            return false;
        }
        
        this.device = await adapter.requestDevice();
        
        // Setup canvas context
        this.context = this.canvas.getContext('webgpu');
        if (!this.context) {
            console.error('[WebGPUQuiltGenerator] Failed to get WebGPU context');
            return false;
        }
        
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: format,
            alphaMode: 'premultiplied'
        });
        
        // Initialize resources
        await this.initPipeline(format);
        this.initBuffers();
        this.initTextures();
        this.initBindGroup();
        
        this.state.initialized = true;
        console.log('[WebGPUQuiltGenerator] Initialized with WebGPU');
        
        return true;
    }
    
    private async initPipeline(format: GPUTextureFormat) {
        if (!this.device) return;
        
        // Create shader modules
        const vertexShader = this.device.createShaderModule({
            label: 'Quilt Vertex Shader',
            code: QUILT_VERTEX_SHADER
        });
        
        const fragmentShader = this.device.createShaderModule({
            label: 'Quilt Fragment Shader',
            code: QUILT_FRAGMENT_SHADER
        });
        
        // Define pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [
                this.device.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.VERTEX,
                            buffer: { type: 'uniform' }
                        },
                        {
                            binding: 1,
                            visibility: GPUShaderStage.FRAGMENT,
                            buffer: { type: 'uniform' }
                        },
                        {
                            binding: 2,
                            visibility: GPUShaderStage.FRAGMENT,
                            texture: { sampleType: 'float' }
                        },
                        {
                            binding: 3,
                            visibility: GPUShaderStage.FRAGMENT,
                            sampler: {}
                        }
                    ]
                })
            ]
        });
        
        // Create render pipeline
        this.pipeline = this.device.createRenderPipeline({
            label: 'Quilt Render Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: vertexShader,
                entryPoint: 'main',
                buffers: [{
                    arrayStride: 32, // 3 pos + 3 normal + 2 uv = 8 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
                        { shaderLocation: 2, offset: 24, format: 'float32x2' }  // uv
                    ]
                }]
            },
            fragment: {
                module: fragmentShader,
                entryPoint: 'main',
                targets: [{
                    format: format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });
    }
    
    private initBuffers() {
        if (!this.device) return;
        
        // Create a torus knot geometry for demonstration
        const { vertices, indices } = this.generateTorusKnot();
        
        // Create vertex buffer
        this.vertexBuffer = this.device.createBuffer({
            label: 'Vertex Buffer',
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices.buffer as ArrayBuffer.buffer);
        
        // Create index buffer
        this.indexBuffer = this.device.createBuffer({
            label: 'Index Buffer',
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, indices.buffer as ArrayBuffer.buffer);
        
        // Create uniform buffer
        const uniformSize = 256; // Enough for matrices and uniforms
        this.uniformBuffer = this.device.createBuffer({
            label: 'Uniform Buffer',
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }
    
    private initTextures() {
        if (!this.device) return;
        
        // Create quilt texture
        this.quiltTexture = this.device.createTexture({
            label: 'Quilt Texture',
            size: {
                width: this.config.width,
                height: this.config.height,
                depthOrArrayLayers: 1
            },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | 
                   GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.COPY_SRC
        });
        
        // Create placeholder mesh texture
        this.meshTexture = this.device.createTexture({
            label: 'Mesh Texture',
            size: { width: 512, height: 512, depthOrArrayLayers: 1 },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        
        // Initialize with gradient
        this.updateMeshTexture();
    }
    
    private initBindGroup() {
        if (!this.device || !this.pipeline) return;
        
        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear'
        });
        
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer! } },
                { binding: 1, resource: { buffer: this.uniformBuffer!, offset: 128 } },
                { binding: 2, resource: this.meshTexture!.createView() },
                { binding: 3, resource: sampler }
            ]
        });
    }
    
    // ========================================================================
    // GEOMETRY GENERATION
    // ========================================================================
    
    private generateTorusKnot(): { vertices: Float32Array, indices: Uint16Array } {
        const p = 2, q = 3;
        const segments = 64;
        const tube = 16;
        const radius = 100;
        const tubeRadius = 30;
        
        const vertices: number[] = [];
        const indices: number[] = [];
        
        for (let i = 0; i <= segments; i++) {
            const u = (i / segments) * p * Math.PI * 2;
            const p1 = this.getTorusKnotPoint(u, q, p, radius);
            const p2 = this.getTorusKnotPoint(u + 0.01, q, p, radius);
            
            const T = new Float32Array([
                p2[0] - p1[0],
                p2[1] - p1[1],
                p2[2] - p1[2]
            ]);
            
            const N = new Float32Array([
                p2[0] + p1[0],
                p2[1] + p1[1],
                p2[2] + p1[2]
            ]);
            
            const B = this.cross(T, N);
            this.normalize(B);
            
            const N2 = this.cross(B, T);
            this.normalize(N2);
            
            for (let j = 0; j <= tube; j++) {
                const v = (j / tube) * Math.PI * 2;
                const cx = -tubeRadius * Math.cos(v);
                const cy = tubeRadius * Math.sin(v);
                
                // Position
                vertices.push(
                    p1[0] + cx * N2[0] + cy * B[0],
                    p1[1] + cx * N2[1] + cy * B[1],
                    p1[2] + cx * N2[2] + cy * B[2]
                );
                
                // Normal
                vertices.push(
                    N2[0] * Math.cos(v) + B[0] * Math.sin(v),
                    N2[1] * Math.cos(v) + B[1] * Math.sin(v),
                    N2[2] * Math.cos(v) + B[2] * Math.sin(v)
                );
                
                // UV
                vertices.push(i / segments, j / tube);
            }
        }
        
        // Generate indices
        for (let i = 0; i < segments; i++) {
            for (let j = 0; j < tube; j++) {
                const a = (tube + 1) * i + j;
                const b = (tube + 1) * (i + 1) + j;
                const c = (tube + 1) * (i + 1) + j + 1;
                const d = (tube + 1) * i + j + 1;
                
                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }
        
        return {
            vertices: new Float32Array(vertices),
            indices: new Uint16Array(indices)
        };
    }
    
    private getTorusKnotPoint(u: number, q: number, p: number, radius: number): Float32Array {
        const cu = Math.cos(u);
        const su = Math.sin(u);
        const quOverP = (q / p) * u;
        const cs = Math.cos(quOverP);
        
        return new Float32Array([
            radius * (2 + cs) * 0.5 * cu,
            radius * (2 + cs) * su * 0.5,
            radius * Math.sin(quOverP) * 0.5
        ]);
    }
    
    private cross(a: Float32Array, b: Float32Array): Float32Array {
        return new Float32Array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]);
    }
    
    private normalize(v: Float32Array) {
        const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 0) {
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        }
    }
    
    // ========================================================================
    // RENDERING
    // ========================================================================
    
    async renderQuilt(): Promise<GPUTexture | null> {
        if (!this.state.initialized || !this.device || !this.pipeline) {
            return null;
        }
        
        this.state.rendering = true;
        
        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();
        
        // Render each view
        const viewsPerRow = this.config.cols;
        const viewWidth = this.config.width / this.config.cols;
        const viewHeight = this.config.height / this.config.rows;
        
        for (let viewIndex = 0; viewIndex < this.config.views; viewIndex++) {
            const row = Math.floor(viewIndex / viewsPerRow);
            const col = viewIndex % viewsPerRow;
            
            // Update uniforms for this view
            this.updateUniforms(viewIndex);
            
            // Begin render pass
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.quiltTexture!.createView(),
                    loadOp: viewIndex === 0 ? 'clear' : 'load',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 }
                }],
                depthStencilAttachment: {
                    view: this.createDepthTexture().createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store'
                }
            });
            
            // Set viewport for this view
            renderPass.setViewport(
                col * viewWidth,
                row * viewHeight,
                viewWidth,
                viewHeight,
                0, 1
            );
            
            // Draw
            renderPass.setPipeline(this.pipeline);
            renderPass.setBindGroup(0, this.bindGroup!);
            renderPass.setVertexBuffer(0, this.vertexBuffer!);
            renderPass.setIndexBuffer(this.indexBuffer!, 'uint16');
            renderPass.drawIndexed(this.indexBuffer!.size / 2);
            
            renderPass.end();
        }
        
        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
        
        this.state.rendering = false;
        
        return this.quiltTexture;
    }
    
    private createDepthTexture(): GPUTexture {
        return this.device!.createTexture({
            size: {
                width: this.config.width,
                height: this.config.height,
                depthOrArrayLayers: 1
            },
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
    
    private updateUniforms(viewIndex: number) {
        if (!this.device) return;
        
        // Calculate view matrix for this index
        const angle = (viewIndex / (this.config.views - 1) - 0.5) * 
                     this.config.viewCone * Math.PI / 180;
        
        // Create matrices (simplified)
        const viewMatrix = this.createViewMatrix(angle);
        const projectionMatrix = this.createProjectionMatrix();
        const modelMatrix = this.createModelMatrix();
        
        // Pack uniforms
        const uniforms = new Float32Array(64); // 16 * 4 for matrices
        uniforms.set(viewMatrix, 0);
        uniforms.set(projectionMatrix, 16);
        uniforms.set(modelMatrix, 32);
        uniforms[48] = viewIndex;
        uniforms[49] = this.state.time;
        uniforms[50] = this.state.coherence;
        uniforms[51] = this.state.meshPhase;
        
        // Write to buffer
        this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniforms.buffer);
    }
    
    private createViewMatrix(angle: number): Float32Array {
        const radius = 500;
        const eye = [Math.sin(angle) * radius, 0, Math.cos(angle) * radius];
        const center = [0, 0, 0];
        const up = [0, 1, 0];
        
        // Simplified lookAt matrix
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            -eye[0], -eye[1], -eye[2], 1
        ]);
    }
    
    private createProjectionMatrix(): Float32Array {
        // Simplified perspective matrix
        const fov = 45 * Math.PI / 180;
        const aspect = this.config.width / this.config.height;
        const near = 0.1;
        const far = 1000;
        
        const f = 1 / Math.tan(fov / 2);
        
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0
        ]);
    }
    
    private createModelMatrix(): Float32Array {
        // Rotation matrix
        const angle = this.state.time;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        
        return new Float32Array([
            c, 0, s, 0,
            0, 1, 0, 0,
            -s, 0, c, 0,
            0, 0, 0, 1
        ]);
    }
    
    // ========================================================================
    // MESH INTEGRATION
    // ========================================================================
    
    updateMeshContext(meshContext: any) {
        this.state.meshContext = meshContext;
        
        // Update mesh phase based on context
        if (meshContext && meshContext.coherence) {
            this.state.meshPhase = meshContext.coherence;
        }
        
        // Update texture if needed
        if (this.config.meshIntegration) {
            this.updateMeshTexture(meshContext);
        }
    }
    
    private updateMeshTexture(meshContext?: any) {
        if (!this.device || !this.meshTexture) return;
        
        // Generate texture data from mesh context
        const width = 512;
        const height = 512;
        const data = new Uint8Array(width * height * 4);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                
                // Create gradient based on mesh coherence
                const coherence = meshContext?.coherence || this.state.coherence;
                
                data[i] = x / width * 255 * (1 - coherence);     // R
                data[i + 1] = y / height * 255 * coherence;      // G
                data[i + 2] = 255 * coherence;                    // B
                data[i + 3] = 255;                                // A
            }
        }
        
        // Write to texture
        this.device.queue.writeTexture(
            { texture: this.meshTexture },
            data,
            { bytesPerRow: width * 4 },
            { width, height, depthOrArrayLayers: 1 }
        );
    }
    
    // ========================================================================
    // API METHODS
    // ========================================================================
    
    setCoherence(coherence: number) {
        this.state.coherence = Math.max(0, Math.min(1, coherence));
    }
    
    animate() {
        this.state.time += 0.01;
        this.renderQuilt();
    }
    
    async getQuiltImage(): Promise<Blob | null> {
        // Render quilt
        const texture = await this.renderQuilt();
        if (!texture) return null;
        
        // Convert to blob (would need canvas readback)
        // This is simplified - actual implementation would read GPU texture
        return new Blob([], { type: 'image/png' });
    }
    
    // ========================================================================
    // LIFECYCLE
    // ========================================================================
    
    destroy() {
        // Clean up GPU resources
        if (this.vertexBuffer) this.vertexBuffer.destroy();
        if (this.indexBuffer) this.indexBuffer.destroy();
        if (this.uniformBuffer) this.uniformBuffer.destroy();
        if (this.quiltTexture) this.quiltTexture.destroy();
        if (this.meshTexture) this.meshTexture.destroy();
        
        this.state.initialized = false;
        
        console.log('[WebGPUQuiltGenerator] Destroyed');
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default WebGPUQuiltGenerator;

export async function createQuiltGenerator(
    canvas: HTMLCanvasElement,
    config?: Partial<QuiltConfig>
): Promise<WebGPUQuiltGenerator | null> {
    const generator = new WebGPUQuiltGenerator(canvas, config);
    const success = await generator.init();
    
    if (!success) {
        console.error('[WebGPUQuiltGenerator] Failed to initialize');
        return null;
    }
    
    return generator;
}

