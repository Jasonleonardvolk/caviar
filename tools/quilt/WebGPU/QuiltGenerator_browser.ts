// QuiltGenerator.ts - Browser-safe version for holographic displays
// This version removes Node.js-specific code for browser compatibility

export interface QuiltConfig {
    rows: number;
    columns: number;
    tileWidth: number;
    tileHeight: number;
    format: 'rgba8' | 'rgba16' | 'rgba32';
}

export class QuiltGenerator {
    private config: QuiltConfig;
    
    constructor(config: Partial<QuiltConfig> = {}) {
        this.config = {
            rows: config.rows || 8,
            columns: config.columns || 6,
            tileWidth: config.tileWidth || 512,
            tileHeight: config.tileHeight || 512,
            format: config.format || 'rgba8'
        };
    }
    
    /**
     * Generate quilt layout for shaders
     */
    generateShaderCode(): string {
        const { rows, columns, tileWidth, tileHeight } = this.config;
        
        return `
// Auto-generated quilt configuration
const QUILT_ROWS: u32 = ${rows}u;
const QUILT_COLS: u32 = ${columns}u;
const TILE_WIDTH: u32 = ${tileWidth}u;
const TILE_HEIGHT: u32 = ${tileHeight}u;
const TOTAL_VIEWS: u32 = ${rows * columns}u;

fn getQuiltCoords(viewIndex: u32, uv: vec2<f32>) -> vec2<f32> {
    let row = viewIndex / QUILT_COLS;
    let col = viewIndex % QUILT_COLS;
    
    let tileUV = vec2<f32>(
        (f32(col) + uv.x) / f32(QUILT_COLS),
        (f32(row) + uv.y) / f32(QUILT_ROWS)
    );
    
    return tileUV;
}

fn sampleQuiltView(texture: texture_2d<f32>, sampler: sampler, viewIndex: u32, uv: vec2<f32>) -> vec4<f32> {
    let quiltUV = getQuiltCoords(viewIndex, uv);
    return textureSample(texture, sampler, quiltUV);
}`;
    }
    
    /**
     * Generate TypeScript interface for the quilt
     */
    generateTypeScript(): string {
        const { rows, columns, tileWidth, tileHeight } = this.config;
        
        return `// Auto-generated TypeScript definitions
export interface QuiltParams {
    rows: ${rows};
    columns: ${columns};
    tileWidth: ${tileWidth};
    tileHeight: ${tileHeight};
    totalViews: ${rows * columns};
}

export const QUILT_CONFIG: QuiltParams = {
    rows: ${rows},
    columns: ${columns},
    tileWidth: ${tileWidth},
    tileHeight: ${tileHeight},
    totalViews: ${rows * columns}
};`;
    }
    
    /**
     * Get quilt configuration
     */
    getConfig(): QuiltConfig {
        return { ...this.config };
    }
    
    /**
     * Update configuration
     */
    updateConfig(config: Partial<QuiltConfig>) {
        this.config = { ...this.config, ...config };
    }
}

// Export a default instance for convenience
export const WebGPUQuiltGenerator = QuiltGenerator;

export default QuiltGenerator;