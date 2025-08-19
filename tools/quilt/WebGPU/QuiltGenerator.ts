// QuiltGenerator.ts - Shader quilt generation for holographic displays

// Note: fs and path imports removed for browser compatibility
// File operations are only needed for CLI usage which is commented out

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

fn sampleQuilt(texture: texture_2d<f32>, sampler: sampler, viewIndex: u32, uv: vec2<f32>) -> vec4<f32> {
    let quiltUV = getQuiltCoords(viewIndex, uv);
    return textureSample(texture, sampler, quiltUV);
}
`;
    }
    
    /**
     * Generate TypeScript interface for quilt parameters
     */
    generateTypeScript(): string {
        const { rows, columns, tileWidth, tileHeight } = this.config;
        
        return `// Auto-generated quilt parameters
export interface QuiltParams {
    rows: number;
    columns: number;
    tileWidth: number;
    tileHeight: number;
    totalViews: number;
    quiltWidth: number;
    quiltHeight: number;
}

export const QUILT_PARAMS: QuiltParams = {
    rows: ${rows},
    columns: ${columns},
    tileWidth: ${tileWidth},
    tileHeight: ${tileHeight},
    totalViews: ${rows * columns},
    quiltWidth: ${columns * tileWidth},
    quiltHeight: ${rows * tileHeight}
};

export function getQuiltUV(viewIndex: number, u: number, v: number): [number, number] {
    const row = Math.floor(viewIndex / ${columns});
    const col = viewIndex % ${columns};
    
    const quiltU = (col + u) / ${columns};
    const quiltV = (row + v) / ${rows};
    
    return [quiltU, quiltV];
}
`;
    }
    
    /**
     * Sync shaders with current configuration
     * Note: This method is only for Node.js CLI usage, not browser
     */
    /*
    async syncShaders(outputDir: string): Promise<void> {
        // Ensure output directory exists
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        // Generate shader code
        const shaderCode = this.generateShaderCode();
        const shaderPath = path.join(outputDir, 'quilt_generated.wgsl');
        fs.writeFileSync(shaderPath, shaderCode);
        console.log(`Generated shader: ${shaderPath}`);
        
        // Generate TypeScript interface
        const tsCode = this.generateTypeScript();
        const tsPath = path.join(outputDir, 'quiltParams.ts');
        fs.writeFileSync(tsPath, tsCode);
        console.log(`Generated TypeScript: ${tsPath}`);
    }
    */
}

// CLI usage - commented out for browser compatibility
// This code is only needed when running as a Node.js script
/*
// import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if running as main module
if (process.argv[1] === __filename) {
    const generator = new QuiltGenerator();
    const outputDir = path.resolve(__dirname, '../../frontend/lib/webgpu/generated');
    
    generator.syncShaders(outputDir).then(() => {
        console.log('Quilt shader sync complete!');
    }).catch(err => {
        console.error('Failed to sync shaders:', err);
        process.exit(1);
    });
}
*/

export default QuiltGenerator;
