#!/usr/bin/env node
/**
 * Shader Code Generation Script
 * Generates WGSL shaders and TypeScript interfaces from configuration
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface ShaderConfig {
    presets: Record<string, PresetConfig>;
    displays: Record<string, DisplayConfig>;
    uniformStructs: Record<string, UniformStruct>;
    lookupTables: Record<string, LookupTable>;
}

interface PresetConfig {
    name: string;
    description: string;
    constants: Record<string, any>;
}

interface DisplayConfig {
    [key: string]: any;
}

interface UniformStruct {
    fields: Array<{
        name: string;
        type: string;
    }>;
}

interface LookupTable {
    size?: number;
    sizes?: number[];
    generator: string;
}

class ShaderGenerator {
    private config!: ShaderConfig;
    private outputDir: string;
    private shaderTemplateDir: string;
    
    constructor() {
        this.outputDir = path.join(__dirname, '../../frontend/shaders/generated');
        this.shaderTemplateDir = path.join(__dirname, '../../frontend/shaders');
    }
    
    async run(preset: string = 'balanced'): Promise<void> {
        console.log(`🔧 Generating shaders for preset: ${preset}`);
        
        // Load configuration
        await this.loadConfig();
        
        // Create output directory
        await fs.mkdir(this.outputDir, { recursive: true });
        
        // Generate files
        await this.generateUniformStructs();
        await this.generateLookupTables();
        await this.generateShaders(preset);
        await this.generateTypeScriptDefinitions();
        
        // Validate generated shaders
        await this.validateShaders();
        
        console.log('✅ Shader generation complete!');
    }
    
    private async loadConfig(): Promise<void> {
        const configPath = path.join(__dirname, 'shader-config.json');
        const configData = await fs.readFile(configPath, 'utf-8');
        this.config = JSON.parse(configData);
    }
    
    /**
     * Generate uniform struct definitions
     */
    private async generateUniformStructs(): Promise<void> {
        let wgslStructs = '// Auto-generated uniform structures\n\n';
        
        for (const [name, struct] of Object.entries(this.config.uniformStructs)) {
            wgslStructs += `struct ${name} {\n`;
            
            for (const field of struct.fields) {
                wgslStructs += `    ${field.name}: ${field.type},\n`;
            }
            
            wgslStructs += `}\n\n`;
        }
        
        await fs.writeFile(
            path.join(this.outputDir, 'uniforms.wgsl'),
            wgslStructs
        );
    }
    
    /**
     * Generate lookup tables
     */
    private async generateLookupTables(): Promise<void> {
        let tables = '// Auto-generated lookup tables\n\n';
        
        // Cos/Sin LUT
        if (this.config.lookupTables.cos_sin_lut) {
            const size = this.config.lookupTables.cos_sin_lut.size!;
            tables += `const COS_SIN_LUT_SIZE: u32 = ${size}u;\n`;
            tables += `const COS_LUT: array<f32, ${size}> = array<f32, ${size}>(\n`;
            
            for (let i = 0; i < size; i++) {
                const angle = (i / size) * 2 * Math.PI;
                tables += `    ${Math.cos(angle).toFixed(8)},\n`;
            }
            
            tables += ');\n\n';
            
            tables += `const SIN_LUT: array<f32, ${size}> = array<f32, ${size}>(\n`;
            
            for (let i = 0; i < size; i++) {
                const angle = (i / size) * 2 * Math.PI;
                tables += `    ${Math.sin(angle).toFixed(8)},\n`;
            }
            
            tables += ');\n\n';
        }
        
        await fs.writeFile(
            path.join(this.outputDir, 'lookup_tables.wgsl'),
            tables
        );
    }
    
    /**
     * Generate shaders from templates
     */
    private async generateShaders(preset: string): Promise<void> {
        const presetConfig = this.config.presets[preset];
        if (!presetConfig) {
            throw new Error(`Unknown preset: ${preset}`);
        }
        
        // Get all .wgsl files in shader directory
        const shaderFiles = await fs.readdir(this.shaderTemplateDir);
        const wgslFiles = shaderFiles.filter(f => f.endsWith('.wgsl'));
        
        for (const file of wgslFiles) {
            const templatePath = path.join(this.shaderTemplateDir, file);
            const template = await fs.readFile(templatePath, 'utf-8');
            
            // Replace placeholders with preset values
            let generated = template;
            
            // Replace constants
            for (const [key, value] of Object.entries(presetConfig.constants)) {
                const regex = new RegExp(`{{${key}}}`, 'g');
                generated = generated.replace(regex, String(value));
            }
            
            // Replace override declarations
            generated = generated.replace(
                /override\s+(\w+)\s*:\s*(\w+)\s*;/g,
                (match, name, type) => {
                    if (presetConfig.constants[name] !== undefined) {
                        return `const ${name}: ${type} = ${presetConfig.constants[name]};`;
                    }
                    return match;
                }
            );
            
            // Add includes
            if (generated.includes('// INCLUDE: uniforms')) {
                const uniforms = await fs.readFile(
                    path.join(this.outputDir, 'uniforms.wgsl'),
                    'utf-8'
                );
                generated = generated.replace('// INCLUDE: uniforms', uniforms);
            }
            
            if (generated.includes('// INCLUDE: lookup_tables')) {
                const tables = await fs.readFile(
                    path.join(this.outputDir, 'lookup_tables.wgsl'),
                    'utf-8'
                );
                generated = generated.replace('// INCLUDE: lookup_tables', tables);
            }
            
            // Write generated shader
            const outputPath = path.join(this.outputDir, `${preset}_${file}`);
            await fs.writeFile(outputPath, generated);
            
            console.log(`  Generated: ${preset}_${file}`);
        }
    }
    
    /**
     * Generate TypeScript definitions
     */
    private async generateTypeScriptDefinitions(): Promise<void> {
        let tsDefinitions = '// Auto-generated TypeScript definitions\n\n';
        
        // Generate preset type
        tsDefinitions += 'export type ShaderPreset = ';
        tsDefinitions += Object.keys(this.config.presets)
            .map(k => `'${k}'`)
            .join(' | ');
        tsDefinitions += ';\n\n';
        
        // Generate uniform interfaces
        for (const [name, struct] of Object.entries(this.config.uniformStructs)) {
            tsDefinitions += `export interface ${name} {\n`;
            
            for (const field of struct.fields) {
                const tsType = this.wgslToTsType(field.type);
                tsDefinitions += `    ${field.name}: ${tsType};\n`;
            }
            
            tsDefinitions += `}\n\n`;
            
            // Calculate byte size
            const byteSize = struct.fields.reduce((sum, field) => {
                return sum + this.getTypeSize(field.type);
            }, 0);
            
            tsDefinitions += `export const ${name}_SIZE = ${byteSize}; // bytes\n\n`;
        }
        
        // Generate preset constants
        tsDefinitions += 'export const SHADER_PRESETS = {\n';
        for (const [key, preset] of Object.entries(this.config.presets)) {
            tsDefinitions += `    ${key}: {\n`;
            tsDefinitions += `        name: '${preset.name}',\n`;
            tsDefinitions += `        description: '${preset.description}',\n`;
            tsDefinitions += `        constants: ${JSON.stringify(preset.constants, null, 12)}\n`;
            tsDefinitions += '    },\n';
        }
        tsDefinitions += '} as const;\n\n';
        
        // Generate display configurations
        tsDefinitions += 'export const DISPLAY_CONFIGS = {\n';
        for (const [key, config] of Object.entries(this.config.displays)) {
            tsDefinitions += `    '${key}': ${JSON.stringify(config, null, 8)},\n`;
        }
        tsDefinitions += '} as const;\n';
        
        await fs.writeFile(
            path.join(this.outputDir, 'shader-types.ts'),
            tsDefinitions
        );
    }
    
    /**
     * Validate generated shaders
     */
    private async validateShaders(): Promise<void> {
        console.log('\n🔍 Validating shaders...');
        
        // Try to find wgsl-analyzer
        try {
            await execAsync('wgsl-analyzer --version');
            
            // Validate each generated shader
            const files = await fs.readdir(this.outputDir);
            const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
            
            for (const file of wgslFiles) {
                const filePath = path.join(this.outputDir, file);
                
                try {
                    await execAsync(`wgsl-analyzer "${filePath}"`);
                    console.log(`  ✅ ${file}`);
                } catch (error: any) {
                    console.error(`  ❌ ${file}: ${error.message}`);
                }
            }
        } catch {
            console.log('  ⚠️  wgsl-analyzer not found. Install with: npm install -g wgsl-analyzer');
        }
    }
    
    private wgslToTsType(wgslType: string): string {
        const typeMap: Record<string, string> = {
            'f32': 'number',
            'f16': 'number',
            'u32': 'number',
            'i32': 'number',
            'bool': 'boolean'
        };
        
        return typeMap[wgslType] || 'any';
    }
    
    private getTypeSize(wgslType: string): number {
        const sizeMap: Record<string, number> = {
            'f32': 4,
            'f16': 2,
            'u32': 4,
            'i32': 4,
            'bool': 4
        };
        
        return sizeMap[wgslType] || 4;
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const preset = args[0] || 'balanced';
    
    const generator = new ShaderGenerator();
    
    try {
        await generator.run(preset);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

export { ShaderGenerator };
