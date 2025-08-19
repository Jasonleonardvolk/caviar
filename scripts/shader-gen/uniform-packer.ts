/**
 * Uniform Buffer Packing Utility
 * Generates TypeScript and WGSL definitions with proper alignment
 */

import fs from 'fs/promises';
import path from 'path';

interface FieldDefinition {
    name: string;
    type: 'f32' | 'u32' | 'i32' | 'vec2f' | 'vec3f' | 'vec4f' | 'mat4x4f';
    arraySize?: number;
}

interface StructDefinition {
    name: string;
    fields: FieldDefinition[];
}

export class UniformPacker {
    // WGSL alignment rules
    private alignmentRules: Record<string, { size: number; align: number }> = {
        'f32': { size: 4, align: 4 },
        'u32': { size: 4, align: 4 },
        'i32': { size: 4, align: 4 },
        'vec2f': { size: 8, align: 8 },
        'vec3f': { size: 12, align: 16 }, // vec3 aligns to 16!
        'vec4f': { size: 16, align: 16 },
        'mat4x4f': { size: 64, align: 16 }
    };
    
    /**
     * Generate both TypeScript and WGSL definitions
     */
    async generateDefinitions(structs: StructDefinition[]): Promise<{
        typescript: string;
        wgsl: string;
    }> {
        let typescript = '// Auto-generated uniform buffer definitions\n\n';
        let wgsl = '// Auto-generated uniform buffer structures\n\n';
        
        for (const struct of structs) {
            const packed = this.packStruct(struct);
            
            // Generate TypeScript
            typescript += this.generateTypeScript(struct, packed);
            
            // Generate WGSL
            wgsl += this.generateWGSL(struct, packed);
        }
        
        return { typescript, wgsl };
    }
    
    /**
     * Pack struct fields with proper alignment
     */
    private packStruct(struct: StructDefinition): {
        fields: Array<{
            field: FieldDefinition;
            offset: number;
            size: number;
            padding: number;
        }>;
        totalSize: number;
    } {
        const packed: any[] = [];
        let currentOffset = 0;
        
        for (const field of struct.fields) {
            const typeInfo = this.alignmentRules[field.type];
            if (!typeInfo) {
                throw new Error(`Unknown type: ${field.type}`);
            }
            
            // Calculate alignment
            const alignment = typeInfo.align;
            const padding = (alignment - (currentOffset % alignment)) % alignment;
            currentOffset += padding;
            
            // Calculate size (including array)
            const elementSize = typeInfo.size;
            const arraySize = field.arraySize || 1;
            const totalFieldSize = elementSize * arraySize;
            
            packed.push({
                field,
                offset: currentOffset,
                size: totalFieldSize,
                padding
            });
            
            currentOffset += totalFieldSize;
        }
        
        // Struct must be 16-byte aligned
        const structPadding = (16 - (currentOffset % 16)) % 16;
        const totalSize = currentOffset + structPadding;
        
        return { fields: packed, totalSize };
    }
    
    /**
     * Generate TypeScript interface and packing function
     */
    private generateTypeScript(
        struct: StructDefinition, 
        packed: any
    ): string {
        let ts = `export interface ${struct.name} {\n`;
        
        // Interface fields
        for (const field of struct.fields) {
            const tsType = this.wgslToTsType(field.type);
            if (field.arraySize) {
                ts += `    ${field.name}: ${tsType}[];\n`;
            } else {
                ts += `    ${field.name}: ${tsType};\n`;
            }
        }
        
        ts += `}\n\n`;
        
        // Size constant
        ts += `export const ${struct.name}_SIZE = ${packed.totalSize}; // bytes\n\n`;
        
        // Packing function
        ts += `export function pack${struct.name}(data: ${struct.name}): ArrayBuffer {\n`;
        ts += `    const buffer = new ArrayBuffer(${packed.totalSize});\n`;
        ts += `    const view = new DataView(buffer);\n\n`;
        
        for (const { field, offset } of packed.fields) {
            const setter = this.getDataViewSetter(field.type);
            
            if (field.arraySize) {
                ts += `    // ${field.name} array\n`;
                ts += `    for (let i = 0; i < ${field.arraySize}; i++) {\n`;
                ts += `        ${this.generateSetter(field, offset, 'i', setter)};\n`;
                ts += `    }\n\n`;
            } else {
                ts += `    // ${field.name}\n`;
                ts += `    ${this.generateSetter(field, offset, null, setter)};\n\n`;
            }
        }
        
        ts += `    return buffer;\n`;
        ts += `}\n\n`;
        
        // Unpacking function
        ts += `export function unpack${struct.name}(buffer: ArrayBuffer): ${struct.name} {\n`;
        ts += `    const view = new DataView(buffer);\n`;
        ts += `    return {\n`;
        
        for (const { field, offset } of packed.fields) {
            const getter = this.getDataViewGetter(field.type);
            
            if (field.arraySize) {
                ts += `        ${field.name}: Array.from({ length: ${field.arraySize} }, (_, i) => `;
                ts += `${this.generateGetter(field, offset, 'i', getter)}),\n`;
            } else {
                ts += `        ${field.name}: ${this.generateGetter(field, offset, null, getter)},\n`;
            }
        }
        
        ts += `    };\n`;
        ts += `}\n\n`;
        
        return ts;
    }
    
    /**
     * Generate WGSL struct definition
     */
    private generateWGSL(struct: StructDefinition, packed: any): string {
        let wgsl = `struct ${struct.name} {\n`;
        
        for (const { field, offset } of packed.fields) {
            // Add offset annotation for clarity
            wgsl += `    @align(${this.alignmentRules[field.type].align}) `;
            wgsl += `/* offset: ${offset} */ `;
            
            if (field.arraySize) {
                wgsl += `${field.name}: array<${field.type}, ${field.arraySize}>,\n`;
            } else {
                wgsl += `${field.name}: ${field.type},\n`;
            }
        }
        
        wgsl += `}\n\n`;
        
        // Add size constant
        wgsl += `// Total size: ${packed.totalSize} bytes\n\n`;
        
        return wgsl;
    }
    
    private wgslToTsType(wgslType: string): string {
        const typeMap: Record<string, string> = {
            'f32': 'number',
            'u32': 'number',
            'i32': 'number',
            'vec2f': '[number, number]',
            'vec3f': '[number, number, number]',
            'vec4f': '[number, number, number, number]',
            'mat4x4f': 'Float32Array' // 16 elements
        };
        
        return typeMap[wgslType] || 'any';
    }
    
    private getDataViewSetter(type: string): string {
        switch (type) {
            case 'f32': return 'setFloat32';
            case 'u32': return 'setUint32';
            case 'i32': return 'setInt32';
            default: return 'setFloat32';
        }
    }
    
    private getDataViewGetter(type: string): string {
        switch (type) {
            case 'f32': return 'getFloat32';
            case 'u32': return 'getUint32';
            case 'i32': return 'getInt32';
            default: return 'getFloat32';
        }
    }
    
    private generateSetter(
        field: FieldDefinition, 
        baseOffset: number, 
        indexVar: string | null,
        setter: string
    ): string {
        const typeInfo = this.alignmentRules[field.type];
        
        switch (field.type) {
            case 'f32':
            case 'u32':
            case 'i32': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 4`
                    : `${baseOffset}`;
                const value = indexVar
                    ? `data.${field.name}[${indexVar}]`
                    : `data.${field.name}`;
                return `view.${setter}(${offset}, ${value}, true)`;
            }
            
            case 'vec2f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 8`
                    : `${baseOffset}`;
                const value = indexVar
                    ? `data.${field.name}[${indexVar}]`
                    : `data.${field.name}`;
                return `view.setFloat32(${offset}, ${value}[0], true);\n` +
                       `        view.setFloat32(${offset} + 4, ${value}[1], true)`;
            }
            
            case 'vec3f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 16` // vec3 aligns to 16!
                    : `${baseOffset}`;
                const value = indexVar
                    ? `data.${field.name}[${indexVar}]`
                    : `data.${field.name}`;
                return `view.setFloat32(${offset}, ${value}[0], true);\n` +
                       `        view.setFloat32(${offset} + 4, ${value}[1], true);\n` +
                       `        view.setFloat32(${offset} + 8, ${value}[2], true)`;
            }
            
            case 'vec4f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 16`
                    : `${baseOffset}`;
                const value = indexVar
                    ? `data.${field.name}[${indexVar}]`
                    : `data.${field.name}`;
                return `view.setFloat32(${offset}, ${value}[0], true);\n` +
                       `        view.setFloat32(${offset} + 4, ${value}[1], true);\n` +
                       `        view.setFloat32(${offset} + 8, ${value}[2], true);\n` +
                       `        view.setFloat32(${offset} + 12, ${value}[3], true)`;
            }
            
            default:
                return `// TODO: ${field.type}`;
        }
    }
    
    private generateGetter(
        field: FieldDefinition, 
        baseOffset: number, 
        indexVar: string | null,
        getter: string
    ): string {
        switch (field.type) {
            case 'f32':
            case 'u32':
            case 'i32': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 4`
                    : `${baseOffset}`;
                return `view.${getter}(${offset}, true)`;
            }
            
            case 'vec2f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 8`
                    : `${baseOffset}`;
                return `[view.getFloat32(${offset}, true), ` +
                       `view.getFloat32(${offset} + 4, true)]`;
            }
            
            case 'vec3f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 16`
                    : `${baseOffset}`;
                return `[view.getFloat32(${offset}, true), ` +
                       `view.getFloat32(${offset} + 4, true), ` +
                       `view.getFloat32(${offset} + 8, true)]`;
            }
            
            case 'vec4f': {
                const offset = indexVar 
                    ? `${baseOffset} + ${indexVar} * 16`
                    : `${baseOffset}`;
                return `[view.getFloat32(${offset}, true), ` +
                       `view.getFloat32(${offset} + 4, true), ` +
                       `view.getFloat32(${offset} + 8, true), ` +
                       `view.getFloat32(${offset} + 12, true)]`;
            }
            
            default:
                return `null // TODO: ${field.type}`;
        }
    }
}

// Example usage
if (require.main === module) {
    async function example() {
        const packer = new UniformPacker();
        
        const structs: StructDefinition[] = [
            {
                name: 'CameraUniforms',
                fields: [
                    { name: 'viewMatrix', type: 'mat4x4f' },
                    { name: 'projMatrix', type: 'mat4x4f' },
                    { name: 'position', type: 'vec3f' },
                    { name: 'time', type: 'f32' }
                ]
            },
            {
                name: 'WavefieldParams',
                fields: [
                    { name: 'phase_modulation', type: 'f32' },
                    { name: 'coherence', type: 'f32' },
                    { name: 'time', type: 'f32' },
                    { name: 'scale', type: 'f32' },
                    { name: 'phases', type: 'f32', arraySize: 32 },
                    { name: 'frequencies', type: 'vec2f', arraySize: 32 }
                ]
            }
        ];
        
        const { typescript, wgsl } = await packer.generateDefinitions(structs);
        
        // Save files
        await fs.writeFile(
            path.join(__dirname, '../../frontend/lib/webgpu/generated/uniforms.ts'),
            typescript
        );
        
        await fs.writeFile(
            path.join(__dirname, '../../frontend/shaders/generated/uniforms.wgsl'),
            wgsl
        );
        
        console.log('✅ Generated uniform definitions');
    }
    
    example().catch(console.error);
}

export default UniformPacker;
