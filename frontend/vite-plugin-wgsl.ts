// vite-plugin-wgsl.ts
// Vite plugin for importing WGSL shader files

import { Plugin } from 'vite';
import * as fs from 'fs';
import * as path from 'path';

export function wgslPlugin(): Plugin {
    return {
        name: 'vite-plugin-wgsl',
        
        transform(code: string, id: string) {
            if (!id.endsWith('.wgsl')) {
                return null;
            }
            
            // Read the shader file
            const shaderContent = fs.readFileSync(id, 'utf-8');
            
            // Generate ES module
            return {
                code: `export default ${JSON.stringify(shaderContent)};`,
                map: null
            };
        },
        
        // Handle HMR for shader files
        handleHotUpdate({ file, server }) {
            if (file.endsWith('.wgsl')) {
                console.log(`Shader updated: ${path.basename(file)}`);
                
                // Invalidate modules that import this shader
                const module = server.moduleGraph.getModuleById(file);
                if (module) {
                    server.moduleGraph.invalidateModule(module);
                    
                    // Trigger HMR
                    server.ws.send({
                        type: 'custom',
                        event: 'shader-update',
                        data: { file: path.basename(file) }
                    });
                }
            }
        }
    };
}

// Type declaration for WGSL imports
declare module '*.wgsl' {
    const content: string;
    export default content;
}
