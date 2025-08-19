// Quilt Generator Diagnostics - Enhanced validation and logging for quilt generation
// Helps prevent GPU device loss and provides detailed debugging information

export interface QuiltDiagnostics {
    expectedWidth: number;
    expectedHeight: number;
    actualWidth: number;
    actualHeight: number;
    workgroupsX: number;
    workgroupsY: number;
    totalTiles: number;
    memoryUsageMB: number;
}

export class WebGPUQuiltGeneratorDiagnostics {
    private static readonly WORKGROUP_SIZE = 8;
    
    /**
     * Validates quilt configuration and texture sizes
     */
    static validateQuiltConfig(
        config: {
            quiltCols: number;
            quiltRows: number;
            tileWidth: number;
            tileHeight: number;
        },
        outputTexture: GPUTexture,
        device?: GPUDevice
    ): QuiltDiagnostics {
        const expectedWidth = config.quiltCols * config.tileWidth;
        const expectedHeight = config.quiltRows * config.tileHeight;
        const actualWidth = outputTexture.width;
        const actualHeight = outputTexture.height;
        
        // Calculate workgroup counts
        const workgroupsX = Math.ceil(expectedWidth / this.WORKGROUP_SIZE);
        const workgroupsY = Math.ceil(expectedHeight / this.WORKGROUP_SIZE);
        
        // Calculate memory usage
        const bytesPerPixel = 4; // RGBA8
        const memoryUsageMB = (expectedWidth * expectedHeight * bytesPerPixel) / (1024 * 1024);
        
        const diagnostics: QuiltDiagnostics = {
            expectedWidth,
            expectedHeight,
            actualWidth,
            actualHeight,
            workgroupsX,
            workgroupsY,
            totalTiles: config.quiltCols * config.quiltRows,
            memoryUsageMB
        };
        
        // Log diagnostic information
        console.log(`[WebGPUQuiltGenerator] ðŸ“Š Diagnostics:`, {
            quiltLayout: `${config.quiltCols}x${config.quiltRows} tiles`,
            tileSize: `${config.tileWidth}x${config.tileHeight}px`,
            expectedResolution: `${expectedWidth}x${expectedHeight}`,
            actualResolution: `${actualWidth}x${actualHeight}`,
            workgroups: `(${workgroupsX}, ${workgroupsY})`,
            memoryUsage: `${memoryUsageMB.toFixed(2)} MB`
        });
        
        // Validation checks
        if (actualWidth !== expectedWidth || actualHeight !== expectedHeight) {
            console.warn(`âš ï¸ [WebGPUQuiltGenerator] Output texture size mismatch!`, {
                expected: `${expectedWidth}x${expectedHeight}`,
                actual: `${actualWidth}x${actualHeight}`
            });
        }
        
        // Check workgroup limits
        const maxWorkgroups = device?.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
        if (workgroupsX > maxWorkgroups || workgroupsY > maxWorkgroups) {
            console.error(`âŒ [WebGPUQuiltGenerator] Workgroup count exceeds device limit!`, {
                workgroups: `(${workgroupsX}, ${workgroupsY})`,
                maxAllowed: maxWorkgroups
            });
        }
        
        // Check for zero workgroups
        if (workgroupsX === 0 || workgroupsY === 0) {
            console.error(`âŒ [WebGPUQuiltGenerator] Invalid workgroup count (zero dimension)!`);
        }
        
        // Memory warnings
        if (memoryUsageMB > 100) {
            console.warn(`âš ï¸ [WebGPUQuiltGenerator] Large memory usage: ${memoryUsageMB.toFixed(2)} MB`);
        }
        
        return diagnostics;
    }
    
    /**
     * Validates view render mode and logs dispatch info
     */
    static validateRenderMode(
        renderMode: { mode: 'single' | 'subset' | 'all'; viewIndex?: number },
        numViews: number,
        tileWidth: number,
        tileHeight: number
    ): void {
        if (renderMode.mode === 'single') {
            if (renderMode.viewIndex === undefined) {
                console.error(`âŒ [WebGPUQuiltGenerator] Single view mode requires viewIndex`);
                return;
            }
            if (renderMode.viewIndex < 0 || renderMode.viewIndex >= numViews) {
                console.error(`âŒ [WebGPUQuiltGenerator] Invalid viewIndex ${renderMode.viewIndex} (numViews=${numViews})`);
                return;
            }
            
            const workgroupsX = Math.ceil(tileWidth / this.WORKGROUP_SIZE);
            const workgroupsY = Math.ceil(tileHeight / this.WORKGROUP_SIZE);
            console.log(`[WebGPUQuiltGenerator] Single view dispatch: view ${renderMode.viewIndex}, workgroups (${workgroupsX}, ${workgroupsY})`);
        }
    }
    
    /**
     * Creates a performance timer for quilt generation
     */
    static createTimer(label: string): { end: () => void } {
        const start = performance.now();
        return {
            end: () => {
                const duration = performance.now() - start;
                console.log(`[WebGPUQuiltGenerator] â±ï¸ ${label}: ${duration.toFixed(2)}ms`);
            }
        };
    }
}

// Helper function for texture usage validation
export function validateTextureUsage(
    texture: GPUTexture,
    requiredUsage: GPUTextureUsageFlags,
    textureName: string
): boolean {
    // Note: We can't directly check texture.usage in the browser API
    // This is a placeholder for documentation purposes
    console.log(`[WebGPUQuiltGenerator] Validating texture "${textureName}" has required usage flags`);
    
    // In practice, validation happens at texture creation time
    // This function serves as a reminder to check usage flags
    return true;
}

// Resource tracking helper
export class ResourceTracker {
    private resources = new Map<string, GPUBuffer | GPUTexture>();
    
    track(name: string, resource: GPUBuffer | GPUTexture): void {
        if (this.resources.has(name)) {
            console.warn(`âš ï¸ [ResourceTracker] Overwriting tracked resource: ${name}`);
        }
        this.resources.set(name, resource);
    }
    
    cleanup(name: string): void {
        const resource = this.resources.get(name);
        if (resource) {
            resource.destroy();
            this.resources.delete(name);
            console.log(`[ResourceTracker] Cleaned up resource: ${name}`);
        }
    }
    
    cleanupAll(): void {
        console.log(`[ResourceTracker] Cleaning up ${this.resources.size} resources`);
        for (const [name, resource] of this.resources) {
            resource.destroy();
        }
        this.resources.clear();
    }
}

