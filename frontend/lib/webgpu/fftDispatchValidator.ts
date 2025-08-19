// FFT Dispatch Validator - Enhanced diagnostics for FFT compute dispatches
// This module adds validation and logging to prevent GPU device loss from invalid dispatches

export interface DispatchParams {
    x: number;
    y: number;
    z: number;
}

export class FFTDispatchValidator {
    private static maxWorkgroupsPerDimension: number = 65535;
    
    /**
     * Validates dispatch parameters and logs diagnostic information
     */
    static validateAndLog(
        kernelName: string,
        dispatch: DispatchParams,
        device?: GPUDevice
    ): boolean {
        // Update max limit from device if available
        if (device?.limits?.maxComputeWorkgroupsPerDimension) {
            this.maxWorkgroupsPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
        }
        
        // Log dispatch information
        console.log(`[FFT] Dispatching ${kernelName}: workgroups = (${dispatch.x}, ${dispatch.y}, ${dispatch.z})`);
        
        // Check for zero dimensions
        if (dispatch.x === 0 || dispatch.y === 0 || dispatch.z === 0) {
            console.error(`[FFT] ❌ Error: dispatch dimension is 0 for ${kernelName}`, dispatch);
            return false;
        }
        
        // Check for exceeding device limits
        if (dispatch.x > this.maxWorkgroupsPerDimension || 
            dispatch.y > this.maxWorkgroupsPerDimension || 
            dispatch.z > this.maxWorkgroupsPerDimension) {
            console.error(`[FFT] ❌ Error: dispatch dimensions exceed device limit (${this.maxWorkgroupsPerDimension})`, {
                kernel: kernelName,
                dispatch,
                maxAllowed: this.maxWorkgroupsPerDimension
            });
            return false;
        }
        
        // Warn about potentially large dispatches
        const totalWorkgroups = dispatch.x * dispatch.y * dispatch.z;
        if (totalWorkgroups > 1000000) {
            console.warn(`[FFT] ⚠️ Warning: Large dispatch for ${kernelName} with ${totalWorkgroups} total workgroups`);
        }
        
        return true;
    }
    
    /**
     * Calculates safe dispatch dimensions with validation
     */
    static calculateSafeDispatch(
        totalElements: number,
        workgroupSize: number,
        dimensions: 1 | 2 = 1
    ): DispatchParams {
        if (dimensions === 1) {
            const x = Math.ceil(totalElements / workgroupSize);
            return { x, y: 1, z: 1 };
        } else {
            // For 2D, try to make a square-ish dispatch
            const sqrtElements = Math.sqrt(totalElements);
            const tilesPerDim = Math.ceil(sqrtElements / workgroupSize);
            return { x: tilesPerDim, y: tilesPerDim, z: 1 };
        }
    }
}

// Export a helper function for easy integration
export function validateFFTDispatch(
    kernelName: string,
    dispatch: DispatchParams,
    device?: GPUDevice
): void {
    const isValid = FFTDispatchValidator.validateAndLog(kernelName, dispatch, device);
    if (!isValid) {
        throw new Error(`Invalid dispatch parameters for ${kernelName}`);
    }
}
