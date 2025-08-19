// WebGPU Device Context
let deviceInstance: GPUDevice | null = null;

export async function getDevice(): Promise<GPUDevice> {
    if (!deviceInstance) {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        deviceInstance = await adapter.requestDevice();
    }
    return deviceInstance;
}

// For synchronous imports, we'll need to initialize this
export let device: GPUDevice;

// Initialize on module load
(async () => {
    device = await getDevice();
})();
