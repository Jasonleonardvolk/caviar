/**
 * Main-thread wrapper for the Basis transcoder worker
 * Handles texture format selection and worker communication
 */
let worker = null;
let sequenceId = 1;
const pendingRequests = new Map();
/**
 * Detect the best texture format for the current device
 */
export function pickTargetFormat(adapter) {
    // Check WebGPU adapter features if available
    if (adapter && adapter.features) {
        if (adapter.features.has('texture-compression-astc')) {
            return 'astc';
        }
        if (adapter.features.has('texture-compression-bc')) {
            return 'bc7';
        }
        if (adapter.features.has('texture-compression-etc2')) {
            return 'etc2';
        }
    }
    // Fallback to user agent detection
    const ua = navigator.userAgent || '';
    // iOS/macOS typically support ASTC
    if (/iPhone|iPad|Mac OS X/.test(ua) && !/Chrome/.test(ua)) {
        return 'astc';
    }
    // Windows typically supports BC7
    if (/Windows/.test(ua)) {
        return 'bc7';
    }
    // Android typically supports ETC2
    if (/Android/.test(ua)) {
        return 'etc2';
    }
    // Default to uncompressed RGBA
    return 'rgba';
}
/**
 * Ensure the worker is initialized
 */
function ensureWorker() {
    if (!worker) {
        // Create worker with module type for ES6 imports
        worker = new Worker(new URL('../workers/basisWorker.ts', import.meta.url), { type: 'module' });
        // Handle messages from worker
        worker.onmessage = (event) => {
            const result = event.data;
            const resolver = pendingRequests.get(result.id);
            if (resolver) {
                pendingRequests.delete(result.id);
                resolver(result);
            }
        };
        // Handle worker errors
        worker.onerror = (error) => {
            console.error('[BasisTranscoder] Worker error:', error);
            // Reject all pending requests
            for (const resolver of pendingRequests.values()) {
                resolver({ id: 0, ok: false, error: String(error) });
            }
            pendingRequests.clear();
        };
    }
    return worker;
}
/**
 * Send a message to the worker and wait for response
 */
async function callWorker(message) {
    const worker = ensureWorker();
    const id = sequenceId++;
    return new Promise((resolve) => {
        pendingRequests.set(id, resolve);
        const fullMessage = { ...message, id };
        // Transfer ArrayBuffer if present
        if (fullMessage.data) {
            worker.postMessage(fullMessage, [fullMessage.data]);
        }
        else {
            worker.postMessage(fullMessage);
        }
    });
}
/**
 * Initialize the Basis transcoder worker
 */
export async function initBasisWorker() {
    const result = await callWorker({ type: 'init' });
    if (!result.ok) {
        throw new Error(result.error || 'Basis initialization failed');
    }
    console.log('[BasisTranscoder] Worker initialized');
}
/**
 * Transcode a KTX2 texture to the target format
 */
export async function transcodeKTX2(buffer, target) {
    // Ensure worker is initialized
    await initBasisWorker();
    // Clone the buffer since it will be transferred
    const bufferCopy = buffer.slice(0);
    const result = await callWorker({
        type: 'transcode',
        data: bufferCopy,
        format: target
    });
    if (!result.ok) {
        throw new Error(result.error || 'Transcode failed');
    }
    if (!result.levels) {
        throw new Error('No levels returned from transcoder');
    }
    return { levels: result.levels };
}
/**
 * Dispose of the transcoder worker
 */
export async function disposeBasisWorker() {
    if (worker) {
        await callWorker({ type: 'dispose' });
        worker.terminate();
        worker = null;
        pendingRequests.clear();
        console.log('[BasisTranscoder] Worker disposed');
    }
}
/**
 * Load and transcode a KTX2 texture from a URL
 */
export async function loadAndTranscodeKTX2(url, adapter) {
    // Fetch the KTX2 file
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch KTX2: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    // Pick the best format for this device
    const format = pickTargetFormat(adapter);
    console.log(`[BasisTranscoder] Loading ${url} as ${format}`);
    // Transcode the texture
    const { levels } = await transcodeKTX2(buffer, format);
    return { levels, format };
}
/**
 * Batch load and transcode multiple KTX2 textures
 */
export async function batchLoadKTX2(urls, adapter) {
    // Initialize worker once for all textures
    await initBasisWorker();
    const format = pickTargetFormat(adapter);
    const results = new Map();
    // Process in parallel with concurrency limit
    const concurrency = 4;
    const queue = [...urls];
    const inProgress = [];
    while (queue.length > 0 || inProgress.length > 0) {
        // Start new tasks up to concurrency limit
        while (inProgress.length < concurrency && queue.length > 0) {
            const url = queue.shift();
            const task = (async () => {
                try {
                    const result = await loadAndTranscodeKTX2(url, adapter);
                    results.set(url, result);
                }
                catch (error) {
                    console.error(`[BasisTranscoder] Failed to load ${url}:`, error);
                    // Store error result
                    results.set(url, { levels: [], format });
                }
            })();
            inProgress.push(task);
        }
        // Wait for at least one task to complete
        if (inProgress.length > 0) {
            await Promise.race(inProgress);
            // Remove completed tasks
            for (let i = inProgress.length - 1; i >= 0; i--) {
                const task = inProgress[i];
                if (await Promise.race([task, Promise.resolve('pending')]) !== 'pending') {
                    inProgress.splice(i, 1);
                }
            }
        }
    }
    return results;
}
// Auto-initialize on first import in production
if (import.meta.env.PROD) {
    initBasisWorker().catch(console.error);
}
