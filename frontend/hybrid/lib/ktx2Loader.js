/**
 * Minimal KTX2 loader using ktx-parse for metadata and Basis Universal transcoder for GPU upload.
 * Expects /wasm/basis_transcoder.{js,wasm} to be precached by the SW.
 * NOTE: This is a starter; wire to your actual texture upload path (WebGPU/ WebGL).
 */
import { read } from 'ktx-parse';
export async function loadKTX2(url) {
    const res = await fetch(url);
    if (!res.ok)
        throw new Error(`KTX2 fetch failed: ${url}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    const ktx = read(buf);
    const levels = ktx.levels.map(l => new Uint8Array(buf.buffer, l.offset, l.length));
    return {
        width: ktx.pixelWidth,
        height: ktx.pixelHeight,
        levels: ktx.levels.length,
        format: ktx.vkFormat?.toString() || 'UNKNOWN',
        levelsData: levels
    };
}
// Example WebGPU upload sketch (you'll adapt to your render pipeline):
export function uploadToWebGPU(device, ktx) {
    const texture = device.createTexture({
        size: { width: ktx.width, height: ktx.height, depthOrArrayLayers: 1 },
        format: 'rgba8unorm',
        mipLevelCount: ktx.levelsData.length,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    let w = ktx.width, h = ktx.height;
    ktx.levelsData.forEach((data, level) => {
        device.queue.writeTexture({ texture, mipLevel: level }, data, { bytesPerRow: Math.max(256, w * 4) }, // align; adjust if you transcode compressed formats
        { width: w, height: h, depthOrArrayLayers: 1 });
        w = Math.max(1, w >> 1);
        h = Math.max(1, h >> 1);
    });
    return texture;
}
