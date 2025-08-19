/**
 * Minimal KTX2 loader using ktx-parse for metadata and Basis Universal transcoder for GPU upload.
 * Expects /wasm/basis_transcoder.{js,wasm} to be precached by the SW.
 * NOTE: This is a starter; wire to your actual texture upload path (WebGPU/ WebGL).
 */
import { read } from 'ktx-parse';

export async function loadKTX2(url: string): Promise<{width:number,height:number,levels:number,format:string,levelsData:Uint8Array[]}> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`KTX2 fetch failed: ${url}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const ktx = read(buf);
  // Using any cast for ktx-parse compatibility - properties may vary by version
  const levels = ktx.levels.map(l => {
    const offset = (l as any).byteOffset || (l as any).levelDataByteOffset || 0;
    const length = (l as any).byteLength || (l as any).levelDataByteLength || 0;
    return new Uint8Array(buf.buffer, offset, length);
  });
  return {
    width: ktx.pixelWidth,
    height: ktx.pixelHeight,
    levels: ktx.levels.length,
    format: ktx.vkFormat?.toString() || 'UNKNOWN',
    levelsData: levels
  };
}

// Example WebGPU upload sketch (you'll adapt to your render pipeline):
export function uploadToWebGPU(device: GPUDevice, ktx: {width:number,height:number,levelsData:Uint8Array[]}) {
  const texture = device.createTexture({
    size: { width: ktx.width, height: ktx.height, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    mipLevelCount: ktx.levelsData.length,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });
  let w = ktx.width, h = ktx.height;
  ktx.levelsData.forEach((data, level) => {
    // Ensure we have a regular ArrayBuffer, not SharedArrayBuffer
    // If it's a SharedArrayBuffer, copy to a new ArrayBuffer
    let safeData: Uint8Array;
    if (data.buffer instanceof SharedArrayBuffer) {
      // Create a new Uint8Array with a regular ArrayBuffer and copy the data
      const newBuffer = new ArrayBuffer(data.byteLength);
      const newArray = new Uint8Array(newBuffer);
      newArray.set(data);
      safeData = newArray;
    } else {
      safeData = data as Uint8Array;
    }
    
    device.queue.writeTexture(
      { texture, mipLevel: level },
      safeData as Uint8Array & { buffer: ArrayBuffer },
      { bytesPerRow: Math.max(256, w * 4) }, // align; adjust if you transcode compressed formats
      { width: w, height: h, depthOrArrayLayers: 1 }
    );
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  });
  return texture;
}
