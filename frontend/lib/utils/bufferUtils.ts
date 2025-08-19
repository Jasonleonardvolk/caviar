// Utility to ensure ArrayBuffer (not SharedArrayBuffer)
export function ensureArrayBuffer(data: Float32Array): Float32Array & { buffer: ArrayBuffer } {
  if (data.buffer instanceof SharedArrayBuffer) {
    const newBuffer = new ArrayBuffer(data.byteLength);
    const newArray = new Float32Array(newBuffer);
    newArray.set(data);
    return newArray as Float32Array & { buffer: ArrayBuffer };
  }
  return data as Float32Array & { buffer: ArrayBuffer };
}

export function ensureArrayBufferUint8(data: Uint8Array): Uint8Array & { buffer: ArrayBuffer } {
  if (data.buffer instanceof SharedArrayBuffer) {
    const newBuffer = new ArrayBuffer(data.byteLength);
    const newArray = new Uint8Array(newBuffer);
    newArray.set(data);
    return newArray as Uint8Array & { buffer: ArrayBuffer };
  }
  return data as Uint8Array & { buffer: ArrayBuffer };
}
