// Fix for buffer upload TypeScript errors
// Create a type-safe wrapper for WebGPU buffer writes

export function writeBufferSafe(
  device: GPUDevice,
  buffer: GPUBuffer,
  bufferOffset: number,
  data: BufferSource | SharedArrayBuffer | ArrayBufferView
): void {
  // Handle all possible input types safely
  if (data instanceof SharedArrayBuffer) {
    // SharedArrayBuffer needs to be wrapped in a view
    device.queue.writeBuffer(buffer, bufferOffset, new Uint8Array(data).buffer as unknown as ArrayBuffer);
  } else if (ArrayBuffer.isView(data)) {
    // For typed arrays, get the underlying ArrayBuffer properly
    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    device.queue.writeBuffer(buffer, bufferOffset, arrayBuffer);
  } else {
    // Plain ArrayBuffer
    device.queue.writeBuffer(buffer, bufferOffset, data as ArrayBuffer);
  }
}

// Alternative simpler approach - just cast
export function writeBufferSimple(
  device: GPUDevice,
  buffer: GPUBuffer,
  bufferOffset: number,
  data: BufferSource | ArrayBufferView
): void {
  device.queue.writeBuffer(buffer, bufferOffset, data as BufferSource);
}
