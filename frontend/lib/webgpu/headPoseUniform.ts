// ${IRIS_ROOT}\frontend\lib\webgpu\headPoseUniform.ts
export type HeadPose = { viewProj: Float32Array }; // 4x4
export class HeadPoseUniform {
  readonly buffer: GPUBuffer;
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
    this.buffer = device.createBuffer({
      size: 64 * 4, // 4x4 f32 matrix (64 bytes) with padding headroom
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'HeadPoseUniform',
    });
  }

  update(matrix4x4: Float32Array) {
    this.device.queue.writeBuffer(this.buffer, 0, matrix4x4.buffer, matrix4x4.byteOffset, 64);
  }
}
