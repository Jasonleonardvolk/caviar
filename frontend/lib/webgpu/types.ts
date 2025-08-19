/**
 * Common types for WebGPU kernels
 */

export interface KernelSpec {
  name: string;
  execute: (commandEncoder: GPUCommandEncoder, inputs: any) => void;
  initialize?: (device: GPUDevice, config?: any) => Promise<void>;
  destroy?: () => void;
}

export interface KernelInputs {
  dims: [number, number];
  [key: string]: any;
}

export interface TimingStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  samples: number[];
}

export interface AccuracyMetrics {
  rmse?: number;
  maxError?: number;
  relativeError?: number;
}

export interface PerformanceMetrics {
  gflops?: number;
  bandwidth?: number;  // GB/s
  efficiency?: number;  // percentage
}
