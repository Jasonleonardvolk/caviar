/// <reference types="@webgpu/types" />

// WebGPU types are globally available
declare module '*.wgsl' {
  const shader: string;
  export default shader;
}

declare module '*.wgsl?raw' {
  const shader: string;
  export default shader;
}

// Additional type declarations for the project
declare namespace App {
  interface Locals {}
  interface PageData {}
  interface Error {}
  interface Platform {}
}

// Ensure WebGPU types are available globally
interface Navigator {
  readonly gpu: GPU;
}

interface WorkerNavigator {
  readonly gpu: GPU;
}

// Ghost interpretation types for photoMorphPipeline
interface GhostInterpretation {
  morphStyle: 'explosive' | 'flowing' | 'structured' | 'crystalline' | 'organic';
  emotionalResonance: {
    clarity: number;
  };
}

// Extend Window for any custom properties
interface Window {
  webgpuRenderer?: any;
  wasmFallbackRenderer?: any;
}
