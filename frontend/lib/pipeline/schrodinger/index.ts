/**
 * Re-export Schrödinger kernels from their actual location
 * This maintains the clean architecture while keeping files where they work
 */
export { SchrodingerRegistry } from '../../webgpu/kernels/schrodingerKernelRegistry';
export { createSplitStepOrchestrator } from '../../webgpu/kernels/splitStepOrchestrator';
export type { SplitStepOrchestrator } from '../../webgpu/kernels/splitStepOrchestrator';

// Export other kernel utilities
export * from '../../webgpu/kernels/schrodingerEvolution';
