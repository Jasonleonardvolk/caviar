// Type definitions for mockGhostEngine.js

export interface MockGhostEngine {
  init(config?: any): Promise<void>;
  process(input: any): Promise<any>;
  getState(): any;
  setState(state: any): void;
  simulateResponse(input: any): any;
  [key: string]: any;
}

declare const mockGhostEngine: MockGhostEngine;
export default mockGhostEngine;
