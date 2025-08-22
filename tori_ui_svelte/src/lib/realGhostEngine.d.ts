// Type definitions for realGhostEngine.js

export interface RealGhostEngine {
  init(config?: any): Promise<void>;
  process(input: any): Promise<any>;
  getState(): any;
  setState(state: any): void;
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  [key: string]: any;
}

export class RealGhostEngine {
  constructor(config?: any);
}

export default RealGhostEngine;
