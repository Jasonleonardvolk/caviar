// Type definitions for ghostEngine.js

export interface GhostEngineConfig {
  [key: string]: any;
}

export interface GhostEngine {
  init(): Promise<void>;
  process(input: any): Promise<any>;
  getState(): any;
  setState(state: any): void;
  [key: string]: any;
}

declare const ghostEngine: GhostEngine;
export default ghostEngine;
