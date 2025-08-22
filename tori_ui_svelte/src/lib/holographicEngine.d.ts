// Type definitions for holographicEngine.js

export interface HolographicEngine {
  init(config?: any): Promise<void>;
  render(data: any): void;
  update(deltaTime: number): void;
  destroy(): void;
  getRenderer(): any;
  getScene(): any;
  getCamera(): any;
  [key: string]: any;
}

export class HolographicEngine {
  constructor(config?: any);
}

export default HolographicEngine;
