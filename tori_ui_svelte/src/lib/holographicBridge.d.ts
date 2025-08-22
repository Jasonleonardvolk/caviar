// Type definitions for holographicBridge.js

export interface HolographicBridge {
  init(config?: any): Promise<void>;
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  send(data: any): Promise<void>;
  receive(): Promise<any>;
  isConnected(): boolean;
  renderHologram(data: any): void;
  [key: string]: any;
}

declare const holographicBridge: HolographicBridge;
export default holographicBridge;
