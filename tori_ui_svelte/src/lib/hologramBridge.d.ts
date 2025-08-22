// Type definitions for hologramBridge.js

export interface HologramBridge {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  send(data: any): Promise<void>;
  receive(): Promise<any>;
  isConnected(): boolean;
  [key: string]: any;
}

declare const hologramBridge: HologramBridge;
export default hologramBridge;
