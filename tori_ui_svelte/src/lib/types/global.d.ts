export {};

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
    TORI?: {
      toggleHologramAudio?: (enabled: boolean) => void;
      toggleHologramVideo?: (enabled: boolean) => void;
      [k: string]: any;
    };
    TORI_DISPLAY_TYPE?: 'webgpu_only' | 'webgpu_then_canvas' | 'canvas_fallback' | string;
    ghostMemoryDemo?: (...args: any[]) => any;
  }
}