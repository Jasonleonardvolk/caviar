// TypeScript augmentation for SvelteKit App.* without touching existing files.
// Ensures all code can safely reference locals.user.name and window.TORI.*

export {};

declare global {
  namespace App {
    interface Locals {
      user?: { id: string; username: string; name?: string; role: 'admin' | 'user' } | null;
    }
    interface PageData {
      user?: Locals['user'] | null;
    }
    interface Window {
      TORI?: {
        updateHologramState?: (state: any) => void;
        setHologramVideoMode?: (enabled: boolean) => void;
        toggleHologramAudio?: (enabled: boolean) => void;
        toggleHologramVideo?: (enabled: boolean) => void;
      };
      ghostMemoryDemo?: () => void;
      webkitAudioContext?: typeof AudioContext;
      TORI_DISPLAY_TYPE?: string;
    }
  }
}
