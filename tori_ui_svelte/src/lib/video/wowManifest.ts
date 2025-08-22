// wowManifest.ts - Load and manage WOW Pack video clips manifest

export type WowClip = {
  id: string;
  label: string;
  sources: { 
    type: 'av1' | 'hevc' | 'h264';
    url: string;
    codecs: string;
    meta?: {
      width: number;
      height: number;
      codec: string;
      pix_fmt: string;
      r_frame_rate: string;
    };
  }[];
  hls?: string;
};

export type WowManifest = {
  version: number;
  updated: string;
  clips: WowClip[];
};

/**
 * Load the WOW Pack manifest from static assets
 */
export async function loadManifest(): Promise<WowClip[]> {
  const res = await fetch('/media/wow/wow.manifest.json', { 
    cache: 'no-store' // Always get fresh manifest
  });
  
  if (!res.ok) {
    throw new Error(`Failed to load WOW manifest: ${res.status}`);
  }
  
  const manifest: WowManifest = await res.json();
  return manifest.clips;
}

/**
 * Find a specific clip by ID
 */
export async function findClip(clipId: string): Promise<WowClip | null> {
  const clips = await loadManifest();
  return clips.find(c => c.id === clipId) || null;
}

/**
 * Get all available clip IDs
 */
export async function getClipIds(): Promise<string[]> {
  const clips = await loadManifest();
  return clips.map(c => c.id);
}

/**
 * Check if a clip exists in the manifest
 */
export async function clipExists(clipId: string): Promise<boolean> {
  const clips = await loadManifest();
  return clips.some(c => c.id === clipId);
}
