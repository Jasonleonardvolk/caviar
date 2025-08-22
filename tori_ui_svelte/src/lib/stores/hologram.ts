import { writable, derived } from 'svelte/store';

// Hologram visualization state (Penrose renderer)
export const hologramEnabled = writable(false);
export const hologramPersona = writable('Enola');

// Webcam state (separate from hologram)
export const webcamEnabled = writable(false);

// Combined hologram state
export const hologram = derived(
  [hologramEnabled, hologramPersona],
  ([$enabled, $persona]) => ({
    enabled: $enabled,
    persona: $persona
  })
);

// Auto-start function for onMount
export function autoStartHologram() {
  hologramEnabled.set(true);
  hologramPersona.set('Enola');
  console.log('🌟 Hologram auto-started with Enola');
}

// Helper to toggle hologram
export function toggleHologram() {
  hologramEnabled.update(n => !n);
}

// Helper to switch persona
export function setHologramPersona(persona: string) {
  hologramPersona.set(persona);
}
