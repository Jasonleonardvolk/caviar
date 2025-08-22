// Stub file for psiMemory store
import { writable } from 'svelte/store';

export const psiMemoryStore = writable<any>({ 
  frames: [], 
  highlights: [] 
});
