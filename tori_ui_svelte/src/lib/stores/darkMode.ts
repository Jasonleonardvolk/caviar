import { writable } from 'svelte/store';
import { browser } from '$app/environment';

// Create a dark mode store with localStorage persistence
function createDarkModeStore() {
  const defaultValue = false;
  const initialValue = browser ? 
    localStorage.getItem('tori-dark-mode') === 'true' : 
    defaultValue;

  const { subscribe, set, update } = writable(initialValue);

  return {
    subscribe,
    toggle: () => update(value => {
      const newValue = !value;
      if (browser) {
        localStorage.setItem('tori-dark-mode', String(newValue));
      }
      return newValue;
    }),
    set: (value: boolean) => {
      if (browser) {
        localStorage.setItem('tori-dark-mode', String(value));
      }
      set(value);
    }
  };
}

export const darkMode = createDarkModeStore();