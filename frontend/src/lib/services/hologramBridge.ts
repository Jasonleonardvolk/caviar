import { writable } from "svelte/store";

export const holoReady = writable(false);

const es = new EventSource("/holo_renderer/events");
es.onopen  = () => holoReady.set(true);
es.onerror = () => holoReady.set(false);
