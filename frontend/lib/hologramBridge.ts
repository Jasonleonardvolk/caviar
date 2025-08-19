import { writable } from "svelte/store";

export const holoReady = writable(false);
export const holoConnected = writable(false);
export const holoMessages = writable<string[]>([]);

let eventSource: EventSource | null = null;

export function connectHologram() {
    if (eventSource) {
        eventSource.close();
    }
    
    try {
        eventSource = new EventSource("/holo_renderer/events");
        
        eventSource.onopen = () => {
            holoReady.set(true);
            holoConnected.set(true);
            console.log("✅ Hologram bridge connected");
        };
        
        eventSource.onerror = (error) => {
            holoReady.set(false);
            holoConnected.set(false);
            console.error("❌ Hologram bridge error:", error);
        };
        
        eventSource.addEventListener("ping", (event) => {
            // Keep-alive ping
            console.log("💓 Hologram ping:", event.data);
        });
        
        eventSource.addEventListener("hologram", (event) => {
            // Hologram data
            holoMessages.update(msgs => [...msgs, event.data].slice(-10));
        });
        
    } catch (error) {
        console.error("Failed to connect hologram bridge:", error);
        holoReady.set(false);
        holoConnected.set(false);
    }
}

export function disconnectHologram() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    holoReady.set(false);
    holoConnected.set(false);
}

// Auto-connect on module load
if (typeof window !== 'undefined') {
    connectHologram();
}
