// ψ-Telemetry Store for Svelte
// Manages real-time curvature and singularity data

import { writable, derived, get } from 'svelte/store';

// Types
interface CurvatureStats {
    region: string;
    kappa_mean: number;
    kappa_sigma: number;
    kappa_max: number;
    kappa_min: number;
    singularities: number;
    timestamp: number;
}

interface EdgeHealthStats {
    avg_tension: number;
    max_tension: number;
    avg_fatigue: number;
    damaged_count: number;
    repair_solitons: number;
    critical_edges: string[];
}

interface TelemetrySnapshot {
    timestamp: number;
    curvature: Record<string, CurvatureStats>;
    edge_health: EdgeHealthStats;
    field_stats: any;
    evolution_time: number;
}

interface SingularityEvent {
    type: 'singularity_spike';
    region: string;
    count: number;
    max_curvature: number;
    timestamp: number;
}

// Stores
export const telemetrySnapshot = writable<TelemetrySnapshot | null>(null);
export const singularityEvents = writable<SingularityEvent[]>([]);
export const connectionStatus = writable<'disconnected' | 'connecting' | 'connected'>('disconnected');

// Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const MAX_SINGULARITY_EVENTS = 100;

// WebSocket management
let telemetrySocket: WebSocket | null = null;
let singularityEventSource: EventSource | null = null;

// Connect to telemetry stream
export function connectTelemetry() {
    connectionStatus.set('connecting');
    
    // WebSocket for telemetry snapshots
    const wsUrl = API_BASE.replace('http', 'ws') + '/api/psi/telemetry/stream';
    telemetrySocket = new WebSocket(wsUrl);
    
    telemetrySocket.onopen = () => {
        console.log('Connected to ψ-telemetry stream');
        connectionStatus.set('connected');
    };
    
    telemetrySocket.onmessage = (event) => {
        try {
            const snapshot: TelemetrySnapshot = JSON.parse(event.data);
            telemetrySnapshot.set(snapshot);
        } catch (error) {
            console.error('Error parsing telemetry:', error);
        }
    };
    
    telemetrySocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        connectionStatus.set('disconnected');
    };
    
    telemetrySocket.onclose = () => {
        console.log('Telemetry WebSocket closed');
        connectionStatus.set('disconnected');
        
        // Reconnect after delay
        setTimeout(() => {
            if (get(connectionStatus) === "disconnected") {
                connectTelemetry();
            }
        }, 5000);
    };
    
    // SSE for singularity events
    const sseUrl = API_BASE + '/api/psi/singularity/stream';
    singularityEventSource = new EventSource(sseUrl);
    
    singularityEventSource.addEventListener('spike', (event) => {
        try {
            const singularity: SingularityEvent = JSON.parse(event.data);
            
            singularityEvents.update(events => {
                const updated = [...events, singularity];
                // Keep only recent events
                if (updated.length > MAX_SINGULARITY_EVENTS) {
                    return updated.slice(-MAX_SINGULARITY_EVENTS);
                }
                return updated;
            });
            
            // Trigger UI pulse effect
            pulseSingularity(singularity.region);
            
        } catch (error) {
            console.error('Error parsing singularity event:', error);
        }
    });
    
    singularityEventSource.onerror = (error) => {
        console.error('SSE error:', error);
    };
}

// Disconnect telemetry
export function disconnectTelemetry() {
    if (telemetrySocket) {
        telemetrySocket.close();
        telemetrySocket = null;
    }
    
    if (singularityEventSource) {
        singularityEventSource.close();
        singularityEventSource = null;
    }
    
    connectionStatus.set('disconnected');
}

// Pulse effect for singularity
function pulseSingularity(region: string) {
    // Emit custom event for UI components
    window.dispatchEvent(new CustomEvent('singularity-pulse', {
        detail: { region }
    }));
}

// Derived stores for specific data
export const curvatureByRegion = derived(
    telemetrySnapshot,
    ($snapshot) => {
        if (!$snapshot) return {};
        return $snapshot.curvature;
    }
);

export const edgeHealth = derived(
    telemetrySnapshot,
    ($snapshot) => {
        if (!$snapshot) return null;
        return $snapshot.edge_health;
    }
);

export const criticalEdges = derived(
    edgeHealth,
    ($health) => {
        if (!$health) return [];
        return $health.critical_edges;
    }
);

// Heatmap data for visualization
export const curvatureHeatmap = derived(
    curvatureByRegion,
    ($curvature) => {
        const regions = Object.keys($curvature);
        if (regions.length === 0) return [];
        
        // Convert to heatmap format
        return regions.map(region => {
            const stats = $curvature[region];
            return {
                region,
                value: stats.kappa_mean,
                intensity: Math.min(1.0, Math.abs(stats.kappa_mean) / 10.0),
                variance: stats.kappa_sigma,
                hasSingularity: stats.singularities > 0
            };
        });
    }
);

// Time series data for charts
export const tensionTimeSeries = (() => {
    const data = writable<Array<{time: number, value: number}>>([]);
    const MAX_POINTS = 100;
    
    // Subscribe to telemetry updates
    telemetrySnapshot.subscribe(snapshot => {
        if (snapshot?.edge_health) {
            data.update(series => {
                const newPoint = {
                    time: snapshot.timestamp,
                    value: snapshot.edge_health.avg_tension
                };
                
                const updated = [...series, newPoint];
                if (updated.length > MAX_POINTS) {
                    return updated.slice(-MAX_POINTS);
                }
                return updated;
            });
        }
    });
    
    return data;
})();

// API functions
export async function fetchRegionCurvature(region: string): Promise<CurvatureStats> {
    const response = await fetch(`${API_BASE}/api/psi/curvature/${region}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch curvature: ${response.statusText}`);
    }
    return response.json();
}

export async function injectStress(epicenter: string, magnitude: number = 1.0) {
    const response = await fetch(`${API_BASE}/api/psi/stress/inject`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ epicenter, magnitude })
    });
    
    if (!response.ok) {
        throw new Error(`Failed to inject stress: ${response.statusText}`);
    }
    return response.json();
}

// Utility functions
export function getRegionColor(curvature: number): string {
    // Map curvature to color gradient
    const normalized = Math.min(1.0, Math.abs(curvature) / 10.0);
    
    if (curvature < 0) {
        // Negative curvature: blue gradient
        const intensity = Math.floor(255 * normalized);
        return `rgb(0, 0, ${intensity})`;
    } else {
        // Positive curvature: red gradient
        const intensity = Math.floor(255 * normalized);
        return `rgb(${intensity}, 0, 0)`;
    }
}

export function formatCurvature(value: number): string {
    if (Math.abs(value) < 0.01) {
        return value.toExponential(2);
    }
    return value.toFixed(3);
}

// Auto-connect on module load
if (typeof window !== 'undefined') {
    connectTelemetry();
    
    // Cleanup on window unload
    window.addEventListener('beforeunload', () => {
        disconnectTelemetry();
    });
}
