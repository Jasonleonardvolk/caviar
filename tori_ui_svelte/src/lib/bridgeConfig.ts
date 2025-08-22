// TORI Bridge Configuration Loader
// Dynamically loads bridge configuration from the launcher

import { writable } from 'svelte/store';
import type { Writable } from 'svelte/store';

// Simplified BridgeConfig interface - base configuration
export interface BridgeConfig {
    mode: 'dev' | 'prod' | 'test';
    api?: { url: string; timeout: number };
    websocket?: { url: string; reconnect: boolean };
}

// Extended BridgeConfig for actual bridge implementations
export interface ExtendedBridgeConfig extends BridgeConfig {
    host: string;
    port: number;
    health_endpoint: string;
    metrics_endpoint: string;
    name: string;
    status: 'healthy' | 'unhealthy' | 'unknown';
}

export interface ToriBridgeConfig {
    bridges: {
        audio: ExtendedBridgeConfig;
        concept: ExtendedBridgeConfig;
        oscillator: ExtendedBridgeConfig;
    };
    backend: {
        host: string;
        port: number;
    };
    websocket: {
        reconnect_interval: number;
        max_reconnect_delay: number;
        heartbeat_interval: number;
        connection_timeout: number;
    };
    zmq: {
        endpoint: string;
    };
    generated_at: string;
    launcher_version: string;
}

// Default configuration (fallback)
const defaultConfig: ToriBridgeConfig = {
    bridges: {
        audio: {
            mode: 'dev',
            host: '127.0.0.1',
            port: 8765,
            health_endpoint: '/health',
            metrics_endpoint: '/metrics',
            name: 'Audio-Hologram Bridge',
            status: 'unknown',
            api: { url: '/api', timeout: 5000 },
            websocket: { url: '/ws', reconnect: true }
        },
        concept: {
            mode: 'dev',
            host: '127.0.0.1',
            port: 8766,
            health_endpoint: '/health',
            metrics_endpoint: '/metrics',
            name: 'Concept Mesh Bridge',
            status: 'unknown',
            api: { url: '/api', timeout: 5000 },
            websocket: { url: '/ws', reconnect: true }
        },
        oscillator: {
            mode: 'dev',
            host: '127.0.0.1',
            port: 8767,
            health_endpoint: '/health',
            metrics_endpoint: '/metrics',
            name: 'Banksy Oscillator',
            status: 'unknown',
            api: { url: '/api', timeout: 5000 },
            websocket: { url: '/ws', reconnect: true }
        }
    },
    backend: {
        host: '127.0.0.1',
        port: 8000
    },
    websocket: {
        reconnect_interval: 1000,
        max_reconnect_delay: 30000,
        heartbeat_interval: 5000,
        connection_timeout: 10000
    },
    zmq: {
        endpoint: 'ipc:///run/tori/bus.sock'
    },
    generated_at: new Date().toISOString(),
    launcher_version: '3.0'
};

// Store for bridge configuration
export const bridgeConfig: Writable<ToriBridgeConfig> = writable(defaultConfig);

// Store for connection status
export const connectionStatus = writable({
    audio: 'disconnected',
    concept: 'disconnected',
    oscillator: 'disconnected',
    backend: 'disconnected'
});

// Load configuration from file
export async function loadBridgeConfig(): Promise<ToriBridgeConfig> {
    try {
        // Try multiple possible locations
        const configPaths = [
            '/bridge_config.json',
            '/src/lib/bridge_config.json',
            '/api/bridge-config',
            'http://127.0.0.1:8000/api/bridge-config'
        ];
        
        for (const path of configPaths) {
            try {
                const response = await fetch(path);
                if (response.ok) {
                    const config = await response.json();
                    console.log('✅ Loaded bridge configuration from:', path);
                    bridgeConfig.set(config);
                    return config;
                }
            } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
                // Try next path
            
}
        }
        
        console.warn('⚠️ Could not load bridge configuration, using defaults');
        return defaultConfig;
        
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('❌ Error loading bridge configuration:', error);
        return defaultConfig;
    
}
}

// WebSocket connection manager with exponential backoff
export class BridgeWebSocket {
    private ws: WebSocket | null = null;
    private url: string;
    private reconnectDelay: number = 1000;
    private maxReconnectDelay: number = 30000;
    private reconnectTimer: number | null = null;
    private heartbeatTimer: number | null = null;
    private isClosing: boolean = false;
    
    constructor(
        private bridgeType: 'audio' | 'concept' | 'oscillator',
        private config: ExtendedBridgeConfig,
        private onMessage: (data: any) => void,
        private onStatusChange?: (status: string) => void
    ) {
        this.url = `ws://${config.host}:${config.port}/hologram`;
    }
    
    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }
        
        this.isClosing = false;
        console.log(`🔌 Connecting to ${this.bridgeType} bridge at ${this.url}`);
        
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log(`✅ Connected to ${this.bridgeType} bridge`);
                this.reconnectDelay = 1000; // Reset delay
                this.updateStatus('connected');
                this.startHeartbeat();
                
                // Send initial message
                this.send({
                    type: 'init',
                    client: 'frontend',
                    timestamp: Date.now()
                });
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.onMessage(data);
                } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
                    console.error(`Error parsing message from ${this.bridgeType
}:`, e);
                }
            };
            
            this.ws.onclose = () => {
                console.log(`🔌 Disconnected from ${this.bridgeType} bridge`);
                this.updateStatus('disconnected');
                this.stopHeartbeat();
                
                if (!this.isClosing) {
                    this.scheduleReconnect();
                }
            };
            
            this.ws.onerror = (error) => {
                console.error(`❌ WebSocket error for ${this.bridgeType}:`, error);
                this.updateStatus('error');
            };
            
        } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
            console.error(`Failed to create WebSocket for ${this.bridgeType
}:`, error);
            this.updateStatus('error');
            this.scheduleReconnect();
        }
    }
    
    private scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        
        console.log(`⏱️ Reconnecting to ${this.bridgeType} in ${this.reconnectDelay}ms`);
        
        this.reconnectTimer = window.setTimeout(() => {
            this.connect();
        }, this.reconnectDelay);
        
        // Exponential backoff
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    }
    
    private startHeartbeat() {
        this.stopHeartbeat();
        
        this.heartbeatTimer = window.setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.send({ type: 'ping', timestamp: Date.now() });
            }
        }, 5000);
    }
    
    private stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    private updateStatus(status: string) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
        
        connectionStatus.update(s => ({
            ...s,
            [this.bridgeType]: status
        }));
    }
    
    send(data: any) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn(`Cannot send to ${this.bridgeType}, not connected`);
        }
    }
    
    close() {
        this.isClosing = true;
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        this.stopHeartbeat();
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.updateStatus('disconnected');
    }
}

// Health check manager
export async function checkBridgeHealth(bridge: ExtendedBridgeConfig): Promise<boolean> {
    try {
        const response = await fetch(`http://${bridge.host}:${bridge.port}${bridge.health_endpoint}`, {
            signal: AbortSignal.timeout(2000)
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.status === 'healthy';
        }
        
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error(`Health check failed for ${bridge.name
}:`, error);
    }
    
    return false;
}

// Initialize and monitor bridge connections
export async function initializeBridges() {
    // Load configuration
    const config = await loadBridgeConfig();
    
    // Start health monitoring
    setInterval(async () => {
        for (const [type, bridge] of Object.entries(config.bridges)) {
            const isHealthy = await checkBridgeHealth(bridge);
            
            bridgeConfig.update(cfg => {
                cfg.bridges[type as keyof typeof cfg.bridges].status = isHealthy ? 'healthy' : 'unhealthy';
                return cfg;
            });
        }
    }, 5000);
    
    console.log('🚀 TORI Bridge connections initialized');
}

// Export convenience functions for creating WebSocket connections
export function createAudioBridge(onMessage: (data: any) => void): BridgeWebSocket {
    const config = defaultConfig.bridges.audio;
    bridgeConfig.subscribe(cfg => {
        if (cfg.bridges.audio) {
            Object.assign(config, cfg.bridges.audio);
        }
    });
    
    return new BridgeWebSocket('audio', config, onMessage);
}

export function createConceptBridge(onMessage: (data: any) => void): BridgeWebSocket {
    const config = defaultConfig.bridges.concept;
    bridgeConfig.subscribe(cfg => {
        if (cfg.bridges.concept) {
            Object.assign(config, cfg.bridges.concept);
        }
    });
    
    return new BridgeWebSocket('concept', config, onMessage);
}

export function createOscillatorBridge(onMessage: (data: any) => void): BridgeWebSocket {
    const config = defaultConfig.bridges.oscillator;
    bridgeConfig.subscribe(cfg => {
        if (cfg.bridges.oscillator) {
            Object.assign(config, cfg.bridges.oscillator);
        }
    });
    
    return new BridgeWebSocket('oscillator', config, onMessage);
}
