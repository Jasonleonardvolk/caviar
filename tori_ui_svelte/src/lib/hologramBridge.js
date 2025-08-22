// hologramBridge.js - Hologram SSE Bridge for TORI Frontend
export class HologramBridge {
    constructor(apiUrl = '') {
        this.apiUrl = apiUrl;
        this.eventSource = null;
        this.isConnected = false;
        this.listeners = new Map();
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;
    }

    connect() {
        if (this.eventSource) {
            this.disconnect();
        }

        const url = `${this.apiUrl}/holo_renderer/events`;
        console.log(`🔌 Connecting to hologram SSE: ${url}`);
        
        try {
            this.eventSource = new EventSource(url);
            
            this.eventSource.addEventListener('open', () => {
                console.log('✅ Hologram SSE connected');
                this.isConnected = true;
                this.reconnectDelay = 1000; // Reset delay
                this.emit('connected');
            });

            this.eventSource.addEventListener('ping', (event) => {
                console.log('💓 Hologram ping received');
                this.emit('ping', event.data);
            });

            this.eventSource.addEventListener('error', (error) => {
                console.error('❌ Hologram SSE error:', error);
                this.isConnected = false;
                this.emit('error', error);
                
                // Auto-reconnect with exponential backoff
                setTimeout(() => {
                    console.log(`🔄 Attempting reconnect in ${this.reconnectDelay}ms...`);
                    this.connect();
                    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
                }, this.reconnectDelay);
            });

            // Generic message handler
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.emit('message', data);
                } catch (e) {
                    console.log('📨 Hologram message:', event.data);
                }
            };

        } catch (error) {
            console.error('❌ Failed to create EventSource:', error);
            this.emit('error', error);
        }
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
            this.isConnected = false;
            this.emit('disconnected');
            console.log('🔌 Hologram SSE disconnected');
        }
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            });
        }
    }
}

// Export singleton instance
export const hologramBridge = new HologramBridge();
