/**
 * SSE Client for real-time event streaming
 */
class SSEClient {
    constructor() {
        this.eventSource = null;
        this.subscribers = new Set();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.url = '/api/v2/hybrid/events/sse';
        this.isConnecting = false;
        // Auto-reconnect on visibility change
        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible' && this.eventSource?.readyState === EventSource.CLOSED) {
                    this.connect();
                }
            });
        }
    }
    async connect(url) {
        if (url)
            this.url = url;
        if (this.isConnecting || this.eventSource?.readyState === EventSource.OPEN) {
            return;
        }
        this.isConnecting = true;
        try {
            this.eventSource = new EventSource(this.url);
            this.eventSource.onopen = () => {
                console.log('[SSE] Connected');
                this.reconnectAttempts = 0;
                this.isConnecting = false;
                this.notify({
                    type: 'connection',
                    data: { status: 'connected' }
                });
            };
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.notify({
                        type: data.type || 'message',
                        data: data.data || data,
                        id: event.lastEventId
                    });
                }
                catch (err) {
                    console.error('[SSE] Failed to parse event:', err);
                }
            };
            this.eventSource.onerror = (error) => {
                console.error('[SSE] Connection error:', error);
                this.isConnecting = false;
                if (this.eventSource?.readyState === EventSource.CLOSED) {
                    this.notify({
                        type: 'connection',
                        data: { status: 'disconnected' }
                    });
                    // Attempt reconnection with exponential backoff
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), 30000);
                        setTimeout(() => {
                            this.reconnectAttempts++;
                            this.connect();
                        }, delay);
                    }
                }
            };
            // Add custom event listeners
            const eventTypes = [
                'mesh_updated',
                'adapter_swap',
                'av_status',
                'log',
                'prompt_result',
                'error'
            ];
            eventTypes.forEach(eventType => {
                this.eventSource.addEventListener(eventType, (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.notify({
                            type: eventType,
                            data,
                            id: event.lastEventId
                        });
                    }
                    catch (err) {
                        console.error(`[SSE] Failed to parse ${eventType} event:`, err);
                    }
                });
            });
        }
        catch (error) {
            console.error('[SSE] Failed to connect:', error);
            this.isConnecting = false;
            throw error;
        }
    }
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
            this.notify({
                type: 'connection',
                data: { status: 'disconnected' }
            });
        }
    }
    subscribe(callback) {
        this.subscribers.add(callback);
        // Return unsubscribe function
        return () => {
            this.subscribers.delete(callback);
        };
    }
    notify(event) {
        this.subscribers.forEach(callback => {
            try {
                callback(event);
            }
            catch (err) {
                console.error('[SSE] Subscriber error:', err);
            }
        });
    }
    getState() {
        if (this.isConnecting)
            return 'connecting';
        if (this.eventSource?.readyState === EventSource.OPEN)
            return 'open';
        return 'closed';
    }
    // Utility method to post events back to server
    async postEvent(type, data) {
        return fetch('/api/v2/hybrid/event', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ type, data })
        });
    }
}
// Export singleton instance
export const sseClient = new SSEClient();
