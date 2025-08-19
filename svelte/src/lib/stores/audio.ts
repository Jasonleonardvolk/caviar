import { writable, derived, get } from 'svelte/store';

// Type definitions
export interface SpectralData {
  centroid: number;
  rms: number;
  spread?: number;
  flux?: number;
}

export interface EmotionData {
  label: string;
  confidence: number;
  valence?: number;
  arousal?: number;
}

export interface HologramHint {
  hue: number;        // 0-360
  intensity: number;  // 0-1
  psi: number;        // 0-1
}

export interface AudioStreamState {
  connected: boolean;
  streaming: boolean;
  error: string | null;
  sessionId?: string;
  lastActivity?: number;
}

export interface StreamStats {
  bytesReceived: number;
  chunksProcessed: number;
  errors: number;
  droppedChunks: number;
  avgProcessingTime?: number;
}

// Stores
export const transcript = writable<string>('');
export const spectral = writable<SpectralData>({ centroid: 0, rms: 0 });
export const emotion = writable<EmotionData>({ label: 'neutral', confidence: 0 });
export const hologramHint = writable<HologramHint>({ hue: 0, intensity: 0, psi: 0 });
export const streamState = writable<AudioStreamState>({
  connected: false,
  streaming: false,
  error: null
});
export const streamStats = writable<StreamStats>({
  bytesReceived: 0,
  chunksProcessed: 0,
  errors: 0,
  droppedChunks: 0
});

// Derived stores
export const isStreaming = derived(streamState, $state => $state.streaming);
export const dominantEmotion = derived(emotion, $emotion => $emotion.label);
export const connectionStatus = derived(streamState, $state => {
  if ($state.connected && $state.streaming) return 'streaming';
  if ($state.connected) return 'connected';
  if ($state.error) return 'error';
  return 'disconnected';
});

// Reset functions
export function resetStream() {
  transcript.set('');
  spectral.set({ centroid: 0, rms: 0 });
  emotion.set({ label: 'neutral', confidence: 0 });
  hologramHint.set({ hue: 0, intensity: 0, psi: 0 });
}

export function resetStats() {
  streamStats.set({
    bytesReceived: 0,
    chunksProcessed: 0,
    errors: 0,
    droppedChunks: 0
  });
}

export function resetAll() {
  resetStream();
  resetStats();
  streamState.set({
    connected: false,
    streaming: false,
    error: null
  });
}

// WebSocket connection management
class AudioStreamManager {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 1000;
  private pingInterval: number | null = null;
  private activityTimeout: number | null = null;
  private transcript_buffer: string[] = [];
  
  // Configuration
  private config = {
    maxTranscriptLength: 500,
    activityTimeout: 60000, // 1 minute
    minAudioSize: 100, // Minimum bytes for valid audio
    maxAudioSize: 16 * 1024 * 1024 // 16MB
  };
  
  constructor() {
    this.sessionId = `audio_${Date.now()}`;
  }
  
  connect(): WebSocket {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/audio/ingest?session_id=${this.sessionId}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('Audio WebSocket connected');
        streamState.update(state => ({
          ...state,
          connected: true,
          error: null,
          sessionId: this.sessionId,
          lastActivity: Date.now()
        }));
        this.reconnectAttempts = 0;
        this.startPing();
        this.startActivityMonitor();
      };
      
      this.ws.onmessage = (event) => {
        this.updateActivity();
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          this.updateStats({ errors: 1 });
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        streamState.update(state => ({
          ...state,
          error: 'Connection error'
        }));
        this.updateStats({ errors: 1 });
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        streamState.update(state => ({
          ...state,
          connected: false,
          streaming: false,
          error: event.wasClean ? null : `Connection lost (${event.code})`
        }));
        this.stopPing();
        this.stopActivityMonitor();
        
        // Only reconnect if it wasn't a clean close
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.attemptReconnect();
        }
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      streamState.update(state => ({
        ...state,
        error: 'Failed to create connection'
      }));
    }
    
    return this.ws;
  }
  
  private handleMessage(message: any) {
    switch (message.type) {
      case 'connected':
        console.log('Connected with session:', message.session_id);
        if (message.config) {
          console.log('Stream config:', message.config);
        }
        break;
        
      case 'partial':
        const data = message.data;
        if (data.transcript !== undefined) {
          this.updateTranscript(data.transcript);
        }
        if (data.spectral) {
          spectral.set(data.spectral);
        }
        if (data.emotion) {
          emotion.set(data.emotion);
        }
        if (data.hologram_hint) {
          hologramHint.set(data.hologram_hint);
        }
        if (data.metrics) {
          this.updateStats({ chunksProcessed: 1 });
        }
        break;
        
      case 'final':
        console.log('Final results received');
        if (message.data.transcript) {
          transcript.set(message.data.transcript);
        }
        if (message.data.metrics) {
          console.log('Processing metrics:', message.data.metrics);
        }
        streamState.update(state => ({ ...state, streaming: false }));
        break;
        
      case 'error':
        console.error('Stream error:', message.error);
        streamState.update(state => ({
          ...state,
          error: message.error,
          streaming: false
        }));
        this.updateStats({ errors: 1 });
        break;
        
      case 'warning':
        console.warn('Stream warning:', message.message);
        if (message.message.includes('dropped')) {
          this.updateStats({ droppedChunks: 1 });
        }
        break;
        
      case 'silence':
        // Silent audio detected, no action needed
        break;
        
      case 'pong':
        // Ping response received
        if (message.stats) {
          streamStats.update(stats => ({
            ...stats,
            avgProcessingTime: message.stats.avg_processing_time
          }));
        }
        break;
        
      case 'stats':
        // Update stream statistics
        if (message.data) {
          streamStats.update(stats => ({
            ...stats,
            ...message.data
          }));
        }
        break;
    }
  }
  
  private updateTranscript(newText: string) {
    // Maintain a buffer of transcript segments
    if (newText && newText.trim()) {
      this.transcript_buffer.push(newText);
      
      // Keep only recent segments
      if (this.transcript_buffer.length > 10) {
        this.transcript_buffer = this.transcript_buffer.slice(-10);
      }
      
      // Update store with concatenated transcript
      const fullTranscript = this.transcript_buffer.join(' ');
      
      // Limit total length
      if (fullTranscript.length > this.config.maxTranscriptLength) {
        transcript.set('...' + fullTranscript.slice(-this.config.maxTranscriptLength));
      } else {
        transcript.set(fullTranscript);
      }
    }
  }
  
  private updateStats(updates: Partial<StreamStats>) {
    streamStats.update(stats => ({
      ...stats,
      ...updates,
      bytesReceived: stats.bytesReceived + (updates.bytesReceived || 0),
      chunksProcessed: stats.chunksProcessed + (updates.chunksProcessed || 0),
      errors: stats.errors + (updates.errors || 0),
      droppedChunks: stats.droppedChunks + (updates.droppedChunks || 0)
    }));
  }
  
  private updateActivity() {
    streamState.update(state => ({
      ...state,
      lastActivity: Date.now()
    }));
  }
  
  private startActivityMonitor() {
    this.activityTimeout = window.setInterval(() => {
      const state = get(streamState);
      if (state.lastActivity && Date.now() - state.lastActivity > this.config.activityTimeout) {
        console.warn('No activity detected, connection may be stale');
        streamState.update(s => ({
          ...s,
          error: 'Connection timeout - no activity'
        }));
        
        // Attempt to reconnect
        this.disconnect();
        this.connect();
      }
    }, 10000); // Check every 10 seconds
  }
  
  private stopActivityMonitor() {
    if (this.activityTimeout) {
      clearInterval(this.activityTimeout);
      this.activityTimeout = null;
    }
  }
  
  private startPing() {
    this.pingInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Ping every 30 seconds
  }
  
  private stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
  
  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      streamState.update(state => ({
        ...state,
        error: 'Failed to reconnect after multiple attempts'
      }));
    }
  }
  
  sendAudio(audioData: ArrayBuffer) {
    // Validate audio data
    if (!audioData || audioData.byteLength === 0) {
      console.error('Invalid audio data: empty buffer');
      streamState.update(state => ({
        ...state,
        error: 'Invalid audio data'
      }));
      return;
    }
    
    if (audioData.byteLength < this.config.minAudioSize) {
      console.warn('Audio data too small, skipping');
      return;
    }
    
    if (audioData.byteLength > this.config.maxAudioSize) {
      console.error('Audio data too large:', audioData.byteLength);
      streamState.update(state => ({
        ...state,
        error: 'Audio data too large'
      }));
      return;
    }
    
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(audioData);
        streamState.update(state => ({ 
          ...state, 
          streaming: true,
          error: null 
        }));
        this.updateStats({ bytesReceived: audioData.byteLength });
        this.updateActivity();
      } catch (error) {
        console.error('Failed to send audio:', error);
        streamState.update(state => ({
          ...state,
          error: 'Failed to send audio data'
        }));
        this.updateStats({ errors: 1 });
      }
    } else {
      console.error('WebSocket not connected');
      streamState.update(state => ({
        ...state,
        error: 'Not connected',
        streaming: false
      }));
    }
  }
  
  finalize() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'finalize' }));
      streamState.update(state => ({ ...state, streaming: false }));
    }
  }
  
  getStats() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stats' }));
    }
  }
  
  reset() {
    resetStream();
    this.transcript_buffer = [];
    this.sessionId = `audio_${Date.now()}`; // New session ID
    this.reconnectAttempts = 0;
    
    // If connected, notify server
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.finalize();
    }
  }
  
  disconnect() {
    this.stopPing();
    this.stopActivityMonitor();
    if (this.ws) {
      this.ws.close(1000, 'User disconnect'); // Clean close
      this.ws = null;
    }
    streamState.update(state => ({
      ...state,
      connected: false,
      streaming: false,
      error: null
    }));
  }
}

// Export singleton instance
export const audioStream = new AudioStreamManager();

// Convenience functions
export function connectAudioWS(): WebSocket {
  return audioStream.connect();
}

export function sendAudioChunk(data: ArrayBuffer) {
  audioStream.sendAudio(data);
}

export function finalizeStream() {
  audioStream.finalize();
}

export function disconnectStream() {
  audioStream.disconnect();
}

export function getStreamStats() {
  audioStream.getStats();
}

export function resetStreamSession() {
  audioStream.reset();
}

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    audioStream.disconnect();
  });
}