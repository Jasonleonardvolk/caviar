/**
 * Banksy API Client - Bridge between React frontend and ALAN backend
 * 
 * This client connects to the simulation_api.py FastAPI server to control
 * and monitor Banksy oscillator simulations in real-time.
 */

export interface BanksyConfig {
  n_oscillators: number;
  run_steps: number;
  spin_substeps: number;
  coupling_type: 'uniform' | 'modular' | 'random';
}

export interface BanksyState {
  step: number;
  time: number;
  order_parameter: number;
  mean_phase: number;
  n_effective: number;
  active_concepts: Record<string, number>;
  trs_loss?: number;
  rollback?: boolean;
}

export interface SimulationResult {
  id: string;
  config: BanksyConfig;
  final_state: BanksyState;
  history_summary: Record<string, number[]>;
  message: string;
}

export class BanksyApiClient {
  private baseUrl: string;
  private wsConnection: WebSocket | null = null;
  private onStateUpdate?: (state: BanksyState) => void;
  private onComplete?: () => void;
  private onError?: (error: string) => void;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  /**
   * Start a new Banksy simulation via HTTP API
   */
  async startSimulation(config: BanksyConfig): Promise<SimulationResult> {
    const response = await fetch(`${this.baseUrl}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Get the current state of a running simulation
   */
  async getSimulation(simId: string): Promise<SimulationResult> {
    const response = await fetch(`${this.baseUrl}/simulate/${simId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Connect to a simulation via WebSocket for real-time updates
   */
  connectRealtime(
    config: BanksyConfig,
    callbacks: {
      onStateUpdate?: (state: BanksyState) => void;
      onComplete?: () => void;
      onError?: (error: string) => void;
    }
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      // Store callbacks
      this.onStateUpdate = callbacks.onStateUpdate;
      this.onComplete = callbacks.onComplete;
      this.onError = callbacks.onError;

      // Create WebSocket connection
      const wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
      this.wsConnection = new WebSocket(`${wsUrl}/ws/simulate`);

      this.wsConnection.onopen = () => {
        console.log('🔗 Banksy WebSocket connected');
        
        // Send configuration to start simulation
        if (this.wsConnection) {
          this.wsConnection.send(JSON.stringify(config));
          resolve();
        }
      };

      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.event === 'complete') {
            console.log('✅ Banksy simulation completed');
            this.onComplete?.();
          } else if (data.error) {
            console.error('❌ Banksy simulation error:', data.error);
            this.onError?.(data.error);
          } else {
            // Regular state update
            this.onStateUpdate?.(data as BanksyState);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          this.onError?.('Failed to parse server message');
        }
      };

      this.wsConnection.onerror = (error) => {
        console.error('❌ Banksy WebSocket error:', error);
        this.onError?.('WebSocket connection failed');
        reject(error);
      };

      this.wsConnection.onclose = (event) => {
        console.log('🔌 Banksy WebSocket disconnected:', event.code, event.reason);
        
        if (!event.wasClean) {
          this.onError?.('Connection lost unexpectedly');
        }
      };
    });
  }

  /**
   * Disconnect from real-time simulation
   */
  disconnect(): void {
    if (this.wsConnection) {
      console.log('🔌 Disconnecting Banksy WebSocket');
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Check if the Banksy backend is available
   */
  async ping(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/`);
      return response.ok;
    } catch (error) {
      console.error('❌ Banksy backend ping failed:', error);
      return false;
    }
  }

  /**
   * Get backend API information
   */
  async getApiInfo(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

/**
 * React hook for using Banksy API client
 */
import { useState, useEffect, useCallback, useRef } from 'react';

export function useBanksyApi(baseUrl?: string) {
  const [client] = useState(() => new BanksyApiClient(baseUrl));
  const [isConnected, setIsConnected] = useState(false);
  const [currentState, setCurrentState] = useState<BanksyState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stateHistoryRef = useRef<BanksyState[]>([]);

  // Check backend connectivity on mount
  useEffect(() => {
    client.ping().then(setIsConnected);
  }, [client]);

  const startRealtime = useCallback(async (config: BanksyConfig) => {
    try {
      setError(null);
      setIsRunning(true);
      stateHistoryRef.current = [];

      await client.connectRealtime(config, {
        onStateUpdate: (state) => {
          setCurrentState(state);
          stateHistoryRef.current.push(state);
        },
        onComplete: () => {
          setIsRunning(false);
          console.log('✅ Banksy simulation completed');
        },
        onError: (errorMsg) => {
          setError(errorMsg);
          setIsRunning(false);
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsRunning(false);
    }
  }, [client]);

  const stop = useCallback(() => {
    client.disconnect();
    setIsRunning(false);
  }, [client]);

  const getStateHistory = useCallback(() => {
    return stateHistoryRef.current;
  }, []);

  return {
    client,
    isConnected,
    currentState,
    isRunning,
    error,
    startRealtime,
    stop,
    getStateHistory,
  };
}

export default BanksyApiClient;
