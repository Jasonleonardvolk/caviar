// ContradictionMonitor.ts - Π(t) & Scar Volatility σ_s Tracking
import { cognitiveState, updateCognitiveState, CognitiveThresholds, addScar } from './cognitiveState';
import { get } from 'svelte/store';

export class ContradictionMonitor {
  private history: number[] = [];
  private windowSize: number;
  private Π: number = 0;              // current contradiction index
  private σ_s: number = 0;            // current scar volatility (variance over window)
  private threshold: number;          // volatility threshold for triggering scar alerts
  private eventCallbacks: Array<(volatility: number) => void> = [];

  constructor(windowSize: number = 50, threshold: number = CognitiveThresholds.SIGMA_ALERT) {
    this.windowSize = windowSize;
    this.threshold = threshold;
    console.log(`🔍 ContradictionMonitor initialized (window=${windowSize}, threshold=${threshold})`);
  }

  /**
   * Update contradiction metric at current time step
   * Called by ConceptDiff or other systems when inconsistency is detected
   */
  updateContradiction(delta: number): void {
    // Clamp delta to reasonable bounds
    const clampedDelta = Math.max(0, Math.min(2.0, delta));
    this.Π = clampedDelta;
    
    // Append to history and maintain window size
    this.history.push(clampedDelta);
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }
    
    // Compute variance over the window for σ_s
    this.σ_s = this.computeVariance(this.history);
    
    // Update cognitive state
    updateCognitiveState({
      contradictionPi: this.Π,
      volatilitySigma: this.σ_s
    });
    
    // Check volatility threshold and handle scar events
    if (this.σ_s > this.threshold) {
      this.handleScarEvent(this.σ_s);
    }
    
    // Log significant changes
    if (this.Π > CognitiveThresholds.PI_HIGH) {
      console.warn(`⚠️ High contradiction detected: Π=${this.Π.toFixed(3)}, σ_s=${this.σ_s.toFixed(3)}`);
    }
  }

  /**
   * Compute variance over the contradiction history
   */
  private computeVariance(data: number[]): number {
    if (data.length < 2) return 0;
    
    const mean = data.reduce((sum, x) => sum + x, 0) / data.length;
    const varSum = data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0);
    return varSum / data.length;
  }

  /**
   * Handle scar volatility events when σ_s exceeds safe limits
   */
  private handleScarEvent(currentVol: number): void {
    console.warn(`🩸 Scar volatility high: σ_s=${currentVol.toFixed(3)}. Initiating stabilization.`);
    
    // Add a scar to the cognitive state
    addScar();
    
    // Notify registered callbacks
    this.eventCallbacks.forEach(callback => {
      try {
        callback(currentVol);
      } catch (error: any) {
        console.error('Error in scar event callback:', error);
      }
    });
    
    // Automatic stabilization actions
    this.attemptStabilization(currentVol);
  }

  /**
   * Attempt to stabilize the system when volatility is high
   */
  private attemptStabilization(volatility: number): void {
    const state = get(cognitiveState);
    
    // Strategy 1: Reduce processing intensity if in a loop
    if (state.activeLoopId) {
      console.log('🩹 Stabilization: Reducing loop processing intensity');
      // Signal to slow down cognitive processing
      this.emitStabilizationSignal('reduce_intensity');
    }
    
    // Strategy 2: If volatility is critical, request pause
    if (volatility > CognitiveThresholds.SIGMA_CRITICAL) {
      console.log('🩹 Stabilization: Requesting cognitive pause');
      this.emitStabilizationSignal('pause_cognition');
    }
    
    // Strategy 3: If coherence is also low, request reset
    if (state.coherenceC < CognitiveThresholds.COHERENCE_MIN) {
      console.log('🩹 Stabilization: Requesting cognitive reset');
      this.emitStabilizationSignal('reset_loop');
    }
  }

  /**
   * Emit stabilization signals to other cognitive components
   */
  private emitStabilizationSignal(signal: 'reduce_intensity' | 'pause_cognition' | 'reset_loop'): void {
    // This would integrate with CadenceController or other systems
    // For now, we update the cognitive state to indicate stabilization is needed
    updateCognitiveState({
      gateActive: false // Disable phase gating during stabilization
    });
    
    // Could emit custom events here for other systems to listen to
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:stabilization', {
        detail: { signal, volatility: this.σ_s, contradiction: this.Π }
      }));
    }
  }

  /**
   * Register a callback for scar events
   */
  onScarEvent(callback: (volatility: number) => void): void {
    this.eventCallbacks.push(callback);
  }

  /**
   * Get current contradiction level
   */
  getPi(): number {
    return this.Π;
  }

  /**
   * Get current scar volatility
   */
  getSigma(): number {
    return this.σ_s;
  }

  /**
   * Get contradiction history for analysis
   */
  getHistory(): readonly number[] {
    return [...this.history];
  }

  /**
   * Check if contradiction is in a safe range
   */
  isSafe(): boolean {
    return this.Π <= CognitiveThresholds.PI_MEDIUM && this.σ_s <= CognitiveThresholds.SIGMA_STABLE;
  }

  /**
   * Get current contradiction status
   */
  getStatus(): 'safe' | 'caution' | 'warning' | 'critical' {
    if (this.Π <= CognitiveThresholds.PI_LOW && this.σ_s <= CognitiveThresholds.SIGMA_STABLE) {
      return 'safe';
    } else if (this.Π <= CognitiveThresholds.PI_MEDIUM && this.σ_s <= CognitiveThresholds.SIGMA_ALERT) {
      return 'caution';
    } else if (this.Π <= CognitiveThresholds.PI_HIGH) {
      return 'warning';
    } else {
      return 'critical';
    }
  }

  /**
   * Reset the contradiction monitor (useful for testing or recovery)
   */
  reset(): void {
    this.history = [];
    this.Π = 0;
    this.σ_s = 0;
    updateCognitiveState({
      contradictionPi: 0,
      volatilitySigma: 0
    });
    console.log('🔄 ContradictionMonitor reset');
  }

  /**
   * Get diagnostic information
   */
  getDiagnostics(): {
    current: number;
    volatility: number;
    status: string;
    historyLength: number;
    trend: 'rising' | 'falling' | 'stable';
  } {
    const recent = this.history.slice(-5);
    let trend: 'rising' | 'falling' | 'stable' = 'stable';
    
    if (recent.length >= 3) {
      const firstHalf = recent.slice(0, Math.floor(recent.length / 2));
      const secondHalf = recent.slice(Math.floor(recent.length / 2));
      const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
      const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
      
      if (secondAvg > firstAvg + 0.1) trend = 'rising';
      else if (secondAvg < firstAvg - 0.1) trend = 'falling';
    }
    
    return {
      current: this.Π,
      volatility: this.σ_s,
      status: this.getStatus(),
      historyLength: this.history.length,
      trend
    };
  }
}

// Singleton instance for global access
export const contradictionMonitor = new ContradictionMonitor();

// Auto-setup event listeners if in browser environment
if (typeof window !== 'undefined') {
  // Listen for concept diff changes and update contradiction
  window.addEventListener('tori:concept-diff', (event: any) => {
    const { delta } = event.detail;
    if (typeof delta === 'number') {
      contradictionMonitor.updateContradiction(delta);
    }
  });
}

console.log('🔍 ContradictionMonitor system ready');