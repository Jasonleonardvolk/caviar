// PhaseController.ts - Φ(t) & Phase Gates Θ_k Management
import { cognitiveState, updateCognitiveState, CognitiveThresholds } from './cognitiveState';
import { get } from 'svelte/store';

interface PhaseGate {
  name: string;
  center: number;      // Center phase [0, 2π]
  halfWidth: number;   // Half-width of gate (radians)
  active: boolean;     // Whether this gate is currently enabled
  priority: number;    // Gate priority (higher = more important)
}

export class PhaseController {
  private phase: number = 0.0;        // current phase angle [0, 2π)
  private phaseIncrement: number;     // phase step per tick
  private gates: Map<string, PhaseGate> = new Map();
  private frequency: number;          // cognitive frequency in Hz
  private tickInterval: number;       // tick interval in ms
  private isRunning: boolean = false;
  private tickTimer: number | null = null;

  constructor(frequencyHz: number = 0.5, tickIntervalMs: number = 100) {
    this.frequency = frequencyHz;
    this.tickInterval = tickIntervalMs;
    
    // Calculate increment per tick: Δφ = 2π * (frequency * interval)
    this.phaseIncrement = 2 * Math.PI * (frequencyHz * tickIntervalMs / 1000);
    
    // Set up default gates
    this.initializeDefaultGates();
    
    console.log(`🌊 PhaseController initialized (f=${frequencyHz}Hz, Δφ=${this.phaseIncrement.toFixed(4)})`);
  }

  /**
   * Initialize default phase gates for common cognitive operations
   */
  private initializeDefaultGates(): void {
    // Gate for loop returns (around π - halfway through cycle)
    this.setPhaseGate('return', Math.PI, CognitiveThresholds.GATE_WIDTH_DEFAULT, 1);
    
    // Gate for memory operations (around 0 - cycle start)
    this.setPhaseGate('memory', 0, CognitiveThresholds.GATE_WIDTH_DEFAULT, 2);
    
    // Gate for feedback (around 3π/2 - near cycle end)
    this.setPhaseGate('feedback', 3 * Math.PI / 2, CognitiveThresholds.GATE_WIDTH_DEFAULT, 3);
    
    // Gate for concept integration (around π/2 - quarter cycle)
    this.setPhaseGate('integration', Math.PI / 2, CognitiveThresholds.GATE_WIDTH_DEFAULT, 1);
  }

  /**
   * Start the phase controller tick cycle
   */
  start(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🌊 Phase controller started');
    
    // Start the tick cycle
    this.scheduleNextTick();
  }

  /**
   * Stop the phase controller
   */
  stop(): void {
    this.isRunning = false;
    if (this.tickTimer) {
      clearTimeout(this.tickTimer);
      this.tickTimer = null;
    }
    console.log('🌊 Phase controller stopped');
  }

  /**
   * Schedule the next tick
   */
  private scheduleNextTick(): void {
    if (!this.isRunning) return;
    
    this.tickTimer = setTimeout(() => {
      this.tick();
      this.scheduleNextTick();
    }, this.tickInterval) as any;
  }

  /**
   * Advance phase by one tick (called automatically when running)
   */
  tick(): void {
    // Advance phase
    this.phase = (this.phase + this.phaseIncrement) % (2 * Math.PI);
    
    // Update cognitive state
    updateCognitiveState({
      phasePhi: this.phase,
      gateActive: this.isAnyGateActive()
    });
    
    // Check for gate transitions and emit events
    this.checkGateTransitions();
    
    // Emit phase update event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:phase-tick', {
        detail: { 
          phase: this.phase, 
          gates: this.getActiveGates(),
          frequency: this.frequency 
        }
      }));
    }
  }

  /**
   * Check if any gates are transitioning and emit events
   */
  private checkGateTransitions(): void {
    const activeGates = this.getActiveGates();
    
    // If we have active gates, log them periodically
    if (activeGates.length > 0) {
      const gateNames = activeGates.map(g => g.name).join(', ');
      // Only log occasionally to avoid spam
      if (Math.random() < 0.01) { // 1% chance per tick
        console.log(`🌊 Phase gates active: ${gateNames} (Φ=${this.phase.toFixed(3)})`);
      }
    }
  }

  /**
   * Define a new phase gate Θ_k by name
   */
  setPhaseGate(name: string, centerPhase: number, width: number, priority: number = 1): void {
    const halfWidth = width / 2;
    const center = centerPhase % (2 * Math.PI); // Normalize to [0, 2π)
    
    const gate: PhaseGate = {
      name,
      center,
      halfWidth,
      active: true,
      priority
    };
    
    this.gates.set(name, gate);
    console.log(`🌊 Phase gate "${name}" set: center=${center.toFixed(3)}, width=${width.toFixed(3)}`);
  }

  /**
   * Check if current phase falls within the specified gate interval
   */
  inGate(gateName: string): boolean {
    const gate = this.gates.get(gateName);
    if (!gate || !gate.active) return false;
    
    // Calculate normalized phase difference to center
    let diff = Math.abs(this.phase - gate.center);
    if (diff > Math.PI) {
      diff = 2 * Math.PI - diff; // shortest distance on circle
    }
    
    return diff <= gate.halfWidth;
  }

  /**
   * Check if any gate is currently active
   */
  isAnyGateActive(): boolean {
    for (const gate of this.gates.values()) {
      if (gate.active && this.inGate(gate.name)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get all currently active gates
   */
  getActiveGates(): PhaseGate[] {
    const active: PhaseGate[] = [];
    for (const gate of this.gates.values()) {
      if (gate.active && this.inGate(gate.name)) {
        active.push(gate);
      }
    }
    // Sort by priority (higher priority first)
    return active.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Get the highest priority active gate
   */
  getPrimaryGate(): PhaseGate | null {
    const activeGates = this.getActiveGates();
    return activeGates.length > 0 ? activeGates[0] : null;
  }

  /**
   * Wait for phase alignment to a specific gate (async)
   */
  async alignToGate(gateName: string, maxWaitMs: number = 5000): Promise<boolean> {
    const startTime = Date.now();
    
    return new Promise((resolve) => {
      const checkAlignment = () => {
        if (this.inGate(gateName)) {
          resolve(true);
          return;
        }
        
        if (Date.now() - startTime > maxWaitMs) {
          console.warn(`⏱️ Phase alignment timeout for gate "${gateName}"`);
          resolve(false);
          return;
        }
        
        // Check again on next tick
        setTimeout(checkAlignment, this.tickInterval);
      };
      
      checkAlignment();
    });
  }

  /**
   * Enable or disable a specific gate
   */
  setGateActive(gateName: string, active: boolean): void {
    const gate = this.gates.get(gateName);
    if (gate) {
      gate.active = active;
      console.log(`🌊 Gate "${gateName}" ${active ? 'enabled' : 'disabled'}`);
    }
  }

  /**
   * Adjust the cognitive frequency (changes tick rate)
   */
  setFrequency(frequencyHz: number): void {
    this.frequency = frequencyHz;
    this.phaseIncrement = 2 * Math.PI * (frequencyHz * this.tickInterval / 1000);
    console.log(`🌊 Cognitive frequency adjusted to ${frequencyHz}Hz (Δφ=${this.phaseIncrement.toFixed(4)})`);
  }

  /**
   * Get current phase value
   */
  getPhase(): number {
    return this.phase;
  }

  /**
   * Get phase as degrees (for easier reading)
   */
  getPhaseDegrees(): number {
    return (this.phase * 180) / Math.PI;
  }

  /**
   * Get current frequency
   */
  getFrequency(): number {
    return this.frequency;
  }

  /**
   * Get all gate definitions
   */
  getGates(): PhaseGate[] {
    return Array.from(this.gates.values());
  }

  /**
   * Remove a gate
   */
  removeGate(gateName: string): boolean {
    const removed = this.gates.delete(gateName);
    if (removed) {
      console.log(`🌊 Gate "${gateName}" removed`);
    }
    return removed;
  }

  /**
   * Check if a specific operation should proceed based on gate status
   */
  shouldProceed(operation: 'return' | 'memory' | 'feedback' | 'integration' | string): boolean {
    // If the specific gate exists and we're in it, proceed
    if (this.inGate(operation)) {
      return true;
    }
    
    // If no specific gate but we're in any gate, proceed with caution
    if (this.isAnyGateActive()) {
      return true;
    }
    
    // Default: allow operation (gates are guidance, not hard blocks)
    return true;
  }

  /**
   * Get phase timing information for an operation
   */
  getOperationTiming(operation: string): {
    inGate: boolean;
    nextWindow: number; // milliseconds until next gate window
    gateCenter: number; // phase center of the gate
  } | null {
    const gate = this.gates.get(operation);
    if (!gate) return null;
    
    const inGate = this.inGate(operation);
    
    // Calculate time until next gate window
    let phaseToCenter = gate.center - this.phase;
    if (phaseToCenter < 0) phaseToCenter += 2 * Math.PI;
    
    const timeToCenter = (phaseToCenter / this.phaseIncrement) * this.tickInterval;
    const nextWindow = Math.max(0, timeToCenter - (gate.halfWidth / this.phaseIncrement) * this.tickInterval);
    
    return {
      inGate,
      nextWindow,
      gateCenter: gate.center
    };
  }

  /**
   * Get diagnostic information about the phase controller
   */
  getDiagnostics(): {
    phase: number;
    phaseDegrees: number;
    frequency: number;
    isRunning: boolean;
    activeGates: string[];
    totalGates: number;
  } {
    return {
      phase: this.phase,
      phaseDegrees: this.getPhaseDegrees(),
      frequency: this.frequency,
      isRunning: this.isRunning,
      activeGates: this.getActiveGates().map(g => g.name),
      totalGates: this.gates.size
    };
  }
}

// Singleton instance for global access
export const phaseController = new PhaseController();

// Auto-start in browser environment
if (typeof window !== 'undefined') {
  // Start phase controller automatically
  phaseController.start();
  
  // Listen for stabilization events to adjust frequency
  window.addEventListener('tori:stabilization', (event: any) => {
    const { signal } = event.detail;
    
    switch (signal) {
      case 'reduce_intensity':
        // Slow down cognitive rhythm
        phaseController.setFrequency(phaseController.getFrequency() * 0.7);
        break;
      case 'pause_cognition':
        // Temporarily stop phase progression
        phaseController.stop();
        setTimeout(() => phaseController.start(), 2000); // Resume after 2s
        break;
      case 'reset_loop':
        // Reset phase to a clean state
        phaseController.stop();
        setTimeout(() => phaseController.start(), 1000); // Quick restart
        break;
    }
  });
}

console.log('🌊 PhaseController system ready');