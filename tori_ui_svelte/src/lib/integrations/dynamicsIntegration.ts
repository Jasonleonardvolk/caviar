// src/lib/integrations/dynamicsIntegration.ts
// -----------------------------------------------------------------------------
// Integration module for KoopmanOperator and LyapunovAnalyzer
// Bridges the gap between TORI's phase events and the dynamics analyzers
// -----------------------------------------------------------------------------

import { koopmanOperator, lyapunovAnalyzer } from '$lib/cognitive/dynamics';
import { browser } from '$app/environment';
import type { KoopmanUpdate, LyapunovSpike } from '$lib/cognitive/dynamics';

export class DynamicsIntegration {
  private isInitialized = false;
  private phaseVector: number[] = [0, 0, 0]; // [phase, amplitude, frequency]
  private updateInterval: NodeJS.Timeout | null = null;
  private koopmanCallbacks: Set<(update: KoopmanUpdate) => void> = new Set();
  private lyapunovCallbacks: Set<(spike: LyapunovSpike) => void> = new Set();

  constructor() {
    if (browser) {
      this.initialize();
    }
  }

  private initialize(): void {
    if (this.isInitialized) return;

    console.log('🌊 Initializing Dynamics Integration...');

    // Listen for phase change events from TORI
    document.addEventListener('tori-soliton-phase-change', this.handlePhaseChange);

    // Listen for Koopman updates
    document.addEventListener('tori-koopman-update', this.handleKoopmanUpdate);

    // Listen for Lyapunov spikes
    document.addEventListener('tori-lyapunov-spike', this.handleLyapunovSpike);

    // Start periodic ingestion (50ms intervals for smooth dynamics)
    this.updateInterval = setInterval(() => {
      this.ingestCurrentState();
    }, 50);

    this.isInitialized = true;
    console.log('✅ Dynamics Integration ready');
  }

  private handlePhaseChange = (event: Event): void => {
    const customEvent = event as CustomEvent;
    const detail = customEvent.detail;

    // Update our phase vector with the new data
    if (detail.phaseAngle !== undefined) {
      this.phaseVector[0] = detail.phaseAngle || 0;
    }
    if (detail.amplitude !== undefined) {
      this.phaseVector[1] = detail.amplitude;
    }
    if (detail.frequency !== undefined) {
      this.phaseVector[2] = detail.frequency;
    }

    // Also capture additional dimensions if available
    if (detail.stability !== undefined) {
      // Extend the phase vector for richer dynamics
      if (this.phaseVector.length === 3) {
        this.phaseVector.push(detail.stability);
      } else {
        this.phaseVector[3] = detail.stability;
      }
    }
    if (detail.valence !== undefined) {
      if (this.phaseVector.length === 4) {
        this.phaseVector.push(detail.valence);
      } else {
        this.phaseVector[4] = detail.valence;
      }
    }

    console.log('📊 Phase vector updated:', this.phaseVector);
  };

  private handleKoopmanUpdate = (event: Event): void => {
    const customEvent = event as CustomEvent<KoopmanUpdate>;
    const update = customEvent.detail;

    console.log('🌀 Koopman update received:', {
      modes: update.eigenmodes.length,
      dominant: update.dominantMode,
      gap: update.spectralGap.toFixed(3)
    });

    // Notify all registered callbacks
    this.koopmanCallbacks.forEach(callback => callback(update));
  };

  private handleLyapunovSpike = (event: Event): void => {
    const customEvent = event as CustomEvent<LyapunovSpike>;
    const spike = customEvent.detail;

    console.warn('⚡ Lyapunov spike detected!', {
      exponent: spike.exponent.toFixed(4),
      instability: spike.instabilityLevel.toFixed(2),
      divergence: spike.divergenceRate.toFixed(4)
    });

    // Notify all registered callbacks
    this.lyapunovCallbacks.forEach(callback => callback(spike));
  };

  private ingestCurrentState(): void {
    // Only ingest if we have meaningful data
    const hasData = this.phaseVector.some(v => v !== 0);
    if (!hasData) return;

    // Feed the phase vector to Koopman operator
    koopmanOperator.ingest([...this.phaseVector]);

    // Feed a scalar (we'll use amplitude) to Lyapunov analyzer
    const amplitude = this.phaseVector[1];
    if (amplitude !== undefined && amplitude !== 0) {
      lyapunovAnalyzer.ingest(amplitude);
    }
  }

  // Public API

  /**
   * Subscribe to Koopman updates
   */
  onKoopmanUpdate(callback: (update: KoopmanUpdate) => void): () => void {
    this.koopmanCallbacks.add(callback);
    return () => this.koopmanCallbacks.delete(callback);
  }

  /**
   * Subscribe to Lyapunov spikes
   */
  onLyapunovSpike(callback: (spike: LyapunovSpike) => void): () => void {
    this.lyapunovCallbacks.add(callback);
    return () => this.lyapunovCallbacks.delete(callback);
  }

  /**
   * Get current phase vector
   */
  getPhaseVector(): number[] {
    return [...this.phaseVector];
  }

  /**
   * Manually trigger analysis with custom data
   */
  ingestCustomVector(vector: number[]): void {
    koopmanOperator.ingest(vector);
    if (vector.length > 0) {
      lyapunovAnalyzer.ingest(vector[0]);
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    document.removeEventListener('tori-soliton-phase-change', this.handlePhaseChange);
    document.removeEventListener('tori-koopman-update', this.handleKoopmanUpdate);
    document.removeEventListener('tori-lyapunov-spike', this.handleLyapunovSpike);

    this.koopmanCallbacks.clear();
    this.lyapunovCallbacks.clear();
    this.isInitialized = false;
  }
}

// Export singleton instance
export const dynamicsIntegration = new DynamicsIntegration();

// Also export for browser console access during development
if (browser && typeof window !== 'undefined') {
  (window as any).dynamicsIntegration = dynamicsIntegration;
}
