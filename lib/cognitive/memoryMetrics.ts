// memoryMetrics.ts — Final Unified Engine (Phase 3 + UI Metrics + Gödelian Monitoring)

import { braidMemory } from './braidMemory';
import { conceptNodes } from '$lib/stores/conceptMesh';
import { get } from 'svelte/store';
import { updateCognitiveState, cognitiveState } from './cognitiveState';

export enum MemoryHealth {
  EXCELLENT = 'Excellent',
  GOOD = 'Good',
  FAIR = 'Fair',
  COLLAPSING = 'Critical'
}

export interface MemoryMetrics {
  loopDensity: number; // ρ_M
  curvature: number;   // κ_I
  scarRatio: number;
  memoryHealth: MemoryHealth;
  godelianRisk: boolean;
  compressionPlateau: boolean;
  recursiveBurstRisk: number;
  condorcetCycleCount: number;

  totalLoops: number;
  closedLoops: number;
  scarCount: number;
  unresolvedLoops: number;
  activeConceptNodes: number;

  lastUpdate: Date;
  trend: 'improving' | 'stable' | 'degrading';
  alertLevel: 'normal' | 'warning' | 'critical';
}

export interface Thresholds {
  densityMin: number;
  curvatureMax: number;
  scarRatioMax: number;
  godelianCurvatureThreshold: number;
  godelianDensityThreshold: number;
  compressionPlateauVariance: number;
  recursiveBurstGamma: number;
}

class MemoryMetricsEngine {
  private metrics: MemoryMetrics;
  private thresholds: Thresholds;
  private history: MemoryMetrics[] = [];
  private compressionHistory: number[] = [];
  private recursiveDepthHistory: number[] = [];
  private updateTimer: ReturnType<typeof setInterval> | null = null;
  private subscribers: Array<(m: MemoryMetrics) => void> = [];

  constructor() {
    this.thresholds = {
      densityMin: 0.2,
      curvatureMax: 0.6,
      scarRatioMax: 0.3,
      godelianCurvatureThreshold: 0.7,
      godelianDensityThreshold: 0.2,
      compressionPlateauVariance: 0.05,
      recursiveBurstGamma: 1.5
    };
    this.metrics = this.emptyMetrics();
    
    // Only start auto-updates in browser environment
    if (typeof window !== 'undefined') {
      this.startAutoUpdates();
    }
    
    console.log('🧮 MemoryMetricsEngine ready');
  }

  private emptyMetrics(): MemoryMetrics {
    return {
      loopDensity: 0,
      curvature: 0,
      scarRatio: 0,
      memoryHealth: MemoryHealth.EXCELLENT,
      godelianRisk: false,
      compressionPlateau: false,
      recursiveBurstRisk: 0,
      condorcetCycleCount: 0,
      totalLoops: 0,
      closedLoops: 0,
      scarCount: 0,
      unresolvedLoops: 0,
      activeConceptNodes: 1,
      lastUpdate: new Date(),
      trend: 'stable',
      alertLevel: 'normal'
    };
  }

  forceUpdate(): MemoryMetrics {
    return this.updateMetrics();
  }

  private updateMetrics(): MemoryMetrics {
    const braidStats = braidMemory.getStats();
    
    // Safe access to conceptNodes with SSR check
    let conceptCount = 1;
    if (typeof window !== 'undefined') {
      const nodes = get(conceptNodes);
      conceptCount = Math.max(1, Object.keys(nodes || {}).length);
    }

    const totalLoops = braidStats.totalLoops;
    const closedLoops = braidStats.closedLoops;
    const scarCount = braidStats.scarredLoops;
    const unresolvedLoops = totalLoops - closedLoops;

    const loopDensity = closedLoops / conceptCount;
    const curvature = (scarCount + unresolvedLoops) / conceptCount;
    const scarRatio = totalLoops > 0 ? scarCount / totalLoops : 0;

    const godelianRisk = curvature > this.thresholds.godelianCurvatureThreshold &&
                         loopDensity < this.thresholds.godelianDensityThreshold;

    const compressionPlateau = this.detectCompressionPlateau(braidStats.compressionRatio);
    const recursiveBurstRisk = this.computeBurstRisk(getDepth(), getContradiction(), getVolatility());

    const condorcetCycleCount = braidStats.paradoxCrossings || 0;

    const health = this.assessHealth(loopDensity, curvature, scarRatio, godelianRisk);
    const trend = this.calculateTrend(loopDensity, curvature);
    const alertLevel = this.calculateAlert(health, godelianRisk, compressionPlateau);

    this.metrics = {
      loopDensity,
      curvature,
      scarRatio,
      memoryHealth: health,
      godelianRisk,
      compressionPlateau,
      recursiveBurstRisk,
      condorcetCycleCount,
      totalLoops,
      closedLoops,
      scarCount,
      unresolvedLoops,
      activeConceptNodes: conceptCount,
      lastUpdate: new Date(),
      trend,
      alertLevel
    };

    this.history.push({ ...this.metrics });
    if (this.history.length > 100) this.history.shift();

    updateCognitiveState({ scarCount: scarCount });
    this.notify();

    return this.metrics;
  }

  private detectCompressionPlateau(current: number): boolean {
    this.compressionHistory.push(current);
    if (this.compressionHistory.length > 10) this.compressionHistory.shift();
    if (this.compressionHistory.length < 5) return false;

    const recent = this.compressionHistory.slice(-5);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const variance = recent.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / recent.length;
    return variance < this.thresholds.compressionPlateauVariance && this.metrics.curvature > 0.5;
  }

  private computeBurstRisk(depth: number, contradiction: number, volatility: number): number {
    this.recursiveDepthHistory.push(depth);
    if (this.recursiveDepthHistory.length > 10) this.recursiveDepthHistory.shift();
    if (this.recursiveDepthHistory.length < 3) return 0;

    const growth = this.recursiveDepthHistory.at(-1)! - this.recursiveDepthHistory[0];
    const risk = (Math.abs(growth) * contradiction * volatility) / this.thresholds.recursiveBurstGamma;
    return Math.min(1, risk);
  }

  private assessHealth(d: number, k: number, s: number, gRisk: boolean): MemoryHealth {
    if (gRisk || k > 0.8 || s > 0.5) return MemoryHealth.COLLAPSING;
    if (k > this.thresholds.curvatureMax || d < this.thresholds.densityMin || s > this.thresholds.scarRatioMax) return MemoryHealth.FAIR;
    if (d > 0.4 && k < 0.3 && s < 0.1) return MemoryHealth.EXCELLENT;
    return MemoryHealth.GOOD;
  }

  private calculateTrend(d: number, k: number): 'improving' | 'stable' | 'degrading' {
    if (this.history.length < 3) return 'stable';
    const old = this.history.at(-3)!;
    const dd = d - old.loopDensity;
    const dk = k - old.curvature;
    if (dd > 0.05 && dk < -0.05) return 'improving';
    if (dd < -0.05 && dk > 0.05) return 'degrading';
    return 'stable';
  }

  private calculateAlert(h: MemoryHealth, godel: boolean, plateau: boolean): 'normal' | 'warning' | 'critical' {
    if (h === MemoryHealth.COLLAPSING || godel) return 'critical';
    if (h === MemoryHealth.FAIR || plateau) return 'warning';
    return h === MemoryHealth.GOOD ? 'normal' : 'warning';
  }

  private notify(): void {
    this.subscribers.forEach(fn => {
      try {
        fn(this.metrics);
      } catch (e) {
        console.error('MemoryMetrics callback error', e);
      }
    });

    // Dispatch events only in browser environment
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:memory:health-update', {
        detail: this.metrics
      }));

      if (this.metrics.alertLevel === 'critical') {
        window.dispatchEvent(new CustomEvent('tori:memory:critical-alert', {
          detail: {
            metrics: this.metrics,
            recommendations: this.getInterventionRecommendations()
          }
        }));
      }
    }
  }

  public getMetrics(): MemoryMetrics {
    return { ...this.metrics };
  }

  public getHealth(): MemoryHealth {
    return this.metrics.memoryHealth;
  }

  public getHistory(): MemoryMetrics[] {
    return [...this.history];
  }

  public reset(): void {
    this.metrics = this.emptyMetrics();
    this.history = [];
    this.compressionHistory = [];
    this.recursiveDepthHistory = [];
  }

  public onUpdate(callback: (metrics: MemoryMetrics) => void): void {
    this.subscribers.push(callback);
  }

  public getInterventionRecommendations(): string[] {
    const rec: string[] = [];
    const m = this.metrics;

    if (m.godelianRisk) rec.push('Gödelian collapse detected — apply recursive limits');
    if (m.compressionPlateau) rec.push('Compression plateau — initiate consolidation');
    if (m.recursiveBurstRisk > 0.7) rec.push('High burst risk — phase realignment recommended');
    if (m.curvature > this.thresholds.curvatureMax) rec.push('High curvature — start scar healing');
    if (m.loopDensity < this.thresholds.densityMin) rec.push('Low density — encourage more closed loops');
    if (m.scarRatio > this.thresholds.scarRatioMax) rec.push('High scar ratio — trigger stabilization protocol');

    return rec;
  }

  public exportMetrics(): object {
    return {
      metrics: this.getMetrics(),
      thresholds: this.thresholds,
      history: this.getHistory(),
      exportedAt: new Date().toISOString()
    };
  }

  private startAutoUpdates(): void {
    this.updateTimer = setInterval(() => {
      this.forceUpdate();
    }, 5000);
  }
}

// Helper stubs (replace if needed)
function getDepth(): number {
  return get(cognitiveState).loopDepth || 0;
}
function getContradiction(): number {
  return get(cognitiveState).contradictionPi || 0.01;
}
function getVolatility(): number {
  return get(cognitiveState).volatilitySigma || 0.01;
}

export const memoryMetrics = new MemoryMetricsEngine();
