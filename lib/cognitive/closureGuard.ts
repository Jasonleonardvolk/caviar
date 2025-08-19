// ClosureGuard.ts - Symbolic Feedback Closure Logic with C(t) & Π(t) thresholds
import { cognitiveState, CognitiveThresholds } from './cognitiveState';
import type { LoopRecord } from './loopRecord';
import { contradictionMonitor } from './contradictionMonitor';
import { phaseController } from './phaseController';
import { get } from 'svelte/store';

export interface ClosureResult {
  allowed: boolean;
  reason: string;
  action: 'proceed' | 'defer' | 'partial' | 'stabilize' | 'abort';
  metadata?: Record<string, any>;
}

export interface FeedbackOptions {
  mode?: 'full' | 'partial' | 'stabilize' | 'minimal';
  message?: string;
  glyph?: string;
  requiresPhaseGate?: boolean;
  minimumCoherence?: number;
  maximumContradiction?: number;
}

export class ClosureGuard {
  private closureHistory: Array<{
    timestamp: Date;
    allowed: boolean;
    coherence: number;
    contradiction: number;
    reason: string;
  }> = [];

  constructor() {
    console.log('🛡️ ClosureGuard initialized for symbolic feedback control');
  }

  /**
   * Main closure evaluation - determines if feedback/loop closure should proceed
   */
  attemptFeedbackClosure(
    loopContext: LoopRecord | null, 
    options: FeedbackOptions = {}
  ): ClosureResult {
    const state = get(cognitiveState);
    
    // Get current metrics
    const coherence = this.computeCoherence(loopContext, state);
    const contradiction = contradictionMonitor.getPi();
    const volatility = contradictionMonitor.getSigma();
    
    // Apply options or use defaults
    const minCoherence = options.minimumCoherence ?? CognitiveThresholds.COHERENCE_MIN;
    const maxContradiction = options.maximumContradiction ?? CognitiveThresholds.PI_MEDIUM;
    
    // Log closure attempt
    const closureAttempt = {
      timestamp: new Date(),
      allowed: false,
      coherence,
      contradiction,
      reason: ''
    };

    // Phase gate check (if required)
    if (options.requiresPhaseGate && !this.checkPhaseAlignment()) {
      closureAttempt.reason = 'Phase gate not aligned';
      this.closureHistory.push(closureAttempt);
      
      return {
        allowed: false,
        reason: 'Waiting for phase gate alignment',
        action: 'defer',
        metadata: { 
          phase: phaseController.getPhase(),
          nextGate: phaseController.getOperationTiming('feedback')
        }
      };
    }

    // Critical contradiction check
    if (contradiction > CognitiveThresholds.PI_CRITICAL) {
      closureAttempt.reason = 'Critical contradiction level';
      this.closureHistory.push(closureAttempt);
      
      return {
        allowed: false,
        reason: `Critical contradiction (Π=${contradiction.toFixed(3)})`,
        action: 'abort',
        metadata: { requiresStabilization: true }
      };
    }

    // High volatility check
    if (volatility > CognitiveThresholds.SIGMA_CRITICAL) {
      closureAttempt.reason = 'Critical scar volatility';
      this.closureHistory.push(closureAttempt);
      
      return {
        allowed: false,
        reason: `Critical volatility (σ_s=${volatility.toFixed(3)})`,
        action: 'stabilize',
        metadata: { stabilizationMode: 'emergency' }
      };
    }

    // Main coherence and contradiction evaluation
    if (coherence >= minCoherence && contradiction <= maxContradiction) {
      // Conditions met: allow feedback loop to close normally
      closureAttempt.allowed = true;
      closureAttempt.reason = 'Safe closure conditions met';
      this.closureHistory.push(closureAttempt);
      
      console.log(`✅ Closure approved: C=${coherence.toFixed(3)}, Π=${contradiction.toFixed(3)}`);
      
      return {
        allowed: true,
        reason: 'Safe closure conditions met',
        action: 'proceed',
        metadata: {
          coherence,
          contradiction,
          volatility,
          confidence: this.calculateConfidence(coherence, contradiction)
        }
      };
    }

    // Conditions not met: determine adaptive action
    if (coherence < minCoherence) {
      closureAttempt.reason = `Low coherence (${coherence.toFixed(3)} < ${minCoherence})`;
      this.closureHistory.push(closureAttempt);
      
      console.log(`⏳ Deferring closure: Low coherence C=${coherence.toFixed(3)} (need>=${minCoherence})`);
      
      return {
        allowed: false,
        reason: `Insufficient coherence (${coherence.toFixed(3)})`,
        action: 'partial',
        metadata: {
          suggestedAction: 'provide_summary',
          coherenceGap: minCoherence - coherence
        }
      };
    }

    if (contradiction > maxContradiction) {
      closureAttempt.reason = `High contradiction (${contradiction.toFixed(3)} > ${maxContradiction})`;
      this.closureHistory.push(closureAttempt);
      
      console.log(`⚠️ Deferring closure: High contradiction Π=${contradiction.toFixed(3)} (need<=${maxContradiction})`);
      
      return {
        allowed: false,
        reason: `Excessive contradiction (${contradiction.toFixed(3)})`,
        action: 'stabilize',
        metadata: {
          suggestedAction: 'inject_calming_symbol',
          contradictionOverage: contradiction - maxContradiction
        }
      };
    }

    // Fallback case
    closureAttempt.reason = 'Unknown closure condition';
    this.closureHistory.push(closureAttempt);
    
    return {
      allowed: false,
      reason: 'Unable to determine closure safety',
      action: 'defer',
      metadata: { requiresManualReview: true }
    };
  }

  /**
   * Compute current coherence from loop context and cognitive state
   */
  private computeCoherence(loopContext: LoopRecord | null, state: any): number {
    // If we have explicit coherence in state, use it
    if (typeof state.coherenceC === 'number') {
      return state.coherenceC;
    }

    // Compute coherence from loop context
    if (loopContext?.coherenceTrace && loopContext.coherenceTrace.length > 0) {
      // Use final coherence value from trace
      return loopContext.coherenceTrace[loopContext.coherenceTrace.length - 1];
    }

    // Estimate coherence from system state
    let estimatedCoherence = 0.5; // baseline

    // Factor in prediction error (lower error = higher coherence)
    if (typeof state.predictionError === 'number') {
      estimatedCoherence += (1 - Math.min(1, state.predictionError)) * 0.3;
    }

    // Factor in active concepts (more active concepts might indicate processing)
    if (Array.isArray(state.activeConcepts)) {
      const conceptCoherence = Math.min(0.2, state.activeConcepts.length * 0.05);
      estimatedCoherence += conceptCoherence;
    }

    // Factor in scar count (more scars = lower coherence)
    if (typeof state.scarCount === 'number') {
      estimatedCoherence -= Math.min(0.3, state.scarCount * 0.1);
    }

    // Factor in loop depth (very deep loops might be less coherent)
    if (typeof state.loopDepth === 'number' && state.loopDepth > 3) {
      estimatedCoherence -= (state.loopDepth - 3) * 0.05;
    }

    return Math.max(0, Math.min(1, estimatedCoherence));
  }

  /**
   * Check if current phase is aligned with appropriate gates
   */
  private checkPhaseAlignment(): boolean {
    // Check if we're in the feedback gate
    return phaseController.inGate('feedback') || phaseController.inGate('return');
  }

  /**
   * Calculate confidence score for closure decision
   */
  private calculateConfidence(coherence: number, contradiction: number): number {
    // Higher coherence and lower contradiction = higher confidence
    const coherenceScore = coherence;
    const contradictionScore = 1 - contradiction;
    
    // Weighted average
    return (coherenceScore * 0.6 + contradictionScore * 0.4);
  }

  /**
   * Execute feedback delivery based on closure result
   */
  async deliverFeedback(
    content: any, 
    closureResult: ClosureResult,
    options: FeedbackOptions = {}
  ): Promise<boolean> {
    const mode = options.mode || this.determineFeedbackMode(closureResult);
    
    console.log(`📤 Delivering feedback (mode: ${mode}, action: ${closureResult.action})`);

    switch (mode) {
      case 'full':
        return this.deliverFullFeedback(content);
        
      case 'partial':
        return this.deliverPartialFeedback(content, closureResult.metadata);
        
      case 'stabilize':
        return this.deliverStabilizingFeedback(options.glyph || '🕊️');
        
      case 'minimal':
        return this.deliverMinimalFeedback(options.message || 'Processing...');
        
      default:
        console.warn(`Unknown feedback mode: ${mode}`);
        return false;
    }
  }

  /**
   * Determine appropriate feedback mode based on closure result
   */
  private determineFeedbackMode(closureResult: ClosureResult): FeedbackOptions['mode'] {
    switch (closureResult.action) {
      case 'proceed':
        return 'full';
      case 'partial':
        return 'partial';
      case 'stabilize':
        return 'stabilize';
      case 'defer':
      case 'abort':
        return 'minimal';
      default:
        return 'minimal';
    }
  }

  /**
   * Deliver full feedback (normal operation)
   */
  private async deliverFullFeedback(content: any): Promise<boolean> {
    // This would interface with the actual feedback system
    console.log('📤 Full feedback delivered:', content);
    
    // Emit event for other systems
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:feedback-delivered', {
        detail: { mode: 'full', content }
      }));
    }
    
    return true;
  }

  /**
   * Deliver partial feedback (reduced complexity)
   */
  private async deliverPartialFeedback(content: any, metadata?: any): Promise<boolean> {
    // Simplify or summarize the content
    const simplifiedContent = this.simplifyContent(content);
    
    console.log('📤 Partial feedback delivered:', simplifiedContent);
    
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:feedback-delivered', {
        detail: { mode: 'partial', content: simplifiedContent, metadata }
      }));
    }
    
    return true;
  }

  /**
   * Deliver stabilizing feedback (calming symbols/messages)
   */
  private async deliverStabilizingFeedback(glyph: string): Promise<boolean> {
    const stabilizingMessage = {
      type: 'stabilization',
      glyph,
      message: 'Taking a moment to process...'
    };
    
    console.log('📤 Stabilizing feedback delivered:', stabilizingMessage);
    
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:feedback-delivered', {
        detail: { mode: 'stabilize', content: stabilizingMessage }
      }));
    }
    
    return true;
  }

  /**
   * Deliver minimal feedback (status messages only)
   */
  private async deliverMinimalFeedback(message: string): Promise<boolean> {
    const minimalContent = { type: 'status', message };
    
    console.log('📤 Minimal feedback delivered:', minimalContent);
    
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:feedback-delivered', {
        detail: { mode: 'minimal', content: minimalContent }
      }));
    }
    
    return true;
  }

  /**
   * Simplify content for partial feedback
   */
  private simplifyContent(content: any): any {
    if (typeof content === 'string') {
      // Truncate long strings
      return content.length > 100 ? content.substring(0, 97) + '...' : content;
    }
    
    if (typeof content === 'object' && content !== null) {
      // Extract key information
      return {
        type: content.type || 'response',
        summary: content.summary || content.message || 'Processing response...',
        partial: true
      };
    }
    
    return content;
  }

  /**
   * Get closure statistics and diagnostics
   */
  getDiagnostics(): {
    totalAttempts: number;
    approvalRate: number;
    recentReasons: string[];
    averageCoherence: number;
    averageContradiction: number;
  } {
    const recent = this.closureHistory.slice(-20); // Last 20 attempts
    
    const approvals = recent.filter(h => h.allowed).length;
    const approvalRate = recent.length > 0 ? approvals / recent.length : 0;
    
    const avgCoherence = recent.length > 0 
      ? recent.reduce((sum, h) => sum + h.coherence, 0) / recent.length 
      : 0;
      
    const avgContradiction = recent.length > 0
      ? recent.reduce((sum, h) => sum + h.contradiction, 0) / recent.length
      : 0;
    
    const recentReasons = recent.slice(-5).map(h => h.reason);
    
    return {
      totalAttempts: this.closureHistory.length,
      approvalRate,
      recentReasons,
      averageCoherence: avgCoherence,
      averageContradiction: avgContradiction
    };
  }

  /**
   * Reset closure history (useful for testing)
   */
  reset(): void {
    this.closureHistory = [];
    console.log('🛡️ ClosureGuard reset');
  }
}

// Singleton instance for global access
export const closureGuard = new ClosureGuard();

console.log('🛡️ ClosureGuard system ready');
