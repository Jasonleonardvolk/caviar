/**
 * DebuggerSentinel.ts - Error detection and recursive debugging agent
 * Hooks to confidence drops and offers contextual fixes
 */

interface DebugSession {
  id: string;
  startTime: Date;
  errorType: string;
  confidence: number;
  context: {
    file?: string;
    line?: number;
    function?: string;
    stackTrace?: string;
    userMessage?: string;
  };
  attempts: DebugAttempt[];
  resolved: boolean;
  successfulSolution?: string;
}

interface DebugAttempt {
  id: string;
  approach: string;
  confidence: number;
  suggestion: string;
  applied: boolean;
  result?: 'success' | 'failure' | 'partial';
  timestamp: Date;
}

interface DebuggerConfig {
  enabled: boolean;
  confidenceThreshold: number;
  maxRecursiveAttempts: number;
  autoApplyHighConfidence: boolean;
  learningEnabled: boolean;
}

class DebuggerSentinel {
  private sessions: Map<string, DebugSession> = new Map();
  private successPatterns: Map<string, any[]> = new Map();
  private config: DebuggerConfig = {
    enabled: true,
    confidenceThreshold: 0.65,
    maxRecursiveAttempts: 5,
    autoApplyHighConfidence: false,
    learningEnabled: true
  };
  private listeners: Array<(session: DebugSession) => void> = [];

  constructor() {
    this.initializeDebugger();
  }

  private initializeDebugger() {
    // Monitor confidence levels from Koopman/Lyapunov analysis
    document.addEventListener('tori-confidence-drop', ((e: CustomEvent) => {
      this.handleConfidenceDrop(e.detail);
    }) as EventListener);

    // Monitor error events
    document.addEventListener('tori-error', ((e: CustomEvent) => {
      this.handleError(e.detail);
    }) as EventListener);

    // Monitor concept coherence issues
    document.addEventListener('tori-phase-instability', ((e: CustomEvent) => {
      this.handlePhaseInstability(e.detail);
    }) as EventListener);

    // Global error handler
    window.addEventListener('error', (e) => {
      this.handleGlobalError(e);
    });

    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (e) => {
      this.handleUnhandledRejection(e);
    });
  }

  private handleConfidenceDrop(confidenceData: {
    currentConfidence: number;
    previousConfidence: number;
    context: any;
    source: string;
  }) {
    const { currentConfidence, context, source } = confidenceData;

    if (currentConfidence < this.config.confidenceThreshold) {
      const sessionId = this.createDebugSession({
        errorType: 'confidence_drop',
        confidence: currentConfidence,
        context: {
          ...context,
          source,
          confidence_threshold: this.config.confidenceThreshold
        }
      });

      this.generateDebugSuggestions(sessionId);
    }
  }

  private handleError(errorData: {
    error: Error;
    context?: any;
    file?: string;
    line?: number;
  }) {
    const { error, context, file, line } = errorData;

    const sessionId = this.createDebugSession({
      errorType: 'runtime_error',
      confidence: 0.2, // Low confidence due to actual error
      context: {
        ...context,
        file,
        line,
        stackTrace: error.stack,
        errorMessage: error.message,
        errorName: error.name
      }
    });

    this.generateDebugSuggestions(sessionId);
  }

  private handlePhaseInstability(phaseData: {
    coherence: number;
    entropy: number;
    concept: string;
    drift: number;
  }) {
    const { coherence, entropy, concept, drift } = phaseData;

    if (coherence < 0.3 || entropy > 0.8 || Math.abs(drift) > 0.5) {
      const sessionId = this.createDebugSession({
        errorType: 'phase_instability',
        confidence: coherence,
        context: {
          concept,
          coherence,
          entropy,
          drift,
          instability_type: this.classifyInstability(coherence, entropy, drift)
        }
      });

      this.generateDebugSuggestions(sessionId);
    }
  }

  private handleGlobalError(e: ErrorEvent) {
    this.handleError({
      error: new Error(e.message),
      context: {
        filename: e.filename,
        lineno: e.lineno,
        colno: e.colno
      },
      file: e.filename,
      line: e.lineno
    });
  }

  private handleUnhandledRejection(e: PromiseRejectionEvent) {
    this.handleError({
      error: new Error(`Unhandled Promise Rejection: ${e.reason}`),
      context: {
        reason: e.reason,
        type: 'promise_rejection'
      }
    });
  }

  private createDebugSession(params: {
    errorType: string;
    confidence: number;
    context: any;
  }): string {
    const sessionId = `debug_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const session: DebugSession = {
      id: sessionId,
      startTime: new Date(),
      errorType: params.errorType,
      confidence: params.confidence,
      context: params.context,
      attempts: [],
      resolved: false
    };

    this.sessions.set(sessionId, session);
    this.notifyListeners(session);

    return sessionId;
  }

  private generateDebugSuggestions(sessionId: string) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const suggestions = this.analyzeAndSuggest(session);
    
    suggestions.forEach((suggestion, index) => {
      const attempt: DebugAttempt = {
        id: `attempt_${Date.now()}_${index}`,
        approach: suggestion.approach,
        confidence: suggestion.confidence,
        suggestion: suggestion.text,
        applied: false,
        timestamp: new Date()
      };

      session.attempts.push(attempt);
    });

    this.notifyListeners(session);
    
    // Auto-apply high confidence suggestions if enabled
    if (this.config.autoApplyHighConfidence) {
      const highConfidenceSuggestion = suggestions.find(s => s.confidence > 0.9);
      if (highConfidenceSuggestion) {
        this.applySuggestion(sessionId, session.attempts[suggestions.indexOf(highConfidenceSuggestion)].id);
      }
    }

    // Emit event for UI
    document.dispatchEvent(new CustomEvent('tori-debugger-suggestions', {
      detail: { sessionId, session }
    }));
  }

  private analyzeAndSuggest(session: DebugSession): Array<{
    approach: string;
    confidence: number;
    text: string;
  }> {
    const suggestions: Array<{
      approach: string;
      confidence: number;
      text: string;
    }> = [];

    switch (session.errorType) {
      case 'confidence_drop':
        suggestions.push(...this.suggestConfidenceImprovements(session));
        break;
      case 'runtime_error':
        suggestions.push(...this.suggestErrorFixes(session));
        break;
      case 'phase_instability':
        suggestions.push(...this.suggestPhaseStabilization(session));
        break;
    }

    // Add suggestions from past successful patterns
    suggestions.push(...this.suggestFromSuccessPatterns(session));

    return suggestions.sort((a, b) => b.confidence - a.confidence);
  }

  private suggestConfidenceImprovements(session: DebugSession): Array<{
    approach: string;
    confidence: number;
    text: string;
  }> {
    return [
      {
        approach: 'context_expansion',
        confidence: 0.8,
        text: 'Try providing more context or breaking down the problem into smaller steps.'
      },
      {
        approach: 'concept_clarification',
        confidence: 0.7,
        text: 'Consider clarifying key concepts or checking for ambiguous terminology.'
      },
      {
        approach: 'phase_realignment',
        confidence: 0.75,
        text: 'Reset phase alignment by reviewing core assumptions and constraints.'
      }
    ];
  }

  private suggestErrorFixes(session: DebugSession): Array<{
    approach: string;
    confidence: number;
    text: string;
  }> {
    const suggestions = [];
    const { errorMessage, stackTrace } = session.context;

    // Common error patterns
    if (errorMessage?.includes('undefined')) {
      suggestions.push({
        approach: 'null_check',
        confidence: 0.9,
        text: 'Add null/undefined checks before accessing object properties.'
      });
    }

    if (errorMessage?.includes('Cannot read property')) {
      suggestions.push({
        approach: 'property_validation',
        confidence: 0.85,
        text: 'Validate object structure before accessing nested properties.'
      });
    }

    if (stackTrace?.includes('async')) {
      suggestions.push({
        approach: 'async_handling',
        confidence: 0.8,
        text: 'Check async/await usage and promise handling.'
      });
    }

    // Generic suggestions
    suggestions.push({
      approach: 'error_boundary',
      confidence: 0.6,
      text: 'Wrap problematic code in try-catch blocks for better error handling.'
    });

    return suggestions;
  }

  private suggestPhaseStabilization(session: DebugSession): Array<{
    approach: string;
    confidence: number;
    text: string;
  }> {
    const { coherence, entropy, drift } = session.context;

    const suggestions = [];

    if (coherence < 0.3) {
      suggestions.push({
        approach: 'coherence_boost',
        confidence: 0.8,
        text: 'Increase coherence by reducing conflicting concepts or clarifying relationships.'
      });
    }

    if (entropy > 0.8) {
      suggestions.push({
        approach: 'entropy_reduction',
        confidence: 0.75,
        text: 'Reduce entropy by organizing thoughts or focusing on core concepts.'
      });
    }

    if (Math.abs(drift) > 0.5) {
      suggestions.push({
        approach: 'drift_correction',
        confidence: 0.7,
        text: 'Correct phase drift by returning to established patterns or constraints.'
      });
    }

    return suggestions;
  }

  private suggestFromSuccessPatterns(session: DebugSession): Array<{
    approach: string;
    confidence: number;
    text: string;
  }> {
    const patterns = this.successPatterns.get(session.errorType) || [];
    
    return patterns.map(pattern => ({
      approach: 'historical_success',
      confidence: pattern.confidence * 0.8, // Slightly reduce confidence for historical patterns
      text: `Previously successful approach: ${pattern.description}`
    }));
  }

  private classifyInstability(coherence: number, entropy: number, drift: number): string {
    if (coherence < 0.2) return 'severe_decoherence';
    if (entropy > 0.9) return 'extreme_chaos';
    if (Math.abs(drift) > 0.8) return 'major_drift';
    if (coherence < 0.5 && entropy > 0.6) return 'mixed_instability';
    return 'mild_instability';
  }

  // Apply a debug suggestion
  async applySuggestion(sessionId: string, attemptId: string): Promise<boolean> {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    const attempt = session.attempts.find(a => a.id === attemptId);
    if (!attempt || attempt.applied) return false;

    attempt.applied = true;

    try {
      const result = await this.executeDebugApproach(session, attempt);
      attempt.result = result;

      if (result === 'success') {
        session.resolved = true;
        session.successfulSolution = attempt.suggestion;
        
        // Learn from success
        if (this.config.learningEnabled) {
          this.recordSuccess(session, attempt);
        }
      }

      this.notifyListeners(session);
      return result === 'success';
    } catch (error) {
      attempt.result = 'failure';
      this.notifyListeners(session);
      return false;
    }
  }

  private async executeDebugApproach(session: DebugSession, attempt: DebugAttempt): Promise<'success' | 'failure' | 'partial'> {
    // In production, this would execute actual debugging actions
    // For now, simulate based on approach and confidence
    
    const { approach, confidence } = attempt;
    
    // Simulate execution time
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    // Higher confidence approaches are more likely to succeed
    const successProbability = confidence * 0.8 + Math.random() * 0.2;
    
    if (successProbability > 0.8) {
      return 'success';
    } else if (successProbability > 0.5) {
      return 'partial';
    } else {
      return 'failure';
    }
  }

  private recordSuccess(session: DebugSession, attempt: DebugAttempt) {
    if (!this.successPatterns.has(session.errorType)) {
      this.successPatterns.set(session.errorType, []);
    }

    this.successPatterns.get(session.errorType)!.push({
      approach: attempt.approach,
      description: attempt.suggestion,
      confidence: attempt.confidence,
      context: session.context,
      timestamp: new Date()
    });

    // Keep only the most recent successful patterns
    const patterns = this.successPatterns.get(session.errorType)!;
    if (patterns.length > 10) {
      patterns.splice(0, patterns.length - 10);
    }
  }

  private notifyListeners(session: DebugSession) {
    this.listeners.forEach(listener => listener(session));
  }

  // Public API
  getActiveSessions(): DebugSession[] {
    return Array.from(this.sessions.values())
      .filter(s => !s.resolved)
      .sort((a, b) => b.startTime.getTime() - a.startTime.getTime());
  }

  getSession(sessionId: string): DebugSession | undefined {
    return this.sessions.get(sessionId);
  }

  dismissSession(sessionId: string) {
    this.sessions.delete(sessionId);
    
    document.dispatchEvent(new CustomEvent('tori-debugger-dismissed', {
      detail: { sessionId }
    }));
  }

  addListener(listener: (session: DebugSession) => void) {
    this.listeners.push(listener);
  }

  removeListener(listener: (session: DebugSession) => void) {
    const index = this.listeners.indexOf(listener);
    if (index > -1) {
      this.listeners.splice(index, 1);
    }
  }

  updateConfig(newConfig: Partial<DebuggerConfig>) {
    this.config = { ...this.config, ...newConfig };
  }

  getConfig(): DebuggerConfig {
    return { ...this.config };
  }

  // Manual debugging trigger
  startDebugSession(params: {
    errorType: string;
    context: any;
    confidence?: number;
  }): string {
    return this.createDebugSession({
      errorType: params.errorType,
      confidence: params.confidence || 0.5,
      context: params.context
    });
  }
}

// Singleton instance
export const debuggerSentinel = new DebuggerSentinel();
export default DebuggerSentinel;
export type { DebugSession, DebugAttempt, DebuggerConfig };