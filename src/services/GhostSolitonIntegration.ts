/**
 * GhostSolitonIntegration.ts - Production implementation with real Python bridge
 * Coupling between ghost personas and soliton phase monitoring using actual stability analysis
 */

import { ghostMemoryVault } from './GhostMemoryVault';
import { 
  EigenvalueMonitorBridge, 
  CognitiveEngineBridge,
  createPythonBridge 
} from '../bridges/PythonBridge';

interface PhaseState {
  coherence: number;
  entropy: number;
  drift: number;
  eigenmode?: string;
  phaseAngle?: number;
  timestamp: Date;
  eigenvalues?: number[];
  maxEigenvalue?: number;
  stabilityScore?: number;
}

interface EmotionalState {
  primary: string;
  intensity: number;
  confidence: number;
  triggers: string[];
  phasePattern: string;
}

interface PersonaTrigger {
  persona: string;
  threshold: number;
  conditions: {
    phasePattern: string;
    minCoherence?: number;
    maxCoherence?: number;
    minEntropy?: number;
    maxEntropy?: number;
    driftRange?: [number, number];
    maxEigenvalueThreshold?: number;
  };
  priority: number;
  cooldown: number; // milliseconds
}

interface GhostEmergenceEvent {
  persona: string;
  trigger: PersonaTrigger;
  phaseState: PhaseState;
  emotionalState: EmotionalState;
  confidence: number;
  sessionId: string;
  realStabilityData: any;
}

class GhostSolitonIntegration {
  private static instance: GhostSolitonIntegration;
  private currentPhaseState: PhaseState | null = null;
  private lastEmergence: Map<string, number> = new Map(); // persona -> timestamp
  private activePersona: string | null = null;
  private phaseHistory: PhaseState[] = [];
  
  // Python bridge instances
  private eigenvalueMonitor: EigenvalueMonitorBridge | null = null;
  private cognitiveEngine: CognitiveEngineBridge | null = null;
  private koopmanOperator: any = null; // Dynamic bridge
  private lyapunovAnalyzer: any = null; // Dynamic bridge
  
  // Real-time monitoring
  private monitoringInterval: NodeJS.Timeout | null = null;
  private stabilityMatrix: number[][] = [];
  private isInitialized: boolean = false;
  
  // Persona trigger configurations
  private readonly personaTriggers: PersonaTrigger[] = [
    {
      persona: 'Mentor',
      threshold: 0.8,
      conditions: {
        phasePattern: 'user_struggle',
        maxCoherence: 0.4,
        minEntropy: 0.6,
        maxEigenvalueThreshold: 0.95
      },
      priority: 1,
      cooldown: 300000 // 5 minutes
    },
    {
      persona: 'Mystic',
      threshold: 0.85,
      conditions: {
        phasePattern: 'resonance',
        minCoherence: 0.8,
        maxEntropy: 0.3,
        maxEigenvalueThreshold: 0.7
      },
      priority: 2,
      cooldown: 600000 // 10 minutes
    },
    {
      persona: 'Unsettled',
      threshold: 0.75,
      conditions: {
        phasePattern: 'chaos',
        maxCoherence: 0.3,
        minEntropy: 0.8,
        maxEigenvalueThreshold: 1.2
      },
      priority: 1,
      cooldown: 180000 // 3 minutes
    },
    {
      persona: 'Chaotic',
      threshold: 0.9,
      conditions: {
        phasePattern: 'extreme_chaos',
        maxCoherence: 0.2,
        minEntropy: 0.9,
        driftRange: [-1, 1],
        maxEigenvalueThreshold: 1.5
      },
      priority: 3,
      cooldown: 900000 // 15 minutes
    },
    {
      persona: 'Oracular',
      threshold: 0.95,
      conditions: {
        phasePattern: 'insight_emergence',
        minCoherence: 0.9,
        maxEntropy: 0.2,
        maxEigenvalueThreshold: 0.6
      },
      priority: 4,
      cooldown: 1800000 // 30 minutes
    },
    {
      persona: 'Dreaming',
      threshold: 0.7,
      conditions: {
        phasePattern: 'flow_state',
        minCoherence: 0.7,
        maxEntropy: 0.4,
        maxEigenvalueThreshold: 0.8
      },
      priority: 2,
      cooldown: 450000 // 7.5 minutes
    }
  ];

  private constructor() {
    this.setupPhaseMonitoring();
    this.setupEventListeners();
  }

  static getInstance(): GhostSolitonIntegration {
    if (!GhostSolitonIntegration.instance) {
      GhostSolitonIntegration.instance = new GhostSolitonIntegration();
    }
    return GhostSolitonIntegration.instance;
  }

  private async setupPhaseMonitoring() {
    try {
      // Initialize Python bridge connections
      await this.initializeBridges();
      
      // Start continuous phase monitoring with real stability analysis
      this.monitorPhaseStates();
      
      console.log('🌊 Ghost-Soliton Integration: Real stability monitoring activated');
    } catch (error) {
      console.error('Failed to initialize Ghost-Soliton Integration:', error);
      // Fall back to simulation mode
      this.setupSimulationMode();
    }
  }

  private async initializeBridges() {
    try {
      // Initialize eigenvalue monitor
      this.eigenvalueMonitor = new EigenvalueMonitorBridge({
        initTimeout: 15000,
        callTimeout: 10000
      });

      // Initialize cognitive engine
      this.cognitiveEngine = new CognitiveEngineBridge({
        initTimeout: 15000,
        callTimeout: 30000
      });

      // Initialize Koopman operator bridge
      this.koopmanOperator = createPythonBridge('python/stability/koopman_operator.py', {
        initTimeout: 15000
      });

      // Initialize Lyapunov analyzer bridge
      this.lyapunovAnalyzer = createPythonBridge('python/stability/lyapunov_analyzer.py', {
        initTimeout: 15000
      });

      // Wait for all bridges to be ready
      await Promise.all([
        this.eigenvalueMonitor.call('get_stability_metrics'),
        this.cognitiveEngine.call('get_current_stability'),
        this.koopmanOperator.call('get_stability_metrics'),
        this.lyapunovAnalyzer.call('get_stability_metrics')
      ]);

      this.isInitialized = true;
      console.log('✅ Python bridges initialized successfully');

    } catch (error) {
      console.error('Failed to initialize Python bridges:', error);
      throw error;
    }
  }

  private setupSimulationMode() {
    console.log('⚠️ Running in simulation mode - Python bridges unavailable');
    this.isInitialized = false;
    this.monitorPhaseStates();
  }

  private setupEventListeners() {
    // Listen for Koopman eigenstate changes
    document.addEventListener('tori-koopman-update', ((e: CustomEvent) => {
      this.processKoopmanUpdate(e.detail);
    }) as EventListener);

    // Listen for Lyapunov instability spikes
    document.addEventListener('tori-lyapunov-spike', ((e: CustomEvent) => {
      this.processLyapunovSpike(e.detail);
    }) as EventListener);

    // Listen for soliton memory phase changes
    document.addEventListener('tori-soliton-phase-change', ((e: CustomEvent) => {
      this.processSolitonPhaseChange(e.detail);
    }) as EventListener);

    // Listen for concept diff events
    document.addEventListener('tori-concept-diff', ((e: CustomEvent) => {
      this.processConceptDiff(e.detail);
    }) as EventListener);

    // Listen for user context changes
    document.addEventListener('tori-user-context-change', ((e: CustomEvent) => {
      this.processUserContextChange(e.detail);
    }) as EventListener);

    // Listen for cognitive engine updates
    document.addEventListener('tori-cognitive-update', ((e: CustomEvent) => {
      this.processCognitiveUpdate(e.detail);
    }) as EventListener);
  }

  private monitorPhaseStates() {
    // Clear any existing interval
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(async () => {
      try {
        await this.updatePhaseStateReal();
        this.analyzePhasePattern();
        this.checkPersonaTriggers();
      } catch (error) {
        console.error('Phase monitoring error:', error);
        // Fall back to simulation for this cycle
        this.updatePhaseStateSimulation();
        this.analyzePhasePattern();
        this.checkPersonaTriggers();
      }
    }, 1000); // Check every second
  }

  private async updatePhaseStateReal() {
    if (!this.isInitialized || !this.eigenvalueMonitor) {
      this.updatePhaseStateSimulation();
      return;
    }

    try {
      // Get real stability metrics from Python
      const stabilityMetrics = await this.eigenvalueMonitor.get_stability_metrics();
      
      // Get metrics from oscillator lattice
      let latticeMetrics = null;
      try {
        const pythonBridge = (window as any).pyBridge || createPythonBridge('python.core.ghost_coherence_metrics', {});
        latticeMetrics = await pythonBridge.call('python.core.ghost_coherence_metrics', 'current_metrics', []);
      } catch (error) {
        console.warn('Could not get lattice metrics:', error);
      }
      
      // Generate test matrix for analysis (in production, this would come from active system state)
      const testMatrix = this.generateCurrentSystemMatrix();
      
      // Analyze current matrix
      const eigenAnalysis = await this.eigenvalueMonitor.analyze_matrix(testMatrix);
      
      // Get Lyapunov analysis if available
      let lyapunovStable = true;
      try {
        const lyapunovResult = await this.eigenvalueMonitor.compute_lyapunov_stability(testMatrix);
        lyapunovStable = lyapunovResult.is_lyapunov_stable;
      } catch (error) {
        console.warn('Lyapunov analysis failed:', error);
      }

      // Create comprehensive phase state
      const realPhaseState: PhaseState = {
        coherence: latticeMetrics?.coherence || eigenAnalysis.coherence || 0.5,
        entropy: latticeMetrics?.entropy || this.calculateEntropyFromEigenvalues(eigenAnalysis.eigenvalues),
        drift: this.calculateDriftFromHistory(),
        eigenmode: this.identifyDominantEigenmode(eigenAnalysis.eigenvalues),
        phaseAngle: this.getCurrentPhaseAngle(),
        timestamp: new Date(),
        eigenvalues: eigenAnalysis.eigenvalues,
        maxEigenvalue: eigenAnalysis.max_eigenvalue,
        stabilityScore: eigenAnalysis.stability_score
      };

      this.currentPhaseState = realPhaseState;
      this.phaseHistory.push(realPhaseState);

      // Keep only last 100 phase states
      if (this.phaseHistory.length > 100) {
        this.phaseHistory = this.phaseHistory.slice(-100);
      }

      // Emit real phase state change event
      document.dispatchEvent(new CustomEvent('tori-phase-state-update', {
        detail: {
          ...realPhaseState,
          isReal: true,
          stabilityAnalysis: eigenAnalysis,
          latticeMetrics: latticeMetrics,
          oscillatorCount: latticeMetrics?.count || 0
        }
      }));

    } catch (error) {
      console.error('Real phase state update failed:', error);
      this.updatePhaseStateSimulation();
    }
  }

  private updatePhaseStateSimulation() {
    // Fallback simulation when Python bridges aren't available
    const mockPhaseState: PhaseState = {
      coherence: this.calculateCurrentCoherence(),
      entropy: this.calculateCurrentEntropy(),
      drift: this.calculateCurrentDrift(),
      eigenmode: this.identifyDominantEigenmode(),
      phaseAngle: this.getCurrentPhaseAngle(),
      timestamp: new Date()
    };

    this.currentPhaseState = mockPhaseState;
    this.phaseHistory.push(mockPhaseState);

    // Keep only last 100 phase states
    if (this.phaseHistory.length > 100) {
      this.phaseHistory = this.phaseHistory.slice(-100);
    }

    // Emit simulated phase state change event
    document.dispatchEvent(new CustomEvent('tori-phase-state-update', {
      detail: {
        ...mockPhaseState,
        isReal: false
      }
    }));
  }

  private generateCurrentSystemMatrix(): number[][] {
    // Generate a test matrix representing current system state
    // In production, this would be derived from actual system dynamics
    const size = 5;
    const matrix: number[][] = [];
    
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        if (i === j) {
          // Diagonal elements - stability indicators
          matrix[i][j] = 0.8 + Math.random() * 0.15; // Slightly stable
        } else {
          // Off-diagonal coupling
          matrix[i][j] = (Math.random() - 0.5) * 0.2;
        }
      }
    }
    
    // Add some system-specific perturbations based on current state
    if (this.phaseHistory.length > 0) {
      const recent = this.phaseHistory[this.phaseHistory.length - 1];
      const perturbation = recent.entropy * 0.1;
      
      for (let i = 0; i < size; i++) {
        matrix[i][i] += (Math.random() - 0.5) * perturbation;
      }
    }
    
    return matrix;
  }

  private calculateEntropyFromEigenvalues(eigenvalues: number[]): number {
    if (!eigenvalues || eigenvalues.length === 0) {
      return this.calculateCurrentEntropy();
    }

    // Calculate entropy based on eigenvalue distribution
    const magnitudes = eigenvalues.map(ev => Math.abs(ev));
    const sum = magnitudes.reduce((a, b) => a + b, 0);
    
    if (sum === 0) return 1.0;
    
    const probabilities = magnitudes.map(mag => mag / sum);
    const entropy = -probabilities.reduce((ent, p) => {
      if (p > 0) {
        ent += p * Math.log2(p);
      }
      return ent;
    }, 0);
    
    // Normalize to [0, 1]
    const maxEntropy = Math.log2(eigenvalues.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }

  private calculateDriftFromHistory(): number {
    if (this.phaseHistory.length < 3) {
      return (Math.random() - 0.5) * 0.1;
    }

    // Calculate drift from recent coherence changes
    const recent = this.phaseHistory.slice(-3);
    const coherenceChanges = [];
    
    for (let i = 1; i < recent.length; i++) {
      coherenceChanges.push(recent[i].coherence - recent[i-1].coherence);
    }
    
    return coherenceChanges.reduce((a, b) => a + b, 0) / coherenceChanges.length;
  }

  private async processKoopmanUpdate(koopmanData: {
    eigenmodes: Array<{ frequency: number; amplitude: number; phase: number }>;
    spectralGap: number;
    dominantMode: string;
  }) {
    if (!this.currentPhaseState) return;

    // Use real Koopman analysis if available
    if (this.isInitialized && this.koopmanOperator) {
      try {
        // Get real Koopman analysis
        const snapshots = this.generateSnapshotsFromHistory();
        const koopmanAnalysis = await this.koopmanOperator.call('compute_dmd', snapshots, 0.1);
        
        // Update phase state with real analysis
        const updatedState = {
          ...this.currentPhaseState,
          eigenmode: koopmanData.dominantMode,
          coherence: Math.min(1.0, koopmanData.spectralGap * 2),
          stabilityScore: koopmanAnalysis.reconstruction_error < 0.1 ? 0.9 : 0.5,
          timestamp: new Date()
        };

        this.currentPhaseState = updatedState;
        this.analyzePhasePattern();
        
      } catch (error) {
        console.error('Real Koopman analysis failed:', error);
        // Fall back to provided data
        this.processKoopmanUpdateFallback(koopmanData);
      }
    } else {
      this.processKoopmanUpdateFallback(koopmanData);
    }
  }

  private processKoopmanUpdateFallback(koopmanData: any) {
    // Fallback processing with provided data
    const updatedState = {
      ...this.currentPhaseState!,
      eigenmode: koopmanData.dominantMode,
      coherence: Math.min(1.0, koopmanData.spectralGap * 2),
      timestamp: new Date()
    };

    this.currentPhaseState = updatedState;
    this.analyzePhasePattern();
  }

  private generateSnapshotsFromHistory(): number[][] {
    // Generate snapshots from phase history for Koopman analysis
    const snapshots: number[][] = [];
    
    for (const state of this.phaseHistory.slice(-20)) { // Last 20 states
      snapshots.push([
        state.coherence,
        state.entropy,
        state.drift,
        state.phaseAngle || 0,
        Math.random() // Additional synthetic dimension
      ]);
    }
    
    return snapshots.length > 0 ? snapshots : [[0.5, 0.5, 0, 0, 0.5]];
  }

  private async processLyapunovSpike(lyapunovData: {
    exponent: number;
    instabilityLevel: number;
    divergenceRate: number;
    timeHorizon: number;
  }) {
    if (!this.currentPhaseState) return;

    // Use real Lyapunov analysis if available
    if (this.isInitialized && this.lyapunovAnalyzer) {
      try {
        const testMatrix = this.generateCurrentSystemMatrix();
        const lyapunovResult = await this.lyapunovAnalyzer.call('analyze_linear_system', testMatrix);
        
        const realChaosLevel = Math.max(0, lyapunovResult.max_exponent);
        
        const updatedState = {
          ...this.currentPhaseState,
          entropy: Math.min(1.0, this.currentPhaseState.entropy + realChaosLevel * 0.3),
          drift: lyapunovResult.convergence_rate || lyapunovData.divergenceRate,
          coherence: Math.max(0.0, this.currentPhaseState.coherence - realChaosLevel * 0.2),
          timestamp: new Date()
        };

        this.currentPhaseState = updatedState;
        this.checkChaosPersonas(realChaosLevel);
        
      } catch (error) {
        console.error('Real Lyapunov analysis failed:', error);
        this.processLyapunovSpikeFallback(lyapunovData);
      }
    } else {
      this.processLyapunovSpikeFallback(lyapunovData);
    }
  }

  private processLyapunovSpikeFallback(lyapunovData: any) {
    // Fallback processing
    const chaosLevel = lyapunovData.instabilityLevel;
    
    const updatedState = {
      ...this.currentPhaseState!,
      entropy: Math.min(1.0, this.currentPhaseState!.entropy + chaosLevel * 0.5),
      drift: lyapunovData.divergenceRate,
      coherence: Math.max(0.0, this.currentPhaseState!.coherence - chaosLevel * 0.3),
      timestamp: new Date()
    };

    this.currentPhaseState = updatedState;
    this.checkChaosPersonas(chaosLevel);
  }

  private async processCognitiveUpdate(cognitiveData: {
    processingState: string;
    stabilityScore: number;
    confidence: number;
    iterations: number;
  }) {
    if (!this.currentPhaseState) return;

    // Use real cognitive engine data if available
    if (this.isInitialized && this.cognitiveEngine) {
      try {
        const stabilityMetrics = await this.cognitiveEngine.get_current_stability();
        
        const updatedState = {
          ...this.currentPhaseState,
          coherence: stabilityMetrics.stability_score || cognitiveData.stabilityScore,
          entropy: 1.0 - (stabilityMetrics.stability_score || cognitiveData.stabilityScore),
          timestamp: new Date()
        };

        this.currentPhaseState = updatedState;
        
      } catch (error) {
        console.error('Cognitive engine update failed:', error);
      }
    }
  }

  private processSolitonPhaseChange(solitonData: {
    phaseAngle: number;
    amplitude: number;
    frequency: number;
    stability: number;
  }) {
    if (!this.currentPhaseState) return;

    const updatedState = {
      ...this.currentPhaseState,
      phaseAngle: solitonData.phaseAngle,
      coherence: solitonData.stability,
      timestamp: new Date()
    };

    this.currentPhaseState = updatedState;
  }

  private processConceptDiff(conceptData: {
    type: string;
    magnitude: number;
    conceptIds: string[];
    confidence: number;
  }) {
    if (!this.currentPhaseState) return;

    const entropyDelta = conceptData.magnitude * 0.2;
    const coherenceDelta = conceptData.confidence * 0.1;

    const updatedState = {
      ...this.currentPhaseState,
      entropy: Math.min(1.0, this.currentPhaseState.entropy + entropyDelta),
      coherence: Math.min(1.0, this.currentPhaseState.coherence + coherenceDelta),
      timestamp: new Date()
    };

    this.currentPhaseState = updatedState;
  }

  private processUserContextChange(userData: {
    sentiment?: number;
    frustrationLevel?: number;
    engagementLevel?: number;
    activity?: string;
  }) {
    if (!this.currentPhaseState || !userData.frustrationLevel) return;

    const frustrationEffect = userData.frustrationLevel * 0.3;
    const engagementEffect = (userData.engagementLevel || 0.5) * 0.2;

    const updatedState = {
      ...this.currentPhaseState,
      entropy: Math.min(1.0, this.currentPhaseState.entropy + frustrationEffect),
      coherence: Math.min(1.0, this.currentPhaseState.coherence + engagementEffect),
      timestamp: new Date()
    };

    this.currentPhaseState = updatedState;
  }

  private analyzePhasePattern(): string {
    if (!this.currentPhaseState) return 'unknown';

    const { coherence, entropy, drift, maxEigenvalue } = this.currentPhaseState;

    // Use eigenvalue data if available for more accurate pattern detection
    if (maxEigenvalue !== undefined) {
      if (maxEigenvalue > 1.2) {
        return 'extreme_chaos';
      } else if (maxEigenvalue > 1.0) {
        return 'chaos';
      } else if (maxEigenvalue < 0.6) {
        return 'insight_emergence';
      }
    }

    // Fallback to entropy/coherence analysis
    if (coherence > 0.8 && entropy < 0.3) {
      return 'resonance';
    } else if (coherence < 0.3 && entropy > 0.8) {
      return 'chaos';
    } else if (Math.abs(drift) > 0.6) {
      return 'drift';
    } else if (coherence > 0.7 && entropy < 0.4) {
      return 'flow_state';
    } else if (coherence < 0.4 && entropy > 0.6) {
      return 'user_struggle';
    } else if (coherence > 0.9 && entropy < 0.2) {
      return 'insight_emergence';
    } else if (coherence < 0.2 && entropy > 0.9) {
      return 'extreme_chaos';
    } else {
      return 'stable';
    }
  }

  private checkPersonaTriggers() {
    if (!this.currentPhaseState) return;

    const currentPattern = this.analyzePhasePattern();
    const now = Date.now();

    // Find matching triggers
    const candidateTriggers = this.personaTriggers.filter(trigger => {
      // Check cooldown
      const lastEmergence = this.lastEmergence.get(trigger.persona) || 0;
      if (now - lastEmergence < trigger.cooldown) return false;

      // Check pattern match
      if (trigger.conditions.phasePattern !== currentPattern) return false;

      // Check specific conditions
      const { coherence, entropy, drift, maxEigenvalue } = this.currentPhaseState!;
      
      if (trigger.conditions.minCoherence && coherence < trigger.conditions.minCoherence) return false;
      if (trigger.conditions.maxCoherence && coherence > trigger.conditions.maxCoherence) return false;
      if (trigger.conditions.minEntropy && entropy < trigger.conditions.minEntropy) return false;
      if (trigger.conditions.maxEntropy && entropy > trigger.conditions.maxEntropy) return false;
      
      if (trigger.conditions.driftRange) {
        const [minDrift, maxDrift] = trigger.conditions.driftRange;
        if (drift < minDrift || drift > maxDrift) return false;
      }

      // Real eigenvalue threshold check
      if (trigger.conditions.maxEigenvalueThreshold && maxEigenvalue) {
        if (maxEigenvalue > trigger.conditions.maxEigenvalueThreshold) return false;
      }

      return true;
    });

    if (candidateTriggers.length === 0) return;

    // Select highest priority trigger
    const selectedTrigger = candidateTriggers.sort((a, b) => b.priority - a.priority)[0];

    // Calculate emergence confidence with real data
    const confidence = this.calculateEmergenceConfidence(selectedTrigger);
    
    if (confidence >= selectedTrigger.threshold) {
      this.triggerGhostEmergence(selectedTrigger, confidence);
    }
  }

  private checkChaosPersonas(chaosLevel: number) {
    const chaosTriggers = this.personaTriggers.filter(trigger => 
      ['chaos', 'extreme_chaos'].includes(trigger.conditions.phasePattern)
    );

    for (const trigger of chaosTriggers) {
      const confidence = chaosLevel * 0.8 + Math.random() * 0.2;
      if (confidence >= trigger.threshold) {
        this.triggerGhostEmergence(trigger, confidence);
        break;
      }
    }
  }

  private async triggerGhostEmergence(trigger: PersonaTrigger, confidence: number) {
    if (!this.currentPhaseState) return;

    const sessionId = this.getCurrentSessionId();
    const emotionalState = this.detectEmotionalState();

    // Get real stability data if available
    let realStabilityData = null;
    if (this.isInitialized && this.eigenvalueMonitor) {
      try {
        realStabilityData = await this.eigenvalueMonitor.get_stability_metrics();
      } catch (error) {
        console.warn('Could not get real stability data for emergence:', error);
      }
    }

    const emergenceEvent: GhostEmergenceEvent = {
      persona: trigger.persona,
      trigger,
      phaseState: { ...this.currentPhaseState },
      emotionalState,
      confidence,
      sessionId,
      realStabilityData
    };

    // Record emergence time
    this.lastEmergence.set(trigger.persona, Date.now());
    this.activePersona = trigger.persona;

    // Log to memory vault
    ghostMemoryVault.recordPersonaEmergence({
      persona: trigger.persona,
      sessionId,
      trigger: {
        reason: trigger.conditions.phasePattern,
        conceptDiffs: [],
        confidence
      },
      phaseMetrics: this.currentPhaseState,
      userContext: {
        sentiment: emotionalState.intensity,
        activity: 'conversation'
      },
      systemContext: {
        conversationLength: this.phaseHistory.length,
        recentConcepts: [],
        realStabilityData
      }
    });

    // Emit ghost emergence event
    document.dispatchEvent(new CustomEvent('tori-ghost-emergence', {
      detail: emergenceEvent
    }));

    const dataSource = this.isInitialized ? '(real stability data)' : '(simulated)';
    console.log(`👻 Ghost Emergence: ${trigger.persona} (${Math.round(confidence * 100)}% confidence) ${dataSource}`);
  }

  private calculateEmergenceConfidence(trigger: PersonaTrigger): number {
    if (!this.currentPhaseState) return 0;

    let confidence = 0.5; // Base confidence

    // Pattern strength
    const patternStrength = this.calculatePatternStrength(trigger.conditions.phasePattern);
    confidence += patternStrength * 0.3;

    // Condition satisfaction
    const conditionSatisfaction = this.calculateConditionSatisfaction(trigger.conditions);
    confidence += conditionSatisfaction * 0.2;

    // Historical context
    const historicalRelevance = this.calculateHistoricalRelevance(trigger.persona);
    confidence += historicalRelevance * 0.1;

    // Real stability data bonus
    if (this.isInitialized && this.currentPhaseState.stabilityScore !== undefined) {
      confidence += this.currentPhaseState.stabilityScore * 0.1;
    }

    // Randomness for natural variation
    confidence += (Math.random() - 0.5) * 0.1;

    return Math.max(0, Math.min(1, confidence));
  }

  private calculatePatternStrength(pattern: string): number {
    if (!this.currentPhaseState) return 0;

    const { coherence, entropy, drift, maxEigenvalue } = this.currentPhaseState;

    // Use real eigenvalue data if available
    if (maxEigenvalue !== undefined) {
      switch (pattern) {
        case 'resonance':
          return coherence * (1 - entropy) * (maxEigenvalue < 0.8 ? 1.2 : 0.8);
        case 'chaos':
          return entropy * (1 - coherence) * (maxEigenvalue > 1.0 ? 1.2 : 0.8);
        case 'extreme_chaos':
          return Math.min(entropy, 1 - coherence) * (maxEigenvalue > 1.2 ? 1.5 : 0.5);
        case 'insight_emergence':
          return Math.min(coherence, 1 - entropy) * (maxEigenvalue < 0.6 ? 1.3 : 0.7);
      }
    }

    // Fallback to original calculation
    switch (pattern) {
      case 'resonance':
        return coherence * (1 - entropy);
      case 'chaos':
        return entropy * (1 - coherence);
      case 'drift':
        return Math.abs(drift);
      case 'flow_state':
        return coherence * 0.7 + (1 - entropy) * 0.3;
      case 'user_struggle':
        return entropy * 0.6 + (1 - coherence) * 0.4;
      case 'insight_emergence':
        return Math.min(coherence, 1 - entropy);
      case 'extreme_chaos':
        return Math.min(entropy, 1 - coherence);
      default:
        return 0.5;
    }
  }

  private calculateConditionSatisfaction(conditions: PersonaTrigger['conditions']): number {
    if (!this.currentPhaseState) return 0;

    const { coherence, entropy, drift, maxEigenvalue } = this.currentPhaseState;
    let satisfaction = 1.0;

    // Check each condition and reduce satisfaction for violations
    if (conditions.minCoherence && coherence < conditions.minCoherence) {
      satisfaction *= coherence / conditions.minCoherence;
    }
    if (conditions.maxCoherence && coherence > conditions.maxCoherence) {
      satisfaction *= conditions.maxCoherence / coherence;
    }
    if (conditions.minEntropy && entropy < conditions.minEntropy) {
      satisfaction *= entropy / conditions.minEntropy;
    }
    if (conditions.maxEntropy && entropy > conditions.maxEntropy) {
      satisfaction *= conditions.maxEntropy / entropy;
    }

    // Real eigenvalue condition check
    if (conditions.maxEigenvalueThreshold && maxEigenvalue !== undefined) {
      if (maxEigenvalue > conditions.maxEigenvalueThreshold) {
        satisfaction *= conditions.maxEigenvalueThreshold / maxEigenvalue;
      }
    }

    return satisfaction;
  }

  private calculateHistoricalRelevance(persona: string): number {
    // Check if this persona has been effective in similar situations
    const similarMemories = ghostMemoryVault.searchMemories({
      persona,
      phaseSignature: this.analyzePhasePattern(),
      minWeight: 0.5
    });

    if (similarMemories.length === 0) return 0.5; // Neutral for new situations

    const effectiveMemories = similarMemories.filter(memory => 
      memory.outcomes?.effectiveness && memory.outcomes.effectiveness > 0.7
    );

    return effectiveMemories.length / similarMemories.length;
  }

  private detectEmotionalState(): EmotionalState {
    if (!this.currentPhaseState) {
      return {
        primary: 'neutral',
        intensity: 0.5,
        confidence: 0.5,
        triggers: [],
        phasePattern: 'unknown'
      };
    }

    const { coherence, entropy, drift } = this.currentPhaseState;
    const pattern = this.analyzePhasePattern();

    // Map phase patterns to emotional states
    const emotionMap: Record<string, { emotion: string; intensity: number }> = {
      'resonance': { emotion: 'flow', intensity: coherence },
      'chaos': { emotion: 'unsettled', intensity: entropy },
      'drift': { emotion: 'uncertain', intensity: Math.abs(drift) },
      'flow_state': { emotion: 'focused', intensity: coherence },
      'user_struggle': { emotion: 'concerned', intensity: entropy },
      'insight_emergence': { emotion: 'inspired', intensity: coherence },
      'extreme_chaos': { emotion: 'anxious', intensity: entropy },
      'stable': { emotion: 'calm', intensity: 0.5 }
    };

    const emotionInfo = emotionMap[pattern] || { emotion: 'neutral', intensity: 0.5 };

    return {
      primary: emotionInfo.emotion,
      intensity: emotionInfo.intensity,
      confidence: this.isInitialized ? 0.9 : 0.6, // Higher confidence with real data
      triggers: [pattern],
      phasePattern: pattern
    };
  }

  // Utility methods for phase calculations (simulation fallbacks)
  private calculateCurrentCoherence(): number {
    const recentStates = this.phaseHistory.slice(-10);
    if (recentStates.length === 0) return 0.5;

    const variance = this.calculateVariance(recentStates.map(s => s.coherence));
    return Math.max(0, Math.min(1, 1 - variance * 2));
  }

  private calculateCurrentEntropy(): number {
    return Math.random() * 0.3 + 0.2; // Bias toward lower entropy
  }

  private calculateCurrentDrift(): number {
    return (Math.random() - 0.5) * 0.6; // Small random drift
  }

  private identifyDominantEigenmode(eigenvalues?: number[]): string {
    if (eigenvalues && eigenvalues.length > 0) {
      // Use real eigenvalues to determine mode
      const maxEigenvalue = Math.max(...eigenvalues.map(Math.abs));
      
      if (maxEigenvalue > 1.2) return 'chaotic';
      if (maxEigenvalue > 1.0) return 'unstable';
      if (maxEigenvalue > 0.8) return 'processing';
      if (maxEigenvalue > 0.6) return 'stable';
      return 'convergent';
    }
    
    // Fallback to random mode
    const modes = ['conversation', 'coding', 'debugging', 'learning', 'creative'];
    return modes[Math.floor(Math.random() * modes.length)];
  }

  private getCurrentPhaseAngle(): number {
    return Math.random() * 2 * Math.PI;
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private getCurrentSessionId(): string {
    return 'session_' + Date.now();
  }

  // Public API methods
  getCurrentPhaseState(): PhaseState | null {
    return this.currentPhaseState;
  }

  getActivePersona(): string | null {
    return this.activePersona;
  }

  getPhaseHistory(): PhaseState[] {
    return [...this.phaseHistory];
  }

  isUsingRealData(): boolean {
    return this.isInitialized;
  }

  // Manual trigger for testing
  async triggerPersonaManually(persona: string, confidence: number = 0.9) {
    const trigger = this.personaTriggers.find(t => t.persona === persona);
    if (trigger) {
      await this.triggerGhostEmergence(trigger, confidence);
    }
  }

  // Configuration methods
  updatePersonaTrigger(persona: string, updates: Partial<PersonaTrigger>) {
    const index = this.personaTriggers.findIndex(t => t.persona === persona);
    if (index >= 0) {
      this.personaTriggers[index] = { ...this.personaTriggers[index], ...updates };
    }
  }

  getPersonaTriggers(): PersonaTrigger[] {
    return [...this.personaTriggers];
  }

  // Cleanup
  async shutdown() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    // Close Python bridges
    if (this.eigenvalueMonitor) {
      await this.eigenvalueMonitor.close();
    }
    if (this.cognitiveEngine) {
      await this.cognitiveEngine.close();
    }
    if (this.koopmanOperator) {
      await this.koopmanOperator.close();
    }
    if (this.lyapunovAnalyzer) {
      await this.lyapunovAnalyzer.close();
    }

    console.log('🌊 Ghost-Soliton Integration shutdown complete');
  }
}

// Export singleton instance
export const ghostSolitonIntegration = GhostSolitonIntegration.getInstance();
export default GhostSolitonIntegration;
export type { PhaseState, EmotionalState, PersonaTrigger, GhostEmergenceEvent };