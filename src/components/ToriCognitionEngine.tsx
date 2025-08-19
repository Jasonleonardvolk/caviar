/**
 * ToriCognitionEngine.tsx - Master integration component for Phase 10
 * Orchestrates holographic projection, ghost personas, agents, memory vault, and Banksy oscillators
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import HologramCanvas from '../ui/HologramCanvas';
import ConceptDiffVisualizer from '../ui/ConceptDiffVisualizer';
import AgentFieldProjector from '../ui/AgentFieldProjector';
import GhostChronicle from '../ghost/GhostChronicle';
import GhostLetterGenerator from '../ghost/GhostLetterGenerator';
import { BanksyOscillatorPanel } from './banksy';
import { ghostSolitonIntegration } from '../../services/GhostSolitonIntegration';
import { ghostMemoryVault } from '../../services/GhostMemoryVault';
import type { SpectralMetadata, GhostOverlay } from '../ui/HologramCanvas';
import type { AgentState } from '../ui/AgentFieldProjector';
import type { GhostEvent } from '../ghost/GhostChronicle';
import type { BanksyState } from './banksy';

interface ToriCognitionEngineProps {
  sessionId: string;
  isEnabled?: boolean;
  showDebugPanel?: boolean;
  onPersonaEmergence?: (persona: string) => void;
  onConceptActivity?: (activity: any) => void;
  className?: string;
}

interface CognitionState {
  // Phase monitoring
  phaseCoherence: number;
  phaseEntropy: number;
  phaseDrift: number;
  
  // Ghost system
  activeGhost: {
    persona: string;
    intensity: number;
    wavelength: number;
  } | null;
  ghostLetterVisible: boolean;
  ghostLetterData: any;
  
  // Agent system
  agents: AgentState[];
  fieldHarmony: number;
  
  // Holographic projection
  spectralMetadata: SpectralMetadata;
  conceptActivity: Array<{
    conceptId: string;
    intensity: number;
    resonance: number;
  }>;
  
  // Memory and continuity
  sessionMemories: any[];
  moodCurve: any[];
  
  // Banksy oscillator system
  banksyState: BanksyState | null;
  oscillatorSynchronization: number;
}

const ToriCognitionEngine: React.FC<ToriCognitionEngineProps> = ({
  sessionId,
  isEnabled = true,
  showDebugPanel = false,
  onPersonaEmergence,
  onConceptActivity,
  className = ''
}) => {
  const [cognitionState, setCognitionState] = useState<CognitionState>({
    phaseCoherence: 0.7,
    phaseEntropy: 0.3,
    phaseDrift: 0.0,
    activeGhost: null,
    ghostLetterVisible: false,
    ghostLetterData: null,
    agents: [],
    fieldHarmony: 0.7,
    spectralMetadata: {},
    conceptActivity: [],
    sessionMemories: [],
    moodCurve: [],
    banksyState: null,
    oscillatorSynchronization: 0.0
  });

  const [isInitialized, setIsInitialized] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState({
    frameRate: 0,
    memoryUsage: 0,
    activeComponents: 0,
    banksyConnected: false
  });

  const hologramCanvasRef = useRef<any>(null);
  const lastFrameTime = useRef(Date.now());
  const frameCount = useRef(0);

  // Initialize the cognition engine
  useEffect(() => {
    if (!isEnabled) return;

    initializeCognitionEngine();
    setIsInitialized(true);

    return () => {
      cleanupCognitionEngine();
    };
  }, [isEnabled, sessionId]);

  // Monitor performance
  useEffect(() => {
    if (!isEnabled) return;

    const performanceInterval = setInterval(() => {
      updatePerformanceMetrics();
    }, 1000);

    return () => clearInterval(performanceInterval);
  }, [isEnabled]);

  const initializeCognitionEngine = () => {
    console.log('🧠 TORI Cognition Engine: Initializing Phase 10 systems with Banksy oscillators...');

    // Setup event listeners for all subsystems
    setupGhostEventListeners();
    setupAgentEventListeners();
    setupPhaseEventListeners();
    setupMemoryEventListeners();
    setupConceptEventListeners();
    setupBanksyEventListeners();

    // Initialize ghost-soliton integration
    ghostSolitonIntegration.getCurrentPhaseState();

    // Load session data
    loadSessionData();

    console.log('✅ TORI Cognition Engine: All systems operational including Banksy oscillators');
  };

  const setupBanksyEventListeners = () => {
    // Banksy state updates from oscillator panel
    const handleBanksyStateUpdate = (state: BanksyState) => {
      setCognitionState(prev => ({
        ...prev,
        banksyState: state,
        oscillatorSynchronization: state.order_parameter,
        // Sync oscillator state with other cognition systems
        phaseCoherence: Math.max(prev.phaseCoherence, state.order_parameter * 0.3),
        fieldHarmony: (prev.fieldHarmony + state.order_parameter) / 2
      }));

      // Trigger holographic effects based on synchronization
      if (state.order_parameter > 0.8 && hologramCanvasRef.current) {
        hologramCanvasRef.current.triggerConceptDiffPulse(
          window.innerWidth / 2,
          window.innerHeight / 2,
          state.order_parameter,
          'synchronization'
        );
      }

      // Update performance metrics
      setPerformanceMetrics(prev => ({
        ...prev,
        banksyConnected: true
      }));

      console.log(`🌀 Banksy Oscillator State: Order=${state.order_parameter.toFixed(3)}, Step=${state.step}`);
    };

    return { handleBanksyStateUpdate };
  };

  const setupGhostEventListeners = () => {
    // Ghost emergence
    const handleGhostEmergence = (event: CustomEvent) => {
      const { persona, confidence, phaseState } = event.detail;
      
      const wavelengthMap: Record<string, number> = {
        'Mentor': 520,
        'Mystic': 450,
        'Unsettled': 620,
        'Chaotic': 680,
        'Oracular': 400,
        'Dreaming': 380
      };

      setCognitionState(prev => ({
        ...prev,
        activeGhost: {
          persona,
          intensity: confidence,
          wavelength: wavelengthMap[persona] || 550
        }
      }));

      if (onPersonaEmergence) {
        onPersonaEmergence(persona);
      }

      console.log(`👻 Ghost Emergence: ${persona} activated`);
    };

    // Ghost letter generation
    const handleGhostLetter = (event: CustomEvent) => {
      setCognitionState(prev => ({
        ...prev,
        ghostLetterVisible: true,
        ghostLetterData: event.detail
      }));
    };

    // Ghost mood updates
    const handleGhostMood = (event: CustomEvent) => {
      const moodData = event.detail;
      setCognitionState(prev => ({
        ...prev,
        moodCurve: [...prev.moodCurve, moodData].slice(-50) // Keep last 50 points
      }));
    };

    document.addEventListener('tori-ghost-emergence', handleGhostEmergence as EventListener);
    document.addEventListener('tori-ghost-letter', handleGhostLetter as EventListener);
    document.addEventListener('tori-ghost-mood', handleGhostMood as EventListener);
  };

  const setupAgentEventListeners = () => {
    // Agent activity updates
    const handleAgentActivity = (event: CustomEvent) => {
      const { agentId, metrics } = event.detail;
      
      setCognitionState(prev => {
        const updatedAgents = prev.agents.map(agent =>
          agent.id === agentId
            ? {
                ...agent,
                isActive: true,
                conceptEntropy: metrics.entropy || agent.conceptEntropy,
                conceptResonance: metrics.resonance || agent.conceptResonance,
                confidence: metrics.confidence || agent.confidence,
                lastActivity: new Date()
              }
            : agent
        );
        return { ...prev, agents: updatedAgents };
      });
    };

    // Field harmony updates
    const handleFieldHarmony = (event: CustomEvent) => {
      setCognitionState(prev => ({
        ...prev,
        fieldHarmony: event.detail.harmony
      }));
    };

    document.addEventListener('tori-agent-activity', handleAgentActivity as EventListener);
    document.addEventListener('tori-field-harmony', handleFieldHarmony as EventListener);
  };

  const setupPhaseEventListeners = () => {
    // Phase state updates
    const handlePhaseUpdate = (event: CustomEvent) => {
      const { coherence, entropy, drift } = event.detail;
      
      setCognitionState(prev => ({
        ...prev,
        phaseCoherence: coherence,
        phaseEntropy: entropy,
        phaseDrift: drift,
        spectralMetadata: {
          ...prev.spectralMetadata,
          phase_signature: determinePhaseSignature(coherence, entropy, drift)
        }
      }));
    };

    // Koopman updates
    const handleKoopmanUpdate = (event: CustomEvent) => {
      console.log('🌊 Koopman Update:', event.detail);
    };

    // Lyapunov spikes
    const handleLyapunovSpike = (event: CustomEvent) => {
      console.log('⚡ Lyapunov Spike:', event.detail);
      
      // Trigger visual disturbance in hologram
      if (hologramCanvasRef.current) {
        hologramCanvasRef.current.triggerConceptDiffPulse(
          window.innerWidth / 2,
          window.innerHeight / 2,
          event.detail.instabilityLevel,
          'phaseShift'
        );
      }
    };

    document.addEventListener('tori-phase-state-update', handlePhaseUpdate as EventListener);
    document.addEventListener('tori-koopman-update', handleKoopmanUpdate as EventListener);
    document.addEventListener('tori-lyapunov-spike', handleLyapunovSpike as EventListener);
  };

  const setupMemoryEventListeners = () => {
    // Memory vault updates
    const handleMemoryUpdate = (event: CustomEvent) => {
      const memoryData = event.detail;
      setCognitionState(prev => ({
        ...prev,
        sessionMemories: [...prev.sessionMemories, memoryData].slice(-20) // Keep last 20
      }));
    };

    document.addEventListener('tori-memory-update', handleMemoryUpdate as EventListener);
  };

  const setupConceptEventListeners = () => {
    // Concept activity updates
    const handleConceptActivity = (activity: Array<{
      conceptId: string;
      intensity: number;
      resonance: number;
    }>) => {
      setCognitionState(prev => ({
        ...prev,
        conceptActivity: activity
      }));

      if (onConceptActivity) {
        onConceptActivity(activity);
      }
    };

    // Concept diff pulses
    const handleConceptDiffPulse = (x: number, y: number, magnitude: number, diffType: string) => {
      if (hologramCanvasRef.current) {
        hologramCanvasRef.current.triggerConceptDiffPulse(x, y, magnitude, diffType);
      }
    };

    // We'll set these up in the ConceptDiffVisualizer component
    return { handleConceptActivity, handleConceptDiffPulse };
  };

  const loadSessionData = async () => {
    try {
      // Load session summary from memory vault
      const sessionSummary = ghostMemoryVault.getSessionSummary(sessionId);
      
      setCognitionState(prev => ({
        ...prev,
        sessionMemories: sessionSummary.memories || [],
        moodCurve: sessionSummary.moodCurve?.timePoints || []
      }));
    } catch (error) {
      console.warn('Failed to load session data:', error);
    }
  };

  const determinePhaseSignature = (coherence: number, entropy: number, drift: number): string => {
    if (coherence > 0.8 && entropy < 0.3) return 'coherence';
    if (coherence < 0.3 && entropy > 0.8) return 'entropy';
    if (Math.abs(drift) > 0.5) return 'drift';
    if (coherence > 0.6 && entropy < 0.6) return 'resonance';
    return 'neutral';
  };

  const createGhostOverlay = (): GhostOverlay | null => {
    if (!cognitionState.activeGhost) return null;

    const patternMap: Record<string, GhostOverlay['pattern']> = {
      'Mentor': 'wave',
      'Mystic': 'ripple',
      'Unsettled': 'distortion',
      'Chaotic': 'crackling',
      'Oracular': 'wave',
      'Dreaming': 'ripple'
    };

    const anomalyMap: Record<string, GhostOverlay['phaseAnomaly']> = {
      'Mentor': 'deviation',
      'Mystic': 'resonance',
      'Unsettled': 'instability',
      'Chaotic': 'chaos',
      'Oracular': 'resonance',
      'Dreaming': 'deviation'
    };

    return {
      persona: cognitionState.activeGhost.persona,
      opacity: cognitionState.activeGhost.intensity,
      pattern: patternMap[cognitionState.activeGhost.persona] || 'ripple',
      wavelength: cognitionState.activeGhost.wavelength,
      phaseAnomaly: anomalyMap[cognitionState.activeGhost.persona] || 'deviation'
    };
  };

  const updatePerformanceMetrics = () => {
    const now = Date.now();
    frameCount.current++;
    
    if (now - lastFrameTime.current >= 1000) {
      const fps = Math.round((frameCount.current * 1000) / (now - lastFrameTime.current));
      
      setPerformanceMetrics(prev => ({
        ...prev,
        frameRate: fps,
        activeComponents: getActiveComponentCount()
      }));
      
      frameCount.current = 0;
      lastFrameTime.current = now;
    }
  };

  const getActiveComponentCount = (): number => {
    let count = 0;
    if (cognitionState.activeGhost) count++;
    if (cognitionState.conceptActivity.length > 0) count++;
    if (cognitionState.agents.some(a => a.isActive)) count++;
    if (cognitionState.banksyState) count++;
    return count;
  };

  const cleanupCognitionEngine = () => {
    console.log('🧠 TORI Cognition Engine: Shutting down...');
    
    // Remove all event listeners
    const events = [
      'tori-ghost-emergence',
      'tori-ghost-letter',
      'tori-ghost-mood',
      'tori-agent-activity',
      'tori-field-harmony',
      'tori-phase-state-update',
      'tori-koopman-update',
      'tori-lyapunov-spike',
      'tori-memory-update'
    ];

    events.forEach(eventName => {
      document.removeEventListener(eventName, () => {});
    });
  };

  const handleGhostLetterClose = () => {
    setCognitionState(prev => ({
      ...prev,
      ghostLetterVisible: false,
      ghostLetterData: null
    }));
  };

  const handleAgentClick = (agent: AgentState) => {
    console.log(`🤖 Agent clicked: ${agent.name}`);
    
    // Show agent details or trigger specific action
    document.dispatchEvent(new CustomEvent('tori-agent-selected', {
      detail: agent
    }));
  };

  const handleGhostEventSelect = (event: GhostEvent) => {
    console.log(`👻 Ghost event selected:`, event);
    
    // Could trigger replay or detailed view
    if (event.content?.message) {
      setCognitionState(prev => ({
        ...prev,
        ghostLetterVisible: true,
        ghostLetterData: {
          persona: event.persona,
          letterContent: event.content.message,
          conceptArc: event.conceptIds || [],
          trigger: event.trigger
        }
      }));
    }
  };

  if (!isEnabled) {
    return null;
  }

  const conceptEventHandlers = setupConceptEventListeners();
  const banksyEventHandlers = setupBanksyEventListeners();

  return (
    <div className={`tori-cognition-engine relative w-full h-full ${className}`}>
      {/* Banksy Oscillator Control Panel - TOP PRIORITY FOR LAUNCH */}
      <motion.div
        className="absolute top-20 left-4 w-96 z-20"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <BanksyOscillatorPanel 
          className="mb-4 shadow-2xl" 
          onStateChange={banksyEventHandlers.handleBanksyStateUpdate}
        />
      </motion.div>

      {/* Holographic Projection Layer */}
      <HologramCanvas
        ref={hologramCanvasRef}
        enabled={isInitialized}
        spectralMetadata={cognitionState.spectralMetadata}
        conceptActivity={cognitionState.conceptActivity}
        ghostPresence={createGhostOverlay()}
        onEffectComplete={(effectId) => console.log(`Effect ${effectId} completed`)}
      />

      {/* Concept Diff Visualizer (Headless) */}
      <ConceptDiffVisualizer
        enabled={isInitialized}
        onConceptActivity={conceptEventHandlers.handleConceptActivity}
        onDiffPulse={conceptEventHandlers.handleConceptDiffPulse}
      />

      {/* Agent Field Projector */}
      <div className="absolute bottom-4 left-4 w-96 h-64">
        <AgentFieldProjector
          agents={cognitionState.agents}
          ghostPresence={cognitionState.activeGhost ? {
            persona: cognitionState.activeGhost.persona,
            compatibility: cognitionState.phaseCoherence,
            fieldEffect: cognitionState.phaseCoherence > 0.7 ? 'boost' : 'harmonize'
          } : undefined}
          userEngagement={{
            typing: false, // Would be connected to actual typing detection
            focus: true,
            energy: cognitionState.phaseCoherence
          }}
          onAgentClick={handleAgentClick}
        />
      </div>

      {/* Ghost Chronicle Timeline */}
      <motion.div
        className="absolute top-4 right-4 w-80 max-h-96 overflow-hidden"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <GhostChronicle
          sessionId={sessionId}
          onEventSelect={handleGhostEventSelect}
          className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm rounded-lg border border-slate-200 dark:border-slate-700"
        />
      </motion.div>

      {/* Ghost Letter Modal */}
      <GhostLetterGenerator
        ghostPersona={cognitionState.ghostLetterData?.persona || 'Mentor'}
        conceptArc={{
          primary: cognitionState.ghostLetterData?.conceptArc?.[0] || 'conversation',
          secondary: cognitionState.ghostLetterData?.conceptArc?.[1] || 'understanding',
          type: 'reflection'
        }}
        phaseState={{
          coherence: cognitionState.phaseCoherence
        }}
        isVisible={cognitionState.ghostLetterVisible}
        onClose={handleGhostLetterClose}
      />

      {/* Enhanced Debug Panel with Banksy Info */}
      {showDebugPanel && (
        <div className="absolute bottom-4 right-4 bg-black/80 text-white p-4 rounded-lg font-mono text-xs max-w-xs">
          <div className="mb-2 font-bold">TORI Cognition Debug</div>
          <div>Phase Coherence: {cognitionState.phaseCoherence.toFixed(2)}</div>
          <div>Phase Entropy: {cognitionState.phaseEntropy.toFixed(2)}</div>
          <div>Phase Drift: {cognitionState.phaseDrift.toFixed(2)}</div>
          <div>Active Ghost: {cognitionState.activeGhost?.persona || 'None'}</div>
          <div>Field Harmony: {cognitionState.fieldHarmony.toFixed(2)}</div>
          <div>Concept Activity: {cognitionState.conceptActivity.length}</div>
          <div className="border-t border-slate-600 mt-2 pt-2">
            <div className="font-bold text-yellow-400">🌀 BANKSY STATUS</div>
            <div>Oscillator Sync: {cognitionState.oscillatorSynchronization.toFixed(3)}</div>
            <div>Step: {cognitionState.banksyState?.step || 0}</div>
            <div>N_eff: {cognitionState.banksyState?.n_effective || 0}</div>
            <div>Connected: {performanceMetrics.banksyConnected ? '✅' : '❌'}</div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-600">
            <div>FPS: {performanceMetrics.frameRate}</div>
            <div>Active Components: {performanceMetrics.activeComponents}</div>
          </div>
        </div>
      )}

      {/* Enhanced Status Indicator with Banksy */}
      <div className="absolute top-4 left-4 flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isInitialized ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-slate-600 dark:text-slate-400">
            {isInitialized ? 'Cognition Active' : 'Initializing...'}
          </span>
        </div>
        
        {/* Banksy Status */}
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${performanceMetrics.banksyConnected ? 'bg-blue-500' : 'bg-yellow-500'}`} />
          <span className="text-sm text-slate-600 dark:text-slate-400">
            🌀 Banksy {performanceMetrics.banksyConnected ? 'Online' : 'Connecting...'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ToriCognitionEngine;
export type { CognitionState };