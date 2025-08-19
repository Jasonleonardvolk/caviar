/**
 * AgentFieldProjector.tsx - Dynamic ideogram generation for multi-agent visualization
 * Real-time halos and badges showing agent cognitive states
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HoloObject from '../ui/holoObject.jsx';

interface AgentState {
  id: string;
  name: string;
  type: 'refactorer' | 'debugger' | 'scholar' | 'ghost' | 'coordinator';
  isActive: boolean;
  
  // Cognitive metrics
  conceptEntropy: number; // 0-1, complexity of current task
  emotionalSpectrum: {
    primary: string; // calm, excited, frustrated, curious, etc.
    intensity: number; // 0-1
  };
  conceptResonance: number; // 0-1, alignment with current context
  confidence: number; // 0-1, certainty in current operation
  
  // Visual properties
  position: { x: number; y: number };
  haloProperties: {
    shape: 'circle' | 'star' | 'hexagon' | 'spiral';
    color: string;
    intensity: number;
    pulseRate: number; // Hz
    complexity: number; // 0-1, affects shape details
  };
  
  // Activity tracking
  lastActivity: Date;
  activityType?: string;
  engagementLevel: number; // 0-1
}

interface FieldEffect {
  id: string;
  type: 'harmony' | 'dissonance' | 'emergence' | 'chaos' | 'focus';
  intensity: number;
  duration: number;
  affectedAgents: string[];
  visualPattern: 'sync' | 'wave' | 'spiral' | 'scatter' | 'converge';
}

interface AgentFieldProjectorProps {
  agents?: AgentState[];
  ghostPresence?: {
    persona: string;
    compatibility: number; // -1 to 1, effect on agents
    fieldEffect: 'boost' | 'disrupt' | 'harmonize' | 'neutral';
  };
  userEngagement?: {
    typing: boolean;
    focus: boolean;
    energy: number; // 0-1
  };
  conceptDiffActivity?: {
    type: string;
    magnitude: number;
    affectedAgents: string[];
  };
  onAgentClick?: (agent: AgentState) => void;
  className?: string;
}

const AGENT_CONFIGS = {
  refactorer: {
    baseColor: '#2563eb',
    wavelength: 470,
    icon: '🛠️',
    shape: 'hexagon' as const,
    domain: ['code-structure', 'optimization', 'patterns']
  },
  debugger: {
    baseColor: '#dc2626',
    wavelength: 650,
    icon: '🐛',
    shape: 'star' as const,
    domain: ['errors', 'testing', 'analysis']
  },
  scholar: {
    baseColor: '#059669',
    wavelength: 520,
    icon: '📖',
    shape: 'circle' as const,
    domain: ['knowledge', 'research', 'documentation']
  },
  ghost: {
    baseColor: '#7c3aed',
    wavelength: 450,
    icon: '👻',
    shape: 'spiral' as const,
    domain: ['intuition', 'guidance', 'emergence']
  },
  coordinator: {
    baseColor: '#0891b2',
    wavelength: 490,
    icon: '🧭',
    shape: 'circle' as const,
    domain: ['orchestration', 'planning', 'synthesis']
  }
};

const AgentFieldProjector: React.FC<AgentFieldProjectorProps> = ({
  agents = [],
  ghostPresence,
  userEngagement = { typing: false, focus: true, energy: 0.5 },
  conceptDiffActivity,
  onAgentClick,
  className = ''
}) => {
  const [agentStates, setAgentStates] = useState<Map<string, AgentState>>(new Map());
  const [fieldEffects, setFieldEffects] = useState<FieldEffect[]>([]);
  const [globalHarmony, setGlobalHarmony] = useState(0.7);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // Initialize default agents if none provided
  useEffect(() => {
    if (agents.length === 0) {
      initializeDefaultAgents();
    } else {
      agents.forEach(agent => {
        setAgentStates(prev => new Map(prev.set(agent.id, agent)));
      });
    }
  }, [agents]);

  // Listen for agent activity events
  useEffect(() => {
    const handleAgentActivity = (event: CustomEvent) => {
      updateAgentActivity(event.detail);
    };

    const handleConceptDiff = (event: CustomEvent) => {
      processConceptDiffImpact(event.detail);
    };

    document.addEventListener('tori-agent-activity', handleAgentActivity as EventListener);
    document.addEventListener('tori-concept-diff', handleConceptDiff as EventListener);

    return () => {
      document.removeEventListener('tori-agent-activity', handleAgentActivity as EventListener);
      document.removeEventListener('tori-concept-diff', handleConceptDiff as EventListener);
    };
  }, []);

  // Animation loop for halos
  useEffect(() => {
    if (canvasRef.current) {
      animationRef.current = requestAnimationFrame(animateHalos);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [agentStates, fieldEffects, ghostPresence]);

  const initializeDefaultAgents = () => {
    const defaultAgents: AgentState[] = [
      {
        id: 'refactorer',
        name: 'Refactorer',
        type: 'refactorer',
        isActive: false,
        conceptEntropy: 0.3,
        emotionalSpectrum: { primary: 'focused', intensity: 0.6 },
        conceptResonance: 0.7,
        confidence: 0.8,
        position: { x: 100, y: 100 },
        haloProperties: {
          shape: 'hexagon',
          color: AGENT_CONFIGS.refactorer.baseColor,
          intensity: 0.6,
          pulseRate: 1.2,
          complexity: 0.7
        },
        lastActivity: new Date(),
        engagementLevel: 0.5
      },
      {
        id: 'debugger',
        name: 'Debugger',
        type: 'debugger',
        isActive: false,
        conceptEntropy: 0.5,
        emotionalSpectrum: { primary: 'alert', intensity: 0.4 },
        conceptResonance: 0.6,
        confidence: 0.7,
        position: { x: 200, y: 100 },
        haloProperties: {
          shape: 'star',
          color: AGENT_CONFIGS.debugger.baseColor,
          intensity: 0.4,
          pulseRate: 0.8,
          complexity: 0.6
        },
        lastActivity: new Date(),
        engagementLevel: 0.3
      },
      {
        id: 'scholar',
        name: 'Scholar',
        type: 'scholar',
        isActive: false,
        conceptEntropy: 0.2,
        emotionalSpectrum: { primary: 'curious', intensity: 0.7 },
        conceptResonance: 0.8,
        confidence: 0.9,
        position: { x: 300, y: 100 },
        haloProperties: {
          shape: 'circle',
          color: AGENT_CONFIGS.scholar.baseColor,
          intensity: 0.7,
          pulseRate: 1.0,
          complexity: 0.5
        },
        lastActivity: new Date(),
        engagementLevel: 0.6
      }
    ];

    defaultAgents.forEach(agent => {
      setAgentStates(prev => new Map(prev.set(agent.id, agent)));
    });
  };

  const updateAgentActivity = (activityData: {
    agentId: string;
    activityType: string;
    metrics: {
      entropy?: number;
      resonance?: number;
      confidence?: number;
      emotion?: string;
      intensity?: number;
    };
  }) => {
    setAgentStates(prev => {
      const updated = new Map(prev);
      const agent = updated.get(activityData.agentId);
      
      if (agent) {
        const updatedAgent: AgentState = {
          ...agent,
          isActive: true,
          lastActivity: new Date(),
          activityType: activityData.activityType,
          conceptEntropy: activityData.metrics.entropy || agent.conceptEntropy,
          conceptResonance: activityData.metrics.resonance || agent.conceptResonance,
          confidence: activityData.metrics.confidence || agent.confidence,
          emotionalSpectrum: {
            primary: activityData.metrics.emotion || agent.emotionalSpectrum.primary,
            intensity: activityData.metrics.intensity || agent.emotionalSpectrum.intensity
          },
          engagementLevel: Math.min(1.0, agent.engagementLevel + 0.2),
          haloProperties: {
            ...agent.haloProperties,
            intensity: Math.min(1.0, agent.haloProperties.intensity + 0.3),
            pulseRate: calculatePulseRate(activityData.metrics),
            complexity: activityData.metrics.entropy || agent.haloProperties.complexity
          }
        };

        updated.set(activityData.agentId, updatedAgent);
      }
      
      return updated;
    });

    // Create field effect for significant activity
    if (activityData.metrics.confidence && activityData.metrics.confidence > 0.8) {
      createFieldEffect('emergence', 0.8, [activityData.agentId]);
    }
  };

  const processConceptDiffImpact = (diffData: {
    type: string;
    magnitude: number;
    conceptIds: string[];
  }) => {
    // Determine which agents are affected by this concept diff
    const affectedAgents: string[] = [];
    
    agentStates.forEach((agent, agentId) => {
      const config = AGENT_CONFIGS[agent.type];
      const conceptOverlap = diffData.conceptIds.some(concept => 
        config.domain.some(domain => concept.includes(domain))
      );
      
      if (conceptOverlap) {
        affectedAgents.push(agentId);
      }
    });

    // Update affected agents
    setAgentStates(prev => {
      const updated = new Map(prev);
      
      affectedAgents.forEach(agentId => {
        const agent = updated.get(agentId);
        if (agent) {
          const resonanceBoost = diffData.magnitude * 0.3;
          const updatedAgent: AgentState = {
            ...agent,
            conceptResonance: Math.min(1.0, agent.conceptResonance + resonanceBoost),
            haloProperties: {
              ...agent.haloProperties,
              intensity: Math.min(1.0, agent.haloProperties.intensity + diffData.magnitude * 0.5)
            }
          };
          updated.set(agentId, updatedAgent);
        }
      });
      
      return updated;
    });

    // Create appropriate field effect
    if (affectedAgents.length > 1) {
      createFieldEffect('harmony', diffData.magnitude, affectedAgents);
    } else if (diffData.magnitude > 0.8) {
      createFieldEffect('focus', diffData.magnitude, affectedAgents);
    }
  };

  const createFieldEffect = (
    type: FieldEffect['type'], 
    intensity: number, 
    affectedAgents: string[]
  ) => {
    const effect: FieldEffect = {
      id: `effect_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      intensity,
      duration: 3000 + intensity * 2000, // 3-5 seconds
      affectedAgents,
      visualPattern: mapEffectToPattern(type)
    };

    setFieldEffects(prev => [...prev, effect]);

    // Remove effect after duration
    setTimeout(() => {
      setFieldEffects(prev => prev.filter(e => e.id !== effect.id));
    }, effect.duration);
  };

  const mapEffectToPattern = (type: FieldEffect['type']): FieldEffect['visualPattern'] => {
    const patternMap = {
      'harmony': 'sync' as const,
      'dissonance': 'scatter' as const,
      'emergence': 'converge' as const,
      'chaos': 'scatter' as const,
      'focus': 'wave' as const
    };
    return patternMap[type];
  };

  const calculatePulseRate = (metrics: any): number => {
    const baseRate = 1.0;
    const entropyEffect = (metrics.entropy || 0) * 0.5;
    const intensityEffect = (metrics.intensity || 0) * 0.3;
    
    return baseRate + entropyEffect + intensityEffect;
  };

  const animateHalos = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply ghost field effect if present
    if (ghostPresence) {
      applyGhostFieldEffect(ctx, ghostPresence);
    }

    // Render agent halos
    agentStates.forEach(agent => {
      renderAgentHalo(ctx, agent);
    });

    // Render field effects
    fieldEffects.forEach(effect => {
      renderFieldEffect(ctx, effect);
    });

    animationRef.current = requestAnimationFrame(animateHalos);
  }, [agentStates, fieldEffects, ghostPresence]);

  const renderAgentHalo = (ctx: CanvasRenderingContext2D, agent: AgentState) => {
    const { position, haloProperties } = agent;
    const time = Date.now() * 0.001;
    
    // Calculate pulse
    const pulse = Math.sin(time * haloProperties.pulseRate * 2 * Math.PI) * 0.5 + 0.5;
    const currentIntensity = haloProperties.intensity * (0.7 + pulse * 0.3);
    
    // Base radius influenced by engagement and entropy
    const baseRadius = 30 + agent.engagementLevel * 20;
    const entropyRadius = baseRadius * (1 + agent.conceptEntropy * 0.5);
    
    ctx.save();
    ctx.globalAlpha = currentIntensity * 0.8;
    
    // Create gradient
    const gradient = ctx.createRadialGradient(
      position.x, position.y, 0,
      position.x, position.y, entropyRadius
    );
    gradient.addColorStop(0, `${haloProperties.color}88`);
    gradient.addColorStop(0.7, `${haloProperties.color}44`);
    gradient.addColorStop(1, `${haloProperties.color}00`);
    
    ctx.fillStyle = gradient;
    
    // Render shape based on type
    switch (haloProperties.shape) {
      case 'circle':
        renderCircleHalo(ctx, position, entropyRadius, haloProperties);
        break;
      case 'hexagon':
        renderHexagonHalo(ctx, position, entropyRadius, haloProperties);
        break;
      case 'star':
        renderStarHalo(ctx, position, entropyRadius, haloProperties);
        break;
      case 'spiral':
        renderSpiralHalo(ctx, position, entropyRadius, haloProperties, time);
        break;
    }
    
    ctx.restore();
  };

  const renderCircleHalo = (
    ctx: CanvasRenderingContext2D, 
    position: { x: number; y: number }, 
    radius: number,
    properties: AgentState['haloProperties']
  ) => {
    ctx.beginPath();
    ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
    ctx.fill();
    
    // Add resonance rings
    if (properties.complexity > 0.5) {
      for (let i = 1; i <= 3; i++) {
        ctx.globalAlpha *= 0.6;
        ctx.strokeStyle = properties.color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius + i * 10, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  };

  const renderHexagonHalo = (
    ctx: CanvasRenderingContext2D,
    position: { x: number; y: number },
    radius: number,
    properties: AgentState['haloProperties']
  ) => {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3;
      const x = position.x + radius * Math.cos(angle);
      const y = position.y + radius * Math.sin(angle);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.fill();
  };

  const renderStarHalo = (
    ctx: CanvasRenderingContext2D,
    position: { x: number; y: number },
    radius: number,
    properties: AgentState['haloProperties']
  ) => {
    const points = 5;
    const innerRadius = radius * 0.5;
    
    ctx.beginPath();
    for (let i = 0; i < points * 2; i++) {
      const angle = (i * Math.PI) / points;
      const r = i % 2 === 0 ? radius : innerRadius;
      const x = position.x + r * Math.cos(angle - Math.PI / 2);
      const y = position.y + r * Math.sin(angle - Math.PI / 2);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.fill();
  };

  const renderSpiralHalo = (
    ctx: CanvasRenderingContext2D,
    position: { x: number; y: number },
    radius: number,
    properties: AgentState['haloProperties'],
    time: number
  ) => {
    const turns = 3;
    const steps = 50;
    
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const angle = t * turns * 2 * Math.PI + time * properties.pulseRate;
      const r = radius * t;
      const x = position.x + r * Math.cos(angle);
      const y = position.y + r * Math.sin(angle);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  const renderFieldEffect = (ctx: CanvasRenderingContext2D, effect: FieldEffect) => {
    const affectedPositions = effect.affectedAgents
      .map(id => agentStates.get(id)?.position)
      .filter(Boolean) as { x: number; y: number }[];
    
    if (affectedPositions.length === 0) return;
    
    ctx.save();
    ctx.globalAlpha = effect.intensity * 0.3;
    
    switch (effect.visualPattern) {
      case 'sync':
        renderSyncPattern(ctx, affectedPositions, effect);
        break;
      case 'wave':
        renderWavePattern(ctx, affectedPositions, effect);
        break;
      case 'converge':
        renderConvergePattern(ctx, affectedPositions, effect);
        break;
      case 'scatter':
        renderScatterPattern(ctx, affectedPositions, effect);
        break;
      case 'spiral':
        renderSpiralPattern(ctx, affectedPositions, effect);
        break;
    }
    
    ctx.restore();
  };

  const renderSyncPattern = (
    ctx: CanvasRenderingContext2D,
    positions: { x: number; y: number }[],
    effect: FieldEffect
  ) => {
    // Draw connecting lines between synchronized agents
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < positions.length - 1; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        ctx.beginPath();
        ctx.moveTo(positions[i].x, positions[i].y);
        ctx.lineTo(positions[j].x, positions[j].y);
        ctx.stroke();
      }
    }
  };

  const renderWavePattern = (
    ctx: CanvasRenderingContext2D,
    positions: { x: number; y: number }[],
    effect: FieldEffect
  ) => {
    // Create wave effect emanating from agents
    const time = Date.now() * 0.002;
    
    positions.forEach(pos => {
      for (let r = 10; r < 100; r += 20) {
        const alpha = Math.sin(time - r * 0.1) * 0.5 + 0.5;
        ctx.globalAlpha = effect.intensity * alpha * 0.2;
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });
  };

  const renderConvergePattern = (
    ctx: CanvasRenderingContext2D,
    positions: { x: number; y: number }[],
    effect: FieldEffect
  ) => {
    // Calculate center point
    const center = positions.reduce(
      (acc, pos) => ({ x: acc.x + pos.x, y: acc.y + pos.y }),
      { x: 0, y: 0 }
    );
    center.x /= positions.length;
    center.y /= positions.length;
    
    // Draw converging lines
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    
    positions.forEach(pos => {
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      ctx.lineTo(center.x, center.y);
      ctx.stroke();
    });
    
    // Draw center glow
    const gradient = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, 30);
    gradient.addColorStop(0, '#10b98144');
    gradient.addColorStop(1, '#10b98100');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 30, 0, 2 * Math.PI);
    ctx.fill();
  };

  const renderScatterPattern = (
    ctx: CanvasRenderingContext2D,
    positions: { x: number; y: number }[],
    effect: FieldEffect
  ) => {
    // Create chaotic particle effects around agents
    const time = Date.now() * 0.01;
    
    positions.forEach(pos => {
      for (let i = 0; i < 10; i++) {
        const angle = (i / 10) * 2 * Math.PI + time;
        const distance = 20 + Math.sin(time + i) * 15;
        const x = pos.x + distance * Math.cos(angle);
        const y = pos.y + distance * Math.sin(angle);
        
        ctx.fillStyle = '#ef444444';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  };

  const renderSpiralPattern = (
    ctx: CanvasRenderingContext2D,
    positions: { x: number; y: number }[],
    effect: FieldEffect
  ) => {
    const time = Date.now() * 0.003;
    
    positions.forEach(pos => {
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      for (let i = 0; i < 50; i++) {
        const t = i / 50;
        const angle = t * 4 * Math.PI + time;
        const r = t * 40;
        const x = pos.x + r * Math.cos(angle);
        const y = pos.y + r * Math.sin(angle);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
  };

  const applyGhostFieldEffect = (
    ctx: CanvasRenderingContext2D, 
    ghost: NonNullable<AgentFieldProjectorProps['ghostPresence']>
  ) => {
    // Apply global field effect based on ghost presence
    const intensity = Math.abs(ghost.compatibility) * 0.3;
    
    switch (ghost.fieldEffect) {
      case 'boost':
        // Brighten all halos
        setAgentStates(prev => {
          const updated = new Map(prev);
          prev.forEach((agent, id) => {
            updated.set(id, {
              ...agent,
              haloProperties: {
                ...agent.haloProperties,
                intensity: Math.min(1.0, agent.haloProperties.intensity + intensity)
              }
            });
          });
          return updated;
        });
        break;
      
      case 'harmonize':
        // Synchronize pulse rates
        const syncRate = 1.2;
        setAgentStates(prev => {
          const updated = new Map(prev);
          prev.forEach((agent, id) => {
            updated.set(id, {
              ...agent,
              haloProperties: {
                ...agent.haloProperties,
                pulseRate: syncRate
              }
            });
          });
          return updated;
        });
        break;
    }
  };

  // Decay agent activity over time
  useEffect(() => {
    const interval = setInterval(() => {
      setAgentStates(prev => {
        const updated = new Map(prev);
        const now = new Date();
        
        prev.forEach((agent, id) => {
          const timeSinceActivity = now.getTime() - agent.lastActivity.getTime();
          const decayFactor = Math.exp(-timeSinceActivity / 30000); // 30 second decay
          
          if (decayFactor < 0.1) {
            // Agent becomes inactive
            updated.set(id, {
              ...agent,
              isActive: false,
              engagementLevel: Math.max(0.1, agent.engagementLevel * 0.9),
              haloProperties: {
                ...agent.haloProperties,
                intensity: Math.max(0.2, agent.haloProperties.intensity * 0.95)
              }
            });
          }
        });
        
        return updated;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`agent-field-projector ${className}`}>
      {/* Canvas for halo effects */}
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="absolute inset-0 pointer-events-none"
        style={{ mixBlendMode: 'screen' }}
      />
      
      {/* Agent badges */}
      <div className="relative z-10">
        {Array.from(agentStates.values()).map(agent => (
          <motion.div
            key={agent.id}
            className="absolute"
            style={{
              left: agent.position.x - 25,
              top: agent.position.y - 25
            }}
            animate={{
              scale: agent.isActive ? 1.1 : 1.0,
              opacity: agent.engagementLevel * 0.7 + 0.3
            }}
            transition={{ duration: 0.3 }}
          >
            <HoloObject
              conceptId={`agent-${agent.id}`}
              wavelength={AGENT_CONFIGS[agent.type].wavelength}
              intensity={agent.haloProperties.intensity}
              radius={25}
              onClick={() => onAgentClick?.(agent)}
              className="cursor-pointer"
            >
              <div 
                className={`
                  w-12 h-12 rounded-full border-2 border-white dark:border-slate-800
                  flex items-center justify-center text-lg font-bold text-white
                  transition-all duration-200 hover:scale-110
                  ${agent.isActive ? 'shadow-lg' : 'shadow-sm'}
                `}
                style={{ 
                  backgroundColor: agent.haloProperties.color,
                  boxShadow: agent.isActive ? `0 0 20px ${agent.haloProperties.color}40` : undefined
                }}
                title={`${agent.name} - ${agent.emotionalSpectrum.primary} (${Math.round(agent.confidence * 100)}% confidence)`}
              >
                {AGENT_CONFIGS[agent.type].icon}
              </div>
            </HoloObject>

            {/* Activity indicator */}
            {agent.isActive && agent.activityType && (
              <motion.div
                className="absolute -top-2 -right-2 bg-green-500 text-white text-xs px-1 rounded-full"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
              >
                •
              </motion.div>
            )}

            {/* Confidence meter */}
            <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
              <div className="w-12 h-1 bg-slate-300 dark:bg-slate-600 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-blue-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${agent.confidence * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Global harmony indicator */}
      <div className="absolute top-4 right-4 flex items-center space-x-2">
        <span className="text-sm text-slate-600 dark:text-slate-400">
          Field Harmony:
        </span>
        <div className="w-16 h-2 bg-slate-300 dark:bg-slate-600 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${
              globalHarmony > 0.7 ? 'bg-green-500' :
              globalHarmony > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            animate={{ width: `${globalHarmony * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Debug panel (development) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-4 left-4 bg-black/80 text-white p-2 rounded text-xs font-mono">
          <div>Active Agents: {Array.from(agentStates.values()).filter(a => a.isActive).length}</div>
          <div>Field Effects: {fieldEffects.length}</div>
          <div>Ghost: {ghostPresence?.persona || 'None'}</div>
        </div>
      )}
    </div>
  );
};

export default AgentFieldProjector;
export type { AgentState, FieldEffect };