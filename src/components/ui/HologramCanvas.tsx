/**
 * HologramCanvas.tsx - Holographic thought projection system
 * Visualizes concept dynamics as ambient visual effects
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';

interface ConceptShimmer {
  id: string;
  x: number;
  y: number;
  intensity: number;
  wavelength: number;
  conceptId: string;
  resonance: number;
  timestamp: number;
}

interface ConceptDiffPulse {
  id: string;
  x: number;
  y: number;
  radius: number;
  magnitude: number;
  diffType: string;
  color: string;
  timestamp: number;
}

interface GhostOverlay {
  persona: string;
  opacity: number;
  pattern: 'ripple' | 'wave' | 'distortion' | 'crackling';
  wavelength: number;
  phaseAnomaly: 'deviation' | 'instability' | 'resonance' | 'chaos';
}

interface SpectralMetadata {
  dominant_frequency?: number;
  emotional_indicators?: {
    brightness: number;
    agitation: number;
    instability: number;
  };
  motion_intensity?: number;
  color_temperature?: number;
  phase_signature?: string;
}

interface HologramCanvasProps {
  width?: number;
  height?: number;
  enabled?: boolean;
  spectralMetadata?: SpectralMetadata;
  conceptActivity?: Array<{
    conceptId: string;
    intensity: number;
    resonance: number;
  }>;
  ghostPresence?: GhostOverlay | null;
  onEffectComplete?: (effectId: string) => void;
}

const HologramCanvas: React.FC<HologramCanvasProps> = ({
  width = window.innerWidth,
  height = window.innerHeight,
  enabled = true,
  spectralMetadata = {},
  conceptActivity = [],
  ghostPresence = null,
  onEffectComplete
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [shimmers, setShimmers] = useState<ConceptShimmer[]>([]);
  const [pulses, setPulses] = useState<ConceptDiffPulse[]>([]);
  const [phaseRing, setPhaseRing] = useState({
    coherence: 0.7,
    phase: 0,
    stability: 0.8
  });

  // Animation loop
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !enabled) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas with subtle background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
    ctx.fillRect(0, 0, width, height);

    // Update and render phase ring
    renderPhaseRing(ctx);

    // Update and render concept shimmers
    renderConceptShimmers(ctx);

    // Update and render ConceptDiff pulses
    renderConceptDiffPulses(ctx);

    // Render ghost overlay if present
    if (ghostPresence) {
      renderGhostOverlay(ctx, ghostPresence);
    }

    // Clean up expired effects
    cleanupExpiredEffects();

    animationRef.current = requestAnimationFrame(animate);
  }, [width, height, enabled, ghostPresence, shimmers, pulses, phaseRing]);

  // Start animation loop
  useEffect(() => {
    if (enabled) {
      animationRef.current = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate, enabled]);

  // Update phase ring based on spectral metadata
  useEffect(() => {
    if (spectralMetadata.phase_signature) {
      const newCoherence = calculateCoherence(spectralMetadata);
      const newStability = calculateStability(spectralMetadata);
      
      setPhaseRing(prev => ({
        coherence: newCoherence,
        phase: (prev.phase + 0.05) % (2 * Math.PI),
        stability: newStability
      }));
    }
  }, [spectralMetadata]);

  // Generate concept shimmers from activity
  useEffect(() => {
    conceptActivity.forEach(concept => {
      if (concept.intensity > 0.5) {
        createConceptShimmer(concept);
      }
    });
  }, [conceptActivity]);

  const calculateCoherence = (metadata: SpectralMetadata): number => {
    const { phase_signature = 'neutral', emotional_indicators } = metadata;
    
    const phaseMap: Record<string, number> = {
      'coherence': 0.9,
      'resonance': 0.8,
      'drift': 0.4,
      'entropy': 0.2,
      'neutral': 0.6
    };

    let coherence = phaseMap[phase_signature] || 0.6;
    
    // Adjust based on emotional indicators
    if (emotional_indicators) {
      coherence *= (1 - emotional_indicators.instability * 0.3);
      coherence = Math.max(0.1, Math.min(1.0, coherence));
    }
    
    return coherence;
  };

  const calculateStability = (metadata: SpectralMetadata): number => {
    const { emotional_indicators, motion_intensity = 0 } = metadata;
    
    let stability = 0.8;
    
    if (emotional_indicators) {
      stability -= emotional_indicators.agitation * 0.4;
    }
    
    stability -= motion_intensity * 0.3;
    
    return Math.max(0.1, Math.min(1.0, stability));
  };

  const renderPhaseRing = (ctx: CanvasRenderingContext2D) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const baseRadius = Math.min(width, height) * 0.3;
    
    // Main coherence ring
    ctx.save();
    ctx.globalAlpha = phaseRing.coherence * 0.6;
    
    const gradient = ctx.createRadialGradient(
      centerX, centerY, baseRadius * 0.8,
      centerX, centerY, baseRadius * 1.2
    );
    
    const hue = phaseRing.coherence * 240; // Blue to red spectrum
    gradient.addColorStop(0, `hsla(${hue}, 70%, 60%, 0.8)`);
    gradient.addColorStop(1, `hsla(${hue}, 70%, 60%, 0)`);
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 3 + phaseRing.stability * 5;
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Stability indicators - additional rings
    if (phaseRing.stability > 0.7) {
      for (let i = 1; i <= 3; i++) {
        ctx.globalAlpha = (phaseRing.stability - 0.7) * 0.3 / i;
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius + i * 20, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    
    ctx.restore();
  };

  const renderConceptShimmers = (ctx: CanvasRenderingContext2D) => {
    shimmers.forEach(shimmer => {
      const age = Date.now() - shimmer.timestamp;
      const maxAge = 5000; // 5 seconds
      
      if (age > maxAge) return;
      
      const alpha = (1 - age / maxAge) * shimmer.intensity;
      const wavelengthColor = wavelengthToRGB(shimmer.wavelength);
      
      ctx.save();
      ctx.globalAlpha = alpha * 0.7;
      
      // Create shimmer effect
      const shimmerRadius = 20 + shimmer.resonance * 30;
      const pulse = Math.sin((Date.now() * 0.01 + shimmer.id.charCodeAt(0)) % (2 * Math.PI));
      
      const gradient = ctx.createRadialGradient(
        shimmer.x, shimmer.y, 0,
        shimmer.x, shimmer.y, shimmerRadius * (1 + pulse * 0.3)
      );
      
      gradient.addColorStop(0, `rgba(${wavelengthColor.r}, ${wavelengthColor.g}, ${wavelengthColor.b}, 0.8)`);
      gradient.addColorStop(1, `rgba(${wavelengthColor.r}, ${wavelengthColor.g}, ${wavelengthColor.b}, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(shimmer.x, shimmer.y, shimmerRadius * (1 + pulse * 0.3), 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.restore();
    });
  };

  const renderConceptDiffPulses = (ctx: CanvasRenderingContext2D) => {
    pulses.forEach(pulse => {
      const age = Date.now() - pulse.timestamp;
      const maxAge = 2000; // 2 seconds
      
      if (age > maxAge) return;
      
      const progress = age / maxAge;
      const currentRadius = pulse.radius * progress;
      const alpha = (1 - progress) * pulse.magnitude;
      
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = pulse.color;
      ctx.lineWidth = 2 + pulse.magnitude * 3;
      
      ctx.beginPath();
      ctx.arc(pulse.x, pulse.y, currentRadius, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Inner glow
      if (pulse.magnitude > 0.7) {
        ctx.globalAlpha = alpha * 0.5;
        ctx.fillStyle = pulse.color;
        ctx.beginPath();
        ctx.arc(pulse.x, pulse.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      ctx.restore();
    });
  };

  const renderGhostOverlay = (ctx: CanvasRenderingContext2D, ghost: GhostOverlay) => {
    ctx.save();
    ctx.globalAlpha = ghost.opacity * 0.4;
    
    switch (ghost.pattern) {
      case 'ripple':
        renderRipplePattern(ctx, ghost);
        break;
      case 'wave':
        renderWavePattern(ctx, ghost);
        break;
      case 'distortion':
        renderDistortionPattern(ctx, ghost);
        break;
      case 'crackling':
        renderCracklingPattern(ctx, ghost);
        break;
    }
    
    ctx.restore();
  };

  const renderRipplePattern = (ctx: CanvasRenderingContext2D, ghost: GhostOverlay) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const time = Date.now() * 0.002;
    
    for (let i = 0; i < 5; i++) {
      const radius = (50 + i * 40) + Math.sin(time + i) * 20;
      const wavelengthColor = wavelengthToRGB(ghost.wavelength);
      
      ctx.strokeStyle = `rgba(${wavelengthColor.r}, ${wavelengthColor.g}, ${wavelengthColor.b}, ${0.3 / (i + 1)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.stroke();
    }
  };

  const renderWavePattern = (ctx: CanvasRenderingContext2D, ghost: GhostOverlay) => {
    const time = Date.now() * 0.003;
    const wavelengthColor = wavelengthToRGB(ghost.wavelength);
    
    ctx.strokeStyle = `rgba(${wavelengthColor.r}, ${wavelengthColor.g}, ${wavelengthColor.b}, 0.3)`;
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    for (let x = 0; x < width; x += 5) {
      const y = height / 2 + Math.sin((x * 0.01) + time) * 50;
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  const renderDistortionPattern = (ctx: CanvasRenderingContext2D, ghost: GhostOverlay) => {
    // Create a subtle distortion effect
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
      if (Math.random() < 0.01) {
        data[i] = Math.min(255, data[i] + Math.random() * 50);     // R
        data[i + 1] = Math.min(255, data[i + 1] + Math.random() * 50); // G
        data[i + 2] = Math.min(255, data[i + 2] + Math.random() * 50); // B
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
  };

  const renderCracklingPattern = (ctx: CanvasRenderingContext2D, ghost: GhostOverlay) => {
    const wavelengthColor = wavelengthToRGB(ghost.wavelength);
    
    for (let i = 0; i < 10; i++) {
      if (Math.random() < 0.3) {
        const x1 = Math.random() * width;
        const y1 = Math.random() * height;
        const x2 = x1 + (Math.random() - 0.5) * 100;
        const y2 = y1 + (Math.random() - 0.5) * 100;
        
        ctx.strokeStyle = `rgba(${wavelengthColor.r}, ${wavelengthColor.g}, ${wavelengthColor.b}, 0.6)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
  };

  const createConceptShimmer = (concept: { conceptId: string; intensity: number; resonance: number }) => {
    const shimmer: ConceptShimmer = {
      id: `shimmer_${concept.conceptId}_${Date.now()}`,
      x: Math.random() * width,
      y: Math.random() * height,
      intensity: concept.intensity,
      wavelength: 400 + concept.resonance * 350, // Map resonance to wavelength
      conceptId: concept.conceptId,
      resonance: concept.resonance,
      timestamp: Date.now()
    };
    
    setShimmers(prev => [...prev, shimmer]);
  };

  const createConceptDiffPulse = useCallback((x: number, y: number, magnitude: number, diffType: string) => {
    const colorMap: Record<string, string> = {
      'create': '#00ff88',
      'update': '#ffaa00', 
      'link': '#0088ff',
      'remove': '#ff4444',
      'phaseShift': '#aa00ff'
    };
    
    const pulse: ConceptDiffPulse = {
      id: `pulse_${Date.now()}_${Math.random()}`,
      x,
      y,
      radius: 100 + magnitude * 100,
      magnitude,
      diffType,
      color: colorMap[diffType] || '#ffffff',
      timestamp: Date.now()
    };
    
    setPulses(prev => [...prev, pulse]);
    
    // Auto-cleanup after effect completes
    setTimeout(() => {
      if (onEffectComplete) {
        onEffectComplete(pulse.id);
      }
    }, 2000);
  }, [onEffectComplete]);

  const cleanupExpiredEffects = () => {
    const now = Date.now();
    
    setShimmers(prev => prev.filter(shimmer => now - shimmer.timestamp < 5000));
    setPulses(prev => prev.filter(pulse => now - pulse.timestamp < 2000));
  };

  const wavelengthToRGB = (wavelength: number): { r: number; g: number; b: number } => {
    let r = 0, g = 0, b = 0;
    
    if (wavelength >= 380 && wavelength < 440) {
      r = -(wavelength - 440) / (440 - 380);
      g = 0.0;
      b = 1.0;
    } else if (wavelength >= 440 && wavelength < 490) {
      r = 0.0;
      g = (wavelength - 440) / (490 - 440);
      b = 1.0;
    } else if (wavelength >= 490 && wavelength < 510) {
      r = 0.0;
      g = 1.0;
      b = -(wavelength - 510) / (510 - 490);
    } else if (wavelength >= 510 && wavelength < 580) {
      r = (wavelength - 510) / (580 - 510);
      g = 1.0;
      b = 0.0;
    } else if (wavelength >= 580 && wavelength < 645) {
      r = 1.0;
      g = -(wavelength - 645) / (645 - 580);
      b = 0.0;
    } else if (wavelength >= 645 && wavelength <= 750) {
      r = 1.0;
      g = 0.0;
      b = 0.0;
    }
    
    return {
      r: Math.round(255 * r),
      g: Math.round(255 * g),
      b: Math.round(255 * b)
    };
  };

  // Expose methods for external triggering
  React.useImperativeHandle(React.createRef(), () => ({
    triggerConceptDiffPulse: createConceptDiffPulse,
    updatePhaseRing: (coherence: number, stability: number) => {
      setPhaseRing(prev => ({ ...prev, coherence, stability }));
    }
  }));

  if (!enabled) {
    return null;
  }

  return (
    <motion.div
      className="fixed inset-0 pointer-events-none z-10"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 2 }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full"
        style={{
          mixBlendMode: 'screen',
          background: 'transparent'
        }}
      />
    </motion.div>
  );
};

export default HologramCanvas;
export type { SpectralMetadata, GhostOverlay, ConceptShimmer, ConceptDiffPulse };