/**
 * ConceptDiffVisualizer.tsx - Real-time concept activity monitor
 * Translates concept changes into visual effects for HologramCanvas
 */

import React, { useEffect, useState, useCallback } from 'react';

interface ConceptDiff {
  id: string;
  type: 'create' | 'update' | 'link' | 'remove' | 'phaseShift';
  conceptId: string;
  magnitude: number;
  confidence: number;
  metadata?: {
    previousValue?: any;
    newValue?: any;
    connections?: string[];
    phaseAngle?: number;
  };
  timestamp: Date;
}

interface ConceptActivity {
  conceptId: string;
  intensity: number;
  resonance: number;
  frequency: number;
  lastUpdate: Date;
}

interface ConceptDiffVisualizerProps {
  onConceptActivity?: (activity: ConceptActivity[]) => void;
  onDiffPulse?: (x: number, y: number, magnitude: number, diffType: string) => void;
  enabled?: boolean;
}

const ConceptDiffVisualizer: React.FC<ConceptDiffVisualizerProps> = ({
  onConceptActivity,
  onDiffPulse,
  enabled = true
}) => {
  const [conceptActivities, setConceptActivities] = useState<Map<string, ConceptActivity>>(new Map());
  const [recentDiffs, setRecentDiffs] = useState<ConceptDiff[]>([]);

  // Listen for concept diff events
  useEffect(() => {
    if (!enabled) return;

    const handleConceptDiff = (event: CustomEvent<ConceptDiff>) => {
      processConcepDiff(event.detail);
    };

    const handleConceptUpdate = (event: CustomEvent<any>) => {
      processConceptUpdate(event.detail);
    };

    // Listen for TORI concept events
    document.addEventListener('tori-concept-diff', handleConceptDiff as EventListener);
    document.addEventListener('tori-concept-update', handleConceptUpdate as EventListener);

    return () => {
      document.removeEventListener('tori-concept-diff', handleConceptDiff as EventListener);
      document.removeEventListener('tori-concept-update', handleConceptUpdate as EventListener);
    };
  }, [enabled]);

  // Decay concept activities over time
  useEffect(() => {
    const interval = setInterval(() => {
      setConceptActivities(prev => {
        const now = new Date();
        const updated = new Map(prev);
        
        for (const [conceptId, activity] of updated) {
          const timeSinceUpdate = now.getTime() - activity.lastUpdate.getTime();
          const decayFactor = Math.exp(-timeSinceUpdate / 30000); // 30 second half-life
          
          if (decayFactor < 0.1) {
            updated.delete(conceptId);
          } else {
            updated.set(conceptId, {
              ...activity,
              intensity: activity.intensity * decayFactor,
              resonance: activity.resonance * decayFactor
            });
          }
        }
        
        return updated;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Emit concept activity updates
  useEffect(() => {
    if (onConceptActivity) {
      const activities = Array.from(conceptActivities.values());
      onConceptActivity(activities);
    }
  }, [conceptActivities, onConceptActivity]);

  const processConcepDiff = useCallback((diff: ConceptDiff) => {
    // Add to recent diffs
    setRecentDiffs(prev => [
      diff,
      ...prev.slice(0, 49) // Keep last 50 diffs
    ]);

    // Update concept activity
    setConceptActivities(prev => {
      const updated = new Map(prev);
      const existing = updated.get(diff.conceptId);
      
      const baseIntensity = calculateDiffIntensity(diff);
      const resonance = calculateResonance(diff, prev);
      
      const newActivity: ConceptActivity = {
        conceptId: diff.conceptId,
        intensity: existing ? Math.min(1.0, existing.intensity + baseIntensity) : baseIntensity,
        resonance: resonance,
        frequency: existing ? existing.frequency + 1 : 1,
        lastUpdate: new Date()
      };
      
      updated.set(diff.conceptId, newActivity);
      return updated;
    });

    // Trigger visual pulse
    if (onDiffPulse && diff.magnitude > 0.3) {
      const position = calculatePulsePosition(diff);
      onDiffPulse(position.x, position.y, diff.magnitude, diff.type);
    }

    // Emit concept diff event for other components
    document.dispatchEvent(new CustomEvent('tori-concept-diff-processed', {
      detail: { diff, timestamp: new Date() }
    }));
  }, [onDiffPulse]);

  const processConceptUpdate = useCallback((updateData: any) => {
    const { conceptId, action, metadata } = updateData;
    
    // Convert concept update to diff format
    const diff: ConceptDiff = {
      id: `update_${Date.now()}_${Math.random()}`,
      type: mapActionToDiffType(action),
      conceptId: conceptId,
      magnitude: calculateUpdateMagnitude(metadata),
      confidence: metadata?.confidence || 0.7,
      metadata: metadata,
      timestamp: new Date()
    };

    processConcepDiff(diff);
  }, [processConcepDiff]);

  const calculateDiffIntensity = (diff: ConceptDiff): number => {
    const baseIntensity = diff.magnitude * diff.confidence;
    
    // Type-based modifiers
    const typeModifiers = {
      'create': 1.0,
      'update': 0.7,
      'link': 0.9,
      'remove': 0.8,
      'phaseShift': 1.2
    };
    
    return Math.min(1.0, baseIntensity * (typeModifiers[diff.type] || 1.0));
  };

  const calculateResonance = (diff: ConceptDiff, existingActivities: Map<string, ConceptActivity>): number => {
    let resonance = diff.confidence;
    
    // Check for related concepts in recent activity
    const relatedConcepts = findRelatedConcepts(diff.conceptId, existingActivities);
    if (relatedConcepts.length > 0) {
      const avgResonance = relatedConcepts.reduce((sum, concept) => sum + concept.resonance, 0) / relatedConcepts.length;
      resonance = Math.min(1.0, resonance + avgResonance * 0.3);
    }
    
    // Phase shift boosts resonance
    if (diff.type === 'phaseShift') {
      resonance = Math.min(1.0, resonance * 1.5);
    }
    
    return resonance;
  };

  const findRelatedConcepts = (conceptId: string, activities: Map<string, ConceptActivity>): ConceptActivity[] => {
    // Simple similarity check - in production would use semantic similarity
    return Array.from(activities.values()).filter(activity => {
      if (activity.conceptId === conceptId) return false;
      
      // Check for similar concept names (basic heuristic)
      const similarity = calculateConceptSimilarity(conceptId, activity.conceptId);
      return similarity > 0.3;
    });
  };

  const calculateConceptSimilarity = (concept1: string, concept2: string): number => {
    // Simple Levenshtein-like similarity for concept names
    const words1 = concept1.toLowerCase().split(/[-_\s]/);
    const words2 = concept2.toLowerCase().split(/[-_\s]/);
    
    const commonWords = words1.filter(word => words2.includes(word));
    const totalWords = new Set([...words1, ...words2]).size;
    
    return commonWords.length / totalWords;
  };

  const calculatePulsePosition = (diff: ConceptDiff): { x: number; y: number } => {
    // Map concept to screen position (simplified)
    const conceptHash = diff.conceptId.split('').reduce((hash, char) => {
      return char.charCodeAt(0) + ((hash << 5) - hash);
    }, 0);
    
    const x = (Math.abs(conceptHash) % (window.innerWidth - 200)) + 100;
    const y = (Math.abs(conceptHash * 3) % (window.innerHeight - 200)) + 100;
    
    return { x, y };
  };

  const mapActionToDiffType = (action: string): ConceptDiff['type'] => {
    const actionMap: Record<string, ConceptDiff['type']> = {
      'create': 'create',
      'add': 'create',
      'update': 'update',
      'modify': 'update',
      'link': 'link',
      'connect': 'link',
      'remove': 'remove',
      'delete': 'remove',
      'phase-shift': 'phaseShift',
      'phase-change': 'phaseShift'
    };
    
    return actionMap[action] || 'update';
  };

  const calculateUpdateMagnitude = (metadata: any): number => {
    if (!metadata) return 0.5;
    
    let magnitude = 0.5;
    
    // Factor in confidence
    if (metadata.confidence !== undefined) {
      magnitude *= metadata.confidence;
    }
    
    // Factor in frequency/importance
    if (metadata.frequency !== undefined) {
      magnitude *= Math.min(1.0, metadata.frequency / 10);
    }
    
    // Factor in connection count
    if (metadata.connections?.length) {
      magnitude *= Math.min(1.5, 1 + metadata.connections.length * 0.1);
    }
    
    return Math.min(1.0, magnitude);
  };

  // Debugging/monitoring interface
  const getActivitySummary = useCallback(() => {
    const activities = Array.from(conceptActivities.values());
    return {
      totalConcepts: activities.length,
      highIntensityConcepts: activities.filter(a => a.intensity > 0.7).length,
      averageResonance: activities.length > 0 
        ? activities.reduce((sum, a) => sum + a.resonance, 0) / activities.length 
        : 0,
      recentDiffCount: recentDiffs.length
    };
  }, [conceptActivities, recentDiffs]);

  // Expose debug methods (development mode)
  React.useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      (window as any).toriConceptDebug = {
        getActivitySummary,
        getActivities: () => Array.from(conceptActivities.values()),
        getRecentDiffs: () => recentDiffs.slice(0, 10),
        triggerTestDiff: (conceptId: string, type: ConceptDiff['type'] = 'create') => {
          const testDiff: ConceptDiff = {
            id: `test_${Date.now()}`,
            type,
            conceptId,
            magnitude: 0.8,
            confidence: 0.9,
            timestamp: new Date()
          };
          processConcepDiff(testDiff);
        }
      };
    }
  }, [getActivitySummary, conceptActivities, recentDiffs, processConcepDiff]);

  // This is a headless component - no direct render
  return null;
};

export default ConceptDiffVisualizer;
export type { ConceptDiff, ConceptActivity };