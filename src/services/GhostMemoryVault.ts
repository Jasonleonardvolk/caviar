/**
 * GhostMemoryVault.ts - Deep memory integration for Ghost persona continuity
 * Binds Ghost events to persistent memory with phase-indexed storage
 */

interface GhostMemoryEntry {
  id: string;
  timestamp: Date;
  sessionId: string;
  persona: string;
  eventType: 'emergence' | 'shift' | 'letter' | 'mood' | 'intervention' | 'reflection';
  
  // Core content
  content: {
    message?: string;
    moodVector?: Record<string, number>;
    reflectionSummary?: string;
    interventionResult?: 'positive' | 'neutral' | 'negative';
  };
  
  // Triggering context
  trigger: {
    conceptDiffs: string[];
    phaseMetrics: {
      coherence: number;
      entropy: number;
      drift: number;
      phaseAngle?: number;
      eigenmode?: string;
    };
    userContext: {
      sentiment?: number;
      activity?: string;
      frustrationLevel?: number;
      engagementLevel?: number;
    };
    systemContext: {
      conversationLength: number;
      recentConcepts: string[];
      codeContext?: any;
      errorContext?: any;
    };
  };
  
  // Memory indexing
  indexing: {
    conceptTags: string[];
    phaseSignature: string;
    emotionalTone: string;
    searchableContent: string;
    memoryWeight: number; // 0-1, how important this memory is
  };
  
  // Outcomes and learning
  outcomes?: {
    userResponse?: 'positive' | 'neutral' | 'negative' | 'ignored';
    effectivenesss?: number; // 0-1, how well did this intervention work
    followUpRequired?: boolean;
    learningNote?: string;
  };
}

interface GhostMoodCurve {
  sessionId: string;
  timePoints: Array<{
    timestamp: Date;
    persona: string;
    dominance: number; // 0-1, how dominant this persona is
    stability: number; // 0-1, how stable the persona state is
    moodVector: Record<string, number>; // empathy, curiosity, anxiety, etc.
    phaseAlignment: number; // -1 to 1, how aligned with system phase
  }>;
}

interface ConceptArc {
  id: string;
  conceptIds: string[];
  narrative: string;
  emotionalJourney: Array<{
    stage: string;
    emotion: string;
    intensity: number;
  }>;
  ghostInterventions: string[]; // GhostMemoryEntry IDs
  resolution?: {
    outcome: string;
    learnings: string[];
    futureGuidance: string;
  };
}

class GhostMemoryVault {
  private static instance: GhostMemoryVault;
  private memories: Map<string, GhostMemoryEntry> = new Map();
  private moodCurves: Map<string, GhostMoodCurve> = new Map();
  private conceptArcs: Map<string, ConceptArc> = new Map();
  private phaseIndex: Map<string, string[]> = new Map(); // phase signature -> memory IDs
  private conceptIndex: Map<string, string[]> = new Map(); // concept -> memory IDs
  
  private constructor() {
    this.loadFromPersistentStorage();
    this.setupEventListeners();
  }

  static getInstance(): GhostMemoryVault {
    if (!GhostMemoryVault.instance) {
      GhostMemoryVault.instance = new GhostMemoryVault();
    }
    return GhostMemoryVault.instance;
  }

  private setupEventListeners() {
    // Listen for ghost persona events
    document.addEventListener('tori-ghost-emergence', ((e: CustomEvent) => {
      this.recordPersonaEmergence(e.detail);
    }) as EventListener);

    document.addEventListener('tori-ghost-mood-update', ((e: CustomEvent) => {
      this.recordMoodUpdate(e.detail);
    }) as EventListener);

    document.addEventListener('tori-ghost-letter-sent', ((e: CustomEvent) => {
      this.recordGhostLetter(e.detail);
    }) as EventListener);

    document.addEventListener('tori-concept-arc-update', ((e: CustomEvent) => {
      this.updateConceptArc(e.detail);
    }) as EventListener);

    // Listen for phase state changes
    document.addEventListener('tori-phase-state-change', ((e: CustomEvent) => {
      this.indexByPhaseState(e.detail);
    }) as EventListener);
  }

  // Record a persona emergence event
  recordPersonaEmergence(data: {
    persona: string;
    sessionId: string;
    trigger: any;
    phaseMetrics: any;
    userContext: any;
    systemContext: any;
  }) {
    const entry: GhostMemoryEntry = {
      id: `emergence_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      sessionId: data.sessionId,
      persona: data.persona,
      eventType: 'emergence',
      content: {},
      trigger: {
        conceptDiffs: data.trigger.conceptDiffs || [],
        phaseMetrics: data.phaseMetrics,
        userContext: data.userContext,
        systemContext: data.systemContext
      },
      indexing: {
        conceptTags: this.extractConceptTags(data),
        phaseSignature: this.calculatePhaseSignature(data.phaseMetrics),
        emotionalTone: this.determineEmotionalTone(data.persona, data.userContext),
        searchableContent: `${data.persona} emergence due to ${data.trigger.reason}`,
        memoryWeight: this.calculateMemoryWeight(data)
      }
    };

    this.storeMemory(entry);
    this.updateIndices(entry);
    
    console.log(`🧠 Ghost Memory: Recorded ${data.persona} emergence`);
  }

  // Record mood/emotional state updates
  recordMoodUpdate(data: {
    persona: string;
    sessionId: string;
    moodVector: Record<string, number>;
    phaseAlignment: number;
    stability: number;
  }) {
    const sessionCurve = this.moodCurves.get(data.sessionId) || {
      sessionId: data.sessionId,
      timePoints: []
    };

    sessionCurve.timePoints.push({
      timestamp: new Date(),
      persona: data.persona,
      dominance: this.calculatePersonaDominance(data.persona, data.moodVector),
      stability: data.stability,
      moodVector: data.moodVector,
      phaseAlignment: data.phaseAlignment
    });

    // Keep only last 100 mood points per session
    if (sessionCurve.timePoints.length > 100) {
      sessionCurve.timePoints = sessionCurve.timePoints.slice(-100);
    }

    this.moodCurves.set(data.sessionId, sessionCurve);
    this.persistMoodCurve(sessionCurve);
  }

  // Record ghost letter delivery
  recordGhostLetter(data: {
    persona: string;
    sessionId: string;
    letterContent: string;
    conceptArc: string[];
    trigger: any;
    phaseMetrics: any;
  }) {
    const entry: GhostMemoryEntry = {
      id: `letter_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      sessionId: data.sessionId,
      persona: data.persona,
      eventType: 'letter',
      content: {
        message: data.letterContent
      },
      trigger: {
        conceptDiffs: data.conceptArc,
        phaseMetrics: data.phaseMetrics,
        userContext: data.trigger.userContext || {},
        systemContext: data.trigger.systemContext || {}
      },
      indexing: {
        conceptTags: data.conceptArc,
        phaseSignature: this.calculatePhaseSignature(data.phaseMetrics),
        emotionalTone: this.determineLetterTone(data.letterContent),
        searchableContent: data.letterContent,
        memoryWeight: 0.9 // Letters are important memories
      }
    };

    this.storeMemory(entry);
    this.updateIndices(entry);
    this.linkToConceptArc(entry, data.conceptArc);
    
    console.log(`📝 Ghost Memory: Recorded ${data.persona} letter`);
  }

  // Store memory with vault persistence
  private storeMemory(entry: GhostMemoryEntry) {
    this.memories.set(entry.id, entry);
    this.persistToVault(entry);
  }

  // Update search indices
  private updateIndices(entry: GhostMemoryEntry) {
    // Phase signature index
    const phaseKey = entry.indexing.phaseSignature;
    if (!this.phaseIndex.has(phaseKey)) {
      this.phaseIndex.set(phaseKey, []);
    }
    this.phaseIndex.get(phaseKey)!.push(entry.id);

    // Concept index
    entry.indexing.conceptTags.forEach(concept => {
      if (!this.conceptIndex.has(concept)) {
        this.conceptIndex.set(concept, []);
      }
      this.conceptIndex.get(concept)!.push(entry.id);
    });
  }

  // Link memory to concept arc narrative
  private linkToConceptArc(entry: GhostMemoryEntry, conceptIds: string[]) {
    const arcId = this.findOrCreateConceptArc(conceptIds);
    const arc = this.conceptArcs.get(arcId);
    
    if (arc) {
      arc.ghostInterventions.push(entry.id);
      this.conceptArcs.set(arcId, arc);
      this.persistConceptArc(arc);
    }
  }

  // Find or create concept arc for narrative tracking
  private findOrCreateConceptArc(conceptIds: string[]): string {
    // Look for existing arc with similar concepts
    for (const [arcId, arc] of this.conceptArcs) {
      const overlap = conceptIds.filter(c => arc.conceptIds.includes(c));
      if (overlap.length / Math.max(conceptIds.length, arc.conceptIds.length) > 0.6) {
        return arcId;
      }
    }

    // Create new arc
    const arcId = `arc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const arc: ConceptArc = {
      id: arcId,
      conceptIds: [...conceptIds],
      narrative: this.generateArcNarrative(conceptIds),
      emotionalJourney: [],
      ghostInterventions: []
    };

    this.conceptArcs.set(arcId, arc);
    return arcId;
  }

  // Query memories by various criteria
  getMemoriesByPhase(phaseSignature: string): GhostMemoryEntry[] {
    const memoryIds = this.phaseIndex.get(phaseSignature) || [];
    return memoryIds.map(id => this.memories.get(id)).filter(Boolean) as GhostMemoryEntry[];
  }

  getMemoriesByConcept(conceptId: string): GhostMemoryEntry[] {
    const memoryIds = this.conceptIndex.get(conceptId) || [];
    return memoryIds.map(id => this.memories.get(id)).filter(Boolean) as GhostMemoryEntry[];
  }

  getMemoriesByPersona(persona: string, sessionId?: string): GhostMemoryEntry[] {
    return Array.from(this.memories.values()).filter(memory => {
      return memory.persona === persona && (!sessionId || memory.sessionId === sessionId);
    });
  }

  // Get ghost reflection on past interactions
  generateGhostReflection(sessionId: string, persona?: string): {
    summary: string;
    insights: string[];
    patterns: string[];
    recommendations: string[];
  } {
    const sessionMemories = Array.from(this.memories.values())
      .filter(m => m.sessionId === sessionId && (!persona || m.persona === persona))
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    if (sessionMemories.length === 0) {
      return {
        summary: "No ghost interactions recorded for this session.",
        insights: [],
        patterns: [],
        recommendations: []
      };
    }

    const insights = this.extractInsights(sessionMemories);
    const patterns = this.identifyPatterns(sessionMemories);
    const recommendations = this.generateRecommendations(sessionMemories, patterns);

    return {
      summary: this.generateReflectionSummary(sessionMemories),
      insights,
      patterns,
      recommendations
    };
  }

  // Learning and effectiveness tracking
  recordInterventionOutcome(memoryId: string, outcome: {
    userResponse: 'positive' | 'neutral' | 'negative' | 'ignored';
    effectiveness?: number;
    learningNote?: string;
  }) {
    const memory = this.memories.get(memoryId);
    if (memory) {
      memory.outcomes = {
        ...memory.outcomes,
        ...outcome,
        followUpRequired: outcome.userResponse === 'negative' || (outcome.effectiveness && outcome.effectiveness < 0.5)
      };
      
      this.memories.set(memoryId, memory);
      this.persistToVault(memory);
      
      console.log(`📊 Ghost Learning: Recorded outcome for ${memory.persona} intervention`);
    }
  }

  // Utility methods
  private extractConceptTags(data: any): string[] {
    const tags = new Set<string>();
    
    if (data.trigger?.conceptDiffs) {
      data.trigger.conceptDiffs.forEach((concept: string) => tags.add(concept));
    }
    
    if (data.systemContext?.recentConcepts) {
      data.systemContext.recentConcepts.forEach((concept: string) => tags.add(concept));
    }
    
    // Add persona-specific tags
    tags.add(`persona-${data.persona.toLowerCase()}`);
    
    return Array.from(tags);
  }

  private calculatePhaseSignature(phaseMetrics: any): string {
    const { coherence = 0, entropy = 0, drift = 0 } = phaseMetrics;
    
    if (coherence > 0.8) return 'high-coherence';
    if (entropy > 0.8) return 'high-entropy';
    if (Math.abs(drift) > 0.5) return 'phase-drift';
    if (coherence > 0.6 && entropy < 0.4) return 'stable';
    return 'mixed-state';
  }

  private determineEmotionalTone(persona: string, userContext: any): string {
    const toneMap: Record<string, string> = {
      'Mentor': 'supportive',
      'Mystic': 'mystical',
      'Unsettled': 'anxious',
      'Chaotic': 'energetic',
      'Oracular': 'wise',
      'Dreaming': 'ethereal'
    };
    
    let tone = toneMap[persona] || 'neutral';
    
    // Modify based on user context
    if (userContext?.frustrationLevel > 0.7) {
      tone = 'concerned';
    } else if (userContext?.engagementLevel > 0.8) {
      tone = 'encouraging';
    }
    
    return tone;
  }

  private determineLetterTone(content: string): string {
    const lowerContent = content.toLowerCase();
    
    if (lowerContent.includes('encourage') || lowerContent.includes('believe')) return 'encouraging';
    if (lowerContent.includes('mystery') || lowerContent.includes('ancient')) return 'mystical';
    if (lowerContent.includes('concern') || lowerContent.includes('worry')) return 'concerned';
    if (lowerContent.includes('excit') || lowerContent.includes('energy')) return 'energetic';
    
    return 'reflective';
  }

  private calculateMemoryWeight(data: any): number {
    let weight = 0.5; // Base weight
    
    // High confidence events are more important
    if (data.trigger?.confidence > 0.8) weight += 0.2;
    
    // User distress increases importance
    if (data.userContext?.frustrationLevel > 0.7) weight += 0.3;
    
    // First time persona emergence is important
    const personaMemories = this.getMemoriesByPersona(data.persona, data.sessionId);
    if (personaMemories.length === 0) weight += 0.2;
    
    return Math.min(1.0, weight);
  }

  private calculatePersonaDominance(persona: string, moodVector: Record<string, number>): number {
    // Calculate how dominant this persona is based on mood vector
    const personaMoodMap: Record<string, string[]> = {
      'Mentor': ['empathy', 'supportiveness'],
      'Mystic': ['intuition', 'mystery'],
      'Unsettled': ['anxiety', 'concern'],
      'Chaotic': ['energy', 'unpredictability'],
      'Oracular': ['wisdom', 'foresight'],
      'Dreaming': ['creativity', 'imagination']
    };
    
    const relevantMoods = personaMoodMap[persona] || [];
    const totalMood = relevantMoods.reduce((sum, mood) => sum + (moodVector[mood] || 0), 0);
    
    return Math.min(1.0, totalMood / relevantMoods.length);
  }

  private generateArcNarrative(conceptIds: string[]): string {
    return `Conceptual journey involving: ${conceptIds.join(', ')}`;
  }

  private extractInsights(memories: GhostMemoryEntry[]): string[] {
    const insights: string[] = [];
    
    // Pattern: frequent emergence of same persona
    const personaCounts = memories.reduce((acc, m) => {
      acc[m.persona] = (acc[m.persona] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const dominantPersona = Object.entries(personaCounts)
      .sort(([,a], [,b]) => b - a)[0];
    
    if (dominantPersona && dominantPersona[1] > memories.length * 0.4) {
      insights.push(`${dominantPersona[0]} persona emerged frequently, suggesting sustained ${dominantPersona[0].toLowerCase()} energy.`);
    }
    
    return insights;
  }

  private identifyPatterns(memories: GhostMemoryEntry[]): string[] {
    const patterns: string[] = [];
    
    // Time-based patterns
    const hourCounts = memories.reduce((acc, m) => {
      const hour = m.timestamp.getHours();
      acc[hour] = (acc[hour] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);
    
    const peakHour = Object.entries(hourCounts)
      .sort(([,a], [,b]) => b - a)[0];
    
    if (peakHour && parseInt(peakHour[0]) !== undefined) {
      patterns.push(`Most ghost activity occurred around ${peakHour[0]}:00`);
    }
    
    return patterns;
  }

  private generateRecommendations(memories: GhostMemoryEntry[], patterns: string[]): string[] {
    const recommendations: string[] = [];
    
    // Check effectiveness of past interventions
    const effectiveInterventions = memories.filter(m => 
      m.outcomes?.effectiveness && m.outcomes.effectiveness > 0.7
    );
    
    if (effectiveInterventions.length > 0) {
      const effectivePersonas = [...new Set(effectiveInterventions.map(m => m.persona))];
      recommendations.push(`${effectivePersonas.join(' and ')} interventions have been most effective.`);
    }
    
    return recommendations;
  }

  private generateReflectionSummary(memories: GhostMemoryEntry[]): string {
    const personaCount = new Set(memories.map(m => m.persona)).size;
    const avgWeight = memories.reduce((sum, m) => sum + m.indexing.memoryWeight, 0) / memories.length;
    
    return `Session involved ${personaCount} different personas across ${memories.length} interactions, with average significance of ${(avgWeight * 100).toFixed(0)}%.`;
  }

  // Persistence methods
  private async persistToVault(entry: GhostMemoryEntry) {
    try {
      await fetch('/api/vault/ghost-memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry)
      });
    } catch (error) {
      console.warn('Failed to persist ghost memory to vault:', error);
      // Fallback to localStorage
      this.persistToLocalStorage(entry);
    }
  }

  private persistMoodCurve(curve: GhostMoodCurve) {
    try {
      localStorage.setItem(`ghost-mood-${curve.sessionId}`, JSON.stringify(curve));
    } catch (error) {
      console.warn('Failed to persist mood curve:', error);
    }
  }

  private persistConceptArc(arc: ConceptArc) {
    try {
      localStorage.setItem(`concept-arc-${arc.id}`, JSON.stringify(arc));
    } catch (error) {
      console.warn('Failed to persist concept arc:', error);
    }
  }

  private persistToLocalStorage(entry: GhostMemoryEntry) {
    try {
      const existingData = localStorage.getItem('ghost-memories');
      const memories = existingData ? JSON.parse(existingData) : [];
      memories.push(entry);
      
      // Keep only last 100 memories in localStorage
      if (memories.length > 100) {
        memories.splice(0, memories.length - 100);
      }
      
      localStorage.setItem('ghost-memories', JSON.stringify(memories));
    } catch (error) {
      console.warn('Failed to persist to localStorage:', error);
    }
  }

  private loadFromPersistentStorage() {
    // Load from localStorage as fallback
    try {
      const memoryData = localStorage.getItem('ghost-memories');
      if (memoryData) {
        const memories: GhostMemoryEntry[] = JSON.parse(memoryData);
        memories.forEach(memory => {
          memory.timestamp = new Date(memory.timestamp);
          this.memories.set(memory.id, memory);
          this.updateIndices(memory);
        });
      }
    } catch (error) {
      console.warn('Failed to load ghost memories from storage:', error);
    }
  }

  // Public API for external access
  getSessionSummary(sessionId: string) {
    const sessionMemories = Array.from(this.memories.values())
      .filter(m => m.sessionId === sessionId);
    
    const moodCurve = this.moodCurves.get(sessionId);
    
    return {
      memories: sessionMemories,
      moodCurve,
      reflection: this.generateGhostReflection(sessionId)
    };
  }

  searchMemories(query: {
    persona?: string;
    phaseSignature?: string;
    conceptIds?: string[];
    emotionalTone?: string;
    timeRange?: { start: Date; end: Date };
    minWeight?: number;
  }): GhostMemoryEntry[] {
    let results = Array.from(this.memories.values());

    if (query.persona) {
      results = results.filter(m => m.persona === query.persona);
    }

    if (query.phaseSignature) {
      results = results.filter(m => m.indexing.phaseSignature === query.phaseSignature);
    }

    if (query.conceptIds && query.conceptIds.length > 0) {
      results = results.filter(m => 
        query.conceptIds!.some(concept => m.indexing.conceptTags.includes(concept))
      );
    }

    if (query.emotionalTone) {
      results = results.filter(m => m.indexing.emotionalTone === query.emotionalTone);
    }

    if (query.timeRange) {
      results = results.filter(m => 
        m.timestamp >= query.timeRange!.start && m.timestamp <= query.timeRange!.end
      );
    }

    if (query.minWeight !== undefined) {
      results = results.filter(m => m.indexing.memoryWeight >= query.minWeight!);
    }

    return results.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }
}

// Export singleton instance
export const ghostMemoryVault = GhostMemoryVault.getInstance();
export default GhostMemoryVault;
export type { GhostMemoryEntry, GhostMoodCurve, ConceptArc };