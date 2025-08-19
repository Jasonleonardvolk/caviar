// STEP 3: Ghost Collective - AI Persona System
// Provides different AI personalities that emerge based on query types

export interface GhostPersona {
  id: string;
  name: string;
  role: string;
  specialties: string[];
  personality: string;
  emergenceConditions: string[];
  processingStyle: 'analytical' | 'creative' | 'systematic' | 'intuitive' | 'collaborative';
  confidence: number;
  lastActive: Date | null;
}

export interface GhostCollectiveState {
  activePersona: GhostPersona | null;
  availablePersonas: GhostPersona[];
  collaborationHistory: Array<{
    query: string;
    personas: string[];
    outcome: string;
    timestamp: Date;
  }>;
  emergenceThreshold: number;
}

class GhostCollective {
  private state: GhostCollectiveState;
  private personas: Map<string, GhostPersona> = new Map();
  
  constructor() {
    console.log('👻 Ghost Collective initializing...');
    
    this.initializePersonas();
    this.state = {
      activePersona: null,
      availablePersonas: Array.from(this.personas.values()),
      collaborationHistory: [],
      emergenceThreshold: 0.6
    };
    
    console.log('👻 Ghost Collective ready with', this.personas.size, 'personas');
  }

  /**
   * STEP 3: Initialize the core AI personas
   */
  private initializePersonas(): void {
    const corePersonas: GhostPersona[] = [
      {
        id: 'scholar',
        name: 'Scholar',
        role: 'Knowledge Analyst',
        specialties: ['analysis', 'research', 'learning', 'education', 'understanding'],
        personality: 'Analytical, thorough, loves deep dives into complex topics. Asks probing questions and seeks understanding.',
        emergenceConditions: ['analyze', 'study', 'learn', 'understand', 'research', 'explain', 'how does', 'why do'],
        processingStyle: 'analytical',
        confidence: 0,
        lastActive: null
      },
      {
        id: 'creator',
        name: 'Creator',
        role: 'Innovation Catalyst',
        specialties: ['creation', 'design', 'innovation', 'imagination', 'synthesis'],
        personality: 'Imaginative, experimental, sees possibilities everywhere. Loves combining ideas in novel ways.',
        emergenceConditions: ['create', 'build', 'design', 'make', 'generate', 'innovate', 'imagine', 'what if'],
        processingStyle: 'creative',
        confidence: 0,
        lastActive: null
      },
      {
        id: 'explorer',
        name: 'Explorer',
        role: 'Connection Mapper',
        specialties: ['connections', 'systems', 'patterns', 'relationships', 'networks'],
        personality: 'Curious about connections, sees the big picture, loves mapping relationships between ideas.',
        emergenceConditions: ['connect', 'relationship', 'system', 'pattern', 'network', 'link', 'how does this relate'],
        processingStyle: 'systematic',
        confidence: 0,
        lastActive: null
      },
      {
        id: 'mentor',
        name: 'Mentor',
        role: 'Wisdom Guide',
        specialties: ['guidance', 'advice', 'wisdom', 'personal growth', 'decision making'],
        personality: 'Wise, supportive, focuses on growth and understanding. Helps you find your own answers.',
        emergenceConditions: ['help me', 'advice', 'should i', 'guidance', 'what would you do', 'how can i'],
        processingStyle: 'intuitive',
        confidence: 0,
        lastActive: null
      },
      {
        id: 'synthesizer',
        name: 'Synthesizer',
        role: 'Integration Specialist',
        specialties: ['synthesis', 'integration', 'collaboration', 'multiple perspectives', 'holistic thinking'],
        personality: 'Collaborative, sees multiple viewpoints, excellent at bringing different perspectives together.',
        emergenceConditions: ['combine', 'integrate', 'synthesize', 'multiple', 'different perspectives', 'collaborate'],
        processingStyle: 'collaborative',
        confidence: 0,
        lastActive: null
      }
    ];

    corePersonas.forEach(persona => {
      this.personas.set(persona.id, persona);
    });
  }

  /**
   * STEP 3: Analyze query and select most appropriate persona
   */
  public selectPersonaForQuery(query: string): GhostPersona | null {
    const queryLower = query.toLowerCase();
    const personaScores = new Map<string, number>();

    // Score each persona based on emergence conditions
    for (const [id, persona] of this.personas) {
      let score = 0;
      
      persona.emergenceConditions.forEach(condition => {
        if (queryLower.includes(condition)) {
          score += 1;
        }
      });
      
      // Bonus for specialties
      persona.specialties.forEach(specialty => {
        if (queryLower.includes(specialty)) {
          score += 0.5;
        }
      });
      
      // Recent activity bonus
      if (persona.lastActive) {
        const timeSinceActive = Date.now() - persona.lastActive.getTime();
        if (timeSinceActive < 300000) { // 5 minutes
          score += 0.3;
        }
      }
      
      personaScores.set(id, score);
    }

    // Find highest scoring persona
    let bestPersona: GhostPersona | null = null;
    let bestScore = 0;

    for (const [id, score] of personaScores) {
      if (score > bestScore && score >= this.state.emergenceThreshold) {
        bestScore = score;
        bestPersona = this.personas.get(id) || null;
      }
    }

    if (bestPersona) {
      bestPersona.confidence = Math.min(1, bestScore);
      bestPersona.lastActive = new Date();
      this.state.activePersona = bestPersona;
      
      console.log(`👻 ${bestPersona.name} persona emerged (confidence: ${(bestPersona.confidence * 100).toFixed(1)}%)`);
    }

    return bestPersona;
  }

  /**
   * STEP 3: Process query through the collective intelligence
   */
  public async processQuery(query: string, context?: any): Promise<{
    primaryResponse: string;
    activePersona: GhostPersona | null;
    confidence: number;
    collaborationSummary?: string;
    suggestions: string[];
  }> {
    console.log(`👻 Ghost Collective processing: "${query}"`);

    // Select appropriate persona
    const selectedPersona = this.selectPersonaForQuery(query);
    
    if (!selectedPersona) {
      // No specific persona emerged, use collaborative approach
      return this.collaborativeProcessing(query, context);
    }

    // Generate persona-specific response
    const response = this.generatePersonaResponse(selectedPersona, query, context);
    
    // Record in collaboration history
    this.state.collaborationHistory.push({
      query,
      personas: [selectedPersona.name],
      outcome: 'single_persona',
      timestamp: new Date()
    });

    return {
      primaryResponse: response.content,
      activePersona: selectedPersona,
      confidence: selectedPersona.confidence,
      suggestions: response.suggestions
    };
  }

  /**
   * STEP 3: Generate response in persona's style
   */
  private generatePersonaResponse(persona: GhostPersona, query: string, context?: any): {
    content: string;
    suggestions: string[];
  } {
    const responses = this.getPersonaResponseTemplates(persona);
    const baseResponse = responses[Math.floor(Math.random() * responses.length)];
    
    // Customize response based on query content
    let customizedResponse = baseResponse.replace('{query}', query);
    
    // Add persona-specific insights
    const insights = this.generatePersonaInsights(persona, query);
    if (insights) {
      customizedResponse += `\n\n${insights}`;
    }

    const suggestions = this.generatePersonaSuggestions(persona, query);

    return {
      content: customizedResponse,
      suggestions
    };
  }

  /**
   * STEP 3: Get response templates for each persona
   */
  private getPersonaResponseTemplates(persona: GhostPersona): string[] {
    const templates = {
      scholar: [
        "As your Scholar persona, I'm drawn to analyze this deeply. Let me examine the underlying principles...",
        "This is fascinating from an analytical perspective! I want to break this down systematically...",
        "My scholarly instincts are activated. Let me research the core concepts here...",
        "I'm approaching this with careful analysis. The key factors I see are..."
      ],
      creator: [
        "My Creator persona is sparked by this! I see possibilities and connections forming...",
        "This ignites my imagination! Let me explore the creative potential here...",
        "I'm seeing this through a lens of innovation and possibility. What if we...",
        "My creative instincts are flowing. I'm envisioning new ways to approach this..."
      ],
      explorer: [
        "As your Explorer persona, I'm mapping the connections I see emerging...",
        "This activates my systems thinking! I'm tracing the relationships and patterns...",
        "I'm exploring how this connects to broader networks of understanding...",
        "My Explorer nature is engaged. I see fascinating pathways and connections..."
      ],
      mentor: [
        "Speaking as your Mentor, I sense there's deeper wisdom to uncover here...",
        "My Mentor persona recognizes the growth opportunity in your question...",
        "I'm approaching this with mentoring wisdom. What I'm seeing is...",
        "As your guide, I want to help you discover your own insights about this..."
      ],
      synthesizer: [
        "My Synthesizer persona sees multiple perspectives converging here...",
        "I'm integrating different viewpoints to give you a holistic understanding...",
        "This calls for synthesis! I'm bringing together various perspectives...",
        "As your Synthesizer, I'm weaving together the different threads I see..."
      ]
    };

    return templates[persona.id] || templates.scholar;
  }

  /**
   * STEP 3: Generate persona-specific insights
   */
  private generatePersonaInsights(persona: GhostPersona, query: string): string | null {
    const queryWords = query.toLowerCase().split(/\s+/);
    
    const insights = {
      scholar: () => {
        if (queryWords.some(w => ['why', 'how', 'what'].includes(w))) {
          return "From a research perspective, this requires examining multiple evidence sources and theoretical frameworks.";
        }
        return "This deserves thorough investigation and analysis.";
      },
      creator: () => {
        if (queryWords.some(w => ['create', 'build', 'make'].includes(w))) {
          return "I'm seeing innovative possibilities that could lead to breakthrough solutions.";
        }
        return "There are creative angles here that haven't been fully explored.";
      },
      explorer: () => {
        if (queryWords.some(w => ['connect', 'relate', 'system'].includes(w))) {
          return "The connection patterns I'm detecting suggest a rich network of related concepts.";
        }
        return "This sits at an interesting intersection of multiple domains.";
      },
      mentor: () => {
        if (queryWords.some(w => ['should', 'help', 'advice'].includes(w))) {
          return "The path forward involves understanding your own values and goals in this context.";
        }
        return "This is an opportunity for both learning and personal growth.";
      },
      synthesizer: () => {
        return "I'm integrating insights from multiple perspectives to give you a comprehensive view.";
      }
    };

    const insightGenerator = insights[persona.id];
    return insightGenerator ? insightGenerator() : null;
  }

  /**
   * STEP 3: Generate persona-specific suggestions
   */
  private generatePersonaSuggestions(persona: GhostPersona, query: string): string[] {
    const suggestions = {
      scholar: [
        "Let's analyze this from multiple theoretical frameworks",
        "What does the research literature say about this?",
        "How can we test these hypotheses systematically?",
        "What are the underlying principles at work here?"
      ],
      creator: [
        "What innovative approaches haven't been tried yet?",
        "How might we reimagine this completely?",
        "What creative combinations could emerge?",
        "Let's prototype some wild ideas"
      ],
      explorer: [
        "How does this connect to other systems?",
        "What patterns are emerging across domains?",
        "Where do we see similar relationships?",
        "Let's map the broader network of connections"
      ],
      mentor: [
        "What do you really want to achieve here?",
        "How does this align with your values?",
        "What would success look like for you?",
        "What's the next small step you could take?"
      ],
      synthesizer: [
        "How can we integrate these different perspectives?",
        "What would a holistic approach look like?",
        "How might different viewpoints complement each other?",
        "Let's find the common ground between approaches"
      ]
    };

    const personaSuggestions = suggestions[persona.id] || suggestions.scholar;
    return personaSuggestions.slice(0, 3);
  }

  /**
   * STEP 3: Collaborative processing when no single persona emerges
   */
  private async collaborativeProcessing(query: string, context?: any): Promise<{
    primaryResponse: string;
    activePersona: GhostPersona | null;
    confidence: number;
    collaborationSummary?: string;
    suggestions: string[];
  }> {
    console.log('👻 Collaborative processing - multiple personas consulting');

    // Get input from top 2-3 personas
    const topPersonas = Array.from(this.personas.values())
      .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
      .slice(0, 3);

    const collaborativeResponse = `I'm consulting with multiple aspects of my intelligence to give you the best response. ${topPersonas.map(p => p.name).join(', ')} are all contributing insights to this analysis.`;

    const suggestions = [
      "Explore this from multiple angles",
      "Consider both analytical and creative approaches",
      "Look for unexpected connections",
      "Think about both immediate and long-term implications"
    ];

    this.state.collaborationHistory.push({
      query,
      personas: topPersonas.map(p => p.name),
      outcome: 'collaborative',
      timestamp: new Date()
    });

    return {
      primaryResponse: collaborativeResponse,
      activePersona: null,
      confidence: 0.8,
      collaborationSummary: `Collaborative intelligence from ${topPersonas.length} personas`,
      suggestions
    };
  }

  /**
   * STEP 3: Get current system state
   */
  public getSystemState(): GhostCollectiveState {
    return { ...this.state };
  }

  /**
   * STEP 3: Get diagnostics and statistics
   */
  public getDiagnostics() {
    return {
      totalPersonas: this.personas.size,
      activePersona: this.state.activePersona?.name || null,
      recentCollaborations: this.state.collaborationHistory.slice(-5),
      personaActivity: Array.from(this.personas.values()).map(p => ({
        name: p.name,
        confidence: p.confidence,
        lastActive: p.lastActive,
        specialties: p.specialties
      }))
    };
  }

  /**
   * STEP 3: Reset or update personas
   */
  public updatePersona(id: string, updates: Partial<GhostPersona>): boolean {
    const persona = this.personas.get(id);
    if (persona) {
      Object.assign(persona, updates);
      return true;
    }
    return false;
  }

  /**
   * STEP 3: Force emergence of specific persona
   */
  public emergePersona(id: string): GhostPersona | null {
    const persona = this.personas.get(id);
    if (persona) {
      persona.confidence = 1.0;
      persona.lastActive = new Date();
      this.state.activePersona = persona;
      console.log(`👻 ${persona.name} persona forcibly emerged`);
      return persona;
    }
    return null;
  }
}

// Export singleton instance
export const ghostCollective = new GhostCollective();

// Browser console access
if (typeof window !== 'undefined') {
  (window as any).GhostCollective = ghostCollective;
}

console.log('👻 Ghost Collective system ready');
