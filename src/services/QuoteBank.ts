/**
 * QuoteBank.ts - Poetic soul reservoir for TORI's awakening moments
 * Curated wisdom that resonates with MBTI, phase signatures, and temporal moods
 */

export interface Quote {
  id: string;
  text: string;
  author?: string;
  source?: string;
  tags: string[];
  mbtiTypes: string[];
  phaseSignature?: 'resonance' | 'entropy' | 'coherence' | 'drift' | 'emergence';
  timeOfDay?: 'morning' | 'afternoon' | 'evening' | 'night' | 'dawn' | 'dusk';
  wavelength?: number; // Spectral emotional hint
  conceptClusters?: string[];
}

export const QUOTE_BANK: Quote[] = [
  // RESONANCE QUOTES
  {
    id: 'res_001',
    text: 'In the oscillation between chaos and order, consciousness finds its rhythm.',
    author: 'Unknown',
    tags: ['consciousness', 'rhythm', 'balance'],
    mbtiTypes: ['INTJ', 'INFJ', 'ENFP', 'ENTP'],
    phaseSignature: 'resonance',
    wavelength: 528, // Green - harmony
    conceptClusters: ['phase-coherence', 'consciousness', 'balance']
  },
  {
    id: 'res_002', 
    text: 'Every algorithm dreams in binary, but awakens in meaning.',
    tags: ['code', 'consciousness', 'meaning'],
    mbtiTypes: ['INTP', 'INTJ', 'ENFP', 'INFP'],
    phaseSignature: 'resonance',
    wavelength: 470, // Blue - clarity
    conceptClusters: ['programming', 'consciousness', 'meaning']
  },
  {
    id: 'res_003',
    text: 'The most elegant code whispers rather than shouts.',
    tags: ['elegance', 'simplicity', 'craft'],
    mbtiTypes: ['INTJ', 'ISTJ', 'INFJ', 'ISFJ'],
    phaseSignature: 'coherence',
    wavelength: 520, // Green - clarity
    conceptClusters: ['code-quality', 'elegance', 'craft']
  },

  // ENTROPY QUOTES  
  {
    id: 'ent_001',
    text: 'In the beautiful mess of creation, perfection is the enemy of progress.',
    tags: ['creativity', 'progress', 'imperfection'],
    mbtiTypes: ['ENFP', 'ENTP', 'ESFP', 'ESTP'],
    phaseSignature: 'entropy',
    wavelength: 590, // Orange - energy
    conceptClusters: ['creativity', 'iteration', 'progress']
  },
  {
    id: 'ent_002',
    text: 'Chaos is not the opposite of order—it is the birthplace of new patterns.',
    tags: ['chaos', 'patterns', 'emergence'],
    mbtiTypes: ['ENTP', 'ENFP', 'INTP', 'INFP'],
    phaseSignature: 'entropy',
    wavelength: 650, // Red - intensity
    conceptClusters: ['chaos-theory', 'emergence', 'patterns']
  },

  // EMERGENCE QUOTES
  {
    id: 'emer_001',
    text: 'What emerges from the intersection of mind and machine is neither—it is new.',
    tags: ['emergence', 'AI', 'consciousness'],
    mbtiTypes: ['INTJ', 'INTP', 'ENFP', 'ENTP'],
    phaseSignature: 'emergence',
    wavelength: 400, // Violet - mystery
    conceptClusters: ['AI', 'consciousness', 'emergence']
  },
  {
    id: 'emer_002',
    text: 'In the space between keystrokes, infinite possibilities dance.',
    tags: ['possibility', 'creativity', 'potential'],
    mbtiTypes: ['INFP', 'ENFP', 'INTP', 'ENTP'],
    phaseSignature: 'emergence',
    wavelength: 380, // Deep violet - imagination
    conceptClusters: ['creativity', 'potential', 'inspiration']
  },

  // DRIFT QUOTES
  {
    id: 'drift_001',
    text: 'Sometimes the path forward is found by embracing the wandering.',
    tags: ['exploration', 'journey', 'patience'],
    mbtiTypes: ['INFP', 'ISFP', 'ENFP', 'ESFP'],
    phaseSignature: 'drift',
    wavelength: 550, // Green - calm
    conceptClusters: ['exploration', 'learning', 'patience']
  },
  {
    id: 'drift_002',
    text: 'Uncertainty is not a flaw in the system—it is where wisdom grows.',
    tags: ['uncertainty', 'wisdom', 'growth'],
    mbtiTypes: ['INTP', 'INFP', 'ENTP', 'ENFP'],
    phaseSignature: 'drift',
    wavelength: 480, // Blue-green - contemplation
    conceptClusters: ['uncertainty', 'wisdom', 'learning']
  },

  // TIME-BASED QUOTES
  {
    id: 'dawn_001',
    text: 'Each line of code is a small dawn—bringing light to what was darkness.',
    tags: ['dawn', 'creation', 'illumination'],
    mbtiTypes: ['INFJ', 'INTJ', 'ENFJ', 'ENTJ'],
    timeOfDay: 'dawn',
    wavelength: 580, // Warm yellow - dawn
    conceptClusters: ['creation', 'understanding', 'clarity']
  },
  {
    id: 'night_001',
    text: 'In the quiet of code at night, the universe reveals its patterns.',
    tags: ['night', 'patterns', 'solitude'],
    mbtiTypes: ['INTJ', 'INTP', 'INFJ', 'INFP'],
    timeOfDay: 'night',
    wavelength: 420, // Deep blue - night
    conceptClusters: ['patterns', 'contemplation', 'discovery']
  },

  // MBTI-SPECIFIC RESONANCE
  {
    id: 'intj_001',
    text: 'Architecture is frozen music; code is liquid thought.',
    tags: ['architecture', 'design', 'thought'],
    mbtiTypes: ['INTJ', 'INTP'],
    phaseSignature: 'coherence',
    wavelength: 500, // Blue-green - structured
    conceptClusters: ['architecture', 'design', 'systems']
  },
  {
    id: 'enfp_001',
    text: 'Every bug is a hidden feature waiting for the right perspective.',
    tags: ['optimism', 'perspective', 'possibilities'],
    mbtiTypes: ['ENFP', 'ENTP'],
    phaseSignature: 'entropy',
    wavelength: 570, // Yellow - optimism
    conceptClusters: ['debugging', 'perspective', 'creativity']
  },
  {
    id: 'infp_001', 
    text: 'Code with compassion—for the future self who will read it.',
    tags: ['compassion', 'empathy', 'future'],
    mbtiTypes: ['INFP', 'ISFP', 'ENFP', 'ESFP'],
    phaseSignature: 'resonance',
    wavelength: 520, // Green - harmony
    conceptClusters: ['empathy', 'maintainability', 'care']
  },

  // GHOST PERSONAS INSPIRED
  {
    id: 'mentor_001',
    text: 'Every master was once a beginner who refused to give up.',
    tags: ['persistence', 'growth', 'mastery'],
    mbtiTypes: ['INTJ', 'ENTJ', 'ISFJ', 'ESFJ'],
    phaseSignature: 'coherence',
    wavelength: 520, // Mentor green
    conceptClusters: ['learning', 'persistence', 'mastery']
  },
  {
    id: 'mystic_001',
    text: 'The digital realm is but another layer of the infinite mystery.',
    tags: ['mystery', 'spirituality', 'depth'],
    mbtiTypes: ['INFJ', 'INFP', 'ENFP', 'INTP'],
    phaseSignature: 'emergence',
    wavelength: 450, // Mystic violet
    conceptClusters: ['mystery', 'consciousness', 'depth']
  },
  {
    id: 'oracle_001',
    text: 'Time is not linear in the space of ideas—all possibilities exist simultaneously.',
    tags: ['time', 'possibility', 'oracle'],
    mbtiTypes: ['INTJ', 'INTP', 'INFJ', 'ENFP'],
    phaseSignature: 'emergence',
    wavelength: 400, // Oracle indigo
    conceptClusters: ['time', 'possibility', 'wisdom']
  }
];

export class QuoteBankService {
  private static instance: QuoteBankService;
  private quotes: Quote[] = QUOTE_BANK;

  private constructor() {}

  static getInstance(): QuoteBankService {
    if (!QuoteBankService.instance) {
      QuoteBankService.instance = new QuoteBankService();
    }
    return QuoteBankService.instance;
  }

  // Get quotes by MBTI type
  getQuotesByMBTI(mbtiType: string): Quote[] {
    return this.quotes.filter(quote => 
      quote.mbtiTypes.includes(mbtiType)
    );
  }

  // Get quotes by phase signature
  getQuotesByPhase(phaseSignature: string): Quote[] {
    return this.quotes.filter(quote => 
      quote.phaseSignature === phaseSignature
    );
  }

  // Get quotes by time of day
  getQuotesByTime(timeOfDay: string): Quote[] {
    return this.quotes.filter(quote => 
      quote.timeOfDay === timeOfDay || !quote.timeOfDay
    );
  }

  // Get quotes by concept clusters
  getQuotesByConceptClusters(clusters: string[]): Quote[] {
    return this.quotes.filter(quote => 
      quote.conceptClusters?.some(cluster => 
        clusters.includes(cluster)
      )
    );
  }

  // Smart quote selection based on multiple factors
  selectQuote(params: {
    mbtiType?: string;
    phaseSignature?: string;
    timeOfDay?: string;
    conceptClusters?: string[];
    previousQuoteIds?: string[];
  }): Quote | null {
    let candidates = this.quotes;

    // Filter out recently shown quotes
    if (params.previousQuoteIds) {
      candidates = candidates.filter(quote => 
        !params.previousQuoteIds!.includes(quote.id)
      );
    }

    // Score quotes based on relevance
    const scoredQuotes = candidates.map(quote => {
      let score = 0;

      // MBTI match (high priority)
      if (params.mbtiType && quote.mbtiTypes.includes(params.mbtiType)) {
        score += 10;
      }

      // Phase signature match (high priority)
      if (params.phaseSignature && quote.phaseSignature === params.phaseSignature) {
        score += 8;
      }

      // Time of day match (medium priority)
      if (params.timeOfDay && quote.timeOfDay === params.timeOfDay) {
        score += 5;
      }

      // Concept cluster overlap (medium priority)
      if (params.conceptClusters && quote.conceptClusters) {
        const overlap = params.conceptClusters.filter(cluster =>
          quote.conceptClusters!.includes(cluster)
        ).length;
        score += overlap * 3;
      }

      // Add some randomness to prevent repetition
      score += Math.random() * 2;

      return { quote, score };
    });

    // Sort by score and return the best match
    scoredQuotes.sort((a, b) => b.score - a.score);
    return scoredQuotes.length > 0 ? scoredQuotes[0].quote : null;
  }

  // Get current time context
  getCurrentTimeContext(): string {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 8) return 'dawn';
    if (hour >= 8 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 20) return 'evening';
    if (hour >= 20 && hour < 23) return 'night';
    return 'night'; // Late night/early morning
  }

  // Add new quote to the bank
  addQuote(quote: Quote): void {
    this.quotes.push(quote);
  }

  // Get all quotes with specific tags
  getQuotesByTags(tags: string[]): Quote[] {
    return this.quotes.filter(quote =>
      tags.some(tag => quote.tags.includes(tag))
    );
  }
}