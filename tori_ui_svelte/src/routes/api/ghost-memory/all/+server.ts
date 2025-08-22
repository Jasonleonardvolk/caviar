import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

/**
 * GET /api/ghost-memory/all
 * Returns all ghost memory data including memories, mood curves, and concept arcs
 */
export const GET: RequestHandler = async ({ locals }) => {
  try {
    // In a real implementation, this would fetch from a database
    // For now, we'll return data from localStorage or a mock response
    
    // Check if we're in the browser environment
    if (typeof window !== 'undefined' && window.localStorage) {
      // Browser environment - try to get from localStorage
      const memories = JSON.parse(localStorage.getItem('ghost-memories') || '[]');
      const moodCurves = JSON.parse(localStorage.getItem('ghost-mood-curves') || '[]');
      const conceptArcs = JSON.parse(localStorage.getItem('ghost-concept-arcs') || '[]');
      
      return json({
        success: true,
        memories,
        moodCurves,
        conceptArcs
      });
    } else {
      // Server environment - return mock data or fetch from backend
      return json({
        success: true,
        memories: getMockMemories(),
        moodCurves: [],
        conceptArcs: []
      });
    }
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('Error fetching ghost memory data:', error);
    
    return json({
      success: false,
      error: 'Failed to fetch ghost memory data',
      memories: [],
      moodCurves: [],
      conceptArcs: []
    
}, { status: 500 });
  }
};

/**
 * POST /api/ghost-memory/all
 * Store new ghost memory entry
 */
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const data = await request.json();
    
    // In a real implementation, this would save to a database
    // For now, we'll just return success
    
    return json({
      success: true,
      message: 'Ghost memory entry stored successfully',
      id: `memory_${Date.now()}`
    });
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('Error storing ghost memory:', error);
    
    return json({
      success: false,
      error: 'Failed to store ghost memory entry'
    
}, { status: 500 });
  }
};

/**
 * Generate mock ghost memory data for demonstration
 */
function getMockMemories() {
  const personas = ['Mentor', 'Mystic', 'Unsettled', 'Chaotic', 'Oracular', 'Dreaming'];
  const eventTypes = ['emergence', 'shift', 'letter', 'mood', 'intervention', 'reflection'];
  const phaseSignatures = ['high-coherence', 'high-entropy', 'phase-drift', 'stable', 'mixed-state'];
  const emotionalTones = ['supportive', 'mystical', 'anxious', 'energetic', 'wise', 'ethereal'];
  
  const memories = [];
  const now = new Date();
  
  // Generate 20 mock memories
  for (let i = 0; i < 20; i++) {
    const timestamp = new Date(now.getTime() - Math.random() * 7 * 24 * 60 * 60 * 1000); // Random time in past week
    const persona = personas[Math.floor(Math.random() * personas.length)];
    const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
    
    memories.push({
      id: `mock_memory_${i}`,
      timestamp: timestamp.toISOString(),
      sessionId: `session_${Math.floor(i / 5)}`,
      persona,
      eventType,
      content: {
        message: eventType === 'letter' 
          ? `Dear friend, I wanted to share some ${persona.toLowerCase()} wisdom with you...`
          : eventType === 'emergence'
          ? `The ${persona} has emerged to guide you through this moment...`
          : undefined,
        moodVector: eventType === 'mood' ? {
          empathy: Math.random(),
          curiosity: Math.random(),
          anxiety: Math.random(),
          energy: Math.random()
        } : undefined
      },
      trigger: {
        conceptDiffs: [`concept-${i}`, `concept-${i+1}`],
        phaseMetrics: {
          coherence: Math.random(),
          entropy: Math.random(),
          drift: (Math.random() - 0.5) * 2,
          phaseAngle: Math.random() * Math.PI * 2
        },
        userContext: {
          sentiment: Math.random() * 2 - 1,
          frustrationLevel: Math.random(),
          engagementLevel: Math.random()
        },
        systemContext: {
          conversationLength: Math.floor(Math.random() * 50) + 1,
          recentConcepts: [`concept-${i-1}`, `concept-${i}`, `concept-${i+1}`]
        }
      },
      indexing: {
        conceptTags: [`concept-${i}`, `persona-${persona.toLowerCase()}`, 'mock-data'],
        phaseSignature: phaseSignatures[Math.floor(Math.random() * phaseSignatures.length)],
        emotionalTone: emotionalTones[Math.floor(Math.random() * emotionalTones.length)],
        searchableContent: `${persona} ${eventType} event with mock data`,
        memoryWeight: Math.random()
      },
      outcomes: Math.random() > 0.5 ? {
        userResponse: ['positive', 'neutral', 'negative', 'ignored'][Math.floor(Math.random() * 4)],
        effectiveness: Math.random(),
        followUpRequired: Math.random() > 0.7,
        learningNote: 'This is a mock learning note for demonstration purposes.'
      } : undefined
    });
  }
  
  return memories;
}
