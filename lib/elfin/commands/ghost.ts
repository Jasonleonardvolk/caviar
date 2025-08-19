// Enhanced Ghost Commands with Existing Persona Engine Integration
import type { ElfinContext, GhostCommand, GhostTrigger, ElfinResult } from '../types.js';
import { ghostState, activeAgents, conversationLog } from '$lib/stores';
import { get } from 'svelte/store';
import { 
  evaluateGhostTriggers, 
  generateGhostMessage, 
  GHOST_PERSONAS,
  type GhostContext 
} from '../ghostPersonaEngine.js';

export async function runGhostCommand(line: string, ctx: ElfinContext): Promise<ElfinResult> {
  const startTime = Date.now();
  const command = parseGhostCommand(line);
  
  if (!command) {
    throw new Error('Invalid ghost command syntax');
  }

  const { persona, action, input, options } = command;
  let result: any = null;
  let sideEffects: any = {};

  try {
    switch (action) {
      case 'emerge':
        result = await handleGhostEmerge(persona, ctx, options);
        sideEffects.ghostStateChange = { activePersona: persona };
        break;
        
      case 'focus':
        result = await handleGhostFocus(persona, input, ctx, options);
        sideEffects.ghostStateChange = { focusConceptId: input };
        break;
        
      case 'search':
        result = await handleGhostSearch(persona, input, ctx, options);
        break;
        
      case 'project':
        result = await handleGhostProject(persona, input, ctx, options);
        sideEffects.thoughtspaceProjections = [{ concept: input, target: 'Thoughtspace' }];
        break;
        
      case 'morph':
        result = await handleGhostMorph(input, ctx, options);
        sideEffects.ghostStateChange = { activePersona: input };
        break;
        
      case 'dismiss':
        result = await handleGhostDismiss(persona, ctx);
        sideEffects.ghostStateChange = { activePersona: null };
        break;
        
      default:
        throw new Error(`Unknown ghost action: ${action}`);
    }

    return {
      success: true,
      command: { type: 'ghost', raw: line, params: command, timestamp: new Date() },
      result,
      context: ctx,
      executionTime: Date.now() - startTime,
      sideEffects
    };

  } catch (error) {
    return {
      success: false,
      command: { type: 'ghost', raw: line, params: command, timestamp: new Date() },
      error: error instanceof Error ? error.message : 'Unknown error',
      context: ctx,
      executionTime: Date.now() - startTime
    };
  }
}

function parseGhostCommand(line: string): GhostCommand | null {
  // Enhanced parsing for multiple ghost command patterns
  
  // Ghost("persona").action("input")
  const basicMatch = line.match(/Ghost\("([^"]+)"\)\.(\w+)\("([^"]*)"\)/);
  if (basicMatch) {
    const [, persona, action, input] = basicMatch;
    return { persona, action, input };
  }
  
  // Ghost("persona").emerge() - no input needed
  const emergeMatch = line.match(/Ghost\("([^"]+)"\)\.emerge\(\)/);
  if (emergeMatch) {
    const [, persona] = emergeMatch;
    return { persona, action: 'emerge', input: '' };
  }
  
  // morphPersona() - context-driven persona change
  const morphMatch = line.match(/morphPersona\(\)/);
  if (morphMatch) {
    return { persona: 'auto', action: 'morph', input: '' };
  }
  
  // morphPersona("specific_persona")
  const morphSpecificMatch = line.match(/morphPersona\("([^"]+)"\)/);
  if (morphSpecificMatch) {
    const [, targetPersona] = morphSpecificMatch;
    return { persona: 'auto', action: 'morph', input: targetPersona };
  }
  
  return null;
}

async function handleGhostEmerge(persona: string, ctx: ElfinContext, options?: any): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost ${persona} emerging...`);
  
  // Get current ghost state
  const currentGhostState = get(ghostState);
  const conversation = get(conversationLog);
  
  // Create ghost context for persona evaluation
  const ghostContext: GhostContext = {
    message: ctx.variables['$LAST_MESSAGE'] || '',
    conversationHistory: conversation,
    phaseCoherence: ctx.phaseMetrics?.coherence || 0.8,
    lyapunovStability: ctx.phaseMetrics?.lyapunovStability || 0.8,
    sessionDuration: ctx.session?.duration || 0,
    conceptIds: ctx.variables['$LAST_CONCEPT_IDS'] || []
  };
  
  // If persona is 'auto', use trigger evaluation to determine best persona
  let targetPersona = persona;
  if (persona === 'auto') {
    const trigger = evaluateGhostTriggers(ghostContext);
    if (trigger) {
      targetPersona = trigger.persona;
      console.log(`👻 [ELFIN++] Auto-selected persona: ${targetPersona} (${trigger.reason})`);
    } else {
      targetPersona = 'mentor'; // Default fallback
      console.log(`👻 [ELFIN++] No trigger found, defaulting to mentor`);
    }
  }
  
  // Validate persona exists
  const personaConfig = Object.values(GHOST_PERSONAS).find(p => p.name === targetPersona);
  if (!personaConfig) {
    throw new Error(`Unknown ghost persona: ${targetPersona}`);
  }
  
  // Update ghost state
  ghostState.update(state => ({
    ...state,
    activePersona: targetPersona,
    auraIntensity: options?.intensity || personaConfig.intensity,
    lastEmergence: new Date(),
    focusConceptId: options?.conceptIds?.[0]
  }));
  
  // Generate ghost message
  const ghostMessage = generateGhostMessage(
    {
      persona: targetPersona,
      probability: 1.0,
      reason: options?.trigger?.reason || 'script_invocation',
      wavelength: personaConfig.wavelength,
      intensity: personaConfig.intensity,
      timestamp: new Date()
    },
    ghostContext
  );
  
  // Dispatch event for UI updates
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('tori-ghost-event', { 
      detail: {
        id: `ghost_${Date.now()}`,
        timestamp: new Date(),
        eventType: 'emergence',
        persona: targetPersona,
        trigger: {
          reason: options?.trigger?.reason || 'script_invocation',
          context: ghostContext,
          confidence: 1.0
        },
        content: {
          message: ghostMessage.ghostMessage
        },
        sessionId: ctx.session?.id || 'default',
        conceptIds: ghostContext.conceptIds,
        wavelength: personaConfig.wavelength
      }
    }));
  }
  
  // API call to backend if available
  try {
    const response = await fetch('/api/ghost/emerge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        persona: targetPersona,
        context: ghostContext,
        trigger: options?.trigger
      })
    });
    
    if (response.ok) {
      const apiResult = await response.json();
      ctx.variables['$GHOST_API_RESULT'] = apiResult;
    }
  } catch (error) {
    console.warn('[ELFIN++] Ghost emerge API unavailable, continuing with local state');
  }
  
  // Store result in context
  ctx.variables['$ACTIVE_GHOST'] = targetPersona;
  ctx.variables['$GHOST_MESSAGE'] = ghostMessage.ghostMessage;
  ctx.variables['$GHOST_WAVELENGTH'] = personaConfig.wavelength;
  
  return {
    persona: targetPersona,
    emerged: true,
    message: ghostMessage.ghostMessage,
    wavelength: personaConfig.wavelength,
    intensity: personaConfig.intensity,
    timestamp: new Date()
  };
}

async function handleGhostFocus(persona: string, input: string, ctx: ElfinContext, options?: any): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost ${persona} focusing on: ${input}`);
  
  // Update ghost state with focus
  ghostState.update(state => ({
    ...state,
    activePersona: persona,
    focusConceptId: input,
    auraIntensity: Math.min(1.0, (state.auraIntensity || 0.5) + 0.2)
  }));
  
  // Dispatch focus event for thoughtspace highlighting
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('thoughtspace:focus', { 
      detail: { 
        conceptId: input,
        persona: persona,
        timestamp: Date.now()
      } 
    }));
  }
  
  // Trigger concept activation in stores
  import('$lib/stores').then(({ activateConcept, focusConcept }) => {
    activateConcept(input);
    focusConcept(input);
  });
  
  // API call for ghost focus
  try {
    const response = await fetch('/api/ghost/focus', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona, concept: input, options })
    });
    
    if (response.ok) {
      const result = await response.json();
      ctx.variables['$FOCUS_RESULT'] = result;
      
      // Update context with search results if provided
      if (result.searchResults) {
        ctx.variables['$LAST_SEARCH_RESULT'] = result.searchResults;
      }
      
      return result;
    }
  } catch (error) {
    console.warn('[ELFIN++] Ghost focus API unavailable');
  }
  
  // Local fallback result
  const result = {
    persona,
    concept: input,
    focused: true,
    insights: `Ghost ${persona} is now focusing cognitive attention on ${input}`,
    timestamp: new Date()
  };
  
  ctx.variables['$FOCUS_RESULT'] = result;
  return result;
}

async function handleGhostSearch(persona: string, input: string, ctx: ElfinContext, options?: any): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost ${persona} searching: ${input}`);
  
  try {
    const response = await fetch('/api/ghost/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona, query: input, options })
    });
    
    if (response.ok) {
      const result = await response.json();
      ctx.variables['$LAST_SEARCH_RESULT'] = result;
      ctx.variables['$SEARCH_QUERY'] = input;
      ctx.lastResult = result;
      
      return result;
    }
  } catch (error) {
    console.warn('[ELFIN++] Ghost search API unavailable, using mock search');
  }
  
  // Mock search result for development
  const mockResult = {
    query: input,
    persona,
    results: [
      {
        title: `Knowledge about ${input}`,
        content: `Relevant information regarding ${input} from the perspective of ${persona} ghost persona.`,
        relevance: 0.9,
        source: 'TORI Knowledge Base',
        conceptIds: [input.toLowerCase().replace(/\s+/g, '-')]
      }
    ],
    timestamp: new Date()
  };
  
  ctx.variables['$LAST_SEARCH_RESULT'] = mockResult;
  ctx.lastResult = mockResult;
  
  return mockResult;
}

async function handleGhostProject(persona: string, input: string, ctx: ElfinContext, options?: any): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost ${persona} projecting: ${input}`);
  
  // Dispatch projection event for thoughtspace
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('thoughtspace:project', { 
      detail: { 
        persona, 
        concept: input, 
        source: 'ghost',
        wavelength: GHOST_PERSONAS[persona.toUpperCase() as keyof typeof GHOST_PERSONAS]?.wavelength || 520,
        intensity: options?.intensity || 0.8
      } 
    }));
  }
  
  const result = {
    persona,
    concept: input,
    projected: true,
    target: 'Thoughtspace',
    timestamp: new Date()
  };
  
  ctx.variables['$PROJECTION_RESULT'] = result;
  return result;
}

async function handleGhostMorph(targetPersona: string, ctx: ElfinContext, options?: any): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost morphing to: ${targetPersona}`);
  
  const currentGhostState = get(ghostState);
  const conversation = get(conversationLog);
  
  // If no target specified, use trigger evaluation
  if (!targetPersona || targetPersona === 'auto') {
    const ghostContext: GhostContext = {
      message: ctx.variables['$LAST_MESSAGE'] || '',
      conversationHistory: conversation,
      phaseCoherence: ctx.phaseMetrics?.coherence || 0.8,
      lyapunovStability: ctx.phaseMetrics?.lyapunovStability || 0.8,
      sessionDuration: ctx.session?.duration || 0
    };
    
    const trigger = evaluateGhostTriggers(ghostContext);
    targetPersona = trigger?.persona || 'mentor';
  }
  
  // Update ghost state
  ghostState.update(state => ({
    ...state,
    activePersona: targetPersona,
    lastEmergence: new Date(),
    auraIntensity: GHOST_PERSONAS[targetPersona.toUpperCase() as keyof typeof GHOST_PERSONAS]?.intensity || 0.8
  }));
  
  // Dispatch morph event
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('ghost:morph', { 
      detail: { 
        from: currentGhostState.activePersona,
        to: targetPersona,
        timestamp: Date.now()
      } 
    }));
  }
  
  const result = {
    from: currentGhostState.activePersona,
    to: targetPersona,
    morphed: true,
    timestamp: new Date()
  };
  
  ctx.variables['$MORPH_RESULT'] = result;
  ctx.variables['$ACTIVE_GHOST'] = targetPersona;
  
  return result;
}

async function handleGhostDismiss(persona: string, ctx: ElfinContext): Promise<any> {
  console.log(`👻 [ELFIN++] Ghost ${persona} dismissing...`);
  
  // Update ghost state
  ghostState.update(state => ({
    ...state,
    activePersona: null,
    focusConceptId: undefined,
    auraIntensity: 0
  }));
  
  // Dispatch dismiss event
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('ghost:dismiss', { 
      detail: { 
        persona,
        timestamp: Date.now()
      } 
    }));
  }
  
  const result = {
    persona,
    dismissed: true,
    timestamp: new Date()
  };
  
  ctx.variables['$DISMISS_RESULT'] = result;
  ctx.variables['$ACTIVE_GHOST'] = null;
  
  return result;
}

export { runGhostCommand };