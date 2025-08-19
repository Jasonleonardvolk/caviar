// Enhanced Vault Commands with Memory Systems Integration
import type { ElfinContext, VaultCommand, ElfinResult } from '../types.js';
import { vaultEntries, sealedArcs } from '$lib/stores';

export async function runVaultCommand(line: string, ctx: ElfinContext): Promise<ElfinResult> {
  const startTime = Date.now();
  const command = parseVaultCommand(line);
  
  if (!command) {
    throw new Error('Invalid vault command syntax');
  }

  let result: any = null;
  let sideEffects: any = {};

  try {
    switch (command.operation) {
      case 'save':
        result = await handleVaultSave(command.key!, command.value, ctx, command.metadata);
        sideEffects.vaultOperations = [`save:${command.key}`];
        break;
        
      case 'load':
        result = await handleVaultLoad(command.key!, command.variable!, ctx);
        break;
        
      case 'seal':
        result = await handleVaultSeal(command.key || command.arcId!, ctx, command.reason);
        sideEffects.vaultOperations = [`seal:${command.key || command.arcId}`];
        break;
        
      case 'unseal':
        result = await handleVaultUnseal(command.key || command.arcId!, ctx);
        sideEffects.vaultOperations = [`unseal:${command.key || command.arcId}`];
        break;
        
      case 'link':
        result = await handleVaultLink(command.key!, command.value, ctx);
        break;
        
      case 'search':
        result = await handleVaultSearch(command.key!, ctx);
        break;
        
      default:
        throw new Error(`Unknown vault operation: ${command.operation}`);
    }

    return {
      success: true,
      command: { type: 'vault', raw: line, params: command, timestamp: new Date() },
      result,
      context: ctx,
      executionTime: Date.now() - startTime,
      sideEffects
    };

  } catch (error) {
    return {
      success: false,
      command: { type: 'vault', raw: line, params: command, timestamp: new Date() },
      error: error instanceof Error ? error.message : 'Unknown error',
      context: ctx,
      executionTime: Date.now() - startTime
    };
  }
}

function parseVaultCommand(line: string): VaultCommand | null {
  // Vault.save("key", $value)
  const saveMatch = line.match(/Vault\.save\("([^"]+)",\s*(.+)\)/);
  if (saveMatch) {
    const [, key, valueExpr] = saveMatch;
    return {
      operation: 'save',
      key,
      value: valueExpr
    };
  }
  
  // $var = Vault.load("key")
  const loadMatch = line.match(/\$(\w+)\s*=\s*Vault\.load\("([^"]+)"\)/);
  if (loadMatch) {
    const [, variable, key] = loadMatch;
    return {
      operation: 'load',
      key,
      variable
    };
  }
  
  // Vault.seal("arcId") or vault.seal(currentArc)
  const sealMatch = line.match(/vault\.seal\("([^"]+)"\)|vault\.seal\((\w+)\)/);
  if (sealMatch) {
    const [, arcId1, arcId2] = sealMatch;
    return {
      operation: 'seal',
      arcId: arcId1 || arcId2
    };
  }
  
  // Vault.unseal("arcId")
  const unsealMatch = line.match(/Vault\.unseal\("([^"]+)"\)/);
  if (unsealMatch) {
    const [, arcId] = unsealMatch;
    return {
      operation: 'unseal',
      arcId
    };
  }
  
  // Vault.link("conceptA", "conceptB")
  const linkMatch = line.match(/Vault\.link\("([^"]+)",\s*"([^"]+)"\)/);
  if (linkMatch) {
    const [, key, value] = linkMatch;
    return {
      operation: 'link',
      key,
      value
    };
  }
  
  // Vault.search("query")
  const searchMatch = line.match(/Vault\.search\("([^"]+)"\)/);
  if (searchMatch) {
    const [, query] = searchMatch;
    return {
      operation: 'search',
      key: query
    };
  }
  
  return null;
}

async function handleVaultSave(key: string, valueExpr: string, ctx: ElfinContext, metadata?: any): Promise<any> {
  console.log(`💾 [ELFIN++] Vault saving: ${key}`);
  
  // Resolve value from context variables
  let value = valueExpr;
  if (valueExpr.startsWith('$')) {
    const varName = valueExpr.slice(1);
    value = ctx.variables[varName] ?? ctx.variables[valueExpr] ?? valueExpr;
  }
  
  // Enhanced metadata with concept analysis
  const enhancedMetadata = {
    timestamp: Date.now(),
    sessionId: ctx.session?.id || 'default',
    conceptIds: extractConceptIds(typeof value === 'string' ? value : JSON.stringify(value)),
    emotionalWeight: calculateEmotionalWeight(value),
    phaseMetrics: ctx.phaseMetrics,
    ...metadata
  };
  
  try {
    const response = await fetch('/api/vault/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        key, 
        value, 
        metadata: enhancedMetadata
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`💾 [ELFIN++] Vault saved successfully: ${key}`);
      
      // Update local vault store
      vaultEntries.update(entries => [...entries, {
        id: result.id || key,
        title: key,
        content: typeof value === 'string' ? value : JSON.stringify(value),
        sealed: false,
        timestamp: new Date(),
        conceptIds: enhancedMetadata.conceptIds,
        emotionalWeight: enhancedMetadata.emotionalWeight
      }]);
      
      ctx.lastResult = result;
      ctx.variables['$VAULT_SAVE_RESULT'] = result;
      
      return result;
    } else {
      throw new Error(`Vault save failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault save API unavailable, using local storage');
    
    // Fallback to local storage for development
    const entry = {
      id: `vault_${Date.now()}`,
      key,
      title: key,
      content: typeof value === 'string' ? value : JSON.stringify(value),
      value,
      sealed: false,
      timestamp: new Date(),
      conceptIds: enhancedMetadata.conceptIds,
      emotionalWeight: enhancedMetadata.emotionalWeight,
      metadata: enhancedMetadata
    };
    
    vaultEntries.update(entries => [...entries, entry]);
    
    ctx.lastResult = entry;
    ctx.variables['$VAULT_SAVE_RESULT'] = entry;
    
    return entry;
  }
}

async function handleVaultLoad(key: string, variable: string, ctx: ElfinContext): Promise<any> {
  console.log(`💾 [ELFIN++] Vault loading: ${key} into $${variable}`);
  
  try {
    const response = await fetch(`/api/vault/load?key=${encodeURIComponent(key)}`);
    
    if (response.ok) {
      const result = await response.json();
      ctx.variables[variable] = result.value;
      ctx.variables[`$${variable}`] = result.value;
      ctx.lastResult = result.value;
      
      console.log(`💾 [ELFIN++] Vault loaded: ${key} -> $${variable}`);
      return result;
    } else if (response.status === 404) {
      console.warn(`💾 [ELFIN++] Vault key not found: ${key}`);
      ctx.variables[variable] = null;
      ctx.variables[`$${variable}`] = null;
      return { error: 'Key not found', key };
    } else {
      throw new Error(`Vault load failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault load API unavailable, checking local storage');
    
    // Fallback: search local vault entries
    import('$lib/stores').then(({ vaultEntries }) => {
      const entries = vaultEntries.subscribe(value => {
        const entry = value.find(e => e.key === key || e.title === key);
        if (entry) {
          ctx.variables[variable] = entry.value || entry.content;
          ctx.variables[`$${variable}`] = entry.value || entry.content;
          console.log(`💾 [ELFIN++] Found in local vault: ${key} -> $${variable}`);
        } else {
          ctx.variables[variable] = null;
          ctx.variables[`$${variable}`] = null;
          console.warn(`💾 [ELFIN++] Key not found in local vault: ${key}`);
        }
      });
      entries(); // Unsubscribe immediately
    });
    
    return { error: 'API unavailable, attempted local lookup', key };
  }
}

async function handleVaultSeal(arcId: string, ctx: ElfinContext, reason?: string): Promise<any> {
  console.log(`🔒 [ELFIN++] Vault sealing arc: ${arcId}`);
  
  // Handle special arc identifiers
  let targetArcId = arcId;
  if (arcId === 'currentArc') {
    // Use current conversation context as arc
    targetArcId = `conversation_${ctx.session?.id}_${Date.now()}`;
    console.log(`🔒 [ELFIN++] Resolved currentArc to: ${targetArcId}`);
  }
  
  const sealReason = reason || 'Emotional protection protocol';
  
  try {
    const response = await fetch('/api/vault/seal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        arcId: targetArcId, 
        reason: sealReason,
        timestamp: Date.now(),
        sessionId: ctx.session?.id,
        phaseMetrics: ctx.phaseMetrics
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      
      // Update sealed arcs store
      sealedArcs.update(arcs => [...arcs, targetArcId]);
      
      // Dispatch seal event for UI updates
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('vault:sealed', { 
          detail: { 
            arcId: targetArcId, 
            reason: sealReason,
            timestamp: Date.now() 
          } 
        }));
      }
      
      console.log(`🔒 [ELFIN++] Arc sealed successfully: ${targetArcId}`);
      ctx.lastResult = result;
      ctx.variables['$SEAL_RESULT'] = result;
      
      return result;
    } else {
      throw new Error(`Vault seal failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault seal API unavailable, updating local state');
    
    // Update local state for immediate feedback
    sealedArcs.update(arcs => [...arcs, targetArcId]);
    
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('vault:sealed', { 
        detail: { 
          arcId: targetArcId, 
          reason: sealReason,
          fallback: true,
          timestamp: Date.now() 
        } 
      }));
    }
    
    const result = {
      arcId: targetArcId,
      sealed: true,
      reason: sealReason,
      timestamp: new Date(),
      fallback: true
    };
    
    ctx.lastResult = result;
    ctx.variables['$SEAL_RESULT'] = result;
    
    return result;
  }
}

async function handleVaultUnseal(arcId: string, ctx: ElfinContext): Promise<any> {
  console.log(`🔓 [ELFIN++] Vault unsealing arc: ${arcId}`);
  
  try {
    const response = await fetch('/api/vault/unseal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ arcId, timestamp: Date.now() })
    });
    
    if (response.ok) {
      const result = await response.json();
      
      // Update sealed arcs store
      sealedArcs.update(arcs => arcs.filter(id => id !== arcId));
      
      // Dispatch unseal event
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('vault:unsealed', { 
          detail: { arcId, timestamp: Date.now() } 
        }));
      }
      
      console.log(`🔓 [ELFIN++] Arc unsealed successfully: ${arcId}`);
      ctx.lastResult = result;
      
      return result;
    } else {
      throw new Error(`Vault unseal failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault unseal API unavailable');
    
    // Update local state
    sealedArcs.update(arcs => arcs.filter(id => id !== arcId));
    
    const result = {
      arcId,
      unsealed: true,
      timestamp: new Date(),
      fallback: true
    };
    
    ctx.lastResult = result;
    return result;
  }
}

async function handleVaultLink(conceptA: string, conceptB: string, ctx: ElfinContext): Promise<any> {
  console.log(`🔗 [ELFIN++] Vault linking: ${conceptA} ↔ ${conceptB}`);
  
  try {
    const response = await fetch('/api/vault/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        conceptA, 
        conceptB, 
        timestamp: Date.now(),
        sessionId: ctx.session?.id
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`🔗 [ELFIN++] Concepts linked successfully`);
      
      // Update concept graph with new link
      import('$lib/stores').then(({ linkConcepts }) => {
        linkConcepts(conceptA, conceptB, 1.0, 'correlation');
      });
      
      ctx.lastResult = result;
      return result;
    } else {
      throw new Error(`Vault link failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault link API unavailable, updating local graph');
    
    // Update local concept graph
    import('$lib/stores').then(({ linkConcepts }) => {
      linkConcepts(conceptA, conceptB, 1.0, 'correlation');
    });
    
    const result = {
      conceptA,
      conceptB,
      linked: true,
      timestamp: new Date(),
      fallback: true
    };
    
    ctx.lastResult = result;
    return result;
  }
}

async function handleVaultSearch(query: string, ctx: ElfinContext): Promise<any> {
  console.log(`🔍 [ELFIN++] Vault searching: ${query}`);
  
  try {
    const response = await fetch(`/api/vault/search?q=${encodeURIComponent(query)}`);
    
    if (response.ok) {
      const result = await response.json();
      ctx.variables['$SEARCH_RESULT'] = result;
      ctx.lastResult = result;
      
      return result;
    } else {
      throw new Error(`Vault search failed: ${response.status}`);
    }
  } catch (error) {
    console.warn('[ELFIN++] Vault search API unavailable, using local search');
    
    // Simple local search fallback
    const mockResult = {
      query,
      results: [
        {
          id: `search_${Date.now()}`,
          title: `Search results for: ${query}`,
          content: `Local search results for "${query}" would appear here.`,
          relevance: 0.8,
          conceptIds: [query.toLowerCase().replace(/\s+/g, '-')]
        }
      ],
      timestamp: new Date(),
      fallback: true
    };
    
    ctx.variables['$SEARCH_RESULT'] = mockResult;
    ctx.lastResult = mockResult;
    
    return mockResult;
  }
}

// Utility functions
function extractConceptIds(text: string): string[] {
  // Enhanced concept extraction using existing patterns
  const concepts = [];
  const lowerText = text.toLowerCase();
  
  if (lowerText.includes('quantum')) concepts.push('quantum-physics');
  if (lowerText.includes('memory')) concepts.push('memory');
  if (lowerText.includes('ghost') || lowerText.includes('persona')) concepts.push('ghost-personas');
  if (lowerText.includes('consciousness')) concepts.push('consciousness');
  if (lowerText.includes('phase')) concepts.push('phase-dynamics');
  if (lowerText.includes('stability')) concepts.push('stability');
  if (lowerText.includes('emotion')) concepts.push('emotion');
  if (lowerText.includes('thought')) concepts.push('cognition');
  
  // Extract potential concept words (nouns)
  const words = text.match(/\b[a-zA-Z]{4,}\b/g) || [];
  const uniqueWords = [...new Set(words.map(w => w.toLowerCase()))];
  
  return [...concepts, ...uniqueWords.slice(0, 5)]; // Limit to avoid noise
}

function calculateEmotionalWeight(value: any): number {
  // Simple emotional weight calculation
  if (typeof value !== 'string') return 0.5;
  
  const text = value.toLowerCase();
  const positiveWords = ['happy', 'joy', 'success', 'good', 'great', 'excellent'];
  const negativeWords = ['sad', 'angry', 'frustrated', 'error', 'problem', 'fail'];
  const anxietyWords = ['worried', 'anxious', 'scared', 'confused', 'lost'];
  
  let weight = 0.5; // Neutral baseline
  
  positiveWords.forEach(word => {
    if (text.includes(word)) weight -= 0.1;
  });
  
  negativeWords.forEach(word => {
    if (text.includes(word)) weight += 0.15;
  });
  
  anxietyWords.forEach(word => {
    if (text.includes(word)) weight += 0.2;
  });
  
  return Math.max(0, Math.min(1, weight));
}

export { runVaultCommand };