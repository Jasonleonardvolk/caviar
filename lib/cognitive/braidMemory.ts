// BraidMemory.ts - Phase 2 Memory Topology with Loop Compression & Digest Fingerprinting

// TEMPORARY - ALL TYPE DEFINITIONS LOCAL TO BYPASS VITE CACHE ISSUE
// Will restore proper imports once cache is resolved

interface LoopRecord {
  // Core identification
  id: string;
  prompt: string;
  
  // Execution trace
  glyphPath: string[];                    // Full glyph sequence
  phaseTrace: number[];                   // Φ(t) trajectory over time
  coherenceTrace: number[];               // C(t) trajectory over time
  contradictionTrace: number[];           // Π(t) trajectory over time
  
  // Closure state
  returnGlyph?: string;                   // R_i: final closure glyph
  closed: boolean;                        // successful closure flag
  scarFlag: boolean;                      // unresolved contradiction flag
  timestamp: Date;                        // creation time
  processingTime: number;                 // milliseconds to complete
  
  // Phase 2: Compression & Memory
  coreGlyphs?: string[];                  // Ĉ: compressed essential glyphs
  cInfinity?: number;                     // C∞: final stable coherence
  digest?: string;                        // SHA-256 fingerprint
  reentryCount?: number;                  // how many times this pattern occurred
  memoryEchoFlag?: boolean;               // flagged as repeating pattern
  
  // Braid topology
  crossingRefs?: string[];                // IDs of loops this crosses with
  braidPosition?: {                       // 3D position in memory braid
    x: number;
    y: number; 
    z: number;
  };
  
  // Integration metadata
  metadata?: {
    createdByPersona?: string;            // Ghost that generated this loop
    scriptId?: string;                    // ELFIN++ script that triggered it
    conceptFootprint?: string[];          // Concepts touched by this loop
    phaseGateHits?: string[];            // Gates that were active during execution
    contradictionPeaks?: number[];       // Timestamps of Π(t) spikes
    coherenceGains?: Array<{             // Major coherence improvement points
      glyphIndex: number;
      deltaC: number;
    }>;
    [key: string]: any;
  };
}

interface LoopCrossing {
  id: string;
  glyph: string;
  loops: [string, string];
  type: 'paradox' | 'harmony';
  timestamp: Date;
  weight: number;
  order: number;
  position?: {
    x: number;
    y: number;
    z: number;
  };
  metadata?: {
    contradictionBefore?: number;
    contradictionAfter?: number;
    coherenceGain?: number;
    phaseAlignment?: number;
  };
}

interface BraidMemoryStats {
  totalLoops: number;
  closedLoops: number;
  scarredLoops: number;
  memoryEchoes: number;
  crossings: number;
  paradoxCrossings: number;
  harmonyCrossings: number;
  compressionRatio: number;
  topDigests: Array<{
    digest: string;
    count: number;
    lastSeen: Date;
  }>;
}

// Compression configuration - copied locally
const CompressionConfig = {
  COHERENCE_DELTA_THRESHOLD: 0.1,
  CONTRADICTION_DROP_THRESHOLD: 0.15,
  PHASE_GATE_BONUS: 0.3,
  GLYPH_TYPE_BONUSES: {
    'return': 0.5,
    'anchor': 0.4,
    'scar-repair': 0.6,
    'intent-bifurcation': 0.3,
    'meta-echo:reflect': 0.4
  },
  COHERENCE_WEIGHT: 1.0,
  CONTRADICTION_WEIGHT: 1.2,
  MINIMUM_CORE_WEIGHT: 0.2,
  REENTRY_MONITOR_THRESHOLD: 2,
  NOVELTY_INJECTION_THRESHOLD: 3,
  PERSONA_SWAP_THRESHOLD: 4,
  ECHO_COLLAPSE_THRESHOLD: 5
} as const;

type NoveltyGlyph = 'interruptor' | 'scar-sealer' | 'intent-bifurcation' | 'meta-echo:reflect' | 'phase-drift' | 'paradox-embrace' | 'coherence-boost' | 'memory-anchor';

// Helper functions - copied locally
function calculateLoopWeight(
  coherenceDelta: number,
  contradictionDelta: number, 
  phaseGateAlignment: number,
  glyphType: string
): number {
  const coherenceComponent = coherenceDelta * CompressionConfig.COHERENCE_WEIGHT;
  const contradictionComponent = -contradictionDelta * CompressionConfig.CONTRADICTION_WEIGHT;
  const phaseComponent = phaseGateAlignment * CompressionConfig.PHASE_GATE_BONUS;
  const typeBonus = CompressionConfig.GLYPH_TYPE_BONUSES[glyphType] || 0;
  
  return coherenceComponent + contradictionComponent + phaseComponent + typeBonus;
}

function classifyCrossingType(
  contradictionBefore: number,
  contradictionAfter: number,
  coherenceGain: number
): 'paradox' | 'harmony' {
  const contradictionIncrease = contradictionAfter - contradictionBefore;
  
  if (contradictionIncrease > 0.2 || coherenceGain < -0.1) {
    return 'paradox';
  }
  
  if (contradictionIncrease < -0.1 && coherenceGain > 0.1) {
    return 'harmony';
  }
  
  return 'harmony';
}

function selectNoveltyGlyph(
  reentryCount: number,
  contradictionLevel: number,
  coherenceLevel: number,
  scarCount: number
): NoveltyGlyph {
  if (contradictionLevel > 0.7) {
    return scarCount > 2 ? 'scar-sealer' : 'paradox-embrace';
  }
  
  if (coherenceLevel < 0.3) {
    return 'coherence-boost';
  }
  
  if (reentryCount >= CompressionConfig.ECHO_COLLAPSE_THRESHOLD) {
    return 'meta-echo:reflect';
  }
  
  if (reentryCount >= CompressionConfig.PERSONA_SWAP_THRESHOLD) {
    return 'intent-bifurcation';
  }
  
  return 'interruptor';
}

// BROWSER-COMPATIBLE CRYPTO FUNCTIONS
async function createBrowserHash(data: string): Promise<string> {
  if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
    // Use modern browser crypto API
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const hashBuffer = await window.crypto.subtle.digest('SHA-256', dataBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } else {
    // Fallback: Simple hash function for environments without crypto
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }
}

export class BraidMemory {
  private loopRegistry = new Map<string, LoopRecord>();
  private digestMap = new Map<string, string>();              // digest → loop ID
  private crossingRegistry = new Map<string, LoopCrossing>(); // crossing ID → crossing
  private crossingCounter = 0;
  private compressionCache = new Map<string, any>();          // loop ID → compressed data

  // Reentry tracking
  private digestSeenCount = new Map<string, number>();        // digest → seen count
  private reentryCallbacks: Array<(digest: string, count: number, loop: LoopRecord) => void> = [];

  constructor() {
    console.log('🧬 BraidMemory initialized for Phase 2 memory topology (browser-compatible crypto)');
    
    // Set up periodic maintenance
    this.setupMaintenance();
  }

  /**
   * Archive a completed loop with compression and digest computation
   */
  archiveLoop(loop: LoopRecord): string {
    const loopId = loop.id || `L${Date.now()}_${Math.random().toString(36).substr(2, 4)}`;
    loop.id = loopId;
    
    // Mark as closed if has return glyph
    loop.closed = !!loop.returnGlyph;
    
    // Compress the loop to extract core components
    const compression = this.compressLoop(loop);
    loop.coreGlyphs = compression.coreGlyphs;
    loop.cInfinity = compression.cInfinity;
    
    // Compute digest fingerprint (async, but we'll handle it)
    this.computeDigest(loop).then(digest => {
      loop.digest = digest;
      this.digestMap.set(digest, loopId);
      this.handleReentry(loop);
    });
    
    // Find and create crossings with existing loops
    this.detectCrossings(loop);
    
    // Store in registry
    this.loopRegistry.set(loopId, loop);
    
    console.log(`🧬 Archived loop ${loopId}: ${loop.coreGlyphs?.length || 0} core glyphs`);
    
    // Cache compressed data for performance
    this.compressionCache.set(loopId, compression);
    
    return loopId;
  }

  /**
   * Compress loop to core symbolic components using multi-signal heuristics
   */
  compressLoop(loop: LoopRecord): { coreGlyphs: string[], cInfinity: number, compressionRatio: number } {
    const coreGlyphs: string[] = [];
    const originalLength = loop.glyphPath.length;
    
    if (originalLength === 0) {
      return { coreGlyphs: [], cInfinity: 0, compressionRatio: 1 };
    }

    // Multi-signal compression analysis
    loop.glyphPath.forEach((glyph, i) => {
      // Calculate coherence differential (dC)
      const coherenceDelta = i > 0 && loop.coherenceTrace[i] && loop.coherenceTrace[i-1]
        ? loop.coherenceTrace[i] - loop.coherenceTrace[i-1]
        : 0;
      
      // Calculate contradiction differential (ΔΠ)  
      const contradictionDelta = i > 0 && loop.contradictionTrace[i] && loop.contradictionTrace[i-1]
        ? loop.contradictionTrace[i] - loop.contradictionTrace[i-1]
        : 0;
      
      // Check if executed in phase gate (phase gate alignment score)
      const phaseAlignment = this.calculatePhaseAlignment(loop, i);
      
      // Calculate total weight using multi-signal heuristic
      const weight = calculateLoopWeight(coherenceDelta, contradictionDelta, phaseAlignment, glyph);
      
      // Include glyph if weight exceeds threshold
      if (weight > CompressionConfig.MINIMUM_CORE_WEIGHT) {
        coreGlyphs.push(glyph);
        
        // Track coherence gains for metadata
        if (coherenceDelta > CompressionConfig.COHERENCE_DELTA_THRESHOLD) {
          if (!loop.metadata) loop.metadata = {};
          if (!loop.metadata.coherenceGains) loop.metadata.coherenceGains = [];
          loop.metadata.coherenceGains.push({
            glyphIndex: i,
            deltaC: coherenceDelta
          });
        }
      }
    });

    // Remove duplicates while preserving order
    const uniqueCoreGlyphs = [...new Set(coreGlyphs)];
    
    // Final stable coherence (C∞)
    const cInfinity = loop.coherenceTrace.length > 0 
      ? loop.coherenceTrace[loop.coherenceTrace.length - 1] 
      : 0;
    
    // Calculate compression ratio
    const compressionRatio = originalLength > 0 ? uniqueCoreGlyphs.length / originalLength : 1;
    
    console.log(`🗜️ Compressed loop: ${originalLength} → ${uniqueCoreGlyphs.length} glyphs (${(compressionRatio * 100).toFixed(1)}% retained)`);
    
    return { 
      coreGlyphs: uniqueCoreGlyphs, 
      cInfinity, 
      compressionRatio 
    };
  }

  /**
   * Calculate phase alignment score for a glyph at specific index
   */
  private calculatePhaseAlignment(loop: LoopRecord, glyphIndex: number): number {
    if (!loop.phaseTrace[glyphIndex] || !loop.metadata?.phaseGateHits) {
      return 0;
    }
    
    // Check if this glyph was executed during an active phase gate
    const phaseAtGlyph = loop.phaseTrace[glyphIndex];
    const gateHits = loop.metadata.phaseGateHits;
    
    // Higher score if executed in return gate or other important gates
    if (gateHits.includes('return')) return 1.0;
    if (gateHits.includes('memory')) return 0.8;
    if (gateHits.includes('feedback')) return 0.6;
    if (gateHits.includes('integration')) return 0.4;
    
    return 0;
  }

  /**
   * Compute SHA-256 digest fingerprint for loop identification (browser-compatible)
   */
  async computeDigest(loop: LoopRecord): Promise<string> {
    const digestData = {
      prompt: loop.prompt.trim(),
      coreGlyphs: loop.coreGlyphs || [],
      cInfinity: loop.cInfinity || 0,
      metadata: {
        createdByPersona: loop.metadata?.createdByPersona,
        conceptFootprint: loop.metadata?.conceptFootprint?.sort() // Sort for consistency
      }
    };
    
    const rawString = JSON.stringify(digestData);
    return await createBrowserHash(rawString);
  }

  /**
   * Handle reentry detection and adaptive responses
   */
  private handleReentry(loop: LoopRecord): void {
    if (!loop.digest) return;
    
    // Update seen count
    const currentCount = this.digestSeenCount.get(loop.digest) || 0;
    const newCount = currentCount + 1;
    this.digestSeenCount.set(loop.digest, newCount);
    
    loop.reentryCount = newCount;
    
    // Flag as memory echo if seen multiple times
    if (newCount >= CompressionConfig.REENTRY_MONITOR_THRESHOLD) {
      loop.memoryEchoFlag = true;
      console.log(`🔁 Memory echo detected: digest ${loop.digest.substring(0, 8)}... (count: ${newCount})`);
      
      // Trigger reentry callbacks
      this.reentryCallbacks.forEach(callback => {
        try {
          callback(loop.digest!, newCount, loop);
        } catch (error) {
          console.error('Error in reentry callback:', error);
        }
      });
    }
  }

  /**
   * Detect crossings with existing loops
   */
  private detectCrossings(newLoop: LoopRecord): void {
    if (!newLoop.glyphPath || newLoop.glyphPath.length === 0) return;
    
    for (const [existingLoopId, existingLoop] of this.loopRegistry) {
      if (!existingLoop.glyphPath) continue;
      
      // Find shared glyphs (potential crossings)
      const sharedGlyphs = newLoop.glyphPath.filter(glyph => 
        existingLoop.glyphPath.includes(glyph)
      );
      
      if (sharedGlyphs.length > 0) {
        sharedGlyphs.forEach(sharedGlyph => {
          this.createCrossing(newLoop, existingLoop, sharedGlyph);
        });
      }
    }
  }

  /**
   * Create a crossing between two loops
   */
  private createCrossing(loopA: LoopRecord, loopB: LoopRecord, sharedGlyph: string): void {
    const crossingId = `C${++this.crossingCounter}_${Date.now()}`;
    
    // Get contradiction levels before and after the shared glyph
    const contradictionBefore = this.getContradictionAtGlyph(loopA, sharedGlyph, 'before');
    const contradictionAfter = this.getContradictionAtGlyph(loopA, sharedGlyph, 'after');
    const coherenceGain = this.getCoherenceGainAtGlyph(loopA, sharedGlyph);
    
    // Classify crossing type
    const crossingType = classifyCrossingType(contradictionBefore, contradictionAfter, coherenceGain);
    
    // Calculate crossing weight
    const weight = (coherenceGain + Math.max(0, contradictionBefore - contradictionAfter)) / 2;
    
    const crossing: LoopCrossing = {
      id: crossingId,
      glyph: sharedGlyph,
      loops: [loopA.id, loopB.id],
      type: crossingType,
      timestamp: new Date(),
      weight,
      order: this.crossingCounter,
      metadata: {
        contradictionBefore,
        contradictionAfter,
        coherenceGain,
        phaseAlignment: this.getPhaseAtGlyph(loopA, sharedGlyph)
      }
    };
    
    this.crossingRegistry.set(crossingId, crossing);
    
    // Update loop references
    if (!loopA.crossingRefs) loopA.crossingRefs = [];
    if (!loopB.crossingRefs) loopB.crossingRefs = [];
    loopA.crossingRefs.push(crossingId);
    loopB.crossingRefs.push(crossingId);
    
    console.log(`🔀 Created ${crossingType} crossing: ${loopA.id} ↔ ${loopB.id} via "${sharedGlyph}"`);
  }

  /**
   * Helper methods for crossing analysis
   */
  private getContradictionAtGlyph(loop: LoopRecord, glyph: string, position: 'before' | 'after'): number {
    const glyphIndex = loop.glyphPath.indexOf(glyph);
    if (glyphIndex === -1 || !loop.contradictionTrace) return 0;
    
    const targetIndex = position === 'before' ? glyphIndex - 1 : glyphIndex + 1;
    return loop.contradictionTrace[targetIndex] || 0;
  }

  private getCoherenceGainAtGlyph(loop: LoopRecord, glyph: string): number {
    const glyphIndex = loop.glyphPath.indexOf(glyph);
    if (glyphIndex === -1 || !loop.coherenceTrace || glyphIndex === 0) return 0;
    
    return (loop.coherenceTrace[glyphIndex] || 0) - (loop.coherenceTrace[glyphIndex - 1] || 0);
  }

  private getPhaseAtGlyph(loop: LoopRecord, glyph: string): number {
    const glyphIndex = loop.glyphPath.indexOf(glyph);
    if (glyphIndex === -1 || !loop.phaseTrace) return 0;
    
    return loop.phaseTrace[glyphIndex] || 0;
  }

  /**
   * Find loop by digest fingerprint
   */
  findByDigest(digest: string): LoopRecord | undefined {
    const loopId = this.digestMap.get(digest);
    return loopId ? this.loopRegistry.get(loopId) : undefined;
  }

  /**
   * Get reentry count for a digest
   */
  getReentryCount(digest: string): number {
    return this.digestSeenCount.get(digest) || 0;
  }

  /**
   * Suggest novelty glyph for reentry optimization
   */
  suggestNoveltyGlyph(
    digest: string, 
    currentContradiction: number, 
    currentCoherence: number,
    scarCount: number
  ): NoveltyGlyph | null {
    const reentryCount = this.getReentryCount(digest);
    
    if (reentryCount < CompressionConfig.NOVELTY_INJECTION_THRESHOLD) {
      return null; // No novelty needed yet
    }
    
    return selectNoveltyGlyph(reentryCount, currentContradiction, currentCoherence, scarCount);
  }

  /**
   * Register callback for reentry events
   */
  onReentry(callback: (digest: string, count: number, loop: LoopRecord) => void): void {
    this.reentryCallbacks.push(callback);
  }

  /**
   * Get all loops by persona
   */
  getLoopsByPersona(persona: string): LoopRecord[] {
    return Array.from(this.loopRegistry.values()).filter(loop => 
      loop.metadata?.createdByPersona === persona
    );
  }

  /**
   * Get all crossings involving a specific loop
   */
  getCrossingsForLoop(loopId: string): LoopCrossing[] {
    return Array.from(this.crossingRegistry.values()).filter(crossing =>
      crossing.loops.includes(loopId)
    );
  }

  /**
   * Get memory statistics
   */
  getStats(): BraidMemoryStats {
    const loops = Array.from(this.loopRegistry.values());
    const crossings = Array.from(this.crossingRegistry.values());
    
    const totalLoops = loops.length;
    const closedLoops = loops.filter(l => l.closed).length;
    const scarredLoops = loops.filter(l => l.scarFlag).length;
    const memoryEchoes = loops.filter(l => l.memoryEchoFlag).length;
    
    const totalCrossings = crossings.length;
    const paradoxCrossings = crossings.filter(c => c.type === 'paradox').length;
    const harmonyCrossings = crossings.filter(c => c.type === 'harmony').length;
    
    // Calculate compression ratio
    const totalOriginalGlyphs = loops.reduce((sum, loop) => sum + loop.glyphPath.length, 0);
    const totalCoreGlyphs = loops.reduce((sum, loop) => sum + (loop.coreGlyphs?.length || 0), 0);
    const compressionRatio = totalOriginalGlyphs > 0 ? totalCoreGlyphs / totalOriginalGlyphs : 1;
    
    // Top digests by frequency
    const digestCounts = Array.from(this.digestSeenCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([digest, count]) => ({
        digest: digest.substring(0, 8) + '...',
        count,
        lastSeen: this.loopRegistry.get(this.digestMap.get(digest)!)?.timestamp || new Date()
      }));
    
    return {
      totalLoops,
      closedLoops,
      scarredLoops,
      memoryEchoes,
      crossings: totalCrossings,
      paradoxCrossings,
      harmonyCrossings,
      compressionRatio,
      topDigests: digestCounts
    };
  }

  /**
   * Export braid memory to JSON
   */
  exportBraid(): object {
    return {
      loops: Array.from(this.loopRegistry.entries()),
      digests: Array.from(this.digestMap.entries()),
      crossings: Array.from(this.crossingRegistry.entries()),
      reentryData: Array.from(this.digestSeenCount.entries()),
      exportTimestamp: new Date().toISOString(),
      version: '2.0'
    };
  }

  /**
   * Import braid memory from JSON
   */
  importBraid(data: any): boolean {
    try {
      // Clear existing data
      this.loopRegistry.clear();
      this.digestMap.clear();
      this.crossingRegistry.clear();
      this.digestSeenCount.clear();
      
      // Import data with deduplication
      if (data.loops) {
        data.loops.forEach(([id, loop]: [string, LoopRecord]) => {
          this.loopRegistry.set(id, loop);
        });
      }
      
      if (data.digests) {
        data.digests.forEach(([digest, id]: [string, string]) => {
          this.digestMap.set(digest, id);
        });
      }
      
      if (data.crossings) {
        data.crossings.forEach(([id, crossing]: [string, LoopCrossing]) => {
          this.crossingRegistry.set(id, crossing);
        });
      }
      
      if (data.reentryData) {
        data.reentryData.forEach(([digest, count]: [string, number]) => {
          this.digestSeenCount.set(digest, count);
        });
      }
      
      console.log('🧬 Braid memory imported successfully');
      return true;
    } catch (error) {
      console.error('Failed to import braid memory:', error);
      return false;
    }
  }

  /**
   * Set up periodic maintenance (LRU pruning, compression)
   */
  private setupMaintenance(): void {
    // Run maintenance every 5 minutes
    setInterval(() => {
      this.runMaintenance();
    }, 5 * 60 * 1000);
  }

  /**
   * Run maintenance tasks
   */
  private runMaintenance(): void {
    const stats = this.getStats();
    console.log(`🧬 Running braid memory maintenance (${stats.totalLoops} loops, ${stats.crossings} crossings)`);
    
    // LRU pruning for non-scarred, low-coherence loops
    this.pruneLowValueLoops();
    
    // Clean up orphaned crossings
    this.cleanupOrphanedCrossings();
    
    // Clear compression cache for old entries
    this.cleanCompressionCache();
  }

  /**
   * Prune low-value loops using LRU strategy
   */
  private pruneLowValueLoops(): void {
    const loops = Array.from(this.loopRegistry.values());
    const maxLoops = 1000; // Keep at most 1000 loops
    
    if (loops.length <= maxLoops) return;
    
    // Sort by value (keep high-coherence, recent, non-scarred loops)
    const sortedLoops = loops.sort((a, b) => {
      const aValue = this.calculateLoopValue(a);
      const bValue = this.calculateLoopValue(b);
      return bValue - aValue; // Descending order
    });
    
    // Remove lowest value loops
    const toRemove = sortedLoops.slice(maxLoops);
    toRemove.forEach(loop => {
      this.loopRegistry.delete(loop.id);
      if (loop.digest) {
        this.digestMap.delete(loop.digest);
      }
    });
    
    console.log(`🗑️ Pruned ${toRemove.length} low-value loops`);
  }

  /**
   * Calculate value score for loop pruning decisions
   */
  private calculateLoopValue(loop: LoopRecord): number {
    let value = 0;
    
    // High coherence is valuable
    value += (loop.cInfinity || 0) * 50;
    
    // Recent loops are more valuable
    const ageInDays = (Date.now() - loop.timestamp.getTime()) / (1000 * 60 * 60 * 24);
    value += Math.max(0, 30 - ageInDays);
    
    // Scarred loops are less valuable (but don't remove completely)
    if (loop.scarFlag) value -= 20;
    
    // Closed loops are more valuable
    if (loop.closed) value += 10;
    
    // Memory echoes might be valuable for pattern recognition
    if (loop.memoryEchoFlag) value += 5;
    
    // Loops with crossings are valuable for topology
    if (loop.crossingRefs && loop.crossingRefs.length > 0) value += loop.crossingRefs.length * 5;
    
    return value;
  }

  /**
   * Clean up crossings that reference deleted loops
   */
  private cleanupOrphanedCrossings(): void {
    const orphanedCrossings: string[] = [];
    
    for (const [crossingId, crossing] of this.crossingRegistry) {
      const loopAExists = this.loopRegistry.has(crossing.loops[0]);
      const loopBExists = this.loopRegistry.has(crossing.loops[1]);
      
      if (!loopAExists || !loopBExists) {
        orphanedCrossings.push(crossingId);
      }
    }
    
    orphanedCrossings.forEach(id => {
      this.crossingRegistry.delete(id);
    });
    
    if (orphanedCrossings.length > 0) {
      console.log(`🧹 Cleaned up ${orphanedCrossings.length} orphaned crossings`);
    }
  }

  /**
   * Clean old entries from compression cache
   */
  private cleanCompressionCache(): void {
    const maxCacheSize = 500;
    if (this.compressionCache.size <= maxCacheSize) return;
    
    // Simple LRU: clear half the cache
    const entries = Array.from(this.compressionCache.keys());
    const toDelete = entries.slice(0, Math.floor(entries.length / 2));
    
    toDelete.forEach(key => {
      this.compressionCache.delete(key);
    });
    
    console.log(`🧹 Cleaned compression cache: ${toDelete.length} entries removed`);
  }
}

// Singleton instance for global access
export const braidMemory = new BraidMemory();

console.log('🧬 BraidMemory system ready for Phase 2 memory topology (browser-compatible crypto)');
