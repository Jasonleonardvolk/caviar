// TORI Cognitive Systems Initialization
// This file ensures all cognitive systems boot properly in browser

import { elfinEngine } from '$lib/elfin/scriptEngine';
import { browserELFIN } from '$lib/browser-elfin';
import { conceptMesh } from '$lib/stores/conceptMesh';
import { ghostPersona } from '$lib/stores/ghostPersona';

/**
 * Initialize all TORI cognitive systems
 */
export function initializeTORI() {
  console.log('🧠 initializeTORI() called');
  
  try {
    // Try to use the main ELFIN engine first
    console.log('🧬 Attempting to use main ELFIN++ engine...');
    
    let activeEngine = null;
    
    try {
      const stats = elfinEngine.getExecutionStats();
      activeEngine = elfinEngine;
      console.log('✅ Main ELFIN++ engine working:', stats);
    } catch (mainError) {
      console.warn('⚠️ Main ELFIN++ engine failed:', mainError);
      
      // Fallback to browser engine
      console.log('🔄 Falling back to browser ELFIN++ engine...');
      if (browserELFIN) {
        activeEngine = browserELFIN;
        console.log('✅ Browser ELFIN++ engine available');
      } else {
        console.error('❌ Browser ELFIN++ engine also unavailable');
        return false;
      }
    }
    
    // Expose to global scope for debugging - FORCED
    if (typeof window !== 'undefined') {
      console.log('🌐 Exposing ELFIN++ to window object...');
      
      (window as any).ELFIN = activeEngine;
      (window as any).TORI = {
        elfin: activeEngine,
        conceptMesh,
        ghostPersona,
        testUpload: () => testELFIN(),
        checkStats: () => activeEngine?.getExecutionStats?.() || 'Engine not available',
        checkConcepts: () => localStorage.getItem('tori-concept-mesh'),
        checkDocs: () => localStorage.getItem('tori-scholarsphere-documents')
      };
      
      console.log('🌐 TORI debugging interface exposed:');
      console.log('  - window.ELFIN (ELFIN++ Engine)');
      console.log('  - window.TORI (Full debug interface)');
      console.log('  - window.TORI.testUpload() (Manual test)');
      console.log('  - window.TORI.checkStats() (Engine stats)');
      console.log('  - window.TORI.checkConcepts() (Concept mesh)');
      console.log('  - window.TORI.checkDocs() (Documents)');
      
      // Verify the assignment worked
      if ((window as any).ELFIN) {
        console.log('✅ window.ELFIN successfully assigned');
        
        // Test the interface immediately
        try {
          const testStats = (window as any).ELFIN.getExecutionStats();
          console.log('✅ window.ELFIN interface test successful:', testStats);
        } catch (testError) {
          console.error('❌ window.ELFIN interface test failed:', testError);
        }
      } else {
        console.error('❌ window.ELFIN assignment failed');
        return false;
      }
    } else {
      console.warn('⚠️ Window object not available (SSR context)');
    }
    
    return true;
  } catch (error) {
    console.error('❌ ELFIN++ initialization failed:', error);
    return false;
  }
}

/**
 * Test ELFIN++ functionality with manual trigger
 */
export function testELFIN() {
  if (typeof window === 'undefined') {
    console.warn('testELFIN: Window not available');
    return;
  }
  
  console.log('🧪 Testing ELFIN++ manual trigger...');
  
  // Dispatch test upload event
  window.dispatchEvent(new CustomEvent('tori:upload', {
    detail: {
      filename: 'TestDocument.pdf',
      text: 'Neural networks use backpropagation for learning. Machine learning algorithms process data to identify patterns. Artificial intelligence systems require large datasets for training.',
      concepts: ['Neural Networks', 'Backpropagation', 'Machine Learning', 'Artificial Intelligence'],
      timestamp: new Date(),
      source: 'manual_test'
    }
  }));
  
  console.log('🧪 Test upload event dispatched');
  
  // Check stats after a delay
  setTimeout(() => {
    try {
      if ((window as any).ELFIN && (window as any).ELFIN.getExecutionStats) {
        const stats = (window as any).ELFIN.getExecutionStats();
        console.log('🔬 ELFIN++ Test Results:', stats);
        console.log('🔬 Recent executions:', stats.recentExecutions);
      } else {
        console.error('❌ ELFIN++ not available for test results');
      }
    } catch (error) {
      console.error('❌ Failed to get test results:', error);
    }
  }, 2000);
}

/**
 * Force ELFIN++ assignment on window (emergency fallback)
 */
export function forceELFINAssignment() {
  if (typeof window !== 'undefined') {
    console.log('🔧 Emergency ELFIN++ assignment starting...');
    
    // Try browser engine first
    if (browserELFIN) {
      (window as any).ELFIN = browserELFIN;
      console.log('🔧 Emergency assignment: browserELFIN → window.ELFIN');
    } else {
      // Create minimal fallback engine
      const fallbackEngine = {
        getExecutionStats: () => ({
          totalScripts: 0,
          totalExecutions: 0,
          averageSuccessRate: 0,
          recentExecutions: [],
          topScripts: [],
          status: 'fallback_engine'
        }),
        triggerTestUpload: () => console.log('🧪 Fallback engine test triggered')
      };
      
      (window as any).ELFIN = fallbackEngine;
      console.log('🔧 Emergency assignment: fallback engine → window.ELFIN');
    }
    
    const success = (window as any).ELFIN !== undefined;
    console.log('🔧 Emergency ELFIN++ assignment result:', success);
    return success;
  }
  return false;
}
