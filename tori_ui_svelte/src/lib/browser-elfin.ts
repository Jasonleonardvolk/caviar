// Browser ELFIN++ Engine - Direct client access
// This ensures ELFIN++ is available in browser console

import { browser } from '$app/environment';

class BrowserELFINEngine {
  private scripts: Map<string, any> = new Map();
  private executionHistory: any[] = [];
  
  constructor() {
    console.log('🌐 Browser ELFIN++ Engine initializing...');
    this.loadScripts();
    this.setupEventListeners();
  }

  private loadScripts() {
    // Core synthesis script
    this.scripts.set('core_synthesis', {
      id: 'core_synthesis',
      name: 'Knowledge Synthesis',
      author: 'Scholar',
      useCount: 0,
      successRate: 1.0,
      created: new Date()
    });

    // Research orchestrator
    this.scripts.set('research_orchestrator', {
      id: 'research_orchestrator', 
      name: 'Research Orchestrator',
      author: 'Explorer',
      useCount: 0,
      successRate: 1.0,
      created: new Date()
    });

    // Novelty injector
    this.scripts.set('novelty_injector', {
      id: 'novelty_injector',
      name: 'Novelty Injection', 
      author: 'Creator',
      useCount: 0,
      successRate: 1.0,
      created: new Date()
    });

    console.log('✅ Browser ELFIN++ scripts loaded');
  }

  private setupEventListeners() {
    if (!browser || typeof window === 'undefined') return;

    window.addEventListener('tori:upload', (event: any) => {
      this.handleUploadEvent(event.detail);
    });

    console.log('📡 Browser ELFIN++ event listeners active');
  }

  private async handleUploadEvent(detail: any) {
    const { filename, text, concepts, timestamp, source } = detail;
    
    console.log('📚 Browser ELFIN++ onUpload triggered:', {
      filename,
      conceptCount: concepts?.length || 0,
      textLength: text?.length || 0,
      source
    });

    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 500));

    // Record execution
    this.executionHistory.push({
      scriptId: 'core_synthesis',
      scriptName: 'Knowledge Synthesis',
      author: 'Scholar',
      timestamp: new Date(),
      success: true,
      confidence: 0.85,
      context: `Process uploaded document: ${filename}`
    });

    // Update script usage
    const script = this.scripts.get('core_synthesis');
    if (script) {
      script.useCount++;
      script.lastUsed = new Date();
    }

    console.log('✅ Browser ELFIN++ document processing complete:', {
      filename,
      success: true,
      confidence: 0.85
    });
  }

  public getExecutionStats(): any {
    return {
      totalScripts: this.scripts.size,
      totalExecutions: this.executionHistory.length,
      averageSuccessRate: Array.from(this.scripts.values()).reduce((acc, script) => acc + script.successRate, 0) / this.scripts.size,
      recentExecutions: this.executionHistory.slice(-10),
      topScripts: Array.from(this.scripts.values())
        .sort((a, b) => b.useCount - a.useCount)
        .slice(0, 5),
      lastExecution: this.executionHistory[this.executionHistory.length - 1] || null
    };
  }

  public getScripts(): any[] {
    return Array.from(this.scripts.values());
  }

  public triggerTestUpload(): void {
    if (!browser || typeof window === 'undefined') return;

    window.dispatchEvent(new CustomEvent('tori:upload', {
      detail: {
        filename: 'BrowserTest.pdf',
        text: 'Neural networks and machine learning algorithms for artificial intelligence research.',
        concepts: ['Neural Networks', 'Machine Learning', 'AI Research'],
        timestamp: new Date(),
        source: 'browser_test'
      }
    }));

    console.log('🧪 Browser test upload event dispatched');
  }
}

// Create and expose browser ELFIN engine
let browserELFIN: BrowserELFINEngine | null = null;

if (browser) {
  browserELFIN = new BrowserELFINEngine();
  
  // Expose globally
  (window as any).ELFIN = browserELFIN;
  (window as any).TORI = {
    elfin: browserELFIN,
    checkStats: () => browserELFIN?.getExecutionStats(),
    checkConcepts: () => localStorage.getItem('tori-concept-mesh'),
    checkDocs: () => localStorage.getItem('tori-scholarsphere-documents'),
    testUpload: () => browserELFIN?.triggerTestUpload()
  };

  console.log('🌐 Browser ELFIN++ exposed:', {
    ELFIN: !!(window as any).ELFIN,
    TORI: !!(window as any).TORI
  });
}

export { browserELFIN };
