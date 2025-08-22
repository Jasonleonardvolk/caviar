// lib/elfin/interpreter.ts - ELFIN++ Scripting Engine Core
import { onUpload } from './scripts/onUpload';
import { onConceptChange } from './scripts/onConceptChange';
import { onGhostStateChange } from './scripts/onGhostStateChange';

// ---- PACK B: minimal context types to satisfy ScriptFunction signature
// We keep these as 'unknown' for now; once the repo stabilizes we can import the real types.
type UploadContext = unknown;
type ConceptChangeContext = unknown;
type GhostStateChangeContext = unknown;

type ScriptContext = Record<string, any>;
type ScriptFunction = (context?: ScriptContext) => Promise<void> | void;

interface ScriptRegistry {
  [scriptName: string]: ScriptFunction;
}

class ElfinInterpreter {
  private scripts: ScriptRegistry = {};
  private isInitialized = false;

  constructor() {
    this.registerBuiltinScripts();
    this.isInitialized = true;
    console.log('🔮 ELFIN++ Scripting Engine Initialized');
  }

  /**
   * Register all built-in ELFIN++ scripts
   */
  private registerBuiltinScripts() {
    // Wrap concrete handlers to the generic signature expected by the interpreter.
    // Cast keeps the call-sites stable while we unify upstream types.
    this.scripts['onUpload'] = (ctx) => onUpload(ctx as UploadContext);
    this.scripts['onConceptChange'] = (ctx) => onConceptChange(ctx as ConceptChangeContext);
    this.scripts['onGhostStateChange'] = (ctx) => onGhostStateChange(ctx as GhostStateChangeContext);
    
    console.log('📜 ELFIN++ Scripts Registered:', Object.keys(this.scripts));
  }

  /**
   * Execute a named script with optional context
   */
  async run(scriptName: string, context: ScriptContext = {}): Promise<boolean> {
    if (!this.isInitialized) {
      console.warn('⚠️ ELFIN++ not initialized, deferring script execution');
      return false;
    }

    const script = this.scripts[scriptName];
    if (!script) {
      console.warn(`⚠️ ELFIN++ Script "${scriptName}" not found`);
      return false;
    }

    try {
      console.log(`🚀 ELFIN++ Executing script: ${scriptName}`, context);
      
      const startTime = performance.now();
      await script(context);
      const endTime = performance.now();
      
      console.log(`✅ ELFIN++ Script "${scriptName}" completed in ${(endTime - startTime).toFixed(2)}ms`);
      return true;
      
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error(`❌ ELFIN++ Script "${scriptName}" execution error:`, error);
      return false;
    }
  }

  /**
   * Register a custom script at runtime
   */
  registerScript(name: string, script: ScriptFunction) {
    this.scripts[name] = script;
    console.log(`📜 ELFIN++ Custom script registered: ${name}`);
  }

  /**
   * Public register method from the patch
   */
  public register(name: string, fn: (ctx: unknown) => Promise<unknown> | unknown) {
    this.scripts[name] = fn as ScriptFunction;
  }

  /**
   * Get list of available scripts
   */
  getAvailableScripts(): string[] {
    return Object.keys(this.scripts);
  }

  /**
   * Check if a script exists
   */
  hasScript(scriptName: string): boolean {
    return scriptName in this.scripts;
  }
}

// Export singleton interpreter instance
export const Elfin = new ElfinInterpreter();

// Export convenience function for running scripts
export const runElfinScript = (scriptName: string, context?: any) => {
  return Elfin.run(scriptName, context);
};

// Export the interpreter for pages that need it
export const globalElfinInterpreter = Elfin;

// Helper functions for common script patterns
export function triggerGhostEvent(ghostName: string, event: string, data: any = {}) {
  return Elfin.run('onGhostStateChange', { ghostName, event, data });
}

export function triggerConceptEvent(concepts: string[], action: string, metadata: any = {}) {
  return Elfin.run('onConceptChange', { concepts, action, metadata });
}

export function triggerUploadEvent(file: File) {
  return Elfin.run('onUpload', { file });
}

// Export the class for type usage
export { ElfinInterpreter };

console.log('🔮 ELFIN++ Interpreter ready for ghost orchestration');