// lib/elfin/interpreter.ts - ELFIN++ Scripting Engine Core
import { onUpload } from './scripts/onUpload';
import { onConceptChange } from './scripts/onConceptChange';
import { onGhostStateChange } from './scripts/onGhostStateChange';

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
    this.scripts['onUpload'] = onUpload;
    this.scripts['onConceptChange'] = onConceptChange;
    this.scripts['onGhostStateChange'] = onGhostStateChange;
    
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

console.log('🔮 ELFIN++ Interpreter ready for ghost orchestration');