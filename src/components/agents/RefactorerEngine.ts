/**
 * RefactorerEngine.ts - Code improvement agent with actionable suggestions
 * Not just badges - autonomous overlays with click hooks
 */

interface RefactorerSuggestion {
  id: string;
  type: 'simplify' | 'optimize' | 'extract' | 'rename' | 'pattern';
  message: string;
  codeLocation: {
    file: string;
    line: number;
    column: number;
  };
  confidence: number;
  appliedDiff?: string;
  conceptDiffs?: string[];
}

interface RefactorerConfig {
  enabled: boolean;
  autoSuggest: boolean;
  confidence_threshold: number;
  max_suggestions_per_session: number;
}

class RefactorerEngine {
  private suggestions: Map<string, RefactorerSuggestion> = new Map();
  private config: RefactorerConfig = {
    enabled: true,
    autoSuggest: true,
    confidence_threshold: 0.7,
    max_suggestions_per_session: 10
  };
  private listeners: Array<(suggestion: RefactorerSuggestion) => void> = [];

  constructor() {
    this.initializeRefactorer();
  }

  private initializeRefactorer() {
    // Listen for code changes in the IDE
    this.setupCodeWatchers();
    
    // Hook into concept diff updates
    this.setupConceptDiffHooks();
  }

  private setupCodeWatchers() {
    // This would integrate with Monaco editor or VS Code API
    // For now, simulate with document listeners
    document.addEventListener('tori-code-change', ((e: CustomEvent) => {
      this.analyzeCodeChange(e.detail);
    }) as EventListener);
  }

  private setupConceptDiffHooks() {
    // Listen for concept updates that might suggest refactoring
    document.addEventListener('tori-concept-update', ((e: CustomEvent) => {
      this.analyzeConceptForRefactoring(e.detail);
    }) as EventListener);
  }

  // Analyze code changes for refactoring opportunities
  analyzeCodeChange(codeChange: {
    file: string;
    content: string;
    line: number;
    column: number;
  }) {
    if (!this.config.enabled || !this.config.autoSuggest) return;

    const suggestions = this.detectRefactoringOpportunities(codeChange);
    
    suggestions.forEach(suggestion => {
      if (suggestion.confidence >= this.config.confidence_threshold) {
        this.addSuggestion(suggestion);
      }
    });
  }

  private detectRefactoringOpportunities(codeChange: any): RefactorerSuggestion[] {
    const suggestions: RefactorerSuggestion[] = [];
    const { content, file, line, column } = codeChange;

    // Pattern detection (simplified - in production this would use AST analysis)
    
    // 1. Long function detection
    if (content.split('\n').length > 20) {
      suggestions.push({
        id: `extract_${Date.now()}`,
        type: 'extract',
        message: 'This function is quite long. Consider extracting smaller functions.',
        codeLocation: { file, line, column },
        confidence: 0.8,
        conceptDiffs: ['function-extraction', 'code-organization']
      });
    }

    // 2. Repeated code patterns
    const lines = content.split('\n');
    const duplicatePatterns = this.findDuplicatePatterns(lines);
    duplicatePatterns.forEach(pattern => {
      suggestions.push({
        id: `pattern_${Date.now()}_${Math.random()}`,
        type: 'pattern',
        message: `Repeated pattern detected: "${pattern.text}". Consider extracting to a helper function.`,
        codeLocation: { file, line: pattern.firstLine, column: 0 },
        confidence: 0.75,
        conceptDiffs: ['code-deduplication', 'helper-functions']
      });
    });

    // 3. Complex conditional logic
    if (content.includes('if') && content.split('if').length > 3) {
      suggestions.push({
        id: `simplify_${Date.now()}`,
        type: 'simplify',
        message: 'Complex conditional logic detected. Consider using early returns or switch statements.',
        codeLocation: { file, line, column },
        confidence: 0.7,
        conceptDiffs: ['conditional-simplification', 'readability']
      });
    }

    // 4. Variable naming suggestions
    const shortVariables = content.match(/\b[a-z]{1,2}\b/g);
    if (shortVariables && shortVariables.length > 2) {
      suggestions.push({
        id: `rename_${Date.now()}`,
        type: 'rename',
        message: 'Consider using more descriptive variable names for better readability.',
        codeLocation: { file, line, column },
        confidence: 0.6,
        conceptDiffs: ['naming-conventions', 'readability']
      });
    }

    return suggestions;
  }

  private findDuplicatePatterns(lines: string[]): Array<{ text: string; firstLine: number }> {
    const patterns: Array<{ text: string; firstLine: number }> = [];
    const seenLines = new Map<string, number>();

    lines.forEach((line, index) => {
      const trimmed = line.trim();
      if (trimmed.length > 10) { // Only check substantial lines
        if (seenLines.has(trimmed)) {
          patterns.push({
            text: trimmed.slice(0, 50) + (trimmed.length > 50 ? '...' : ''),
            firstLine: seenLines.get(trimmed)! + 1
          });
        } else {
          seenLines.set(trimmed, index);
        }
      }
    });

    return patterns;
  }

  private analyzeConceptForRefactoring(conceptUpdate: any) {
    // If certain concepts are frequently accessed, suggest refactoring
    const { conceptId, frequency, context } = conceptUpdate;
    
    if (frequency > 10 && context === 'code-review') {
      this.addSuggestion({
        id: `concept_${conceptId}_${Date.now()}`,
        type: 'optimize',
        message: `The concept "${conceptId}" is heavily used. Consider creating dedicated utilities.`,
        codeLocation: { file: 'current', line: 0, column: 0 },
        confidence: 0.8,
        conceptDiffs: [conceptId, 'utility-creation']
      });
    }
  }

  // Add a new suggestion and notify listeners
  addSuggestion(suggestion: RefactorerSuggestion) {
    if (this.suggestions.size >= this.config.max_suggestions_per_session) {
      // Remove oldest suggestion
      const oldestKey = this.suggestions.keys().next().value;
      this.suggestions.delete(oldestKey);
    }

    this.suggestions.set(suggestion.id, suggestion);
    
    // Notify listeners (UI components)
    this.listeners.forEach(listener => listener(suggestion));
    
    // Emit event for UI
    document.dispatchEvent(new CustomEvent('tori-refactorer-suggestion', {
      detail: suggestion
    }));
  }

  // Apply a refactoring suggestion
  async applySuggestion(suggestionId: string): Promise<boolean> {
    const suggestion = this.suggestions.get(suggestionId);
    if (!suggestion) return false;

    try {
      // In production, this would interact with the code editor
      const applied = await this.performRefactoring(suggestion);
      
      if (applied) {
        suggestion.appliedDiff = `Applied ${suggestion.type} refactoring`;
        
        // Update concept diffs
        if (suggestion.conceptDiffs) {
          document.dispatchEvent(new CustomEvent('tori-concept-update', {
            detail: {
              conceptIds: suggestion.conceptDiffs,
              action: 'refactoring-applied',
              metadata: { suggestionId, type: suggestion.type }
            }
          }));
        }
        
        // Remove from active suggestions
        this.suggestions.delete(suggestionId);
        return true;
      }
    } catch (error) {
      console.error('Failed to apply refactoring:', error);
    }
    
    return false;
  }

  private async performRefactoring(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Placeholder for actual refactoring logic
    // In production, this would interface with language servers, AST manipulation, etc.
    
    switch (suggestion.type) {
      case 'extract':
        return this.performExtraction(suggestion);
      case 'simplify':
        return this.performSimplification(suggestion);
      case 'optimize':
        return this.performOptimization(suggestion);
      case 'rename':
        return this.performRenaming(suggestion);
      case 'pattern':
        return this.performPatternExtraction(suggestion);
      default:
        return false;
    }
  }

  private async performExtraction(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Simulate function extraction
    console.log(`Extracting function at ${suggestion.codeLocation.file}:${suggestion.codeLocation.line}`);
    return true;
  }

  private async performSimplification(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Simulate code simplification
    console.log(`Simplifying code at ${suggestion.codeLocation.file}:${suggestion.codeLocation.line}`);
    return true;
  }

  private async performOptimization(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Simulate optimization
    console.log(`Optimizing code at ${suggestion.codeLocation.file}:${suggestion.codeLocation.line}`);
    return true;
  }

  private async performRenaming(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Simulate variable renaming
    console.log(`Renaming variables at ${suggestion.codeLocation.file}:${suggestion.codeLocation.line}`);
    return true;
  }

  private async performPatternExtraction(suggestion: RefactorerSuggestion): Promise<boolean> {
    // Simulate pattern extraction
    console.log(`Extracting pattern at ${suggestion.codeLocation.file}:${suggestion.codeLocation.line}`);
    return true;
  }

  // Public API methods
  getSuggestions(): RefactorerSuggestion[] {
    return Array.from(this.suggestions.values());
  }

  dismissSuggestion(suggestionId: string) {
    this.suggestions.delete(suggestionId);
    
    document.dispatchEvent(new CustomEvent('tori-refactorer-dismissed', {
      detail: { suggestionId }
    }));
  }

  addListener(listener: (suggestion: RefactorerSuggestion) => void) {
    this.listeners.push(listener);
  }

  removeListener(listener: (suggestion: RefactorerSuggestion) => void) {
    const index = this.listeners.indexOf(listener);
    if (index > -1) {
      this.listeners.splice(index, 1);
    }
  }

  updateConfig(newConfig: Partial<RefactorerConfig>) {
    this.config = { ...this.config, ...newConfig };
  }

  getConfig(): RefactorerConfig {
    return { ...this.config };
  }
}

// Singleton instance
export const refactorerEngine = new RefactorerEngine();
export default RefactorerEngine;