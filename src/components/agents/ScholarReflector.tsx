/**
 * ScholarReflector.tsx - Knowledge discovery agent that links past ConceptDiffs
 * Activates on PDF context and suggests relevant connections
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HoloObject from '../ui/holoObject.jsx';

interface ScholarSuggestion {
  id: string;
  type: 'concept-link' | 'reference' | 'definition' | 'example' | 'contradiction';
  title: string;
  description: string;
  conceptIds: string[];
  confidence: number;
  sourceDocument?: string;
  createdAt: Date;
  wavelength?: number;
}

interface ScholarReflectorProps {
  isActive?: boolean;
  currentContext?: {
    documentId?: string;
    selectedText?: string;
    focusedConcept?: string;
  };
  onSuggestionClick?: (suggestion: ScholarSuggestion) => void;
  className?: string;
}

class ScholarEngine {
  private suggestions: Map<string, ScholarSuggestion> = new Map();
  private conceptHistory: Map<string, any[]> = new Map();
  private listeners: Array<(suggestions: ScholarSuggestion[]) => void> = [];

  constructor() {
    this.initializeScholar();
  }

  private initializeScholar() {
    // Listen for PDF focus events
    document.addEventListener('tori-pdf-focus', ((e: CustomEvent) => {
      this.analyzePDFContext(e.detail);
    }) as EventListener);

    // Listen for concept updates
    document.addEventListener('tori-concept-update', ((e: CustomEvent) => {
      this.trackConceptHistory(e.detail);
    }) as EventListener);

    // Listen for text selection in documents
    document.addEventListener('tori-text-selection', ((e: CustomEvent) => {
      this.analyzeTextSelection(e.detail);
    }) as EventListener);
  }

  private analyzePDFContext(pdfContext: {
    documentId: string;
    page: number;
    extractedConcepts: string[];
    selectedText?: string;
  }) {
    const { documentId, extractedConcepts, selectedText } = pdfContext;

    // Find related concepts from history
    extractedConcepts.forEach(conceptId => {
      const relatedConcepts = this.findRelatedConcepts(conceptId);
      
      relatedConcepts.forEach(related => {
        this.generateScholarSuggestion({
          type: 'concept-link',
          primaryConcept: conceptId,
          relatedConcept: related.conceptId,
          context: {
            documentId,
            selectedText,
            relationship: related.relationship,
            confidence: related.confidence
          }
        });
      });
    });

    // Check for contradictions with past learning
    this.checkForContradictions(extractedConcepts, documentId);

    // Suggest definitions for complex terms
    this.suggestDefinitions(extractedConcepts);
  }

  private findRelatedConcepts(conceptId: string): Array<{
    conceptId: string;
    relationship: string;
    confidence: number;
  }> {
    const related: Array<{
      conceptId: string;
      relationship: string;
      confidence: number;
    }> = [];

    // Search through concept history
    this.conceptHistory.forEach((history, historicalConceptId) => {
      if (historicalConceptId === conceptId) return;

      // Calculate relationship strength based on co-occurrence
      const coOccurrence = this.calculateCoOccurrence(conceptId, historicalConceptId);
      
      if (coOccurrence > 0.3) {
        related.push({
          conceptId: historicalConceptId,
          relationship: this.determineRelationshipType(conceptId, historicalConceptId),
          confidence: coOccurrence
        });
      }
    });

    return related.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
  }

  private calculateCoOccurrence(concept1: string, concept2: string): number {
    const history1 = this.conceptHistory.get(concept1) || [];
    const history2 = this.conceptHistory.get(concept2) || [];

    if (history1.length === 0 || history2.length === 0) return 0;

    // Find overlapping sessions/documents
    const sessions1 = new Set(history1.map(h => h.sessionId));
    const sessions2 = new Set(history2.map(h => h.sessionId));
    
    const intersection = [...sessions1].filter(s => sessions2.has(s));
    const union = new Set([...sessions1, ...sessions2]);

    return intersection.length / union.size;
  }

  private determineRelationshipType(concept1: string, concept2: string): string {
    // Simple heuristics - in production this would be more sophisticated
    const relationships = [
      'builds upon',
      'contrasts with',
      'exemplifies',
      'is prerequisite for',
      'is applied in',
      'relates to'
    ];

    // For now, return a random relationship
    // In production, this would analyze the semantic relationship
    return relationships[Math.floor(Math.random() * relationships.length)];
  }

  private checkForContradictions(newConcepts: string[], documentId: string) {
    newConcepts.forEach(conceptId => {
      const history = this.conceptHistory.get(conceptId) || [];
      
      // Look for potential contradictions in past learning
      const contradictoryEntries = history.filter(entry => 
        entry.metadata?.contradicts || entry.confidence < 0.3
      );

      if (contradictoryEntries.length > 0) {
        this.generateScholarSuggestion({
          type: 'contradiction',
          primaryConcept: conceptId,
          context: {
            documentId,
            contradictions: contradictoryEntries,
            confidence: 0.8
          }
        });
      }
    });
  }

  private suggestDefinitions(concepts: string[]) {
    // Suggest definitions for complex or new concepts
    concepts.forEach(conceptId => {
      if (!this.conceptHistory.has(conceptId)) {
        this.generateScholarSuggestion({
          type: 'definition',
          primaryConcept: conceptId,
          context: {
            confidence: 0.7,
            reason: 'new_concept'
          }
        });
      }
    });
  }

  private generateScholarSuggestion(params: {
    type: ScholarSuggestion['type'];
    primaryConcept: string;
    relatedConcept?: string;
    context: any;
  }) {
    const { type, primaryConcept, relatedConcept, context } = params;

    let title = '';
    let description = '';
    let wavelength = 520; // Default green for knowledge

    switch (type) {
      case 'concept-link':
        title = `Connect "${primaryConcept}" with "${relatedConcept}"`;
        description = `These concepts ${context.relationship}. Confidence: ${Math.round(context.confidence * 100)}%`;
        wavelength = 480; // Blue for connections
        break;
      
      case 'contradiction':
        title = `Potential contradiction in "${primaryConcept}"`;
        description = `This concept conflicts with previous learning in ${context.contradictions.length} instance(s)`;
        wavelength = 620; // Orange for warnings
        break;
      
      case 'definition':
        title = `Define "${primaryConcept}"`;
        description = `This appears to be a new concept. Would you like me to find a definition?`;
        wavelength = 450; // Violet for new knowledge
        break;
      
      case 'reference':
        title = `Reference for "${primaryConcept}"`;
        description = `Found related information in previous documents`;
        wavelength = 520; // Green for references
        break;
      
      case 'example':
        title = `Example of "${primaryConcept}"`;
        description = `This concept has been used in similar contexts before`;
        wavelength = 550; // Yellow-green for examples
        break;
    }

    const suggestion: ScholarSuggestion = {
      id: `scholar_${type}_${Date.now()}_${Math.random()}`,
      type,
      title,
      description,
      conceptIds: [primaryConcept, ...(relatedConcept ? [relatedConcept] : [])],
      confidence: context.confidence || 0.7,
      sourceDocument: context.documentId,
      createdAt: new Date(),
      wavelength
    };

    this.suggestions.set(suggestion.id, suggestion);
    this.notifyListeners();

    // Emit event for UI
    document.dispatchEvent(new CustomEvent('tori-scholar-suggestion', {
      detail: suggestion
    }));
  }

  private trackConceptHistory(conceptUpdate: any) {
    const { conceptId, action, metadata, sessionId } = conceptUpdate;
    
    if (!this.conceptHistory.has(conceptId)) {
      this.conceptHistory.set(conceptId, []);
    }

    this.conceptHistory.get(conceptId)!.push({
      action,
      metadata,
      sessionId: sessionId || 'default',
      timestamp: new Date(),
      confidence: metadata?.confidence || 1.0
    });
  }

  private analyzeTextSelection(selectionData: {
    text: string;
    documentId: string;
    position: { start: number; end: number };
  }) {
    const { text, documentId } = selectionData;

    // Extract potential concepts from selected text
    const extractedConcepts = this.extractConceptsFromText(text);
    
    extractedConcepts.forEach(conceptId => {
      this.analyzePDFContext({
        documentId,
        page: 0,
        extractedConcepts: [conceptId],
        selectedText: text
      });
    });
  }

  private extractConceptsFromText(text: string): string[] {
    // Simplified concept extraction
    // In production, this would use NLP techniques
    const words = text.toLowerCase().split(/\s+/);
    return words.filter(word => word.length > 4); // Simple heuristic
  }

  private notifyListeners() {
    const currentSuggestions = Array.from(this.suggestions.values());
    this.listeners.forEach(listener => listener(currentSuggestions));
  }

  // Public API
  getSuggestions(): ScholarSuggestion[] {
    return Array.from(this.suggestions.values())
      .sort((a, b) => b.confidence - a.confidence);
  }

  dismissSuggestion(suggestionId: string) {
    this.suggestions.delete(suggestionId);
    this.notifyListeners();
  }

  addListener(listener: (suggestions: ScholarSuggestion[]) => void) {
    this.listeners.push(listener);
  }

  removeListener(listener: (suggestions: ScholarSuggestion[]) => void) {
    const index = this.listeners.indexOf(listener);
    if (index > -1) {
      this.listeners.splice(index, 1);
    }
  }
}

// Singleton instance
const scholarEngine = new ScholarEngine();

const ScholarReflector: React.FC<ScholarReflectorProps> = ({
  isActive = true,
  currentContext,
  onSuggestionClick,
  className = ''
}) => {
  const [suggestions, setSuggestions] = useState<ScholarSuggestion[]>([]);
  const [expandedSuggestion, setExpandedSuggestion] = useState<string | null>(null);

  useEffect(() => {
    if (!isActive) return;

    const handleSuggestions = (newSuggestions: ScholarSuggestion[]) => {
      setSuggestions(newSuggestions);
    };

    scholarEngine.addListener(handleSuggestions);
    setSuggestions(scholarEngine.getSuggestions());

    return () => {
      scholarEngine.removeListener(handleSuggestions);
    };
  }, [isActive]);

  // Filter suggestions based on current context
  const contextualSuggestions = suggestions.filter(suggestion => {
    if (!currentContext) return true;
    
    if (currentContext.documentId && suggestion.sourceDocument !== currentContext.documentId) {
      return false;
    }
    
    if (currentContext.focusedConcept && 
        !suggestion.conceptIds.includes(currentContext.focusedConcept)) {
      return false;
    }
    
    return true;
  });

  const handleSuggestionClick = (suggestion: ScholarSuggestion) => {
    if (onSuggestionClick) {
      onSuggestionClick(suggestion);
    }
    
    setExpandedSuggestion(
      expandedSuggestion === suggestion.id ? null : suggestion.id
    );
  };

  const handleDismiss = (e: React.MouseEvent, suggestionId: string) => {
    e.stopPropagation();
    scholarEngine.dismissSuggestion(suggestionId);
  };

  if (!isActive || contextualSuggestions.length === 0) {
    return null;
  }

  return (
    <div className={`tori-scholar-reflector ${className}`}>
      <div className="mb-4 flex items-center space-x-2">
        <HoloObject
          conceptId="scholar-agent"
          wavelength={520}
          intensity={0.8}
          radius={16}
        >
          <span className="text-sm font-medium text-green-600 dark:text-green-400">
            📖 Scholar
          </span>
        </HoloObject>
        <span className="text-xs text-slate-500">
          {contextualSuggestions.length} insight{contextualSuggestions.length !== 1 ? 's' : ''}
        </span>
      </div>

      <AnimatePresence>
        <div className="space-y-2">
          {contextualSuggestions.slice(0, 5).map(suggestion => (
            <motion.div
              key={suggestion.id}
              layout
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="group"
            >
              <div
                className={`
                  tori-panel p-3 cursor-pointer transition-all duration-200
                  hover:shadow-md hover:scale-[1.02]
                  ${expandedSuggestion === suggestion.id ? 'ring-2 ring-green-500/30' : ''}
                `}
                onClick={() => handleSuggestionClick(suggestion)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <HoloObject
                        conceptId={suggestion.id}
                        wavelength={suggestion.wavelength}
                        intensity={0.6}
                        radius={8}
                      >
                        <div className="w-2 h-2 rounded-full"></div>
                      </HoloObject>
                      
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {suggestion.title}
                      </span>
                      
                      <span className={`
                        text-xs px-2 py-0.5 rounded-full
                        ${suggestion.type === 'contradiction' 
                          ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
                          : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        }
                      `}>
                        {suggestion.type}
                      </span>
                    </div>
                    
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                      {suggestion.description}
                    </p>

                    <AnimatePresence>
                      {expandedSuggestion === suggestion.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700"
                        >
                          <div className="text-xs text-slate-500 space-y-1">
                            <div>Concepts: {suggestion.conceptIds.join(', ')}</div>
                            <div>Confidence: {Math.round(suggestion.confidence * 100)}%</div>
                            <div>Created: {suggestion.createdAt.toLocaleTimeString()}</div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <button
                    onClick={(e) => handleDismiss(e, suggestion.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                    title="Dismiss suggestion"
                  >
                    ×
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </AnimatePresence>
    </div>
  );
};

export default ScholarReflector;
export { scholarEngine };
export type { ScholarSuggestion };