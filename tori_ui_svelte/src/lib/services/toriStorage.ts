type StorageConfig = any;
/**
 * TORI Storage Manager - Simple localStorage Implementation
 * =========================================================
 * 
 * Simplified storage that avoids IndexedDB issues
 * Uses localStorage with automatic fallback to memory
 */

import { browser } from '$app/environment';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  concepts?: string[];
  processingMethod?: string;
  confidence?: number;
  attachments?: any[];
  memoryContext?: any;
  typingMetrics?: TypingMetrics;
  intentContext?: IntentContext;
}

export interface TypingMetrics {
  wpm: number;
  pausePatterns: Array<{ duration: number; position: number }>;
  deleteRatio: number;
  burstiness: number;
  totalKeystrokes: number;
  typingDuration: number;
}

export interface IntentContext {
  primaryIntent: string;
  confidence: number;
  alternatives: string[];
  naturalLanguagePattern: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  persona?: string;
  metadata?: any;
}

export interface Document {
  id: string;
  name: string;
  content: string;
  type: string;
  size: number;
  uploadedAt: Date;
  concepts?: string[];
  processingStatus: 'pending' | 'processing' | 'completed' | 'error';
  error?: string;
}

export interface ConceptNode {
  id: string;
  label: string;
  type: string;
  weight: number;
  connections: string[];
  metadata?: any;
}

export interface Memory {
  id: string;
  type: 'episodic' | 'semantic' | 'procedural';
  content: any;
  timestamp: Date;
  relevance: number;
  associations: string[];
}

class ToriStorageManager {
  private prefix = 'tori_';
  private memoryStore = new Map<string, any>();
  private useMemoryOnly = false;

  constructor() {
    if (browser) {
      this.initialize();
    }
  }

  private initialize() {
    console.log('🗄️ Initializing TORI Storage (localStorage mode)...');
    
    try {
      // Test localStorage
      const testKey = this.prefix + 'test';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      console.log('✅ localStorage available');
    } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
      console.warn('⚠️ localStorage not available, using memory storage');
      this.useMemoryOnly = true;
    
}
    
    console.log('✅ TORI Storage ready');
  }

  // Helper methods for storage
  private getKey(store: string, id?: string): string {
    return id ? `${this.prefix}${store}_${id}` : `${this.prefix}${store}`;
  }

  private save(key: string, data: any): void {
    const serialized = JSON.stringify(data);
    
    if (this.useMemoryOnly) {
      this.memoryStore.set(key, serialized);
    } else {
      try {
        localStorage.setItem(key, serialized);
      } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
        console.warn('localStorage full, using memory:', e);
        this.memoryStore.set(key, serialized);
      
}
    }
  }

  private load(key: string): any {
    try {
      const data = this.useMemoryOnly 
        ? this.memoryStore.get(key)
        : localStorage.getItem(key);
      
      return data ? JSON.parse(data) : null;
    } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
      console.error('Error loading data:', e);
      return null;
    
}
  }

  private delete(key: string): void {
    if (this.useMemoryOnly) {
      this.memoryStore.delete(key);
    } else {
      localStorage.removeItem(key);
    }
  }

  private list(prefix: string): string[] {
    if (this.useMemoryOnly) {
      return Array.from(this.memoryStore.keys())
        .filter(k => k.startsWith(prefix));
    } else {
      return Object.keys(localStorage)
        .filter(k => k.startsWith(prefix));
    }
  }

  // Conversation methods
  async saveConversation(conversation: Conversation): Promise<void> {
    const key = this.getKey('conversation', conversation.id);
    this.save(key, conversation);
  }

  async getConversation(id: string): Promise<Conversation | null> {
    const key = this.getKey('conversation', id);
    return this.load(key);
  }

  async getAllConversations(): Promise<Conversation[]> {
    const keys = this.list(this.prefix + 'conversation_');
    const conversations: Conversation[] = [];
    
    for (const key of keys) {
      const conv = this.load(key);
      if (conv) conversations.push(conv);
    }
    
    return conversations.sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }

  async deleteConversation(id: string): Promise<void> {
    const key = this.getKey('conversation', id);
    this.delete(key);
  }

  // Document methods
  async saveDocument(document: Document): Promise<void> {
    const key = this.getKey('document', document.id);
    this.save(key, document);
  }

  async getDocument(id: string): Promise<Document | null> {
    const key = this.getKey('document', id);
    return this.load(key);
  }

  async getAllDocuments(): Promise<Document[]> {
    const keys = this.list(this.prefix + 'document_');
    const documents: Document[] = [];
    
    for (const key of keys) {
      const doc = this.load(key);
      if (doc) documents.push(doc);
    }
    
    return documents.sort((a, b) => 
      new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
    );
  }

  async deleteDocument(id: string): Promise<void> {
    const key = this.getKey('document', id);
    this.delete(key);
  }

  // Concept methods
  async saveConcept(concept: ConceptNode): Promise<void> {
    const key = this.getKey('concept', concept.id);
    this.save(key, concept);
  }

  async getConcept(id: string): Promise<ConceptNode | null> {
    const key = this.getKey('concept', id);
    return this.load(key);
  }

  async getAllConcepts(): Promise<ConceptNode[]> {
    const keys = this.list(this.prefix + 'concept_');
    const concepts: ConceptNode[] = [];
    
    for (const key of keys) {
      const concept = this.load(key);
      if (concept) concepts.push(concept);
    }
    
    return concepts;
  }

  // Memory methods
  async saveMemory(memory: Memory): Promise<void> {
    const key = this.getKey('memory', memory.id);
    this.save(key, memory);
  }

  async getMemory(id: string): Promise<Memory | null> {
    const key = this.getKey('memory', id);
    return this.load(key);
  }

  async getAllMemories(): Promise<Memory[]> {
    const keys = this.list(this.prefix + 'memory_');
    const memories: Memory[] = [];
    
    for (const key of keys) {
      const memory = this.load(key);
      if (memory) memories.push(memory);
    }
    
    return memories.sort((a, b) => b.relevance - a.relevance);
  }

  // Preferences methods
  async savePreference(key: string, value: any): Promise<void> {
    const prefKey = this.getKey('pref', key);
    this.save(prefKey, value);
  }

  async getPreference(key: string): Promise<any> {
    const prefKey = this.getKey('pref', key);
    return this.load(prefKey);
  }

  async getAllPreferences(): Promise<Record<string, any>> {
    const keys = this.list(this.prefix + 'pref_');
    const preferences: Record<string, any> = {};
    
    for (const key of keys) {
      const prefKey = key.replace(this.prefix + 'pref_', '');
      preferences[prefKey] = this.load(key);
    }
    
    return preferences;
  }

  // Utility methods
  async clearAll(): Promise<void> {
    console.log('🧹 Clearing all TORI storage...');
    
    if (this.useMemoryOnly) {
      this.memoryStore.clear();
    } else {
      const keys = Object.keys(localStorage)
        .filter(k => k.startsWith(this.prefix));
      
      for (const key of keys) {
        localStorage.removeItem(key);
      }
    }
    
    console.log('✅ Storage cleared');
  }

  async exportData(): Promise<any> {
    const data = {
      conversations: await this.getAllConversations(),
      documents: await this.getAllDocuments(),
      concepts: await this.getAllConcepts(),
      memories: await this.getAllMemories(),
      preferences: await this.getAllPreferences(),
      exportedAt: new Date().toISOString()
    };
    
    return data;
  }

  async importData(data: any): Promise<void> {
    // Clear existing data first
    await this.clearAll();
    
    // Import conversations
    if (data.conversations) {
      for (const conv of data.conversations) {
        await this.saveConversation(conv);
      }
    }
    
    // Import documents
    if (data.documents) {
      for (const doc of data.documents) {
        await this.saveDocument(doc);
      }
    }
    
    // Import concepts
    if (data.concepts) {
      for (const concept of data.concepts) {
        await this.saveConcept(concept);
      }
    }
    
    // Import memories
    if (data.memories) {
      for (const memory of data.memories) {
        await this.saveMemory(memory);
      }
    }
    
    // Import preferences
    if (data.preferences) {
      for (const [key, value] of Object.entries(data.preferences)) {
        await this.savePreference(key, value);
      }
    }
    
    console.log('✅ Data imported successfully');
  }
}

// Export singleton instance
export const toriStorage = new ToriStorageManager();

// Export types
export type { StorageConfig };
