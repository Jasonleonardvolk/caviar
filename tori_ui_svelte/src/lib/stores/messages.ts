import { writable } from 'svelte/store';
import { browser } from '$app/environment';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  streaming?: boolean;
  processingMethod?: string;
  confidence?: number;
  concepts?: string[];
  replyTo?: string; // Threading support
  memoryContext?: {
    relatedMemories: number;
    phaseCoherence: number;
    valence?: number;
    phaseTag?: number;
    amplitude?: number;
    frequency?: number;
  };
}

function createMessageStore() {
  const { subscribe, update, set } = writable<Message[]>([]);
  let page = 0;
  let hasMoreHistory = true;
  let isLoadingHistory = false;

  // Load conversation history from localStorage on init (SSR safe)
  if (browser) {
    const saved = localStorage.getItem('tori-conversation-history');
    if (saved) {
      try {
        const loadedHistory = JSON.parse(saved).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        set(loadedHistory);
      } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
        console.warn('Failed to load conversation history:', e);
      
}
    }

    // Listen for storage events to sync across tabs
    window.addEventListener('storage', (e) => {
      if (e.key === 'tori-conversation-history' && e.newValue) {
        try {
          const updatedHistory = JSON.parse(e.newValue).map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
          set(updatedHistory);
        } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
          console.warn('Failed to sync from storage event:', err);
        
}
      }
    });
  }

  // Save to localStorage whenever messages change (SSR safe)
  if (browser) {
    subscribe(messages => {
      if (messages.length > 0) {
        localStorage.setItem('tori-conversation-history', JSON.stringify(messages));
      }
    });
  }

  /** Pull in older messages when the top sentinel fires */
  async function loadOlder() {
    // Prevent multiple loads
    if (!hasMoreHistory || isLoadingHistory) return;
    
    isLoadingHistory = true;
    page += 1;
    
    try {
      // TODO: Replace with actual Soliton Memory fetch
      const response = await fetch(`/api/messages?page=${page}`);
      const older = await response.json();
      
      if (older.length > 0) {
        update(list => [...older, ...list]);
        // Keep hasMoreHistory true if we got results
      } else {
        hasMoreHistory = false; // No more history available
      }
    } catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
      console.error('Failed to load older messages:', e);
      hasMoreHistory = false; // Disable on error
    
} finally {
      isLoadingHistory = false;
    }
  }

  /** Append streamed chunks to the last assistant bubble */
  function addStreamChunk(id: string, chunk: string) {
    update(list =>
      list.map(m => (m.id === id ? { ...m, content: m.content + chunk } : m))
    );
  }

  /** Add a new message */
  function addMessage(message: Message) {
    update(list => [...list, message]);
  }

  /** Clear all messages */
  function clearMessages() {
    set([]);
    if (browser) {
      localStorage.removeItem('tori-conversation-history');
    }
    page = 0; // Reset pagination pointer
    hasMoreHistory = true; // Reset history flag
    isLoadingHistory = false;
  }

  /** Get message by ID for threading */
  function getMessageById(id: string): Message | undefined {
    let foundMessage: Message | undefined;
    subscribe(messages => {
      foundMessage = messages.find(m => m.id === id);
    })();
    return foundMessage;
  }

  /** Get thread context for a message */
  function getThreadContext(messageId: string): Message[] {
    const context: Message[] = [];
    let currentId: string | undefined = messageId;
    
    subscribe(messages => {
      while (currentId) {
        const msg = messages.find(m => m.id === currentId);
        if (msg) {
          context.unshift(msg);
          currentId = msg.replyTo;
        } else {
          break;
        }
      }
    })();
    
    return context;
  }

  return { 
    subscribe, 
    loadOlder, 
    set, 
    addStreamChunk, 
    addMessage, 
    clearMessages,
    getMessageById,
    getThreadContext
  };
}

export const messages = createMessageStore();
