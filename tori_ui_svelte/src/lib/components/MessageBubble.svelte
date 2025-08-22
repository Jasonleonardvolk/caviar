<script lang="ts">
  import type { Message } from '$lib/stores/messages';
  import { darkMode } from '$lib/stores/darkMode';
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  
  export let message: Message;
  
  let DOMPurify: any;
  let safeHtml = '';
  
  onMount(async () => {
    if (browser) {
      // Dynamic import to reduce server bundle size
      const purify = await import('isomorphic-dompurify');
      DOMPurify = purify.default;
      
      // Sanitize content for all assistant messages
      if (message.role === 'assistant') {
        safeHtml = DOMPurify.sanitize(message.content, { 
          ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'blockquote'],
          ALLOWED_ATTR: ['class'],
          KEEP_CONTENT: true
        });
      }
    }
  });
  
  function getProcessingIcon(method: string): string {
    switch (method) {
      case 'document_grounded_ai': return '📚';
      case 'error_fallback': return '⚠️';
      case 'revolutionary_synthesis': return '🌌';
      case 'holographic_synthesis': return '🎯';
      case 'ghost_collective': return '👻';
      case 'cognitive_engine': return '🧬';
      case 'braid_memory': return '🔗';
      default: return '🤖';
    }
  }
  
  function getProcessingName(method: string): string {
    switch (method) {
      case 'document_grounded_ai': return 'Document AI';
      case 'error_fallback': return 'Error Recovery';
      case 'revolutionary_synthesis': return 'Revolutionary';
      case 'holographic_synthesis': return 'Holographic';
      case 'ghost_collective': return 'Ghost Collective';
      case 'cognitive_engine': return 'Cognitive Engine';
      case 'braid_memory': return 'BraidMemory';
      default: return 'TORI AI';
    }
  }
  
  // Update safe HTML when content changes (for streaming)
  $: if (DOMPurify && message.role === 'assistant') {
    safeHtml = DOMPurify.sanitize(message.content, { 
      ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'blockquote'],
      ALLOWED_ATTR: ['class'],
      KEEP_CONTENT: true
    });
  }
</script>

<article 
  role="article"
  aria-live={message.streaming ? 'polite' : undefined}
  aria-label="{message.role === 'user' ? 'Your message' : 'Assistant message'}"
  class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'} mb-2"
>
  <div 
    class="max-w-[80%] rounded-lg message-bubble"
    style="
      background-color: {message.role === 'user' ? 'var(--color-accent)' : $darkMode ? '#374151' : 'var(--color-secondary)'};
      color: {message.role === 'user' ? 'white' : $darkMode ? '#f3f4f6' : 'var(--color-text-primary)'};
      padding: var(--space-2) var(--space-3);
      border-radius: var(--border-radius);
      box-shadow: var(--shadow-sm);
    "
  >
    <div class="message-content" style="font-size: var(--text-sm); line-height: 1.6;">
      {#if message.role === 'assistant' && safeHtml}
        {@html safeHtml}
      {:else}
        {message.content}
      {/if}
    </div>
    
    {#if message.role === 'assistant' && message.processingMethod}
      <div class="mt-2" style="font-size: var(--text-xs); opacity: 0.8;">
        {getProcessingIcon(message.processingMethod)} {getProcessingName(message.processingMethod)}
        {#if message.confidence}
          • {Math.round(message.confidence * 100)}% confidence
        {/if}
      </div>
    {/if}
    
    {#if message.memoryContext}
      <div class="mt-1" style="font-size: var(--text-xs); opacity: 0.6;">
        🌊 {message.memoryContext.relatedMemories} memories • φ={message.memoryContext.phaseTag?.toFixed(3) || 'N/A'}
      </div>
    {/if}
  </div>
</article>

<style>
  /* Ensure code blocks and lists are styled properly */
  .message-content :global(pre) {
    background: rgba(0, 0, 0, 0.1);
    padding: var(--space-2);
    border-radius: var(--border-radius-sm);
    overflow-x: auto;
    margin: var(--space-1) 0;
  }
  
  .message-content :global(code) {
    background: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 0.9em;
  }
  
  .message-content :global(ul),
  .message-content :global(ol) {
    margin: var(--space-1) 0;
    padding-left: var(--space-3);
  }
  
  .message-content :global(blockquote) {
    border-left: 3px solid var(--color-accent);
    padding-left: var(--space-2);
    margin: var(--space-1) 0;
    opacity: 0.9;
  }
</style>
