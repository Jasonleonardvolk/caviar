<script lang="ts">
  import { tick } from 'svelte';
  import { messages } from '$lib/stores/messages';
  import MessageBubble from './MessageBubble.svelte';
  import JumpToLatest from './JumpToLatest.svelte';
  import { darkMode } from '$lib/stores/darkMode';

  let container: HTMLDivElement;
  let atBottom = true;
  let debounceTimer: ReturnType<typeof setTimeout>;

  /** Pause auto-follow when user scrolls up > 64 px */
  function handleScroll() {
    const threshold = 64;
    const diff =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    atBottom = diff < threshold;
  }

  /** Snap to newest whenever we're pinned at the bottom */
  $: if (atBottom && $messages.length > 0) {
    tick().then(() => {
      if (container) {
        container.scrollTo({ top: container.scrollHeight });
      }
    });
  }

  /** Re-check bottom pin after streaming chunks */
  export function recheckScroll() {
    if (atBottom && container) {
      container.scrollTo({ top: container.scrollHeight });
    }
  }

  /** Force scroll to bottom */
  export function scrollToBottom() {
    if (container) {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    }
  }

  /** Invisible sentinel that triggers `loadOlder()` when visible */
  function topSentinel(node: HTMLElement) {
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        // Debounce to prevent rapid fire
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          messages.loadOlder();
        }, 100);
      }
    });
    io.observe(node);
    return {
      destroy() {
        io.disconnect();
        clearTimeout(debounceTimer);
      }
    };
  }
</script>

<!-- Outer scroll container -->
<div 
  id="chat-feed"
  bind:this={container} 
  on:scroll={handleScroll}
  class="flex-1 overflow-y-auto"
  style="padding: var(--space-3); position: relative;"
  aria-label="Conversation"
  aria-live="polite"
  role="log"
>
  <!-- Top sentinel -->
  <div use:topSentinel aria-hidden="true" />
  
  <!-- List of messages -->
  {#each $messages as message (message.id)}
    <MessageBubble {message} />
  {/each}
  
  <!-- "Jump to latest" pill -->
  {#if !atBottom && $messages.length > 0}
    <div 
      class="sticky bottom-4 flex justify-center"
      style="pointer-events: none;"
    >
      <JumpToLatest
        on:click={scrollToBottom}
      />
    </div>
  {/if}
</div>

<style>
  /* Custom scrollbar styling */
  #chat-feed::-webkit-scrollbar {
    width: var(--space-1);
  }
  
  #chat-feed::-webkit-scrollbar-track {
    background: var(--color-secondary);
  }
  
  #chat-feed::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: calc(var(--space-1) / 2);
  }
  
  #chat-feed::-webkit-scrollbar-thumb:hover {
    background: var(--color-text-secondary);
  }

  :global(.dark) #chat-feed::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
  }
  
  :global(.dark) #chat-feed::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
  }
  
  :global(.dark) #chat-feed::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }
</style>
