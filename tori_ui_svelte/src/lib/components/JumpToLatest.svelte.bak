<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();
  
  let buttonElement: HTMLButtonElement;
  
  function handleClick() {
    dispatch('click');
  }
  
  function handleMouseOver(e: MouseEvent) {
    const target = e.currentTarget as HTMLElement;
    target.style.transform = 'translateY(-2px)';
    target.style.boxShadow = 'var(--shadow-xl)';
  }
  
  function handleMouseOut(e: MouseEvent) {
    const target = e.currentTarget as HTMLElement;
    target.style.transform = 'translateY(0)';
    target.style.boxShadow = 'var(--shadow-lg)';
  }
  
  function handleFocus(e: FocusEvent) {
    const target = e.currentTarget as HTMLElement;
    target.style.outline = '2px solid var(--color-accent)';
    target.style.outlineOffset = '2px';
  }
  
  function handleBlur(e: FocusEvent) {
    const target = e.currentTarget as HTMLElement;
    target.style.outline = 'none';
  }
</script>

<div style="pointer-events: auto;">
  <button
    bind:this={buttonElement}
    on:click={handleClick}
    on:mouseover={handleMouseOver}
    on:mouseout={handleMouseOut}
    on:focus={handleFocus}
    on:blur={handleBlur}
    tabindex="0"
    aria-label="Jump to latest message"
    aria-controls="chat-feed"
    style="
      background-color: var(--color-accent);
      color: white;
      border-radius: 9999px;
      padding: var(--space-2) var(--space-3);
      box-shadow: var(--shadow-lg);
      border: none;
      cursor: pointer;
      font-size: var(--text-sm);
      font-weight: 500;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: var(--space-1);
    "
  >
    <span aria-hidden="true">↓</span>
    <span>Jump to latest</span>
  </button>
</div>
