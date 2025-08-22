<script lang="ts">
    import { currentGroup } from '$lib/stores/session';
    import { get } from 'svelte/store';
    
    let showInviteModal = false;
    let inviteCode = '';
    let inviteLink = '';
    let loading = false;
    let copied = false;
    let useReadableCode = true;
    let expiresHours = 168; // 7 days default
    
    async function createInvite() {
        const group = get(currentGroup);
        if (!group) return;
        
        loading = true;
        
        try {
            const response = await fetch(`/api/groups/${group.id}/invite`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    readable: useReadableCode,
                    expires_hours: expiresHours
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to create invite');
            }
            
            const data = await response.json();
            
            if (data.type === 'code') {
                inviteCode = data.invite;
                inviteLink = '';
            } else {
                inviteCode = '';
                inviteLink = data.link || data.invite;
            }
            
        } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
            console.error('Failed to create invite:', error);
            alert('Failed to create invite');
        
} finally {
            loading = false;
        }
    }
    
    async function copyToClipboard(text: string) {
        try {
            await navigator.clipboard.writeText(text);
            copied = true;
            setTimeout(() => copied = false, 2000);
        } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
            console.error('Failed to copy:', err);
        
}
    }
    
    function toggleInviteModal() {
        showInviteModal = !showInviteModal;
        if (showInviteModal && !inviteCode && !inviteLink) {
            createInvite();
        }
    }
    
    $: isOwner = $currentGroup && $currentGroup.owner_id === 'admin_user'; // TODO: Get real user ID
</script>

{#if $currentGroup && isOwner}
    <button class="invite-btn" on:click={toggleInviteModal}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0a1 1 0 011 1v6h6a1 1 0 110 2H9v6a1 1 0 11-2 0V9H1a1 1 0 110-2h6V1a1 1 0 011-1z"/>
        </svg>
        Invite Members
    </button>
{/if}

{#if showInviteModal}
    <div\ class="modal-backdrop"\ role="button"\ tabindex="0"\ on:click=\{toggleInviteModal}\ on:keydown=\{\(e\)\ =>\ e\.key\ ===\ 'Escape'\ &&\ toggleInviteModal\(\)}>
        <div\ class="modal"\ on:click\|stopPropagation>
            <h3>Invite to {$currentGroup?.name}</h3>
            
            <div class="invite-options">
                <label>
                    <input 
                        type="radio" 
                        bind:group={useReadableCode} 
                        value={true}
                        on:change={createInvite}
                    />
                    Simple Code (easy to share)
                </label>
                <label>
                    <input 
                        type="radio" 
                        bind:group={useReadableCode} 
                        value={false}
                        on:change={createInvite}
                    />
                    Secure Link (more secure)
                </label>
            </div>
            
            <div class="expiry-options">
                <label>
                    Expires in:
                    <select bind:value={expiresHours} on:change={createInvite}>
                        <option value={24}>24 hours</option>
                        <option value={168}>7 days</option>
                        <option value={720}>30 days</option>
                        <option value={8760}>1 year</option>
                    </select>
                </label>
            </div>
            
            {#if loading}
                <div class="loading">Generating invite...</div>
            {:else if inviteCode}
                <div class="invite-display">
                    <div class="code-display">{inviteCode}</div>
                    <button 
                        class="copy-btn" 
                        on:click={() => copyToClipboard(inviteCode)}
                    >
                        {copied ? '✓ Copied!' : 'Copy'}
                    </button>
                </div>
                <p class="hint">Share this code with people you want to invite</p>
            {:else if inviteLink}
                <div class="invite-display">
                    <input 
                        type="text" 
                        value={inviteLink} 
                        readonly
                        on:click={(e) => e.currentTarget.select()}
                    />
                    <button 
                        class="copy-btn" 
                        on:click={() => copyToClipboard(inviteLink)}
                    >
                        {copied ? '✓ Copied!' : 'Copy'}
                    </button>
                </div>
                <p class="hint">Share this link to invite members</p>
            {/if}
            
            <div class="modal-actions">
                <button class="btn-secondary" on:click={toggleInviteModal}>
                    Close
                </button>
            </div>
        </div>
    </div>
{/if}

<style>
    .invite-btn {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid #3498db;
        background: white;
        color: #3498db;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.875rem;
        transition: all 0.2s;
    }
    
    .invite-btn:hover {
        background: #3498db;
        color: white;
    }
    
    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .modal {
        background: white;
        border-radius: 0.5rem;
        padding: 2rem;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    h3 {
        margin: 0 0 1.5rem 0;
        font-size: 1.25rem;
    }
    
    .invite-options {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .invite-options label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
    }
    
    .expiry-options {
        margin-bottom: 1.5rem;
    }
    
    .expiry-options label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .expiry-options select {
        padding: 0.25rem 0.5rem;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
    }
    
    .loading {
        text-align: center;
        color: #666;
        padding: 2rem;
    }
    
    .invite-display {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .code-display {
        flex: 1;
        padding: 1rem;
        background: #f8f9fa;
        border: 2px dashed #ddd;
        border-radius: 0.25rem;
        font-family: monospace;
        font-size: 1.5rem;
        text-align: center;
        letter-spacing: 0.1em;
    }
    
    .invite-display input {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        font-family: monospace;
        font-size: 0.875rem;
    }
    
    .copy-btn {
        padding: 0.5rem 1rem;
        border: 1px solid #ddd;
        background: white;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .copy-btn:hover {
        background: #f8f9fa;
    }
    
    .hint {
        color: #666;
        font-size: 0.875rem;
        margin: 0 0 1rem 0;
    }
    
    .modal-actions {
        display: flex;
        justify-content: flex-end;
        margin-top: 1.5rem;
    }
    
    .btn-secondary {
        padding: 0.5rem 1rem;
        border: 1px solid #ddd;
        background: white;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    
    .btn-secondary:hover {
        background: #f8f9fa;
    }
</style>
