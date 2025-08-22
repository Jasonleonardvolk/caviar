<script lang="ts">
    import { currentGroupId, fetchUserGroups } from '$lib/stores/session';
    
    export let show = false;
    
    let inviteCode = '';
    let loading = false;
    let error = '';
    let success = false;
    
    // Get or generate CSRF token
    function getCSRFToken(): string {
        // In production, get this from a secure cookie or meta tag
        let token = sessionStorage.getItem('csrf_token');
        if (!token) {
            // Generate a random token for this session
            const array = new Uint8Array(32);
            crypto.getRandomValues(array);
            token = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
            sessionStorage.setItem('csrf_token', token);
        }
        return token;
    }
    
    async function joinGroup() {
        if (!inviteCode.trim()) {
            error = 'Please enter an invite code';
            return;
        }
        
        loading = true;
        error = '';
        
        try {
            // Use v2 endpoint with Bearer token
            const csrfToken = getCSRFToken();
            
            const response = await fetch('/api/groups/join/v2', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${csrfToken}` // CSRF protection
                },
                body: JSON.stringify({ token: inviteCode.trim() })
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to join group');
            }
            
            const result = await response.json();
            
            // Success!
            success = true;
            
            // Refresh groups list
            await fetchUserGroups();
            
            // Set as current group
            currentGroupId.set(result.joined);
            
            // Close modal after delay
            setTimeout(() => {
                show = false;
                inviteCode = '';
                success = false;
            }, 1500);
            
        } catch (err) {
            error = err instanceof Error ? err.message : 'Failed to join group';
        } finally {
            loading = false;
        }
    }
    
    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Enter' && !loading) {
            joinGroup();
        } else if (e.key === 'Escape') {
            show = false;
        }
    }
    
    function close() {
        show = false;
        inviteCode = '';
        error = '';
        success = false;
    }
    
    // Clear sensitive data on unmount
    function cleanup() {
        inviteCode = '';
    }
</script>

{#if show}
    <div class="modal-backdrop" on:click={close} on:keydown={(e) =>
        <div class="modal" on:click|stopPropagation on:keydown={(e) =>
            <h2>Join Group with Invite</h2>
            
            {#if success}
                <div class="success-message">
                    <span class="icon">✓</span>
                    Successfully joined group!
                </div>
            {:else}
                <p>Enter the invite code or link you received:</p>
                
                <div class="security-notice">
                    <span class="icon">🔒</span>
                    Secure single-use invite
                </div>
                
                <input
                    type="text"
                    bind:value={inviteCode}
                    placeholder="ABCD-EFGH or full invite link"
                    on:keydown={handleKeydown}
                    disabled={loading}
                    class:error={error}
                    autocomplete="off"
                    spellcheck="false"
                />
                
                {#if error}
                    <div class="error-message">{error}</div>
                {/if}
                
                <div class="modal-actions">
                    <button
                        type="button"
                        class="btn-secondary"
                        on:click={close}
                        disabled={loading}
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        class="btn-primary"
                        on:click={joinGroup}
                        disabled={loading || !inviteCode.trim()}
                    >
                        {loading ? 'Joining...' : 'Join Group'}
                    </button>
                </div>
                
                <div class="security-info">
                    <small>
                        • Invites can only be used once<br>
                        • Expire after set duration<br>
                        • Transmitted securely over HTTPS
                    </small>
                </div>
            {/if}
        </div>
    </div>
{/if}

<svelte:window on:beforeunload={cleanup} />

<style>
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
        max-width: 400px;
        width: 90%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    h2 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
    }
    
    p {
        margin: 0 0 1rem 0;
        color: #666;
    }
    
    .security-notice {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #f0f9ff;
        border: 1px solid #3498db;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        color: #3498db;
    }
    
    input {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        font-size: 1rem;
        font-family: monospace;
        text-align: center;
        text-transform: uppercase;
    }
    
    input:focus {
        outline: none;
        border-color: #3498db;
    }
    
    input.error {
        border-color: #e74c3c;
    }
    
    input:disabled {
        background: #f5f5f5;
        cursor: not-allowed;
    }
    
    .error-message {
        color: #e74c3c;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .success-message {
        text-align: center;
        color: #27ae60;
        font-size: 1.125rem;
        padding: 2rem 0;
    }
    
    .success-message .icon {
        display: block;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .modal-actions {
        display: flex;
        gap: 1rem;
        justify-content: flex-end;
        margin-top: 1.5rem;
    }
    
    .btn-primary,
    .btn-secondary {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        font-size: 1rem;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    
    .btn-primary {
        background: #3498db;
        color: white;
    }
    
    .btn-secondary {
        background: #95a5a6;
        color: white;
    }
    
    .btn-primary:hover:not(:disabled),
    .btn-secondary:hover:not(:disabled) {
        opacity: 0.9;
    }
    
    .btn-primary:disabled,
    .btn-secondary:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .security-info {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        color: #666;
        line-height: 1.4;
    }
    
    .security-info small {
        font-size: 0.75rem;
    }
</style>
