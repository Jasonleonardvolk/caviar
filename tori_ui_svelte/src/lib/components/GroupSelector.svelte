<script lang="ts">
    import { onMount } from 'svelte';
    import { currentGroupId, userGroups, fetchUserGroups, type Group } from '$lib/stores/session';
    
    let isLoading = true;
    let showCreateModal = false;
    
    // New group form
    let newGroupId = '';
    let newGroupName = '';
    let newGroupDescription = '';
    let createError = '';
    let isCreating = false;
    
    onMount(async () => {
        await fetchUserGroups();
        isLoading = false;
    });
    
    function selectGroup(groupId: string | null) {
        currentGroupId.set(groupId);
    }
    
    async function createGroup() {
        createError = '';
        
        // Validate
        if (!newGroupId || newGroupId.length < 3) {
            createError = 'Group ID must be at least 3 characters';
            return;
        }
        
        if (!newGroupName.trim()) {
            createError = 'Group name is required';
            return;
        }
        
        isCreating = true;
        
        try {
            const response = await fetch('/api/groups', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: newGroupId.toLowerCase().replace(/[^a-z0-9-_]/g, '-'),
                    name: newGroupName,
                    description: newGroupDescription || undefined
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create group');
            }
            
            const newGroup = await response.json();
            
            // Refresh groups and select the new one
            await fetchUserGroups();
            currentGroupId.set(newGroup.id);
            
            // Reset form and close modal
            newGroupId = '';
            newGroupName = '';
            newGroupDescription = '';
            showCreateModal = false;
            
        } catch (error) {
            createError = error instanceof Error ? error.message : 'Failed to create group';
        } finally {
            isCreating = false;
        }
    }
    
    function formatGroupId(value: string) {
        // Auto-format group ID as user types
        newGroupId = value.toLowerCase().replace(/[^a-z0-9-_]/g, '-');
    }
</script>

<div class="group-selector">
    {#if isLoading}
        <div class="loading">Loading groups...</div>
    {:else}
        <select 
            value={$currentGroupId || ''} 
            on:change={(e) => selectGroup(e.currentTarget.value || null)}
            class="group-select"
        >
            <option value="">Personal Workspace</option>
            
            {#if $userGroups.length > 0}
                <optgroup label="Your Groups">
                    {#each $userGroups as group}
                        <option value={group.id}>
                            {group.name} ({group.member_count} members)
                        </option>
                    {/each}
                </optgroup>
            {/if}
        </select>
        
        <button 
            class="create-btn"
            on:click={() => showCreateModal = true}
            title="Create new group"
        >
            +
        </button>
    {/if}
</div>

<!-- Create Group Modal -->
{#if showCreateModal}
    <div\ class="modal-backdrop"\ role="button"\ tabindex="0"\ on:click=\{\(\)\ =>\ \(showCreateModal\ =\ false\)}\ on:keydown=\{\(e\)\ =>\ e\.key\ ===\ 'Escape'\ &&\ \(showCreateModal\ =\ false\)}> showCreateModal = false} on:keydown={(e) =>
        <div\ class="modal"\ on:click\|stopPropagation>
            <h2>Create New Group</h2>
            
            <form on:submit|preventDefault={createGroup}>
                <div class="form-group">
                    <label for="group-id">Group ID</label>
                    <input
                        id="group-id"
                        type="text"
                        bind:value={newGroupId}
                        on:input={(e) => formatGroupId(e.currentTarget.value)}
                        placeholder="my-awesome-group"
                        required
                        minlength="3"
                        maxlength="32"
                        pattern="[a-z0-9-_]+"
                        disabled={isCreating}
                    />
                    <small>Lowercase letters, numbers, dashes, and underscores only</small>
                </div>
                
                <div class="form-group">
                    <label for="group-name">Display Name</label>
                    <input
                        id="group-name"
                        type="text"
                        bind:value={newGroupName}
                        placeholder="My Awesome Group"
                        required
                        disabled={isCreating}
                    />
                </div>
                
                <div class="form-group">
                    <label for="group-desc">Description (optional)</label>
                    <textarea
                        id="group-desc"
                        bind:value={newGroupDescription}
                        placeholder="What's this group about?"
                        rows="3"
                        disabled={isCreating}
                    />
                </div>
                
                {#if createError}
                    <div class="error">{createError}</div>
                {/if}
                
                <div class="modal-actions">
                    <button
                        type="button"
                        class="btn-secondary"
                        on:click={() => showCreateModal = false}
                        disabled={isCreating}
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        class="btn-primary"
                        disabled={isCreating}
                    >
                        {isCreating ? 'Creating...' : 'Create Group'}
                    </button>
                </div>
            </form>
        </div>
    </div>
{/if}

<style>
    .group-selector {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .group-select {
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-color, #ddd);
        border-radius: 0.25rem;
        background: var(--bg-color, white);
        font-size: 0.875rem;
        min-width: 200px;
    }
    
    .create-btn {
        padding: 0.5rem;
        border: 1px solid var(--border-color, #ddd);
        border-radius: 0.25rem;
        background: var(--bg-color, white);
        cursor: pointer;
        font-weight: bold;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .create-btn:hover {
        background: var(--hover-bg, #f5f5f5);
    }
    
    .loading {
        color: var(--text-muted, #666);
        font-size: 0.875rem;
    }
    
    /* Modal styles */
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
        max-height: 90vh;
        overflow-y: auto;
    }
    
    .modal h2 {
        margin: 0 0 1.5rem 0;
        font-size: 1.5rem;
    }
    
    .form-group {
        margin-bottom: 1.5rem;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .form-group input,
    .form-group textarea {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid var(--border-color, #ddd);
        border-radius: 0.25rem;
        font-size: 1rem;
    }
    
    .form-group small {
        display: block;
        margin-top: 0.25rem;
        color: var(--text-muted, #666);
        font-size: 0.875rem;
    }
    
    .error {
        color: var(--error-color, #e74c3c);
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: var(--error-bg, #fee);
        border-radius: 0.25rem;
    }
    
    .modal-actions {
        display: flex;
        gap: 1rem;
        justify-content: flex-end;
        margin-top: 2rem;
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
        background: var(--primary-color, #3498db);
        color: white;
    }
    
    .btn-secondary {
        background: var(--secondary-color, #95a5a6);
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
</style>
