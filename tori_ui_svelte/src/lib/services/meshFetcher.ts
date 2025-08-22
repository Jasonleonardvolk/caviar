// lib/services/meshFetcher.ts
import { get } from 'svelte/store';
import { currentGroupId, currentUser } from '$lib/stores/session';

// Cache for ETags per user/group combination
const etagCache = new Map<string, string>();

// Get cache key for current user/group
function getCacheKey(): string {
    const user = get(currentUser);
    const groupId = get(currentGroupId);
    return `${user.id}:${groupId || 'personal'}`;
}

export interface MeshInfo {
    etag: string;
    userId: string;
    groupId: string | null;
    size: number;
    modified: number;
}

export interface DeltaResponse {
    etag: string;
    type: 'full' | 'delta';
    data?: any;
    changes?: {
        concepts_added: any[];
        concepts_modified: any[];
        concepts_removed: any[];
        edges_added: any[];
        edges_removed: any[];
    };
}

/**
 * Check if mesh has changed using ETag
 */
export async function checkMeshChanged(): Promise<boolean> {
    const user = get(currentUser);
    const groupId = get(currentGroupId);
    const cacheKey = getCacheKey();
    const cachedEtag = etagCache.get(cacheKey);
    
    try {
        const params = new URLSearchParams({
            userId: user.id,
            ...(groupId && { groupId })
        });
        
        const response = await fetch(`/api/mesh/etag?${params}`, {
            headers: cachedEtag ? { 'If-None-Match': cachedEtag } : {}
        });
        
        if (response.status === 304) {
            // Not modified
            return false;
        }
        
        if (response.ok) {
            const data: MeshInfo = await response.json();
            etagCache.set(cacheKey, data.etag);
            return true;
        }
        
        throw new Error(`Failed to check mesh: ${response.statusText}`);
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Error checking mesh:', error);
        throw error;
    
}
}

/**
 * Fetch mesh with delta support
 */
export async function fetchMeshDelta(): Promise<DeltaResponse | null> {
    const user = get(currentUser);
    const groupId = get(currentGroupId);
    const cacheKey = getCacheKey();
    const cachedEtag = etagCache.get(cacheKey);
    
    try {
        const params = new URLSearchParams({
            userId: user.id,
            ...(groupId && { groupId }),
            ...(cachedEtag && { since_etag: cachedEtag })
        });
        
        const response = await fetch(`/api/mesh/delta?${params}`, {
            headers: cachedEtag ? { 'If-None-Match': cachedEtag } : {}
        });
        
        if (response.status === 304) {
            // Not modified
            return null;
        }
        
        if (response.ok) {
            const data: DeltaResponse = await response.json();
            etagCache.set(cacheKey, data.etag);
            return data;
        }
        
        throw new Error(`Failed to fetch mesh delta: ${response.statusText}`);
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Error fetching mesh delta:', error);
        throw error;
    
}
}

/**
 * Fetch mesh if changed
 */
export async function fetchMeshIfChanged(): Promise<boolean> {
    const changed = await checkMeshChanged();
    
    if (changed) {
        // Fetch the updated mesh
        const delta = await fetchMeshDelta();
        
        if (delta) {
            if (delta.type === 'full') {
                // Full update
                await updateConceptStore(delta.data);
            } else {
                // Apply delta changes
                await applyDeltaChanges(delta.changes!);
            }
            
            return true;
        }
    }
    
    return false;
}

/**
 * Update concept store with new data
 */
async function updateConceptStore(data: any) {
    // TODO: Import and update your concept store
    console.log('Updating concept store with full data:', data);
    
    // Example:
    // import { conceptStore } from '$lib/stores/concepts';
    // conceptStore.set(data);
}

/**
 * Apply delta changes to concept store
 */
async function applyDeltaChanges(changes: DeltaResponse['changes']) {
    // TODO: Apply incremental changes
    console.log('Applying delta changes:', changes);
    
    // Example:
    // import { conceptStore } from '$lib/stores/concepts';
    // conceptStore.update(store => {
    //     // Add new concepts
    //     // Modify existing concepts
    //     // Remove deleted concepts
    //     return store;
    // });
}

/**
 * Start polling for changes (mobile mode)
 */
export function startMeshPolling(intervalMs: number = 10000) {
    let pollInterval: number;
    
    const poll = async () => {
        try {
            await fetchMeshIfChanged();
        } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
            console.error('Polling error:', error);
        
}
    };
    
    // Poll immediately
    poll();
    
    // Set up interval
    pollInterval = window.setInterval(poll, intervalMs);
    
    // Poll on visibility change
    const handleVisibilityChange = () => {
        if (!document.hidden) {
            poll();
        }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Return cleanup function
    return () => {
        clearInterval(pollInterval);
        document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
}

/**
 * Clear all cached ETags
 */
export function clearEtagCache() {
    etagCache.clear();
}

/**
 * Get mesh info without fetching content
 */
export async function getMeshInfo(): Promise<MeshInfo> {
    const user = get(currentUser);
    const groupId = get(currentGroupId);
    
    const params = new URLSearchParams({
        userId: user.id,
        ...(groupId && { groupId })
    });
    
    const response = await fetch(`/api/mesh/info?${params}`);
    
    if (!response.ok) {
        throw new Error(`Failed to get mesh info: ${response.statusText}`);
    }
    
    return response.json();
}

// Auto-start polling on mobile devices
if (typeof window !== 'undefined') {
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    
    if (isMobile) {
        // Start polling on mobile
        const cleanup = startMeshPolling();
        
        // Store cleanup function
        if (typeof window !== 'undefined') {
            (window as any).__meshPollingCleanup = cleanup;
        }
    }
}
