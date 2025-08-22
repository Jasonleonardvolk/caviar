// stores/session.ts
import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';

// Current group ID (persisted to localStorage)
function createCurrentGroupId() {
    const STORAGE_KEY = 'tori_current_group_id';
    
    // Initialize from localStorage
    const stored = browser ? localStorage.getItem(STORAGE_KEY) : null;
    const { subscribe, set, update } = writable<string | null>(stored);
    
    return {
        subscribe,
        set: (value: string | null) => {
            if (browser && value !== null) {
                localStorage.setItem(STORAGE_KEY, value);
            } else if (browser) {
                localStorage.removeItem(STORAGE_KEY);
            }
            set(value);
        },
        update: (updater: (value: string | null) => string | null) => {
            update(value => {
                const newValue = updater(value);
                if (browser && newValue !== null) {
                    localStorage.setItem(STORAGE_KEY, newValue);
                } else if (browser) {
                    localStorage.removeItem(STORAGE_KEY);
                }
                return newValue;
            });
        },
        clear: () => {
            if (browser) {
                localStorage.removeItem(STORAGE_KEY);
            }
            set(null);
        }
    };
}

// User session info
export const currentUser = writable({
    id: 'admin_user',  // TODO: Get from auth
    name: 'Admin User',
    email: 'admin@example.com'
});

// Current group selection
export const currentGroupId = createCurrentGroupId();

// Group list (fetched from API)
export const userGroups = writable<Group[]>([]);

// Currently selected group details
export const currentGroup = derived(
    [currentGroupId, userGroups],
    ([$currentGroupId, $userGroups]) => {
        if (!$currentGroupId) return null;
        return $userGroups.find(g => g.id === $currentGroupId) || null;
    }
);

// Types
export interface Group {
    id: string;
    name: string;
    description?: string;
    owner_id: string;
    members: string[];
    created_at: string;
    updated_at: string;
    member_count: number;
}

// API functions
export async function fetchUserGroups() {
    try {
        const response = await fetch('/api/groups');
        if (!response.ok) throw new Error('Failed to fetch groups');
        
        const groups = await response.json();
        userGroups.set(groups);
        
        // If current group is not in list, clear selection
        const groupIds = groups.map((g: Group) => g.id);
        currentGroupId.update(id => {
            if (id && !groupIds.includes(id)) {
                return null;
            }
            return id;
        });
        
        return groups;
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        console.error('Failed to fetch groups:', error);
        return [];
    
}
}

// Helper to inject groupId into API calls
export function withGroupId(url: string, groupId?: string | null): string {
    const gid = groupId ?? get(currentGroupId);
    if (!gid) return url;
    
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}groupId=${encodeURIComponent(gid)}`;
}

// Re-export for convenience
// get is already imported at the top of the file
