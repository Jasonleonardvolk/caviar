// API Error Handler with Rate Limit Support
export class ApiError extends Error {
    constructor(
        message: string, 
        public status: number,
        public detail?: string,
        public retryAfter?: number
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

export async function handleApiResponse(response: Response) {
    if (!response.ok) {
        let detail = '';
        let retryAfter = 60; // default retry time
        
        try {
            const errorData = await response.json();
            detail = errorData.detail || errorData.message || '';
        } catch {
            detail = response.statusText;
        }
        
        // Handle rate limiting
        if (response.status === 429) {
            // Check for Retry-After header
            const retryHeader = response.headers.get('Retry-After');
            if (retryHeader) {
                retryAfter = parseInt(retryHeader) || 60;
            }
            
            throw new ApiError(
                'Rate limit exceeded',
                response.status,
                detail,
                retryAfter
            );
        }
        
        throw new ApiError(
            `API Error: ${response.status}`,
            response.status,
            detail
        );
    }
    
    return response;
}

// Example usage in a component
export async function joinGroupWithInvite(inviteCode: string) {
    try {
        const response = await fetch('/api/groups/join/v2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getCSRFToken()}`
            },
            body: JSON.stringify({ token: inviteCode })
        });
        
        await handleApiResponse(response);
        return await response.json();
        
    } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
        if (error instanceof ApiError && error.status === 429) {
            // Show rate limit toast
            showRateLimitToast(error.detail || 'Too many attempts', error.retryAfter);
        
}
        throw error;
    }
}

// Helper to get CSRF token
function getCSRFToken(): string {
    let token = sessionStorage.getItem('csrf_token');
    if (!token) {
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        token = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
        sessionStorage.setItem('csrf_token', token);
    }
    return token;
}

// Toast notification store
import { writable } from 'svelte/store';

export const rateLimitToast = writable({
    show: false,
    message: '',
    retryIn: 60
});

export function showRateLimitToast(message: string, retryIn: number = 60) {
    rateLimitToast.set({
        show: true,
        message,
        retryIn
    });
}
