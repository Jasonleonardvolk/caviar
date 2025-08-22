// src/hooks.server.ts
import type { Handle } from '@sveltejs/kit';
import { redirect } from '@sveltejs/kit';
import fs from 'node:fs/promises';
import path from 'node:path';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002';
const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const handle: Handle = async ({ event, resolve }) => {
  // Serve uploaded files statically from LOCAL_UPLOAD_DIR
  if (event.url.pathname.startsWith('/uploads/')) {
    const file = event.url.pathname.replace('/uploads/', '');
    const full = path.join(LOCAL_UPLOAD_DIR, file);
    try {
      const data = await fs.readFile(full);
      const ext = path.extname(file).toLowerCase();
      const mime = ext === '.pdf' ? 'application/pdf' : 'application/octet-stream';
      return new Response(data, { status: 200, headers: { 'content-type': mime } });
    } catch {
      // File not found, continue to normal handling
    }
  }

  // Read session cookie on every request
  const session = event.cookies.get('session');
  const username = event.cookies.get('username');
  const role = event.cookies.get('role');
  
  // Populate event.locals.user based on session
  if (session && username) {
    // Set user object matching the interface in app.d.ts
    event.locals.user = { 
      id: `user_${username}`,
      username: username,
      name: username, // Use username as display name
      role: (role === 'admin' || role === 'user') ? role : 'user'
    };
  }
  // DO NOT set null - user is optional in app.d.ts
  // If you need to explicitly clear it later, use:
  // delete event.locals.user;
  
  // Optional: Protect routes (uncomment if you want server-side protection)
  // Note: The layout already handles this client-side
  /*
  if (!event.locals.user && 
      event.url.pathname !== '/login' && 
      event.url.pathname !== '/logout' &&
      !event.url.pathname.startsWith('/api/')) {
    throw redirect(303, '/login');
  }
  */
  
  // CRITICAL: Add Authorization header for backend API calls
  // This bridges the cookie-to-header gap since backend expects Bearer token
  const originalFetch = event.fetch;
  event.fetch = async (input, init) => {
    // Properly extract URL string from various input types
    const urlStr = 
      typeof input === 'string'
        ? input
        : input instanceof Request
          ? input.url
          : input instanceof URL
            ? input.href
            : String(input);
    
    // If calling our backend API and we have a session token
    if (session && urlStr.startsWith(API_URL)) {
      init = init || {};
      init.headers = new Headers(init.headers || {});
      
      // Add Bearer token from session cookie
      // This allows backend to authenticate even though we use cookies
      (init.headers as Headers).set('Authorization', `Bearer ${session}`);
      
      // Also ensure credentials are included for cookies
      init.credentials = 'include';
    }
    
    return originalFetch(input, init);
  };
  
  // Continue to route handler
  const response = await resolve(event);
  return response;
};
