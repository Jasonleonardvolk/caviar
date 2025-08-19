import { fail, redirect } from '@sveltejs/kit';
import type { Actions } from './$types';

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const data = await request.formData();
    const username = data.get('username')?.toString()?.trim();
    const password = data.get('password')?.toString();
    
    // Validate input
    if (!username || !password) {
      return fail(400, { 
        error: 'Username and password are required' 
      });
    }
    
    // Admin authentication
    if (username.toLowerCase() === 'admin') {
      // For launch: simple admin password check
      // TODO: Replace with proper hashed password system
      if (password === 'tori2025admin') {  // Change this password!
        cookies.set('session', 'admin', {
          path: '/',
          httpOnly: true,
          secure: false, // Set to true in production with HTTPS
          sameSite: 'strict',
          maxAge: 60 * 60 * 24 * 7 // 7 days
        });
        
        throw redirect(303, '/');
      } else {
        return fail(401, { 
          error: 'Invalid admin credentials' 
        });
      }
    }
    
    // Regular user authentication/registration
    // For launch: auto-register any valid username
    if (username.length < 2) {
      return fail(400, { 
        error: 'Username must be at least 2 characters' 
      });
    }
    
    if (password.length < 3) {
      return fail(400, { 
        error: 'Password must be at least 3 characters' 
      });
    }
    
    // Set user session cookie
    cookies.set('session', username, {
      path: '/',
      httpOnly: true,
      secure: false, // Set to true in production with HTTPS
      sameSite: 'strict',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });
    
    // Redirect to main app
    throw redirect(303, '/');
  }
};
