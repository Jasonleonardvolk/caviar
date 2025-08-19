import { redirect } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ cookies }) => {
  // Clear session cookie with correct path
  cookies.delete('session', { path: '/' });
  
  // Redirect to login page
  throw redirect(303, '/login');
};
