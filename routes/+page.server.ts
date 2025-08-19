import { redirect } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = ({ locals }) => {
  // If user is not logged in, redirect to login
  if (!locals.user) {
    throw redirect(303, '/login');
  }
  
  // User is authenticated, continue to main page
  return {};
};
