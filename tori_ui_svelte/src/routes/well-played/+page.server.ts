import { redirect } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

// Legacy route redirect to canonical renderer
export const load: PageServerLoad = async () => {
  // 308 Permanent Redirect to the canonical renderer route
  throw redirect(308, '/renderer');
};
