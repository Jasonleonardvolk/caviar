import type { PageServerLoad } from './$types';
import { getHealth } from '$lib/health/checks.server';

export const load: PageServerLoad = async () => {
  const report = await getHealth();
  return { report };
};