// @ts-nocheck
import type { PageServerLoad } from './$types';
import { getHealth } from '$lib/health/checks.server';

export const load = async () => {
  const report = await getHealth();
  return { report };
};;null as any as PageServerLoad;