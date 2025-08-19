import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { getHealth } from '$lib/health/checks.server';

export const GET: RequestHandler = async () => {
  const report = await getHealth();
  return json(report, { status: report.ok ? 200 : 503 });
};