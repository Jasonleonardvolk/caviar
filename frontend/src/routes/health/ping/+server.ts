import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';

// Fast, allocation-light. Suitable for LB/uptime checks.
export const GET: RequestHandler = async () => json({ ok: true }, { status: 200 });
export const HEAD: RequestHandler = async () => new Response(null, { status: 200 });