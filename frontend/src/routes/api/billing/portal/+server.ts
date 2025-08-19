import type { RequestHandler } from './$types';
import Stripe from 'stripe';
import { json, error } from '@sveltejs/kit';

const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY || '';
const stripe = STRIPE_SECRET_KEY ? new Stripe(STRIPE_SECRET_KEY, { apiVersion: '2024-04-10' }) : null;

/**
 * POST /api/billing/portal
 * Body: { customerId: string, returnUrl?: string }
 */
export const POST: RequestHandler = async ({ request, url }) => {
  if (!stripe) throw error(500, 'Stripe not configured');

  const body = await request.json().catch(() => ({}));
  const customerId = String(body?.customerId || '');
  if (!customerId) throw error(400, 'Missing customerId');

  const session = await stripe.billingPortal.sessions.create({
    customer: customerId,
    return_url: body?.returnUrl || `${url.origin}/account/manage`
  });
  return json({ url: session.url });
};