import type { RequestHandler } from './$types';
import Stripe from 'stripe';
import { json, error } from '@sveltejs/kit';
import fs from 'node:fs';
import path from 'node:path';

const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY || '';
const stripe = STRIPE_SECRET_KEY ? new Stripe(STRIPE_SECRET_KEY, { apiVersion: '2024-04-10' }) : null;

function loadPlans() {
  // Read repo-level plans.json (../.. from frontend during dev)
  const candidates = [
    path.resolve(process.cwd(), '..', 'config', 'plans.json'),
    path.resolve(process.cwd(), 'config', 'plans.json')
  ];
  for (const p of candidates) {
    try { if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch {}
  }
  return null;
}

export const POST: RequestHandler = async ({ request, url }) => {
  if (!stripe) throw error(500, 'Stripe not configured');

  const body = await request.json().catch(() => ({}));
  const planId = String(body?.planId ?? 'plus');

  const plans = loadPlans();
  if (!plans) throw error(500, 'plans.json not found');
  const plan = plans.plans.find((p: any) => p.id === planId);
  if (!plan || !plan.stripePriceId) throw error(400, 'Invalid plan');

  const success = url.origin + '/account/success';
  const cancel  = url.origin + '/pricing?cancel=1';

  const session = await stripe.checkout.sessions.create({
    mode: 'subscription',
    line_items: [{ price: plan.stripePriceId, quantity: 1 }],
    success_url: success,
    cancel_url: cancel,
    // You can pass a user id here as client_reference_id if available.
  });

  return json({ url: session.url });
};