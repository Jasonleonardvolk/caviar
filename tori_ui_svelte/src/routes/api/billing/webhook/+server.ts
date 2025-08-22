import type { RequestHandler } from './$types';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, { apiVersion: '2024-06-20' });

// OPTIONAL: if you configure a webhook secret, verify signature.
// const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.text();
  try {
    // Simple: just parse event without signature for now (dev only).
    const event = JSON.parse(body);
    console.log('Stripe webhook event:', event.type);

    // TODO: on 'checkout.session.completed', mark user as paid in your user system.

    return new Response('ok');
  } catch (e:any) {
    console.error('webhook error', e);
    return new Response('bad request', { status: 400 });
  }
};