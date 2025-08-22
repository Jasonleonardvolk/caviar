import type { RequestHandler } from './$types';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, { apiVersion: '2024-06-20' });

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const { planId } = await request.json();
    const price = planId === 'pro'
      ? process.env.STRIPE_PRICE_PRO
      : process.env.STRIPE_PRICE_PLUS;

    if (!price) return new Response('Missing price', { status: 400 });

    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      line_items: [{ price, quantity: 1 }],
      success_url: process.env.STRIPE_SUCCESS_URL ?? `${url.origin}/thank-you?plan=${planId}`,
      cancel_url: process.env.STRIPE_CANCEL_URL ?? `${url.origin}/pricing?canceled=1`,
      allow_promotion_codes: true
    });

    return new Response(JSON.stringify({ url: session.url }), { headers: { 'content-type':'application/json' } });
  } catch (e:any) {
    console.error('checkout error', e);
    return new Response('Checkout error', { status: 500 });
  }
};