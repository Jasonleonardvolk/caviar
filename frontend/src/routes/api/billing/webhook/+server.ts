import type { RequestHandler } from './$types';
import Stripe from 'stripe';
import { env } from '$env/dynamic/private';

const stripe = new Stripe(env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2023-10-16'
});

const webhookSecret = env.STRIPE_WEBHOOK_SECRET || '';

export const POST: RequestHandler = async ({ request }) => {
  if (!webhookSecret) {
    console.warn('Stripe webhook secret not configured');
    return new Response('Webhook not configured', { status: 500 });
  }

  const signature = request.headers.get('stripe-signature');
  if (!signature) {
    return new Response('No signature', { status: 400 });
  }

  const body = await request.text();

  try {
    const event = stripe.webhooks.constructEvent(body, signature, webhookSecret);

    // Handle the event
    switch (event.type) {
      case 'checkout.session.completed':
        const session = event.data.object as Stripe.Checkout.Session;
        console.log('Checkout completed:', session.id);
        // TODO: Update user's plan in database
        break;

      case 'customer.subscription.created':
      case 'customer.subscription.updated':
        const subscription = event.data.object as Stripe.Subscription;
        console.log('Subscription event:', subscription.id, event.type);
        // TODO: Update subscription status
        break;

      case 'customer.subscription.deleted':
        const deletedSub = event.data.object as Stripe.Subscription;
        console.log('Subscription cancelled:', deletedSub.id);
        // TODO: Downgrade user to free plan
        break;

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return new Response(JSON.stringify({ received: true }), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (err) {
    console.error('Webhook error:', err);
    return new Response(`Webhook Error: ${err instanceof Error ? err.message : 'Unknown'}`, {
      status: 400
    });
  }
};