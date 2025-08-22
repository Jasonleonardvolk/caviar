// D:\Dev\kha\tori_ui_svelte\src\routes\api\health\+server.ts
import type { RequestHandler } from './$types';

export const prerender = false;

export const GET: RequestHandler = async () => {
  return new Response(
    JSON.stringify({
      ok: true,
      app: 'iris',
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development'
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    }
  );
};