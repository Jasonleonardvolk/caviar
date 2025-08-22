import type { RequestHandler } from './$types';

// Respond to GET /health with a 200 OK and minimal content.
export const GET: RequestHandler = async () => {
  return new Response('OK', {
    status: 200,
    headers: {
      'Content-Type': 'text/plain',
      'Cache-Control': 'no-cache'
    }
  });
};

// Also handle HEAD requests for some health checkers
export const HEAD: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Content-Type': 'text/plain',
      'Cache-Control': 'no-cache'
    }
  });
};