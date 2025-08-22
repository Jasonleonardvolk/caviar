import { dev } from '$app/environment';

/**
 * If IRIS_ALLOW_UNAUTH=1 (or we're in dev), allow all.
 * If INTERNAL_API_KEY is set, require header x-api-key to match it.
 * Otherwise allow.
 */
export function requireApiKey(request: Request) {
  if (dev || process.env.IRIS_ALLOW_UNAUTH === '1') return null;

  const expected = process.env.INTERNAL_API_KEY;
  if (!expected) return null;

  const got = request.headers.get('x-api-key');
  if (got !== expected) {
    return new Response('Unauthorized', { status: 401 });
  }
  return null;
}