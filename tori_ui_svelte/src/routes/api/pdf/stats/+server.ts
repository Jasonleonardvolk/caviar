import { json } from '@sveltejs/kit';
import { isMock } from '$lib/server/env';

export async function GET() {
  if (isMock()) {
    return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
  }
  
  // Real path - TODO: Connect to real PDF processing service
  return json({
    ok: false,
    error: 'PDF processing service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
