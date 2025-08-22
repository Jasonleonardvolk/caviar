import { json } from '@sveltejs/kit';
import { isMock } from '$lib/server/env';

export async function GET() {
  if (isMock()) {
    return json({
      ok: true,
      state: 'idle',
      concepts: 0,
      vault: { connected: false, mode: 'mock' },
      note: 'mock'
    });
  }
  
  // Real path - TODO: Connect to real memory vault service
  return json({
    ok: false,
    error: 'Memory vault service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
