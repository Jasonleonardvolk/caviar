import type { PageServerLoad } from './$types';
import { env } from '$env/dynamic/private';

export const load: PageServerLoad = async ({ fetch }) => {
  // If using mocks, return mock data directly without fetching
  if (env.IRIS_USE_MOCKS === '1') {
    return {
      pdf: { ok: true, docs: 3, pages: 42, note: 'mock' },
      memory: { ok: true, state: 'idle', concepts: 0, vault: { connected: false, mode: 'mock' }, note: 'mock' }
    };
  }
  
  // In real mode, try to fetch from the API endpoints
  try {
    const [pdfResponse, memoryResponse] = await Promise.all([
      fetch('/api/pdf/stats'),
      fetch('/api/memory/state')
    ]);
    
    const pdf = await pdfResponse.json();
    const memory = await memoryResponse.json();
    
    return {
      pdf,
      memory
    };
  } catch (error) {
    // If there's an error in real mode, return safe defaults
    console.error('Error fetching data for renderer:', error);
    return {
      pdf: { ok: false, error: 'Unable to fetch PDF stats' },
      memory: { ok: false, error: 'Unable to fetch memory state' }
    };
  }
};
