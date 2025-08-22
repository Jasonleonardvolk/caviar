import { json } from '@sveltejs/kit';
import { getAllGhosts } from '$lib/services/GhostMemoryVaultAdapter';

export const GET = async () => {
  try {
    const ghosts = await getAllGhosts();
    return json({
      success: true,
      memories: ghosts,
      moodCurves: [],  // TODO: Implement when available in mesh
      conceptArcs: []  // TODO: Implement when available in mesh
    });
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('Error fetching ghost memories:', error);
    return json({
      success: false,
      error: 'Failed to fetch ghost memories',
      memories: [],
      moodCurves: [],
      conceptArcs: []
    
}, { status: 500 });
  }
};

// Optional: POST endpoint for compatibility
export const POST = async ({ request }) => {
  try {
    const ghostEvent = await request.json();
    
    // Since the GhostMemoryVault service already handles this,
    // this endpoint is just for compatibility
    return json({ success: true });
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.error('Error in ghost memory POST:', error);
    return json({ success: false, error: msg 
}, { status: 500 });
  }
};
