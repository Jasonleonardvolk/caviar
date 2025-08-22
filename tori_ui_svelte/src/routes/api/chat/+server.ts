import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readFileSync } from 'fs';
import { join } from 'path';

// Get dynamic API port (same as other routes)
function getApiUrl(): string {
  try {
    const configPath = join(process.cwd(), 'api_port.json');
    const config = JSON.parse(readFileSync(configPath, 'utf-8'));
    return `http://localhost:${config.port}`;
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    console.warn('❌ Could not read API port, falling back to 8002:', error);
    return 'http://localhost:8002';
  
}
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { message, userId } = await request.json();
    
    if (!message || typeof message !== 'string') {
      return json({ error: 'Message is required' }, { status: 400 });
    }
    
    const apiUrl = getApiUrl();
    console.log(`🤖 Chat API: Processing message for user ${userId || 'anonymous'}`);
    console.log(`📝 Message: "${message}"`);
    console.log(`🔗 Backend URL: ${apiUrl}`);
    
    // Forward to backend chat endpoint
    const response = await fetch(`${apiUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        userId: userId || 'anonymous',
        timestamp: new Date().toISOString()
      })
    });
    
    if (!response.ok) {
      console.error(`❌ Backend chat error: ${response.status}`);
      
      if (response.status === 404) {
        return json({ 
          error: 'Chat endpoint not available',
          suggestion: 'Chat functionality requires backend update'
        }, { status: 503 });
      }
      
      return json({ 
        error: 'Backend communication failed',
        status: response.status 
      }, { status: 502 });
    }
    
    const result = await response.json();
    console.log(`✅ Chat response received from backend`);
    
    return json(result);
    
  } catch (error) {
    console.error('❌ Chat API error:', error);
    return json({ 
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  return json({
    message: 'TORI Chat API',
    status: 'available',
    endpoints: ['POST /api/chat'],
    features: ['concept_search', 'soliton_memory', 'knowledge_synthesis']
  });
};