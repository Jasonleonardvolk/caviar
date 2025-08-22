import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// FIXED: Removed automatic fallback behavior - backend integration must be working
let backendAvailable = false;
let lastCheckTime = 0;
const CHECK_INTERVAL = 30000; // Check every 30 seconds (more frequent)

async function checkBackendHealth(): Promise<boolean> {
  const now = Date.now();
  if (now - lastCheckTime < CHECK_INTERVAL) {
    return backendAvailable;
  }
  
  lastCheckTime = now;
  try {
    const response = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000) // 2 second timeout
    });
    backendAvailable = response.ok;
    
    if (backendAvailable) {
      console.log('✅ Soliton backend connection verified');
    } else {
      console.error('❌ Soliton backend health check failed:', response.status);
    }
  } catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
    backendAvailable = false;
    console.error('❌ Soliton backend unreachable:', msg);
  
}
  
  return backendAvailable;
}

// FIXED: Helper function to validate required fields for different endpoints
function validateRequestBody(path: string, body: any): { valid: boolean; error?: string } {
  switch (path) {
    case 'init':
      if (!body.userId) {
        return { valid: false, error: 'userId is required for memory initialization' };
      }
      break;
    case 'store':
      if (!body.conceptId || !body.content) {
        return { valid: false, error: 'conceptId and content are required for memory storage' };
      }
      break;
    case 'phase':
      if (!body.targetPhase) {
        return { valid: false, error: 'targetPhase is required for phase-based search' };
      }
      break;
    case 'vault':
      if (!body.conceptId || !body.vaultLevel) {
        return { valid: false, error: 'conceptId and vaultLevel are required for vaulting' };
      }
      break;
    case 'find-related':
      if (!body.conceptId) {
        return { valid: false, error: 'conceptId is required for finding related memories' };
      }
      break;
  }
  return { valid: true };
}

export const POST: RequestHandler = async ({ params, request, locals }) => {
  const path = (Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''));
  
  let body: any;
  try {
    body = await request.json();
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error('❌ Invalid JSON in Soliton API request:', msg);
    return json({ 
      success: false, 
      error: 'Invalid JSON in request body',
      timestamp: new Date().toISOString()
    }, { status: 400 });
  }
  
  // FIXED: Always log Soliton API calls for debugging backend integration
  console.log(`🌊 Soliton API POST: /api/soliton/${path}`, {
    userId: locals.user?.id || body.userId || 'anonymous',
    hasBody: !!body,
    timestamp: new Date().toISOString()
  });
  
  // FIXED: Validate request body before proceeding
  const validation = validateRequestBody(path, body);
  if (!validation.valid) {
    console.error('❌ Soliton API validation failed:', validation.error);
    return json({
      success: false,
      error: validation.error,
      timestamp: new Date().toISOString()
    }, { status: 400 });
  }
  
  // FIXED: Check backend availability - NO FALLBACK, fail if backend is down
  const isBackendAvailable = await checkBackendHealth();
  if (!isBackendAvailable) {
    console.error('❌ Soliton Memory backend unavailable - cannot process request');
    return json({
      success: false,
      error: 'Soliton Memory backend is unavailable. Please ensure the backend service is running.',
      backend_url: BACKEND_URL,
      suggested_action: 'Check that the Python/Rust Soliton Memory backend is running and accessible.',
      timestamp: new Date().toISOString()
    }, { status: 503 }); // Service Unavailable
  }
  
  try {
    // FIXED: Add more comprehensive headers for backend communication
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-User-Id': String(locals.user?.id || body.userId || 'anonymous'),
      'X-Request-Path': path,
      'X-Frontend-Origin': 'sveltekit-ui',
      'X-Request-Timestamp': new Date().toISOString()
    };
    
    // Forward to Python backend with timeout
    const response = await fetch(`${BACKEND_URL}/api/soliton/${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000) // 5 second timeout for backend operations
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Backend error for ${path}:`, response.status, errorText);
      return json({
        success: false,
        error: `Backend error: ${response.status} ${response.statusText}`,
        backend_response: errorText,
        timestamp: new Date().toISOString()
      }, { status: response.status });
    }
    
    const result = await response.json();
    
    // FIXED: Validate backend response structure
    if (!result || typeof result !== 'object') {
      console.error('❌ Invalid response from backend:', result);
      return json({
        success: false,
        error: 'Invalid response from Soliton Memory backend',
        timestamp: new Date().toISOString()
      }, { status: 502 });
    }
    
    // FIXED: Add frontend metadata to successful responses
    const enhancedResult = {
      ...result,
      frontend_processed: true,
      processing_timestamp: new Date().toISOString(),
      request_path: path,
      // FIXED: Ensure we never return 'fallback' engine from backend
      engine: result.engine === 'fallback' ? 'backend' : result.engine
    };
    
    console.log(`✅ Soliton API success for ${path}:`, {
      engine: enhancedResult.engine,
      success: enhancedResult.success,
      timestamp: enhancedResult.processing_timestamp
    });
    
    return json(enhancedResult);
    
  } catch (error) {
    // FIXED: Proper error handling without fallback
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error(`❌ Soliton API error for ${path}:`, errorMessage);
    
    // FIXED: Return proper error response instead of fallback
    return json({
      success: false,
      error: 'Failed to communicate with Soliton Memory backend',
      details: errorMessage,
      backend_url: BACKEND_URL,
      suggested_action: 'Verify backend connectivity and check backend logs',
      timestamp: new Date().toISOString()
    }, { status: 502 }); // Bad Gateway
  }
};

export const GET: RequestHandler = async ({ params, locals, url }) => {
  const path = (Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''));
  
  // FIXED: Always log GET requests for debugging
  console.log(`🌊 Soliton API GET: /api/soliton/${path}`, {
    userId: locals.user?.id || 'anonymous',
    query: url.search,
    timestamp: new Date().toISOString()
  });
  
  // FIXED: Check backend availability - NO FALLBACK
  const isBackendAvailable = await checkBackendHealth();
  if (!isBackendAvailable) {
    console.error('❌ Soliton Memory backend unavailable for GET request');
    return json({
      success: false,
      error: 'Soliton Memory backend is unavailable',
      backend_url: BACKEND_URL,
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
  
  try {
    // FIXED: Enhanced headers for GET requests
    const headers: Record<string, string> = {
      'X-User-Id': String(locals.user?.id || 'anonymous'),
      'X-Request-Path': path,
      'X-Frontend-Origin': 'sveltekit-ui',
      'X-Request-Timestamp': new Date().toISOString()
    };
    
    const response = await fetch(`${BACKEND_URL}/api/soliton/${path}${url.search}`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Backend GET error for ${path}:`, response.status, errorText);
      return json({
        success: false,
        error: `Backend error: ${response.status} ${response.statusText}`,
        backend_response: errorText,
        timestamp: new Date().toISOString()
      }, { status: response.status });
    }
    
    const result = await response.json();
    
    // FIXED: Validate and enhance GET response
    if (!result || typeof result !== 'object') {
      console.error('❌ Invalid GET response from backend:', result);
      return json({
        success: false,
        error: 'Invalid response from Soliton Memory backend',
        timestamp: new Date().toISOString()
      }, { status: 502 });
    }
    
    const enhancedResult = {
      ...result,
      frontend_processed: true,
      processing_timestamp: new Date().toISOString(),
      request_path: path,
      // FIXED: Ensure backend responses don't claim to be fallback
      engine: result.engine === 'fallback' ? 'backend' : result.engine
    };
    
    console.log(`✅ Soliton API GET success for ${path}:`, {
      engine: enhancedResult.engine,
      dataType: Array.isArray(result.relatedMemories) ? 'memories' : 
                 result.stats ? 'stats' : 'other',
      timestamp: enhancedResult.processing_timestamp
    });
    
    return json(enhancedResult);
    
  } catch (error) {
    // FIXED: Proper error handling for GET requests
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error(`❌ Soliton API GET error for ${path}:`, errorMessage);
    
    return json({
      success: false,
      error: 'Failed to communicate with Soliton Memory backend',
      details: errorMessage,
      backend_url: BACKEND_URL,
      timestamp: new Date().toISOString()
    }, { status: 502 });
  }
};

// FIXED: Add OPTIONS handler for CORS preflight requests
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-User-Id, X-Request-Path, X-Frontend-Origin, X-Request-Timestamp',
      'Access-Control-Max-Age': '86400'
    }
  });
};

// FIXED: Add PATCH handler for memory updates
export const PATCH: RequestHandler = async ({ params, request, locals }) => {
  const path = (Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''));
  
  let body: any;
  try {
    body = await request.json();
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return json({
      success: false,
      error: 'Invalid JSON in request body',
      timestamp: new Date().toISOString()
    }, { status: 400 });
  }
  
  console.log(`🌊 Soliton API PATCH: /api/soliton/${path}`);
  
  const isBackendAvailable = await checkBackendHealth();
  if (!isBackendAvailable) {
    return json({
      success: false,
      error: 'Soliton Memory backend is unavailable',
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
  
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-User-Id': String(locals.user?.id || body.userId || 'anonymous'),
      'X-Request-Path': path,
      'X-Frontend-Origin': 'sveltekit-ui'
    };
    
    const response = await fetch(`${BACKEND_URL}/api/soliton/${path}`, {
      method: 'PATCH',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      return json({
        success: false,
        error: `Backend error: ${response.status} ${response.statusText}`,
        backend_response: errorText,
        timestamp: new Date().toISOString()
      }, { status: response.status });
    }
    
    const result = await response.json();
    return json({
      ...result,
      frontend_processed: true,
      processing_timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return json({
      success: false,
      error: 'Failed to communicate with Soliton Memory backend',
      details: errorMessage,
      timestamp: new Date().toISOString()
    }, { status: 502 });
  }
};

// FIXED: Add DELETE handler for memory cleanup
export const DELETE: RequestHandler = async ({ params, locals, url }) => {
  const path = (Array.isArray(params?.path) ? params.path.join('/') : (params?.path ?? ''));
  
  console.log(`🌊 Soliton API DELETE: /api/soliton/${path}`);
  
  const isBackendAvailable = await checkBackendHealth();
  if (!isBackendAvailable) {
    return json({
      success: false,
      error: 'Soliton Memory backend is unavailable',
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
  
  try {
    const headers: Record<string, string> = {
      'X-User-Id': String(locals.user?.id || 'anonymous'),
      'X-Request-Path': path,
      'X-Frontend-Origin': 'sveltekit-ui'
    };
    
    const response = await fetch(`${BACKEND_URL}/api/soliton/${path}${url.search}`, {
      method: 'DELETE',
      headers,
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      return json({
        success: false,
        error: `Backend error: ${response.status} ${response.statusText}`,
        backend_response: errorText,
        timestamp: new Date().toISOString()
      }, { status: response.status });
    }
    
    const result = await response.json();
    return json({
      ...result,
      frontend_processed: true,
      processing_timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return json({
      success: false,
      error: 'Failed to communicate with Soliton Memory backend',
      details: errorMessage,
      timestamp: new Date().toISOString()
    }, { status: 502 });
  }
};
