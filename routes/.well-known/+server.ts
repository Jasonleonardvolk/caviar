// Handle .well-known requests to silence 404 errors from browser extensions
export async function GET() {
  return new Response('{"status":"not implemented"}', {
    status: 404,
    headers: { 'Content-Type': 'application/json' }
  });
}
