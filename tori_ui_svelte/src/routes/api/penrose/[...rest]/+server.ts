// D:\Dev\kha\tori_ui_svelte\src\routes\api\penrose\[...rest]\+server.ts
import { env } from '$env/dynamic/private';
import type { RequestHandler } from './$types';

const PENROSE_BASE = env.PENROSE_URL ?? 'http://127.0.0.1:7401';
const HOP_BY_HOP = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length', 'host'
]);

function makeInit(req: Request): RequestInit & { duplex?: 'half' } {
  const method = req.method.toUpperCase();
  const hasBody = !(method === 'GET' || method === 'HEAD');
  // clone & sanitize headers
  const headers = new Headers();
  req.headers.forEach((v, k) => { if (!HOP_BY_HOP.has(k.toLowerCase())) headers.set(k, v); });

  const init: RequestInit & { duplex?: 'half' } = { method, headers };
  if (hasBody) { init.body = req.body as any; init.duplex = 'half'; }
  return init;
}

async function handle({ request, params, fetch, url }: Parameters<RequestHandler>[0]) {
  const tail = params.rest ? `/${params.rest}` : '/';
  // preserve query string
  const target = `${PENROSE_BASE}${tail}${url.search}`;
  const init = makeInit(request);

  const resp = await fetch(target, init);
  // clone response headers; drop content-encoding to avoid double compression
  const out = new Headers();
  resp.headers.forEach((v, k) => { if (k.toLowerCase() !== 'content-encoding') out.set(k, v); });

  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: out });
}

export const GET:     RequestHandler = (e) => handle(e);
export const POST:    RequestHandler = (e) => handle(e);
export const PUT:     RequestHandler = (e) => handle(e);
export const PATCH:   RequestHandler = (e) => handle(e);
export const DELETE:  RequestHandler = (e) => handle(e);
export const OPTIONS: RequestHandler = (e) => handle(e);