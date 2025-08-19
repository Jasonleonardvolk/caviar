import type { RequestHandler } from './$types';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';

const ROOT = path.resolve(process.cwd(), '..');
const TEMPLATES_DIR = path.join(ROOT, 'exports', 'templates');

export const GET: RequestHandler = async ({ params, url }) => {
  const raw = params.name || '';
  const name = raw.replace(/[\/\\]+/g, ''); // sanitize
  const filePath = path.join(TEMPLATES_DIR, name);
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    return new Response('Not found', { status: 404 });
  }

  const asAttachment = url.searchParams.get('download') === '1';
  const headers = new Headers({
    'Content-Type': name.toLowerCase().endsWith('.glb') ? 'model/gltf-binary' : 'application/octet-stream',
    'Content-Length': String(fs.statSync(filePath).size),
    'Content-Disposition': `${asAttachment ? 'attachment' : 'inline'}; filename="${name}"`
  });

  // Node -> Web stream
  const webStream = Readable.toWeb(fs.createReadStream(filePath)) as unknown as ReadableStream;
  return new Response(webStream, { headers });
};