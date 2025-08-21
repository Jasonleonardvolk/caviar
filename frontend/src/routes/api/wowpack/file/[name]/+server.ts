import type { RequestHandler } from './$types';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';

const ROOT = path.resolve(process.cwd(), '..');
const OUT_DIR = path.join(ROOT, 'content', 'wowpack', 'output');

function contentTypeFor(name: string) {
  const ext = path.extname(name).toLowerCase();
  if (ext === '.mp4' || ext === '.m4v') return 'video/mp4';
  if (ext === '.webm') return 'video/webm';
  if (ext === '.mov') return 'video/quicktime';
  if (ext === '.mkv') return 'video/x-matroska';
  if (ext === '.hevc') return 'video/mp4';
  return 'application/octet-stream';
}

export const GET: RequestHandler = async ({ params, url }) => {
  const raw = params.name || '';
  const safe = raw.replace(/[\/\\]+/g, ''); // basic sanitization
  const abs = path.join(OUT_DIR, safe);
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) {
    return new Response('Not found', { status: 404 });
  }
  const asDownload = url.searchParams.get('download') === '1';
  const headers = new Headers({
    'content-type': contentTypeFor(safe),
    'content-length': String(fs.statSync(abs).size),
    'content-disposition': `${asDownload ? 'attachment' : 'inline'}; filename="${safe}"`
  });
  const stream = Readable.toWeb(fs.createReadStream(abs)) as unknown as ReadableStream;
  return new Response(stream, { headers });
};