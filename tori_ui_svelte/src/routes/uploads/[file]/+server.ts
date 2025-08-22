import type { RequestHandler } from './$types';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const GET: RequestHandler = async ({ params }) => {
  const file = params.file;
  const full = path.join(LOCAL_UPLOAD_DIR, file);
  try {
    const data = await fs.readFile(full);
    const ext = path.extname(file).toLowerCase();
    const mime =
      ext === '.pdf' ? 'application/pdf' :
      ext === '.txt' ? 'text/plain; charset=utf-8' :
      (ext === '.png' ? 'image/png' :
      ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
      'application/octet-stream');
    return new Response(data, { status: 200, headers: { 'content-type': mime } });
  } catch {
    return new Response('Not found', { status: 404 });
  }
};