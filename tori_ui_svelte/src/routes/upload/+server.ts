import { json } from '@sveltejs/kit';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const LOCAL_UPLOAD_DIR = process.env.LOCAL_UPLOAD_DIR || 'var/uploads';

export const POST = async ({ request }) => {
  await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });

  const ctype = request.headers.get('content-type') || '';
  let buf: Buffer;
  let filename = `upload-${Date.now()}.bin`;

  if (ctype.startsWith('multipart/form-data')) {
    const form = await request.formData();
    const file: any = form.get('file');
    if (!file?.arrayBuffer) return new Response('No file', { status: 400 });
    buf = Buffer.from(await file.arrayBuffer());
    filename = file.name || filename;
  } else {
    buf = Buffer.from(await request.arrayBuffer());
  }

  const safe = filename.replace(/[^\w.\-]+/g, '_');
  const full = path.join(LOCAL_UPLOAD_DIR, safe);
  await fs.writeFile(full, buf);

  return json({ ok: true, key: safe, size: buf.length });
};

export const GET = async () => {
  await fs.mkdir(LOCAL_UPLOAD_DIR, { recursive: true });
  const entries = await fs.readdir(LOCAL_UPLOAD_DIR, { withFileTypes: true });
  const files = entries.filter(e => e.isFile()).map(e => e.name);
  return json({ ok: true, files });
};