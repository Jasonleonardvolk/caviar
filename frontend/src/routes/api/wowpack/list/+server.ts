import type { RequestHandler } from './$types';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..'); // D:\Dev\kha (when running from frontend)
const OUT_DIR = path.join(ROOT, 'content', 'wowpack', 'output'); // encoded files live here
const IN_DIR  = path.join(ROOT, 'content', 'wowpack', 'input');  // ProRes masters live here

const BASES = [
  { id: 'holo_flux_loop',      title: 'HOLO FLUX' },
  { id: 'mach_lightfield',     title: 'MACH LIGHTFIELD' },
  { id: 'kinetic_logo_parade', title: 'KINETIC LOGO PARADE' }
];

// Most browsers: H264 MP4; then AV1 WebM; HEVC may work in Safari/iOS 26; .mov as last resort (often won't play).
const ALLOWED_EXTS = ['.mp4', '.webm', '.m4v', '.hevc', '.mov', '.mkv'];

function guessContentType(name: string) {
  const ext = path.extname(name).toLowerCase();
  if (ext === '.mp4' || ext === '.m4v') return 'video/mp4';
  if (ext === '.webm') return 'video/webm';
  if (ext === '.mov') return 'video/quicktime';
  if (ext === '.mkv') return 'video/x-matroska';
  if (ext === '.hevc') return 'video/mp4'; // container may vary; serve as mp4
  return 'application/octet-stream';
}

export const GET: RequestHandler = async () => {
  const items: Array<{
    id: string;
    title: string;
    files: Array<{ name: string; size: number; mtime: string; contentType: string }>;
    mastersPresent: boolean;
  }> = [];

  // Make sure directories exist (don't throw)
  const outExists = fs.existsSync(OUT_DIR);
  const inExists  = fs.existsSync(IN_DIR);

  for (const base of BASES) {
    const files: Array<{ name: string; size: number; mtime: string; contentType: string }> = [];
    if (outExists) {
      const dirents = await fsp.readdir(OUT_DIR, { withFileTypes: true }).catch(() => []);
      for (const d of dirents) {
        if (!d.isFile()) continue;
        const ext = path.extname(d.name).toLowerCase();
        if (!ALLOWED_EXTS.includes(ext)) continue;
        // match either prefix or exact base somewhere in the filename
        if (!(d.name.startsWith(base.id) || d.name.includes(`${base.id}`))) continue;
        const abs = path.join(OUT_DIR, d.name);
        const st = await fsp.stat(abs).catch(() => null);
        if (!st) continue;
        files.push({
          name: d.name,
          size: st.size,
          mtime: st.mtime.toISOString(),
          contentType: guessContentType(d.name)
        });
      }
      // sort by preferred playability: mp4 > webm > hevc > mov > mkv
      const order = ['.mp4','.m4v','.webm','.hevc','.mov','.mkv'];
      files.sort((a,b) => order.indexOf(path.extname(a.name).toLowerCase()) - order.indexOf(path.extname(b.name).toLowerCase()));
    }

    let mastersPresent = false;
    if (inExists) {
      // presence of ProRes master (may not be web-playable)
      mastersPresent = fs.existsSync(path.join(IN_DIR, `${base.id}.mov`));
    }

    items.push({ id: base.id, title: base.title, files, mastersPresent });
  }

  return new Response(JSON.stringify({
    root: ROOT,
    outputDir: OUT_DIR,
    inputDir: IN_DIR,
    items
  }), { headers: { 'content-type': 'application/json' }});
};