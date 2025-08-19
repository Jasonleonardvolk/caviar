import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname } from 'node:path';

const src = 'D:\\Dev\\kha\\config\\plans.json';
const dst = 'D:\\Dev\\kha\\frontend\\static\\config\\plans.json';

function sha256(buf) { return createHash('sha256').update(buf).digest('hex'); }

const a = await readFile(src);
let b; try { b = await readFile(dst); } catch {}
if (b && sha256(a) === sha256(b)) {
  console.log('✔ plans.json already in sync');
} else {
  await mkdir(dirname(dst), { recursive: true });
  await writeFile(dst, a);
  console.log('🔄 plans.json synced → frontend/static/config/plans.json');
}