import fs from 'node:fs';
import path from 'node:path';
import { json } from '@sveltejs/kit';

const ROOT = path.resolve(process.cwd(), '..');

function resolveInput(p: string) {
  // absolute → as-is; relative → under project root
  return path.isAbsolute(p) ? p : path.join(ROOT, p);
}

export const POST = async ({ request }) => {
  const raw = await request.json().catch(() => ({}));
  const inputReq = raw?.input ?? path.join('data','concept_graph.json');
  const input = resolveInput(String(inputReq));

  if (!fs.existsSync(input)) {
    return json({ ok:false, error:`Input not found: ${input}` }, { status: 400 });
  }

  // Continue with existing logic (invoke tsx exporter, ktx2 script, zip option)
  // For now, returning a simple success response to indicate the path validation works
  return json({ 
    ok: true, 
    message: 'Export endpoint ready',
    input: input,
    exists: true
  });
};