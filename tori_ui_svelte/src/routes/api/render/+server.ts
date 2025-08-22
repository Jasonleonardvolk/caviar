import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

export const POST = async ({ request, fetch }) => {
  try {
    const payload = await request.json();
    const url = `${env.IRIS_SERVER_ASSIST_URL ?? 'http://127.0.0.1:7401'}/api/penrose/solve`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) return json({ ok:false, reason:'server_assist_failed' }, { status: 502 });

    const data = await resp.json();
    return json({ ok:true, data });
  } catch (e) {
    return json({ ok:false, reason:'proxy_exception' }, { status: 502 });
  }
};