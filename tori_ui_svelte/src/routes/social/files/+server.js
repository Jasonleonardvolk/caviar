import { json } from '@sveltejs/kit';
import { readdir } from 'fs/promises';
import { join } from 'path';

export async function GET() {
  try {
    const socialPath = join(process.cwd(), 'static', 'social');
    
    const tiktokFiles = await readdir(join(socialPath, 'tiktok')).catch(() => []);
    const snapFiles = await readdir(join(socialPath, 'snap')).catch(() => []);
    
    return json({
      tiktok: tiktokFiles.filter(f => f.endsWith('.mp4')),
      snap: snapFiles.filter(f => f.endsWith('.mp4'))
    });
  } catch (error) {
    return json({ tiktok: [], snap: [] });
  }
}
