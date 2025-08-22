import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import manifest from '../../../../assets/3d/luxury/ASSET_MANIFEST.json';

export const GET: RequestHandler = async ({ url }) => {
  // Optional filtering by category
  const category = url.searchParams.get('category');
  
  let items = manifest;
  
  if (category) {
    items = manifest.filter((asset: any) => asset.category === category);
  }
  
  return json({
    ok: true,
    count: items.length,
    items,
    categories: [...new Set(manifest.map((a: any) => a.category))],
    licenses: [...new Set(manifest.map((a: any) => a.license))]
  });
};