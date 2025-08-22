// D:\Dev\kha\tori_ui_svelte\src\routes\+page.server.ts
import { redirect } from '@sveltejs/kit';

export const prerender = false;

export const load = () => {
  // Always land on the iRis HUD, never the legacy TORI splash.
  throw redirect(307, '/hologram');
};