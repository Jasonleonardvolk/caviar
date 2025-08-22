import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Preprocess TypeScript, PostCSS, etc.
  preprocess: vitePreprocess(),
  
  kit: {
    adapter: adapter({
      out: 'build'            // Node server output (server+client)
    }),
    // Keep endpoints server-side; avoid forcing prerender
    prerender: { entries: [] }
  }
};

export default config;
