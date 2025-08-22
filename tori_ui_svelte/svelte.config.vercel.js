import adapterNode from '@sveltejs/adapter-node';
import adapterAuto from '@sveltejs/adapter-auto';
import adapterVercel from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// Detect deployment target
const deployTarget = process.env.DEPLOY_TARGET || process.env.VERCEL ? 'vercel' : 
                    process.env.BUILD_TARGET === 'node' ? 'node' : 'auto';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	
	kit: {
		adapter: deployTarget === 'vercel' ? adapterVercel({
			// Vercel adapter configuration
			runtime: 'nodejs20.x',
			regions: ['iad1'], // US East by default
			maxDuration: 10, // 10 seconds for API routes
		}) : deployTarget === 'node' ? adapterNode() : adapterAuto(),
		
		// Ensure consistent paths
		files: {
			routes: 'src/routes',
			lib: 'src/lib'
		}
	}
};

export default config;