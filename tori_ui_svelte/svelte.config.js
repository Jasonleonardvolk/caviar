import adapterNode from '@sveltejs/adapter-node';
import adapterAuto from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// Use node adapter for production builds
const useNode = process.env.BUILD_TARGET === 'node';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	
	kit: {
		adapter: useNode ? adapterNode() : adapterAuto(),
		
		// Ensure consistent paths
		files: {
			routes: 'src/routes',
			lib: 'src/lib'
		}
	}
};

export default config;