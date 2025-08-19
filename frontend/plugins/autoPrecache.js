import { glob } from 'glob';
/**
 * Vite plugin to automatically generate a precache manifest for Service Worker
 * Scans specified globs and emits a JSON manifest with all matching files
 */
export default function autoPrecache(opts = {}) {
    const manifestName = opts.manifestName || 'precache.manifest.json';
    const include = opts.include || [
        'public/wasm/**/*.{wasm,js}',
        'public/hybrid/wgsl/**/*.wgsl',
        'public/assets/**/*.{ktx2,ktx,png,jpg,jpeg,webp}',
        'public/assets/**/*.{bin,json}',
        'public/assets/**/*.{mp4,webm,av1,ivf}',
    ];
    const extra = opts.extra || ['/'];
    const excludePatterns = opts.excludePatterns || [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/*.map'
    ];
    return {
        name: 'auto-precache',
        apply: 'build',
        enforce: 'post',
        async generateBundle(_, bundle) {
            const root = process.cwd();
            const files = new Set(extra);
            // Scan all include patterns
            for (const pattern of include) {
                const entries = glob.sync(pattern, {
                    nodir: true,
                    cwd: root,
                    absolute: false,
                    ignore: excludePatterns
                });
                for (const entry of entries) {
                    // Convert to web path
                    let webPath = entry.replace(/\\/g, '/');
                    // Remove 'public/' prefix if present
                    if (webPath.startsWith('public/')) {
                        webPath = webPath.substring(6);
                    }
                    // Ensure leading slash
                    if (!webPath.startsWith('/')) {
                        webPath = '/' + webPath;
                    }
                    files.add(webPath);
                }
            }
            // Add generated bundle files
            for (const [fileName, asset] of Object.entries(bundle)) {
                // Skip source maps and other non-cacheable files
                if (fileName.endsWith('.map') || fileName.endsWith('.html')) {
                    continue;
                }
                // Add JS, CSS, and other assets
                if (fileName.endsWith('.js') ||
                    fileName.endsWith('.css') ||
                    fileName.endsWith('.wasm') ||
                    fileName.endsWith('.wgsl')) {
                    files.add(`/${fileName}`);
                }
            }
            // Sort for consistent output
            const sortedFiles = Array.from(files).sort();
            // Generate manifest
            const manifest = {
                version: new Date().toISOString(),
                files: sortedFiles,
                total: sortedFiles.length
            };
            const payload = JSON.stringify(manifest, null, 2);
            // Emit the manifest file
            this.emitFile({
                type: 'asset',
                fileName: manifestName,
                source: payload
            });
            console.log(`[auto-precache] Generated ${manifestName} with ${files.size} entries`);
            // Log some stats in development
            if (process.env.NODE_ENV !== 'production') {
                const stats = {
                    wasm: sortedFiles.filter(f => f.endsWith('.wasm')).length,
                    shaders: sortedFiles.filter(f => f.endsWith('.wgsl')).length,
                    textures: sortedFiles.filter(f => /\.(ktx2?|png|jpg|jpeg|webp)$/.test(f)).length,
                    videos: sortedFiles.filter(f => /\.(mp4|webm|av1|ivf)$/.test(f)).length,
                    scripts: sortedFiles.filter(f => f.endsWith('.js')).length,
                    styles: sortedFiles.filter(f => f.endsWith('.css')).length,
                };
                console.log('[auto-precache] File breakdown:', stats);
            }
        },
    };
}
/**
 * After-build plugin to sweep hashed assets from dist
 * This ensures fingerprinted files are included in the cache
 */
export function afterBuildPrecache(opts = {}) {
    const manifestName = opts.manifestName ?? 'precache.manifest.json';
    const includeExt = [
        '.js', '.css', '.wasm', '.wgsl', '.ktx2', '.ktx',
        '.webm', '.mp4', '.av1', '.ivf', '.json', '.bin'
    ].map(e => e.toLowerCase());
    const extra = new Set(opts.extra ?? ['/']);
    let base = '/';
    const files = new Set();
    return {
        name: 'after-build-precache',
        apply: 'build',
        enforce: 'post',
        configResolved(config) {
            base = config.base || '/';
        },
        generateBundle(_, bundle) {
            // Collect all generated files
            for (const [fileName, asset] of Object.entries(bundle)) {
                const lower = fileName.toLowerCase();
                const shouldInclude = includeExt.some(ext => lower.endsWith(ext));
                if (shouldInclude) {
                    // Ensure proper path format
                    const path = (base.endsWith('/') ? base : base + '/') + fileName;
                    files.add(path);
                }
            }
            // Add extra files
            for (const extraFile of extra) {
                files.add(extraFile);
            }
            // Generate manifest
            const manifest = {
                version: new Date().toISOString(),
                files: Array.from(files).sort(),
                total: files.size
            };
            const payload = JSON.stringify(manifest, null, 2);
            // Emit manifest
            this.emitFile({
                type: 'asset',
                fileName: manifestName,
                source: payload
            });
            console.log(`[after-build-precache] Generated ${manifestName} with ${files.size} entries`);
        },
    };
}
