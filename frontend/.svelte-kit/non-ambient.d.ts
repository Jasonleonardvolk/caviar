
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	export interface AppTypes {
		RouteId(): "/" | "/account" | "/account/manage" | "/api" | "/api/billing" | "/api/billing/checkout" | "/api/billing/portal" | "/api/billing/webhook" | "/api/templates" | "/api/templates/export" | "/api/templates/file" | "/api/templates/file/[name]" | "/api/templates/upload" | "/health" | "/health/ping" | "/health/raw" | "/hologram" | "/pricing" | "/publish" | "/templates" | "/templates/upload";
		RouteParams(): {
			"/api/templates/file/[name]": { name: string }
		};
		LayoutParams(): {
			"/": { name?: string };
			"/account": Record<string, never>;
			"/account/manage": Record<string, never>;
			"/api": { name?: string };
			"/api/billing": Record<string, never>;
			"/api/billing/checkout": Record<string, never>;
			"/api/billing/portal": Record<string, never>;
			"/api/billing/webhook": Record<string, never>;
			"/api/templates": { name?: string };
			"/api/templates/export": Record<string, never>;
			"/api/templates/file": { name?: string };
			"/api/templates/file/[name]": { name: string };
			"/api/templates/upload": Record<string, never>;
			"/health": Record<string, never>;
			"/health/ping": Record<string, never>;
			"/health/raw": Record<string, never>;
			"/hologram": Record<string, never>;
			"/pricing": Record<string, never>;
			"/publish": Record<string, never>;
			"/templates": Record<string, never>;
			"/templates/upload": Record<string, never>
		};
		Pathname(): "/" | "/account" | "/account/manage" | "/api" | "/api/billing" | "/api/billing/checkout" | "/api/billing/portal" | "/api/billing/webhook" | "/api/templates" | "/api/templates/export" | "/api/templates/file" | `/api/templates/file/${string}` & {} | "/api/templates/upload" | "/health" | "/health/ping" | "/health/raw" | "/hologram" | "/pricing" | "/publish" | "/templates" | "/templates/upload";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): "/assets/quilt/demo_5x9/index.json" | "/assets/quilt/demo_5x9/tile_0.ktx2" | "/assets/quilt/demo_5x9/tile_1.ktx2" | "/assets/quilt/demo_5x9/tile_2.ktx2" | "/config/plans.json" | "/dev/probe_webgpu_limits.html" | "/hybrid/wgsl/avatarShader.wgsl" | "/hybrid/wgsl/bitReversal.wgsl" | "/hybrid/wgsl/butterflyStage.wgsl" | "/hybrid/wgsl/fftShift.wgsl" | "/hybrid/wgsl/hybridWavefieldBlend.wgsl" | "/hybrid/wgsl/lenticularInterlace.wgsl" | "/hybrid/wgsl/lightFieldComposer.wgsl" | "/hybrid/wgsl/lightFieldComposerEnhanced.wgsl" | "/hybrid/wgsl/multiDepthWaveSynth.wgsl" | "/hybrid/wgsl/multiViewSynthesis.wgsl" | "/hybrid/wgsl/normalize.wgsl" | "/hybrid/wgsl/phaseOcclusion.wgsl" | "/hybrid/wgsl/propagation.wgsl" | "/hybrid/wgsl/topologicalOverlay.wgsl" | "/hybrid/wgsl/transpose.wgsl" | "/hybrid/wgsl/velocityField.wgsl" | "/hybrid/wgsl/wavefieldEncoder.wgsl" | "/hybrid/wgsl/wavefieldEncoder_optimized.wgsl" | "/manifest.webmanifest" | "/service-worker.js" | "/templates/index.json" | "/tests/quilt_display.html" | "/tests/schrodinger_bench.html" | "/tests/waveop_dashboard.html" | "/wasm/basis_transcoder.js" | "/wasm/basis_transcoder.wasm" | "/wasm/mesh_ops.wasm" | "/wasm/phase_core.wasm" | string & {};
	}
}