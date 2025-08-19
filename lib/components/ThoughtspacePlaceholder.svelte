<script lang="ts">
	import { onMount } from 'svelte';
	import { conceptMesh } from '$lib/stores/conceptMesh';
	import { ghostState } from '$lib/stores/ghostPersona';

	let canvasContainer: HTMLDivElement;
	let statusMessage = 'Initializing thoughtspace...';

	onMount(() => {
		statusMessage = 'Ready for holographic projection';
		
		// Listen for concept updates
		const unsubscribe = conceptMesh.subscribe(mesh => {
			const nodeCount = Object.keys(mesh.nodes).length;
			const activeCount = mesh.activeNodes.length;
			const entangledCount = Object.values(mesh.nodes).filter(n => n.entangled).length;
			
			statusMessage = `${nodeCount} concepts • ${activeCount} active • ${entangledCount} entangled`;
		});

		return unsubscribe;
	});

	function getPhaseColor(signature: string): string {
		switch (signature) {
			case 'resonance': return '#06b6d4';
			case 'chaos': return '#dc2626';
			case 'drift': return '#f59e0b';
			case 'stable': return '#059669';
			default: return '#6b7280';
		}
	}

	$: phaseSignature = getPhaseSignature($ghostState.coherence, $ghostState.entropy, $ghostState.drift);

	function getPhaseSignature(coherence: number, entropy: number, drift: number): string {
		if (coherence > 0.8 && entropy < 0.3) return 'resonance';
		if (coherence < 0.3 && entropy > 0.8) return 'chaos';
		if (Math.abs(drift) > 0.5) return 'drift';
		if (coherence > 0.6 && entropy < 0.6) return 'stable';
		return 'neutral';
	}
</script>

<section class="tori-panel h-full flex flex-col">
	<header class="flex items-center justify-between mb-4">
		<div class="flex items-center space-x-2">
			<h2 class="text-lg font-bold text-tori-primary">🧠 Thoughtspace</h2>
			<div 
				class="w-3 h-3 rounded-full animate-pulse"
				style="background-color: {getPhaseColor(phaseSignature)}"
				title="Phase: {phaseSignature}"
			></div>
		</div>
		<div class="text-xs text-slate-400">Phase {phaseSignature}</div>
	</header>

	<div class="flex-1 relative bg-slate-900 rounded-lg overflow-hidden">
		<!-- Canvas Container - Ready for Three.js -->
		<div 
			bind:this={canvasContainer}
			class="w-full h-full flex items-center justify-center"
			id="thoughtspace-canvas"
		>
			<!-- Placeholder visualization -->
			<div class="text-center space-y-4">
				<div class="relative">
					<!-- Central Phase Ring -->
					<div 
						class="w-32 h-32 rounded-full border-4 border-opacity-60 animate-pulse"
						style="border-color: {getPhaseColor(phaseSignature)}"
					>
						<div class="w-full h-full flex items-center justify-center">
							<span class="text-lg">🌌</span>
						</div>
					</div>
					
					<!-- Concept Nodes (mock) -->
					{#each Object.values($conceptMesh.nodes).slice(0, 6) as node, i}
						<div 
							class="absolute w-4 h-4 rounded-full border-2 border-tori-primary bg-tori-primary/20 animate-pulse"
							style="
								top: {50 + 30 * Math.cos(i * Math.PI / 3)}%; 
								left: {50 + 30 * Math.sin(i * Math.PI / 3)}%;
								transform: translate(-50%, -50%);
								animation-delay: {i * 0.2}s;
							"
							title={node.label}
						>
							{#if node.entangled}
								<div class="absolute -inset-2 rounded-full bg-yellow-400/30 animate-ping"></div>
							{/if}
						</div>
					{/each}

					<!-- Ghost Aura -->
					{#if $ghostState.activePersona}
						<div 
							class="absolute inset-0 rounded-full border-2 border-opacity-40 animate-spin"
							style="
								border-color: {$ghostState.personas[$ghostState.activePersona]?.wavelength === 520 ? '#059669' : 
											   $ghostState.personas[$ghostState.activePersona]?.wavelength === 450 ? '#7c3aed' : 
											   $ghostState.personas[$ghostState.activePersona]?.wavelength === 620 ? '#dc2626' : '#06b6d4'};
								animation-duration: 8s;
							"
						></div>
					{/if}
				</div>

				<div class="space-y-2">
					<p class="text-sm text-slate-300">Ready for 3D holographic rendering</p>
					<p class="text-xs text-slate-500">{statusMessage}</p>
				</div>
			</div>
		</div>

		<!-- Phase Metrics Overlay -->
		<div class="absolute bottom-4 left-4 bg-slate-800/80 rounded-lg p-3 text-xs">
			<div class="grid grid-cols-3 gap-3">
				<div>
					<div class="text-slate-400">Coherence</div>
					<div class="text-white font-mono">{$ghostState.coherence.toFixed(2)}</div>
				</div>
				<div>
					<div class="text-slate-400">Entropy</div>
					<div class="text-white font-mono">{$ghostState.entropy.toFixed(2)}</div>
				</div>
				<div>
					<div class="text-slate-400">Drift</div>
					<div class="text-white font-mono">{$ghostState.drift.toFixed(2)}</div>
				</div>
			</div>
		</div>

		<!-- Active Ghost Indicator -->
		{#if $ghostState.activePersona}
			<div class="absolute top-4 right-4 bg-slate-800/80 rounded-lg p-2 text-xs">
				<div class="flex items-center space-x-2">
					<div 
						class="w-2 h-2 rounded-full animate-pulse"
						style="background-color: {$ghostState.personas[$ghostState.activePersona]?.wavelength === 520 ? '#059669' : 
											   $ghostState.personas[$ghostState.activePersona]?.wavelength === 450 ? '#7c3aed' : 
											   $ghostState.personas[$ghostState.activePersona]?.wavelength === 620 ? '#dc2626' : '#06b6d4'}"
					></div>
					<span class="text-white">
						👻 {$ghostState.personas[$ghostState.activePersona].name}
					</span>
				</div>
			</div>
		{/if}
	</div>
</section>