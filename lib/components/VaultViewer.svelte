<script lang="ts">
	import { onMount } from 'svelte';
	import { toriApi, type VaultEntry } from '$lib/services/api';
	import { conceptMesh, addConcept } from '$lib/stores/conceptMesh';

	let vault: VaultEntry[] = [];
	let selectedEntry: VaultEntry | null = null;
	let searchQuery = '';
	let loading = false;

	onMount(async () => {
		loading = true;
		vault = await toriApi.getVaultEntries();
		loading = false;
	});

	$: filteredVault = vault.filter(entry => 
		entry.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
		entry.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
	);

	async function selectEntry(entry: VaultEntry) {
		if (entry.isSealed) {
			// Handle sealed entries specially
			if (confirm(`This memory is sealed: ${entry.sealReason || 'Protected content'}. Access anyway?`)) {
				selectedEntry = entry;
			}
		} else {
			selectedEntry = await toriApi.getVaultEntry(entry.id) || entry;
		}

		// Add concepts to mesh
		entry.conceptIds.forEach(conceptId => {
			addConcept({
				id: conceptId,
				label: conceptId.replace(/-/g, ' '),
				position: { x: 0, y: 0, z: 0 },
				wavelength: 520,
				intensity: 0.7,
				connections: [],
				entangled: false,
				stability: 0.8,
				conceptType: 'vault-concept'
			});
		});
	}

	function formatDate(date: Date): string {
		return date.toLocaleDateString([], { 
			month: 'short', 
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	function getEntryIcon(entry: VaultEntry): string {
		if (entry.isSealed) return '🔒';
		if (entry.tags.includes('quantum')) return '⚛️';
		if (entry.tags.includes('ghost')) return '👻';
		if (entry.tags.includes('memory')) return '🧠';
		return '📄';
	}

	function getTagColor(tag: string): string {
		if (tag === 'sealed') return 'bg-red-900/30 text-red-400';
		if (tag === 'quantum') return 'bg-purple-900/30 text-purple-400';
		if (tag === 'ghost') return 'bg-indigo-900/30 text-indigo-400';
		if (tag === 'memory') return 'bg-green-900/30 text-green-400';
		return 'bg-slate-700 text-slate-300';
	}

	async function sealEntry(entry: VaultEntry) {
		const reason = prompt('Reason for sealing this memory:');
		if (reason) {
			const success = await toriApi.sealMemory(entry.conceptIds, reason);
			if (success) {
				entry.isSealed = true;
				entry.sealReason = reason;
				vault = vault; // Trigger reactivity
			}
		}
	}
</script>

<section class="tori-panel h-full flex flex-col">
	<header class="flex items-center justify-between mb-4">
		<h2 class="text-lg font-bold text-tori-primary">🗄️ Memory Vault</h2>
		<div class="text-sm text-slate-400">
			{vault.length} entries ({vault.filter(e => e.isSealed).length} sealed)
		</div>
	</header>

	<div class="mb-4">
		<input
			bind:value={searchQuery}
			placeholder="Search vault..."
			class="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400"
		/>
	</div>

	{#if loading}
		<div class="flex items-center justify-center py-8">
			<div class="animate-spin w-6 h-6 border-2 border-tori-primary border-t-transparent rounded-full"></div>
		</div>
	{:else}
		<div class="flex-1 overflow-y-auto space-y-2">
			{#each filteredVault as entry}
				<div 
					class="bg-slate-700/50 border border-slate-600 rounded-lg p-3 cursor-pointer hover:bg-slate-600/50 transition-colors {selectedEntry?.id === entry.id ? 'ring-2 ring-tori-primary' : ''}"
					on:click={() => selectEntry(entry)}
				>
					<div class="flex items-start justify-between mb-2">
						<div class="flex items-center space-x-2">
							<span class="text-lg">{getEntryIcon(entry)}</span>
							<h3 class="font-medium text-sm text-white">{entry.title}</h3>
						</div>
						<span class="text-xs text-slate-500">{formatDate(entry.createdAt)}</span>
					</div>

					{#if entry.isSealed}
						<div class="text-xs text-red-400 mb-2">
							🔒 SEALED: {entry.sealReason || 'Protected content'}
						</div>
					{:else}
						<p class="text-xs text-slate-300 mb-2 line-clamp-2">
							{entry.content.substring(0, 100)}...
						</p>
					{/if}

					<div class="flex flex-wrap gap-1">
						{#each entry.tags as tag}
							<span class="px-2 py-0.5 text-xs rounded {getTagColor(tag)}">
								{tag}
							</span>
						{/each}
					</div>
				</div>
			{/each}

			{#if filteredVault.length === 0}
				<div class="text-center py-8 text-slate-400">
					{searchQuery ? 'No entries match your search' : 'Vault is empty'}
				</div>
			{/if}
		</div>
	{/if}

	{#if selectedEntry}
		<div class="mt-4 border-t border-slate-600 pt-4">
			<div class="flex items-center justify-between mb-2">
				<h3 class="font-medium text-white">{selectedEntry.title}</h3>
				<div class="flex space-x-2">
					{#if !selectedEntry.isSealed}
						<button
							on:click={() => sealEntry(selectedEntry)}
							class="text-xs text-red-400 hover:text-red-300"
							title="Seal this memory"
						>
							🔒 Seal
						</button>
					{/if}
					<button
						on:click={() => selectedEntry = null}
						class="text-xs text-slate-400 hover:text-slate-300"
					>
						✕
					</button>
				</div>
			</div>
			
			{#if selectedEntry.isSealed}
				<div class="text-sm text-red-400 bg-red-900/20 rounded p-2">
					This memory is sealed for protection. Content access restricted.
				</div>
			{:else}
				<div class="text-sm text-slate-300 bg-slate-800/50 rounded p-2 max-h-32 overflow-y-auto">
					{selectedEntry.content}
				</div>
			{/if}

			<div class="mt-2 text-xs text-slate-500">
				Concepts: {selectedEntry.conceptIds.join(', ')}
			</div>
		</div>
	{/if}
</section>