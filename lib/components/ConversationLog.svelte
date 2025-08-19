<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { toriApi, type ConversationMessage } from '$lib/services/api';
	import { ghostState, emergeGhost } from '$lib/stores/ghostPersona';
	import { conceptMesh, addConceptDiff, activateConcept } from '$lib/stores/conceptMesh';

	let conversation: ConversationMessage[] = [];
	let newMessage = '';
	let loading = false;
	let websocket: WebSocket | null = null;

	onMount(async () => {
		// Load initial conversation
		conversation = await toriApi.getConversationLogs();
		
		// Initialize WebSocket connection for real-time updates
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const wsUrl = `${protocol}//${window.location.host}/ws/conversation`;
		
		websocket = new WebSocket(wsUrl);
		
		websocket.onmessage = (event) => {
			const message = JSON.parse(event.data);
			
			if (message.type === 'conversation-update') {
				conversation = [...conversation, message.data];
				
				// Process concepts from new message
				if (message.data.concepts && message.data.concepts.length > 0) {
					addConceptDiff({
						type: 'chat',
						title: `Chat Message: ${message.data.author}`,
						concepts: message.data.concepts,
						summary: message.data.content?.substring(0, 100) + '...',
						metadata: {
							author: message.data.author,
							timestamp: message.data.timestamp,
							messageId: message.data.id
						}
					});
				}
			}
		};
		
		websocket.onerror = (error) => {
			console.error('WebSocket error:', error);
		};
		
		websocket.onclose = () => {
			console.log('WebSocket connection closed');
		};
	});

	onDestroy(() => {
		if (websocket) {
			websocket.close();
		}
	});

	async function sendMessage() {
		if (!newMessage.trim() || loading) return;
		
		loading = true;
		
		try {
			const message = await toriApi.sendMessage(newMessage);
			conversation = [...conversation, message];
			
			// Process concepts from response
			if (message.concepts && message.concepts.length > 0) {
				addConceptDiff({
					type: 'chat',
					title: `TORI Response`,
					concepts: message.concepts,
					summary: message.content?.substring(0, 100) + '...',
					metadata: {
						author: 'TORI',
						timestamp: message.timestamp,
						messageId: message.id,
						userMessage: newMessage
					}
				});
				
				// Trigger ghost emergence based on response concepts
				const dominantConcept = message.concepts[0];
				if (dominantConcept) {
					activateConcept(dominantConcept);
					emergeGhost(dominantConcept);
				}
			}
			
			newMessage = '';
		} catch (error) {
			console.error('Failed to send message:', error);
		} finally {
			loading = false;
		}
	}

	function handleKeyDown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}

	// Auto-scroll to bottom when new messages arrive
	let chatContainer: HTMLElement;
	$: if (chatContainer && conversation) {
		setTimeout(() => {
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}, 100);
	}
</script>

<div class="flex flex-col h-full bg-gray-50">
	<!-- Chat header -->
	<div class="flex-shrink-0 bg-white border-b border-gray-200 px-6 py-4">
		<div class="flex items-center justify-between">
			<div>
				<h2 class="text-lg font-medium text-gray-900">Conversation with TORI</h2>
				<p class="text-sm text-gray-500">
					{conversation.length} messages • 
					{$conceptMesh.length} concept diffs processed
				</p>
			</div>
			
			<!-- Connection status -->
			<div class="flex items-center space-x-2">
				<div class="w-2 h-2 rounded-full {websocket && websocket.readyState === WebSocket.OPEN ? 'bg-green-500' : 'bg-red-500'}"></div>
				<span class="text-xs text-gray-500">
					{websocket && websocket.readyState === WebSocket.OPEN ? 'Connected' : 'Disconnected'}
				</span>
			</div>
		</div>
	</div>

	<!-- Chat messages -->
	<div class="flex-1 overflow-y-auto px-6 py-4 space-y-4" bind:this={chatContainer}>
		{#each conversation as message (message.id)}
			<div class="flex {message.author === 'user' ? 'justify-end' : 'justify-start'}">
				<div class="max-w-2xl {message.author === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-900'} rounded-lg px-4 py-3 shadow-sm border">
					<!-- Message header -->
					<div class="flex items-center justify-between mb-2">
						<div class="flex items-center space-x-2">
							<span class="text-sm font-medium">
								{message.author === 'user' ? 'You' : 'TORI'}
							</span>
							
							{#if message.ghost}
								<span class="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs">
									👻 {message.ghost}
								</span>
							{/if}
						</div>
						
						<span class="text-xs opacity-70">
							{new Date(message.timestamp).toLocaleTimeString()}
						</span>
					</div>

					<!-- Message content -->
					<div class="text-sm leading-relaxed">
						{message.content}
					</div>

					<!-- Concepts -->
					{#if message.concepts && message.concepts.length > 0}
						<div class="mt-3 pt-3 border-t border-gray-200 border-opacity-20">
							<div class="flex flex-wrap gap-1">
								{#each message.concepts as concept}
									<button
										on:click={() => activateConcept(concept)}
										class="px-2 py-1 bg-gray-100 {message.author === 'user' ? 'bg-blue-500 bg-opacity-20 text-blue-100' : 'text-gray-600'} rounded text-xs hover:bg-opacity-30 transition-colors"
									>
										{concept}
									</button>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Processing indicator -->
					{#if message.processing}
						<div class="mt-2 flex items-center space-x-2 text-xs opacity-70">
							<div class="w-2 h-2 bg-current rounded-full animate-pulse"></div>
							<span>Processing...</span>
						</div>
					{/if}
				</div>
			</div>
		{/each}

		<!-- Loading indicator -->
		{#if loading}
			<div class="flex justify-start">
				<div class="bg-white text-gray-900 rounded-lg px-4 py-3 shadow-sm border">
					<div class="flex items-center space-x-2">
						<div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
						<div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
						<div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
						<span class="text-sm text-gray-500 ml-2">TORI is thinking...</span>
					</div>
				</div>
			</div>
		{/if}
	</div>

	<!-- Message input -->
	<div class="flex-shrink-0 bg-white border-t border-gray-200 px-6 py-4">
		<div class="flex space-x-4">
			<div class="flex-1">
				<textarea
					bind:value={newMessage}
					on:keydown={handleKeyDown}
					placeholder="Type your message to TORI..."
					class="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
					rows="2"
					disabled={loading}
				></textarea>
			</div>
			
			<button
				on:click={sendMessage}
				disabled={!newMessage.trim() || loading}
				class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
			>
				{loading ? 'Sending...' : 'Send'}
			</button>
		</div>
		
		<!-- Input tips -->
		<div class="mt-2 flex items-center justify-between">
			<div class="text-xs text-gray-500">
				Press Enter to send • Shift+Enter for new line
			</div>
			
			<div class="text-xs text-gray-500">
				Concepts will be automatically extracted and visualized
			</div>
		</div>
	</div>
</div>
