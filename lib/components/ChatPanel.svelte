<!-- components/ChatPanel.svelte (Updated with suggestion support) -->
<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';
  import { addConceptDiff } from '$lib/stores/conceptMesh';
  import { updateUserStats } from '$lib/stores/user';
  import { ghostPersona, setMood } from '$lib/stores/ghostPersona';
  
  // Simple message structure
  type Message = { 
    id: string;
    sender: 'user' | 'ghost'; 
    text: string; 
    timestamp: Date;
    concepts?: string[];
  };
  
  let messages: Message[] = [];
  let userInput: string = "";
  let messagesContainer: HTMLDivElement;
  let isTyping = false;
  
  // Auto-scroll to bottom when messages update
  afterUpdate(() => {
    scrollToBottom();
  });
  
  onMount(() => {
    // Load saved messages
    loadMessagesFromMemory();
    
    // Add welcome message
    if (messages.length === 0) {
      addGhostMessage("Hello! 👻 I'm your TORI Ghost AI. I'm here to help you explore ideas, manage knowledge, and enhance your thinking. What would you like to discuss?");
    }
  });
  
  function scrollToBottom() {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }
  
  // Export function for suggestions to use
  export function sendSuggestionMessage(message: string) {
    userInput = message;
    sendMessage();
  }
  
  async function sendMessage() {
    const text = userInput.trim();
    if (!text || isTyping) return;
    
    // Add user's message to chat
    const userMessage: Message = {
      id: `msg_${Date.now()}_user`,
      sender: 'user',
      text,
      timestamp: new Date()
    };
    
    messages = [...messages, userMessage];
    
    // Clear input field
    userInput = "";
    
    // Show typing indicator
    isTyping = true;
    
    // Extract concepts from user message
    const concepts = extractConceptsFromMessage(text);
    
    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    // Generate AI response
    const responseText = await generateGhostResponse(text, concepts);
    
    // Add ghost response
    addGhostMessage(responseText, concepts);
    
    isTyping = false;
    
    // Create a ConceptDiff from this interaction
    if (concepts.length > 0) {
      addConceptDiff({
        type: 'chat',
        title: 'Conversation',
        concepts,
        summary: `Discussion about: ${concepts.join(', ')}`
      });
    }
    
    // Update user stats
    updateUserStats({
      conversationsCount: undefined // Will increment in update function
    });
    
    // Save messages to memory
    saveMessagesToMemory();
  }
  
  function addGhostMessage(text: string, concepts?: string[]) {
    const ghostMessage: Message = {
      id: `msg_${Date.now()}_ghost`,
      sender: 'ghost',
      text,
      timestamp: new Date(),
      concepts
    };
    
    messages = [...messages, ghostMessage];
    saveMessagesToMemory();
  }
  
  // Enhanced ghost response generator
  async function generateGhostResponse(userMessage: string, concepts: string[]): Promise<string> {
    const message = userMessage.toLowerCase();
    
    // Update ghost mood based on message content
    if (message.includes('help') || message.includes('explain')) {
      setMood('helpful');
    } else if (message.includes('complex') || message.includes('difficult')) {
      setMood('focused');
    } else if (message.includes('thanks') || message.includes('great')) {
      setMood('pleased');
    }
    
    // Contextual responses
    if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
      return "Hello! 👻 I'm delighted to meet you. I can help you with document analysis, concept exploration, and knowledge synthesis. What interests you today?";
    }
    
    if (message.includes('what can you do') || message.includes('help')) {
      return "I can help you in several ways: 📄 Analyze documents and extract key concepts, 💭 Explore ideas and their connections, 🧠 Build and navigate your knowledge graph, 💡 Suggest insights and connections, 🔍 Search through your accumulated knowledge. What would you like to try first?";
    }
    
    if (message.includes('document') || message.includes('upload') || message.includes('file')) {
      return "Great! You can upload documents using the panel on the right. I'll automatically extract key concepts and add them to your knowledge graph. This helps build a comprehensive understanding of your information landscape.";
    }
    
    if (message.includes('concept') || message.includes('idea')) {
      return `Interesting concept exploration! ${concepts.length > 0 ? `I notice you mentioned: ${concepts.join(', ')}. ` : ''}Let me help you dive deeper into these ideas and discover their connections to your existing knowledge.`;
    }
    
    if (message.includes('memory') || message.includes('remember')) {
      return "My memory system works differently than traditional storage. Instead of saving raw data, I build a living concept mesh that captures relationships and insights. This allows for more meaningful knowledge retrieval and synthesis.";
    }
    
    if (message.includes('think') || message.includes('analyze')) {
      return "Let's think through this together. I'll apply structured analysis while maintaining awareness of the broader context from your knowledge base. What specific aspect would you like to explore first?";
    }
    
    // Default intelligent response based on concepts
    if (concepts.length > 0) {
      const conceptList = concepts.join(', ');
      return `👻 I've analyzed your message and identified these key concepts: ${conceptList}. This adds valuable context to our conversation. How would you like to explore these ideas further?`;
    }
    
    // Fallback responses with personality
    const fallbackResponses = [
      "👻 That's fascinating! I'm processing this information and updating my understanding. Please tell me more about your thoughts on this.",
      "Intriguing perspective! I'm integrating this into my concept mesh. What implications do you see from this?",
      "I'm following your reasoning. This connects to several concepts in our shared knowledge space. Would you like me to explore those connections?",
      "👻 Your insight is valuable. I'm noting the conceptual patterns here. How does this relate to your broader objectives?",
      "This adds important context to our dialogue. I'm curious about your perspective on the deeper implications."
    ];
    
    return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
  }
  
  // Extract concepts from message (enhanced logic)
  function extractConceptsFromMessage(message: string): string[] {
    const text = message.toLowerCase();
    const concepts: string[] = [];
    
    // Technical concepts
    if (text.includes('ai') || text.includes('artificial intelligence')) concepts.push('AI');
    if (text.includes('machine learning') || text.includes('ml')) concepts.push('Machine Learning');
    if (text.includes('data') || text.includes('dataset')) concepts.push('Data');
    if (text.includes('algorithm')) concepts.push('Algorithm');
    if (text.includes('neural network')) concepts.push('Neural Networks');
    if (text.includes('deep learning')) concepts.push('Deep Learning');
    
    // Business concepts
    if (text.includes('strategy') || text.includes('strategic')) concepts.push('Strategy');
    if (text.includes('market') || text.includes('marketing')) concepts.push('Marketing');
    if (text.includes('finance') || text.includes('financial')) concepts.push('Finance');
    if (text.includes('project') || text.includes('management')) concepts.push('Project Management');
    if (text.includes('analysis') || text.includes('analyze')) concepts.push('Analysis');
    
    // Knowledge concepts
    if (text.includes('research') || text.includes('study')) concepts.push('Research');
    if (text.includes('learn') || text.includes('education')) concepts.push('Learning');
    if (text.includes('memory') || text.includes('knowledge')) concepts.push('Knowledge Management');
    if (text.includes('concept') || text.includes('idea')) concepts.push('Conceptual Thinking');
    
    // Process concepts
    if (text.includes('design') || text.includes('create')) concepts.push('Design');
    if (text.includes('develop') || text.includes('build')) concepts.push('Development');
    if (text.includes('test') || text.includes('evaluate')) concepts.push('Testing');
    if (text.includes('implement') || text.includes('deploy')) concepts.push('Implementation');
    
    return [...new Set(concepts)]; // Remove duplicates
  }
  
  // Allow pressing Enter to send the message
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }
  
  // Memory management
  function saveMessagesToMemory() {
    try {
      localStorage.setItem('tori-chat-messages', JSON.stringify(messages));
    } catch (error) {
      console.warn('Failed to save messages:', error);
    }
  }
  
  function loadMessagesFromMemory() {
    try {
      const saved = localStorage.getItem('tori-chat-messages');
      if (saved) {
        messages = JSON.parse(saved).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
      }
    } catch (error) {
      console.warn('Failed to load messages:', error);
    }
  }
  
  function clearChat() {
    if (confirm('Clear all chat messages?')) {
      messages = [];
      localStorage.removeItem('tori-chat-messages');
    }
  }
  
  // Format timestamp for display
  function formatTime(date: Date): string {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
</script>

<div class="flex flex-col h-full bg-tori-background">
  <!-- Chat header -->
  <div class="bg-white border-b border-gray-200 p-3 flex justify-between items-center">
    <div>
      <h2 class="font-semibold text-tori-text">Conversation with Ghost AI</h2>
      <p class="text-xs text-gray-500">Persona: {$ghostPersona.persona} • Mood: {$ghostPersona.mood}</p>
    </div>
    <button 
      class="text-xs text-gray-500 hover:text-red-600 transition-colors"
      on:click={clearChat}
      title="Clear chat"
    >
      🗑️ Clear
    </button>
  </div>
  
  <!-- Messages display area -->
  <div class="flex-1 overflow-y-auto p-4 space-y-3" bind:this={messagesContainer}>
    {#each messages as msg (msg.id)}
      <div class="flex {msg.sender === 'user' ? 'justify-end' : 'justify-start'}">
        <div class="max-w-[80%] {msg.sender === 'user' ? 'order-2' : 'order-1'}">
          <!-- Message bubble -->
          <div class="{msg.sender === 'user' ? 'bg-blue-100 border-blue-200' : 'bg-white border-gray-200'} 
                      border rounded-lg px-4 py-2 shadow-sm">
            
            <!-- Sender label and timestamp -->
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs font-medium text-gray-600">
                {msg.sender === 'user' ? 'You' : '👻 Ghost'}
              </span>
              <span class="text-xs text-gray-400">{formatTime(msg.timestamp)}</span>
            </div>
            
            <!-- Message content -->
            <div class="text-sm text-tori-text break-words">
              {msg.text}
            </div>
            
            <!-- Concepts (if any) -->
            {#if msg.concepts && msg.concepts.length > 0}
              <div class="mt-2 flex flex-wrap gap-1">
                {#each msg.concepts as concept}
                  <span class="inline-block bg-purple-100 text-purple-700 text-xs 
                               rounded-full px-2 py-0.5 border border-purple-200">
                    {concept}
                  </span>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/each}
    
    <!-- Typing indicator -->
    {#if isTyping}
      <div class="flex justify-start">
        <div class="bg-gray-100 border border-gray-200 rounded-lg px-4 py-2 shadow-sm">
          <div class="flex items-center space-x-1">
            <span class="text-xs font-medium text-gray-600">👻 Ghost is thinking</span>
            <div class="flex space-x-1">
              <div class="w-1 h-1 bg-gray-400 rounded-full animate-bounce"></div>
              <div class="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Input area -->
  <div class="border-t border-gray-200 bg-white p-3">
    <div class="flex items-end space-x-2">
      <div class="flex-1">
        <textarea 
          class="w-full border border-gray-300 rounded-md p-2 text-sm resize-none
                 focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-blue-300
                 placeholder-gray-400"
          bind:value={userInput} 
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)" 
          on:keydown={handleKeydown}
          rows="1"
          disabled={isTyping}
        ></textarea>
      </div>
      <button 
        class="tori-button-primary px-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
        on:click={sendMessage}
        disabled={!userInput.trim() || isTyping}
      >
        {isTyping ? '⏳' : '📤'}
      </button>
    </div>
    
    <div class="mt-2 flex justify-between items-center text-xs text-gray-500">
      <span>{messages.length} messages in conversation</span>
      <span>Ghost stability: {($ghostPersona.stability * 100).toFixed(0)}%</span>
    </div>
  </div>
</div>