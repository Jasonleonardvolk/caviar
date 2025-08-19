<!-- TEMPORARY CHAT TEST COMPONENT -->
<script lang="ts">
  let userInput = "";
  let messages = [{
    id: "welcome", 
    text: "🔥 DIRECT API TEST LOADED! Type anything to test real API connection.",
    sender: "system",
    timestamp: new Date()
  }];
  
  async function sendMessage() {
    if (!userInput.trim()) return;
    
    // Add user message
    messages = [...messages, {
      id: Date.now().toString(),
      text: userInput,
      sender: "user", 
      timestamp: new Date()
    }];
    
    const query = userInput;
    userInput = "";
    
    try {
      console.log("🔥 TESTING API CALL TO /api/chat");
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, userId: 'test_user' })
      });
      
      console.log("🔥 API Response status:", response.status);
      
      if (response.ok) {
        const result = await response.json();
        console.log("🔥 API Response data:", result);
        
        messages = [...messages, {
          id: Date.now().toString(),
          text: `✅ REAL API WORKS! Response: ${result.response}`,
          sender: "api",
          timestamp: new Date(),
          concepts: result.concepts_found,
          confidence: result.confidence,
          processing_time: result.processing_time
        }];
      } else {
        throw new Error(`Status ${response.status}`);
      }
    } catch (error) {
      console.error("🔥 API ERROR:", error);
      messages = [...messages, {
        id: Date.now().toString(),
        text: `❌ API FAILED: ${error.message}`,
        sender: "error",
        timestamp: new Date()
      }];
    }
  }
</script>

<div class="fixed top-4 right-4 w-96 h-96 bg-white border-2 border-red-500 rounded-lg shadow-lg z-50 p-4">
  <div class="flex justify-between items-center mb-4">
    <h1 class="text-lg font-bold text-red-600">🔥 DIRECT API TEST</h1>
    <button onclick="this.parentElement.parentElement.remove()">❌</button>
  </div>
  
  <div class="space-y-2 mb-4 max-h-48 overflow-y-auto">
    {#each messages as msg}
      <div class="p-2 rounded text-sm {msg.sender === 'user' ? 'bg-blue-100' : msg.sender === 'error' ? 'bg-red-100' : msg.sender === 'api' ? 'bg-green-100' : 'bg-gray-100'}">
        <strong>{msg.sender}:</strong> {msg.text}
        {#if msg.concepts && msg.concepts.length > 0}
          <div class="mt-1">
            <strong>Concepts:</strong> {msg.concepts.join(', ')}
          </div>
        {/if}
        {#if msg.confidence}
          <div class="text-xs text-gray-600">
            Confidence: {Math.round(msg.confidence * 100)}% | Time: {msg.processing_time?.toFixed(3)}s
          </div>
        {/if}
      </div>
    {/each}
  </div>
  
  <div class="flex gap-2">
    <input 
      class="flex-1 border rounded px-2 py-1 text-sm" 
      bind:value={userInput} 
      placeholder="Type 'what do you know about darwin'"
      on:keydown={e => e.key === 'Enter' && sendMessage()}
    />
    <button class="bg-red-500 text-white px-3 py-1 rounded text-sm" on:click={sendMessage}>Test</button>
  </div>
</div>