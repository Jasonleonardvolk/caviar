<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { runElfinScript, globalElfinInterpreter } from '$lib/elfin/interpreter.js';
  import type { ElfinContext, ToriEvent } from '$lib/elfin/types.js';
  import { ghostState, conceptGraph, conversationLog } from '$lib/stores';
  
  let scriptInput = `Ghost("Scholar").focus("quantum research") -> search("quantum memory entanglement");
Vault.save("Research:Quantum", $LAST_SEARCH_RESULT);
project("Research:Quantum") -> Thoughtspace.display(at="currentNode");`;

  let output: string[] = [];
  let context: ElfinContext | null = null;
  let executing = false;
  let savedScripts: Array<{name: string, script: string}> = [];
  let realTimeMode = true;

  // Enhanced example scripts with existing system integration
  const exampleScripts = [
    {
      name: "Ghost Research Flow",
      script: `Ghost("Scholar").focus("quantum research") -> search("quantum memory entanglement");
Vault.save("Research:Quantum", $LAST_SEARCH_RESULT);
project("Research:Quantum") -> Thoughtspace.display(at="currentNode");`
    },
    {
      name: "Emotional Support Sequence", 
      script: `if (coherence < 0.3) -> Ghost("Unsettled").emerge();
vault.seal("currentArc");
Ghost("Unsettled").project("SupportMessage");`
    },
    {
      name: "Agent Coordination",
      script: `Ghost("Mentor").emerge();
$analysis = Vault.load("ProblemContext");
Ghost("Mentor").focus($analysis);
project("Solution") -> Thoughtspace.display(at="center");`
    },
    {
      name: "Memory Gating Sequence",
      script: `Memory.gate("emotional_content");
$gated_concepts = $MEMORY_RESULT;
Vault.save("GatedMemory", $gated_concepts);
project($gated_concepts) -> Thoughtspace.display(at="edge");`
    },
    {
      name: "Stability Analysis",
      script: `Lyapunov.analyze("system_state");
if (stability < 0.5) -> Ghost("Chaotic").emerge();
$stability_report = $STABILITY_RESULT;
Vault.save("StabilityReport", $stability_report);`
    }
  ];

  let eventListener: ((event: ToriEvent) => void) | null = null;

  onMount(() => {
    // Load saved scripts from localStorage
    const saved = localStorage.getItem('elfin-scripts');
    if (saved) {
      try {
        savedScripts = JSON.parse(saved);
      } catch (e) {
        console.warn('Could not load saved scripts');
      }
    }
    
    addOutput('⚡ ELFIN++ Enhanced Console Ready');
    addOutput('🔗 Integrated with existing Ghost, Vault, and Memory systems');
    addOutput('Type your consciousness script and press Execute');
    
    // Listen for real-time TORI events
    if (realTimeMode) {
      eventListener = (event: ToriEvent) => {
        handleToriEvent(event);
      };
      globalElfinInterpreter.addEventListener(eventListener);
    }
  });

  onDestroy(() => {
    if (eventListener) {
      globalElfinInterpreter.removeEventListener(eventListener);
    }
  });

  function handleToriEvent(event: ToriEvent) {
    switch (event.type) {
      case 'ghost:emerge':
        addOutput(`👻 Ghost ${event.persona} emerged (${event.trigger.reason})`);
        break;
      case 'vault:sealed':
        addOutput(`🔒 Memory arc sealed: ${event.arcId}`);
        break;
      case 'elfin:executed':
        if (event.result.success) {
          addOutput(`✅ ${event.command.type}: ${event.command.raw}`);
        } else {
          addOutput(`❌ ${event.command.type} failed: ${event.result.error}`);
        }
        break;
    }
  }

  async function executeScript() {
    if (!scriptInput.trim() || executing) return;
    
    executing = true;
    addOutput(`\n> Executing Enhanced ELFIN++ Script:`);
    addOutput(scriptInput);
    addOutput('---');
    
    try {
      const result = await runElfinScript(scriptInput);
      context = result;
      
      addOutput('✅ Script executed successfully');
      
      if (result.lastResult) {
        addOutput(`Last result: ${JSON.stringify(result.lastResult, null, 2)}`);
      }
      
      addOutput(`Variables: ${Object.keys(result.variables).length} defined`);
      Object.entries(result.variables).forEach(([key, value]) => {
        const displayValue = typeof value === 'object' ? JSON.stringify(value) : value;
        addOutput(`  ${key}: ${displayValue}`);
      });
      
    } catch (error) {
      addOutput(`❌ Execution error: ${error.message}`);
      console.error('[ELFIN++] Console execution error:', error);
    }
    
    executing = false;
    addOutput('---\n');
  }

  function addOutput(text: string) {
    output = [...output, text];
    setTimeout(() => {
      const outputEl = document.getElementById('elfin-output');
      if (outputEl) {
        outputEl.scrollTop = outputEl.scrollHeight;
      }
    }, 10);
  }

  function clearOutput() {
    output = [];
  }

  function loadExample(script: string) {
    scriptInput = script;
  }

  function saveScript() {
    const name = prompt('Enter script name:');
    if (name && scriptInput.trim()) {
      savedScripts = [...savedScripts, { name, script: scriptInput }];
      localStorage.setItem('elfin-scripts', JSON.stringify(savedScripts));
    }
  }

  function loadSavedScript(script: string) {
    scriptInput = script;
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'F5' || (event.ctrlKey && event.key === 'Enter')) {
      event.preventDefault();
      executeScript();
    }
  }
</script>

<svelte:head>
  <title>ELFIN++ Enhanced Console - TORI</title>
</svelte:head>

<svelte:window on:keydown={handleKeydown} />

<div class="elfin-console h-full flex">
  <!-- Script Editor -->
  <div class="editor-panel flex-1 flex flex-col p-4">
    <div class="editor-header mb-4">
      <h1 class="text-xl font-bold text-cyan-400 mb-2">⚡ ELFIN++ Enhanced Console</h1>
      <p class="text-sm opacity-60">Consciousness Scripting Language with Live System Integration</p>
    </div>

    <div class="script-editor flex-1 flex flex-col glass-panel p-4">
      <div class="editor-toolbar flex justify-between items-center mb-4 pb-2 border-b border-white/10">
        <div class="toolbar-left flex gap-2">
          <button
            on:click={executeScript}
            disabled={executing || !scriptInput.trim()}
            class="execute-btn px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            {executing ? '⏳ Executing...' : '▶ Execute (F5)'}
          </button>
          
          <button
            on:click={clearOutput}
            class="clear-btn px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm transition-colors"
          >
            Clear Output
          </button>
          
          <button
            on:click={saveScript}
            class="save-btn px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
          >
            💾 Save
          </button>
        </div>
        
        <div class="status text-sm opacity-60">
          {executing ? 'Executing...' : 'Ready'}
        </div>
      </div>

      <textarea
        bind:value={scriptInput}
        placeholder="Enter your ELFIN++ script here..."
        class="script-textarea flex-1 w-full bg-black/20 border border-white/10 rounded-lg p-4 font-mono text-sm resize-none focus:outline-none focus:border-cyan-500/50"
        disabled={executing}
      />
    </div>
  </div>

  <!-- Output & Examples -->
  <div class="output-panel w-1/2 flex flex-col p-4 border-l border-white/10">
    <!-- Examples -->
    <div class="examples-section mb-4">
      <h3 class="text-lg font-semibold mb-3">📋 Enhanced Example Scripts</h3>
      <div class="examples-grid grid gap-2">
        {#each exampleScripts as example}
          <button
            on:click={() => loadExample(example.script)}
            class="example-btn p-3 glass-panel text-left hover:bg-white/10 transition-colors rounded-lg"
          >
            <div class="font-semibold text-sm text-cyan-400">{example.name}</div>
            <div class="text-xs opacity-60 mt-1 line-clamp-2">{example.script.split('\n')[0]}...</div>
          </button>
        {/each}
      </div>

      {#if savedScripts.length > 0}
        <div class="saved-scripts mt-4">
          <h4 class="text-sm font-semibold mb-2 opacity-80">💾 Saved Scripts</h4>
          <div class="space-y-1">
            {#each savedScripts as saved}
              <button
                on:click={() => loadSavedScript(saved.script)}
                class="saved-script-btn w-full p-2 text-left bg-black/20 hover:bg-black/30 rounded text-sm transition-colors"
              >
                {saved.name}
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- Console Output -->
    <div class="output-section flex-1 flex flex-col">
      <h3 class="text-lg font-semibold mb-3">📺 Console Output</h3>
      <div 
        id="elfin-output"
        class="output-display flex-1 glass-panel p-4 overflow-y-auto font-mono text-sm"
      >
        {#each output as line}
          <div class="output-line {line.startsWith('❌') ? 'text-red-400' : line.startsWith('✅') ? 'text-green-400' : line.startsWith('>') ? 'text-cyan-400' : ''}">
            {line}
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .script-textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 1.5;
  }
  
  .script-textarea::placeholder {
    @apply opacity-40;
  }
  
  .output-display {
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
  }
  
  .output-display::-webkit-scrollbar {
    width: 6px;
  }
  
  .output-display::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 3px;
  }
  
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  
  .output-line {
    margin-bottom: 2px;
    white-space: pre-wrap;
  }
</style>