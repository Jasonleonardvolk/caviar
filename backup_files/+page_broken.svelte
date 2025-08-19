<script lang="ts">
  import { onMount, afterUpdate, tick } from 'svelte';
  import { conceptMesh, addConceptDiff } from '$lib/stores/conceptMesh';
  import ConceptDebugPanel from '$lib/components/ConceptDebugPanel.svelte';
  import { dynamicsIntegration } from '$lib/integrations/dynamicsIntegration';
  import type { KoopmanUpdate, LyapunovSpike } from '$lib/cognitive/dynamics';
  
  
  // ✨ Import Soliton Memory System - FIXED: Use proper service imports
  import { 
    initializeUserMemory, 
    storeMemory, 
    fetchMemoryStats, 
    findRelatedMemories,
    vaultMemory,
    analyzeValence,
    extractPhaseTags,
    measureAmplitude,
    estimateFrequency,
    phaseChangeEvent
  } from '$lib/services/solitonMemory';
  
  // STEP 1-4: Import ALL systems
  let braidMemory: any = null;
  let cognitiveEngine: any = null;
  let holographicMemory: any = null;
  let ghostCollective: any = null;

  let files;

  // Get user data from server via layout
  export let data: { user: { name: string; role: 'admin' | 'user' } | null };
  
  let mounted = false;
  let messageInput = '';
  let isTyping = false;
  let showDebugPanel = false;
  let conversationHistory: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    concepts?: string[];
    loopId?: string;
    braidStats?: any;
    braidLoopId?: string;
    processingMethod?: string;
    confidence?: number;
    systemInsights?: string[];
    activePersona?: any;
    holographicData?: any;
    conceptNodes?: any[];
    // NEW: Added memory context and phase data
    memoryContext?: {
      relatedMemories: number;
      phaseCoherence: number;
      valence?: number;
      phaseTag?: number;
      amplitude?: number;
      frequency?: number;
    };
    sensitiveContent?: boolean;
  }> = [];
  
  // System stats - FIXED: Proper Soliton integration
  let solitonStats: any = null;
  let braidStats: any = null;
  let holographicStats: any = null;
  let ghostStats: any = null;
  let memoryInitialized = false;
  let currentUserId = '';
  
  // FIXED: Phase monitoring for Ghost AI
  let currentPhase = 'idle';
  let phaseAmplitude = 0;
  let phaseFrequency = 0;
  let systemCoherence = 1.0;
  
  // Enhanced holographic memory tracking
  let userConceptNode: any = null;
  let recentCrossings: any[] = [];
  let currentCoherence = 0.5;
  let currentContradiction = 0.0;
  
  // Dynamics analysis tracking
  let latestKoopmanUpdate: KoopmanUpdate | null = null;
  let recentLyapunovSpikes: LyapunovSpike[] = [];
  let showDynamicsInfo = false;
  
  // ✅ AUTO-SCROLL FUNCTIONALITY
  let scrollContainer: HTMLElement;
  let isUserScrolledUp = false;
  let showScrollToBottom = false;
  let lastMessageCount = 0;
  let shouldAutoScroll = true;
  
  // Track scroll position to detect manual scrolling
  function handleScroll() {
    if (!scrollContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50; // 50px threshold
    
    isUserScrolledUp = !isAtBottom;
    showScrollToBottom = isUserScrolledUp && conversationHistory.length > 0;
    shouldAutoScroll = isAtBottom;
  }
  
  // Auto-scroll to bottom function
  function scrollToBottom(force = false) {
    if (!scrollContainer) return;
    
    if (force || shouldAutoScroll) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
      isUserScrolledUp = false;
      showScrollToBottom = false;
      shouldAutoScroll = true;
    }
  }
  
  // Auto-scroll when new messages are added
  $: if (conversationHistory.length > lastMessageCount && mounted) {
    lastMessageCount = conversationHistory.length;
    // Use tick to ensure DOM is updated before scrolling
    tick().then(() => {
      if (shouldAutoScroll) {
        scrollToBottom(false);
      } else {
        // Show scroll to bottom button if user is scrolled up
        showScrollToBottom = true;
      }
    });
  }
  
  // Force scroll to bottom (for button click)
  function forceScrollToBottom() {
    scrollToBottom(true);
  }
  
  onMount(() => {
    mounted = true;
    
    let unsubscribeKoopman: (() => void) | undefined;
    let unsubscribeLyapunov: (() => void) | undefined;
    
    (async () => {
      // STEP 1-4: Load ALL cognitive systems
      try {
        const cognitive = await import('$lib/cognitive');
        braidMemory = cognitive.braidMemory;
        cognitiveEngine = cognitive.cognitiveEngine;
        holographicMemory = cognitive.holographicMemory;
        ghostCollective = cognitive.ghostCollective;
        
        console.log('🧬 ALL SYSTEMS LOADED:', {
          braidMemory: !!braidMemory,
          cognitiveEngine: !!cognitiveEngine,
          holographicMemory: !!holographicMemory,
          ghostCollective: !!ghostCollective
        });
      } catch (error) {
        console.warn('⚠️ Some cognitive systems not available:', error);
      }
      
      // 🌊 INITIALIZE SOLITON MEMORY SYSTEM - FIXED: Proper backend-first approach
      console.log('🌊 Initializing Soliton Memory System...');
      try {
        currentUserId = data.user?.name || 'default_user';
        
        // FIXED: Remove fallback behavior - use backend first
        await initializeUserMemory(currentUserId);
        memoryInitialized = true;
        console.log('✨ Soliton Memory initialized for user:', currentUserId);
        
        // Get initial memory stats
        try {
          solitonStats = await fetchMemoryStats(currentUserId);
          console.log('📊 Initial Memory Stats:', solitonStats);
        } catch (error) {
          console.error('❌ Failed to get memory stats - backend may be down:', error);
          // FIXED: Don't use fallback stats - log error instead
          throw error;
        }
        
        // Store foundational memory about this session with proper metadata
        const sessionMemoryId = `session_${Date.now()}`;
        const sessionContent = `New session started for ${data.user?.name || 'User'} with TORI consciousness interface`;
        
        await storeMemory(
          sessionMemoryId,
          sessionContent,
          1.0, // Maximum importance
          ['session', 'initialization', 'system']
        );
        
        console.log('🌊 Session memory stored with ID:', sessionMemoryId);
        
      } catch (error) {
        console.error('❌ Failed to initialize Soliton Memory - backend integration issue:', error);
        // FIXED: Don't continue with fallback - show error to user
        memoryInitialized = false;
      }
      
      // 🧬 INITIALIZE BRAID MEMORY
      if (braidMemory) {
        try {
          console.log('🧬 Initializing Braid Memory...');
          
          // Set up reentry callback to detect memory loops
          braidMemory.onReentry((digest: string, count: number, loop: any) => {
            console.log(`🔁 Memory loop detected! Pattern seen ${count} times`);
            
            // If we're in a memory loop, suggest novelty
            if (count >= 3) {
              const noveltyGlyph = braidMemory.suggestNoveltyGlyph(
                digest,
                0.5, // current contradiction
                0.7, // current coherence
                0    // scar count
              );
              console.log(`💡 Suggested novelty glyph: ${noveltyGlyph}`);
            }
          });
          
          console.log('✅ Braid Memory initialized and monitoring for loops');
        } catch (error) {
          console.warn('Failed to initialize Braid Memory:', error);
        }
      }
      
     // 🔮 INITIALIZE HOLOGRAPHIC MEMORY
      if (holographicMemory && typeof holographicMemory.initialize === 'function') {
        holographicMemory.initialize();
      } else {
        console.warn('HolographicMemory: initialize() not found');
      }

      // 👻 INITIALIZE GHOST COLLECTIVE
      if (ghostCollective) {
        try {
          console.log('👻 Initializing Ghost Collective...');
          // Set up phase change listeners for Ghost AI
          if (typeof window !== 'undefined') {
            document.addEventListener('tori-soliton-phase-change', (event: any) => {
              const phaseData = event.detail;
              currentPhase = phaseData.phase || 'active';
              phaseAmplitude = phaseData.amplitude || 0;
              phaseFrequency = phaseData.frequency || 0;
              systemCoherence = phaseData.stability || 1.0;
              
              console.log('👻 Ghost AI notified of phase change:', phaseData);
            });
          }
          console.log('✅ Ghost Collective ready with phase monitoring');
        } catch (error) {
          console.warn('Failed to initialize Ghost Collective:', error);
        }
      }
      
      // 🧠 INITIALIZE COGNITIVE ENGINE
      if (cognitiveEngine) {
        try {
          console.log('🧠 Initializing Cognitive Engine...');
          console.log('✅ Cognitive Engine ready');
        } catch (error) {
          console.warn('Failed to initialize Cognitive Engine:', error);
        }
      }
      
      // STEP 2-4: Initialize Enhanced API Service
      console.log('🚀 Enhanced API Service v4.0 initialized with full system integration');
      
      // 🧬 BRAID MEMORY INTEGRATION - Load conversation history from BraidMemory
      if (braidMemory) {
        try {
          // First try to restore from BraidMemory
          const braidHistory = braidMemory.getConversationHistory?.();
          if (braidHistory && braidHistory.length > 0) {
            console.log('🧬 Restoring conversation from BraidMemory:', braidHistory.length, 'messages');
            conversationHistory = braidHistory.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }));
            lastMessageCount = braidHistory.length;
            
            // Auto-scroll to bottom after loading history
            tick().then(() => scrollToBottom(true));
          } else {
            // Fallback to localStorage if BraidMemory is empty
            const saved = localStorage.getItem('tori-conversation-history');
            if (saved) {
              try {
                const loadedHistory = JSON.parse(saved).map((msg: any) => ({
                  ...msg,
                  timestamp: new Date(msg.timestamp)
                }));
                conversationHistory = loadedHistory;
                lastMessageCount = loadedHistory.length;
                
                // Sync to BraidMemory
                if (braidMemory.setConversationHistory) {
                  braidMemory.setConversationHistory(loadedHistory);
                  console.log('🧬 Synced localStorage history to BraidMemory');
                }
                
                // Auto-scroll to bottom after loading history
                tick().then(() => scrollToBottom(true));
              } catch (e) {
                console.warn('Failed to load conversation history from localStorage:', e);
              }
            }
          }
        } catch (error) {
          console.warn('🧬 BraidMemory history restoration failed:', error);
          // Fallback to localStorage
          const saved = localStorage.getItem('tori-conversation-history');
          if (saved) {
            try {
              const loadedHistory = JSON.parse(saved).map((msg: any) => ({
                ...msg,
                timestamp: new Date(msg.timestamp)
              }));
              conversationHistory = loadedHistory;
              lastMessageCount = loadedHistory.length;
              tick().then(() => scrollToBottom(true));
            } catch (e) {
              console.warn('Failed to load conversation history:', e);
            }
          }
        }
      } else {
        // No BraidMemory available, use localStorage only
        const saved = localStorage.getItem('tori-conversation-history');
        if (saved) {
          try {
            const loadedHistory = JSON.parse(saved).map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp)
            }));
            conversationHistory = loadedHistory;
            lastMessageCount = loadedHistory.length;
            
            // Auto-scroll to bottom after loading history
            tick().then(() => scrollToBottom(true));
          } catch (e) {
            console.warn('Failed to load conversation history:', e);
          }
        }
      }
      
      // 🌊 Set up dynamics analysis listeners
      unsubscribeKoopman = dynamicsIntegration.onKoopmanUpdate((update) => {
        latestKoopmanUpdate = update;
        console.log('🌊 Koopman modes:', update.eigenmodes.length, 'Dominant:', update.dominantMode);
      });
      
      unsubscribeLyapunov = dynamicsIntegration.onLyapunovSpike((spike) => {
        recentLyapunovSpikes = [...recentLyapunovSpikes.slice(-4), spike]; // Keep last 5 spikes
        console.warn('⚡ Chaos detected! Exponent:', spike.exponent.toFixed(3));
        
        // Optional: Trigger some UI response to chaos
        if (spike.instabilityLevel > 0.8) {
          // High chaos - maybe adjust UI behavior
          console.error('🌀 EXTREME CHAOS DETECTED');
        }
      });
      
      console.log('🎯 TORI main page loaded with FULL SYSTEM INTEGRATION and auto-scroll');
    })();
    
    // Poll for memory stats every 5 seconds - FIXED: Better error handling
    const statsInterval = setInterval(async () => {
      if (!memoryInitialized) return;
      
      try {
        solitonStats = await fetchMemoryStats(currentUserId);
        
        // Also update other system stats
        if (braidMemory) {
          braidStats = braidMemory.getStats();
        }
        if (holographicMemory) {
          holographicStats = holographicMemory.getVisualizationData();
        }
        if (ghostCollective) {
          ghostStats = ghostCollective.getDiagnostics?.() || null;
        }
      } catch (error) {
        console.warn('Failed to get memory stats:', error);
        // FIXED: Don't reset stats on error - keep last known good state
      }
    }, 5000);
    
    // Cleanup on unmount
    return () => {
      clearInterval(statsInterval);
      if (unsubscribeKoopman) unsubscribeKoopman();
      if (unsubscribeLyapunov) unsubscribeLyapunov();
    };
  });
  
  // 🧬 Auto-save conversation history to BraidMemory and localStorage
  $: if (mounted && conversationHistory.length > 0) {
    // Save to localStorage as backup
    localStorage.setItem('tori-conversation-history', JSON.stringify(conversationHistory));
    
    // Also sync to BraidMemory if available
    if (braidMemory && braidMemory.setConversationHistory) {
      try {
        braidMemory.setConversationHistory(conversationHistory);
        console.log('🧬 Auto-saved', conversationHistory.length, 'messages to BraidMemory');
      } catch (error) {
        console.warn('🧬 Failed to sync to BraidMemory:', error);
      }
    }
  }
  
  async function sendMessage() {
    if (!messageInput.trim() || isTyping || !data.user) return;
    
    // FIXED: Check if Soliton Memory is initialized before proceeding
    if (!memoryInitialized) {
      console.error('❌ Cannot send message - Soliton Memory not initialized');
      return;
    }
    
    const userMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
      role: 'user' as const,
      content: messageInput.trim(),
      timestamp: new Date(),
      braidLoopId: undefined as string | undefined,
      memoryContext: undefined as any,
      sensitiveContent: false
    };
    
    conversationHistory = [...conversationHistory, userMessage];
    const currentMessage = messageInput;
    messageInput = '';
    isTyping = true;
    
    // Auto-scroll for user's message
    shouldAutoScroll = true;
    
    // ✨ FIXED: Store user message in Soliton Memory with proper analysis
    let solitonResult: any = null;
    try {
      // Analyze emotional valence and extract phase tags
      const valence = analyzeValence(currentMessage);
      const phaseTags = extractPhaseTags(currentMessage);
      const amplitude = measureAmplitude(currentMessage);
      const frequency = estimateFrequency(currentMessage);
      
      // FIXED: Use proper parameter order and metadata
      solitonResult = await storeMemory(
        userMessage.id,     // conceptId
        currentMessage,     // content
        0.8,               // importance
        ['chat', 'user_input', ...phaseTags] // tags including phase information
      );
      
      // Store analysis results in message context
      userMessage.memoryContext = {
        relatedMemories: 0, // Will be updated below
        phaseCoherence: 0.5,
        valence,
        phaseTag: solitonResult.phaseTag,
        amplitude,
        frequency
      };
      
      console.log('🌊 User message stored in Soliton Memory:', solitonResult);
      
      // FIXED: Proper phase change event with all required data
      phaseChangeEvent({
        userId: currentUserId,
        phase: 'user_input',
        amplitude,
        frequency,
        phaseTag: solitonResult.phaseTag,
        valence,
        timestamp: Date.now()
      });
      
      // Trigger phase monitoring for Ghost AI
      if (typeof window !== 'undefined') {
        document.dispatchEvent(new CustomEvent('tori-soliton-phase-change', {
          detail: {
            userId: currentUserId,
            phase: 'user_input',
            phaseAngle: solitonResult.phaseTag,
            amplitude,
            frequency,
            stability: 0.8,
            valence
          }
        }));
      }
    } catch (error) {
      console.error('❌ Failed to store user message in Soliton Memory:', error);
      // FIXED: Don't continue if memory storage fails - this indicates backend issues
      isTyping = false;
      return;
    }
    
    // 🧬 Store in Braid Memory for loop analysis
    if (braidMemory && solitonResult) {
      try {
        // Create a loop record for this interaction
        const loopRecord = {
          id: `loop_${userMessage.id}`,
          prompt: currentMessage,
          glyphPath: currentMessage.split(' ').filter(w => w.length > 3), // Simple tokenization
          phaseTrace: [solitonResult.phaseTag || 0],
          coherenceTrace: [0.5], // Starting coherence
          contradictionTrace: [0.0], // No contradiction yet
          closed: false,
          scarFlag: false,
          timestamp: new Date(),
          processingTime: 0,
          metadata: {
            createdByPersona: 'user',
            conceptFootprint: [],
            phaseGateHits: [],
            solitonPhase: solitonResult.phaseTag
          }
        };
        
        const loopId = braidMemory.archiveLoop(loopRecord);
        console.log(`🧬 Archived user loop: ${loopId}`);
        
        // Store loop ID for response correlation
        userMessage.braidLoopId = loopId;
      } catch (error) {
        console.warn('Failed to store in Braid Memory:', error);
      }
    }
    
    // 🔮 Store in Holographic Memory with enhanced integration
    userConceptNode = null;
    if (holographicMemory) {
      try {
        // Create a concept node for this message
        const essence = currentMessage.split(' ').slice(0, 3).join(' ') || `Message_${Date.now()}`;
        userConceptNode = holographicMemory.createConceptNode(essence, 0.7);
        
        // Update position based on conversation flow
        userConceptNode.position = {
          x: conversationHistory.length * 2,
          y: solitonResult?.phaseTag || 0,
          z: (userMessage.memoryContext?.phaseCoherence || 0.5) * 10
        };
        
        // Add phase metadata
        if (solitonResult?.phaseTag) {
          userConceptNode.metadata.memoryReferences.push(`phase_${solitonResult.phaseTag}`);
        }
        
        // Connect to previous concepts if they exist
        const previousNodes = holographicMemory.getAllNodes();
        if (previousNodes.length > 1) {
          const lastNode = previousNodes[previousNodes.length - 2];
          holographicMemory.createConnection(
            lastNode.id,
            userConceptNode.id,
            0.6,
            'temporal'
          );
        }
        
        // Activate the concept to create visual wave
        holographicMemory.activateConcept(userConceptNode.id, 0.5);
        
        console.log('🔮 Created holographic concept:', essence, 'at', userConceptNode.position);
      } catch (error) {
        console.warn('Failed to store in Holographic Memory:', error);
      }
    }
    
    try {
      // 🔍 FIXED: Find related memories using phase correlation
      let relatedMemories: any[] = [];
      try {
        relatedMemories = await findRelatedMemories(
          userMessage.id,
          5 // max results
        );
        
        // Update memory context with related memories count
        if (userMessage.memoryContext) {
          userMessage.memoryContext.relatedMemories = relatedMemories.length;
          userMessage.memoryContext.phaseCoherence = relatedMemories.length > 0 ? 0.8 : 0.0;
        }
        
        console.log(`🔗 Found ${relatedMemories.length} related memories`);
      } catch (error) {
        console.warn('Failed to find related memories:', error);
      }
      
      // 🚀 NEW: Use our REAL answer API for document-grounded responses
      console.log('🤖 Calling REAL answer API:', currentMessage);
      const chatResponse = await fetch('/api/answer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_query: currentMessage,
          persona: { name: data.user?.name || 'anonymous' },
          // FIXED: Include memory context for better responses
          memory_context: {
            related_memories: relatedMemories,
            user_id: currentUserId,
            phase_tag: solitonResult?.phaseTag
          }
        })
      });

      if (!chatResponse.ok) {
        throw new Error(`Answer API error: ${chatResponse.status}`);
      }

      const chatResult = await chatResponse.json();
      console.log('✅ Real answer API response:', chatResult);

      // Create enhanced response object to match expected format
      const enhancedResponse = {
        response: chatResult.answer,
        newConcepts: chatResult.sources || [],
        confidence: chatResult.confidence || 0.8,
        processingMethod: 'document_grounded_ai',
        systemInsights: [
          `Context: ${chatResult.context_used || 'general_knowledge'}`,
          `Sources consulted: ${(chatResult.sources || []).length}`,
          `Confidence: ${Math.round((chatResult.confidence || 0.8) * 100)}%`,
          `Processing time: ${(chatResult.processing_time || 0).toFixed(3)}s`,
          `Documents used: ${chatResult.documents_consulted || 0}`,
          `Tokens generated: ${chatResult.tokens_generated || 0}`
        ],
        activePersona: { name: 'TORI AI', id: 'tori' },
        conceptNodes: (chatResult.sources || []).map((source, i) => ({
          id: `source_${i}`,
          name: source,
          position: { x: i, y: 0, z: 0 }
        })),
        loopId: `answer_${Date.now()}`,
        braidMetrics: { crossings: 0 },
        emergentConnections: [],
        holographicData: null
      };
      
      // FIXED: Analyze AI response for sensitivity and phase data
      const aiValence = analyzeValence(enhancedResponse.response);
      const aiPhaseTags = extractPhaseTags(enhancedResponse.response);
      const aiAmplitude = measureAmplitude(enhancedResponse.response);
      const aiFrequency = estimateFrequency(enhancedResponse.response);
      
      // Check for sensitive content
      const isSensitive = enhancedResponse.response.toLowerCase().includes('sensitive') ||
                          enhancedResponse.response.toLowerCase().includes('protect') ||
                          enhancedResponse.response.toLowerCase().includes('vault') ||
                          Math.abs(aiValence) > 0.7; // High emotional content
      
      const assistantMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
        role: 'assistant' as const,
        content: enhancedResponse.response,
        timestamp: new Date(),
        concepts: enhancedResponse.newConcepts,
        loopId: enhancedResponse.loopId,
        braidStats: enhancedResponse.braidMetrics,
        processingMethod: enhancedResponse.processingMethod,
        confidence: enhancedResponse.confidence,
        systemInsights: enhancedResponse.systemInsights,
        activePersona: enhancedResponse.activePersona,
        holographicData: enhancedResponse.holographicData,
        conceptNodes: enhancedResponse.conceptNodes,
        sensitiveContent: isSensitive,
        memoryContext: {
          relatedMemories: relatedMemories.length,
          phaseCoherence: relatedMemories.length > 0 ? 0.8 : 0.0,
          valence: aiValence,
          phaseTag: undefined, // Will be set after storing
          amplitude: aiAmplitude,
          frequency: aiFrequency
        }
      };
      
      conversationHistory = [...conversationHistory, assistantMessage];
      
      // ✨ FIXED: Store assistant response in Soliton Memory with proper analysis
      try {
        const aiMemoryResult = await storeMemory(
          assistantMessage.id,        // conceptId
          enhancedResponse.response,  // content
          0.9,                       // Higher importance for AI responses
          ['chat', 'ai_response', ...aiPhaseTags] // tags including AI phase analysis
        );
        
        // Update memory context with stored data
        if (assistantMessage.memoryContext) {
          assistantMessage.memoryContext.phaseTag = aiMemoryResult.phaseTag;
        }
        
        console.log('🌊 AI response stored in Soliton Memory:', aiMemoryResult);
        
        // FIXED: Auto-vault sensitive content with proper protection
        if (isSensitive) {
          try {
            await vaultMemory(assistantMessage.id, 'UserSealed');
            console.log('🔐 Sensitive memory auto-vaulted for protection');
          } catch (vaultError) {
            console.warn('Failed to vault sensitive memory:', vaultError);
          }
        }
        
        // Emit phase change event for AI response
        phaseChangeEvent({
          userId: currentUserId,
          phase: 'ai_response',
          amplitude: aiAmplitude,
          frequency: aiFrequency,
          phaseTag: aiMemoryResult.phaseTag,
          valence: aiValence,
          timestamp: Date.now()
        });
        
      } catch (error) {
        console.error('❌ Failed to store AI response in Soliton Memory:', error);
      }
      
      // 🧬 Complete the Braid Memory loop
      if (braidMemory && userMessage.braidLoopId) {
        try {
          // Get the original loop (need to access the Map correctly)
          const loopRegistry = braidMemory.loopRegistry;
          let originalLoop = null;
          
          // Try to get the loop from the registry
          if (loopRegistry && typeof loopRegistry.get === 'function') {
            originalLoop = loopRegistry.get(userMessage.braidLoopId);
          }
          
          if (originalLoop) {
            // Update with AI response
            originalLoop.returnGlyph = 'ai_response';
            originalLoop.closed = true;
            originalLoop.coherenceTrace.push(enhancedResponse.confidence || 0.8);
            originalLoop.contradictionTrace.push(0); // Assuming no contradiction
            originalLoop.processingTime = Date.now() - originalLoop.timestamp.getTime();
            
            // Add AI concepts to glyph path
            if (enhancedResponse.newConcepts) {
              originalLoop.glyphPath.push(...enhancedResponse.newConcepts);
            }
            
            // Re-archive to trigger compression and crossing detection
            braidMemory.archiveLoop(originalLoop);
            
            // Check for crossings with other loops
            const crossings = braidMemory.getCrossingsForLoop(originalLoop.id);
            if (crossings.length > 0) {
              console.log(`🔀 Found ${crossings.length} memory crossings!`);
              crossings.forEach(crossing => {
                console.log(`  - ${crossing.type} crossing via "${crossing.glyph}"`);
              });
            }
          }
        } catch (error) {
          console.warn('Failed to complete Braid loop:', error);
        }
      }
      
      // 🔮 Update Holographic Memory with AI response
      if (holographicMemory && userConceptNode) {
        try {
          // Create AI response concept node
          const aiEssence = enhancedResponse.response.split(' ').slice(0, 3).join(' ') || `AI_Response_${Date.now()}`;
          const aiConceptNode = holographicMemory.createConceptNode(aiEssence, enhancedResponse.confidence || 0.8);
          
          // Position relative to user node
          aiConceptNode.position = {
            x: userConceptNode.position.x + 2,
            y: aiAmplitude * 10,
            z: (enhancedResponse.confidence || 0.5) * 10
          };
          
          // Connect user query to AI response
          holographicMemory.createConnection(
            userConceptNode.id,
            aiConceptNode.id,
            enhancedResponse.confidence || 0.8,
            'causal',
            'source-to-target'
          );
          
          // Connect to related concepts
          enhancedResponse.newConcepts?.forEach((concept, index) => {
            const conceptNode = holographicMemory.createConceptNode(concept, 0.5);
            conceptNode.position = {
              x: aiConceptNode.position.x + (index + 1) * 1.5,
              y: aiConceptNode.position.y + Math.sin(index) * 2,
              z: aiConceptNode.position.z + Math.cos(index) * 2
            };
            
            holographicMemory.createConnection(
              aiConceptNode.id,
              conceptNode.id,
              0.4,
              'semantic'
            );
          });
          
          // Add persona touch if available
          if (enhancedResponse.activePersona) {
            holographicMemory.addPersonaTouch(
              aiConceptNode.id,
              enhancedResponse.activePersona.id,
              0.6,
              `Response via ${enhancedResponse.processingMethod}`
            );
          }
          
          // Activate AI node
          holographicMemory.activateConcept(aiConceptNode.id, 0.6);
          
          // Detect emergent clusters
          const clusters = holographicMemory.detectEmergentClusters();
          if (clusters.length > 0) {
            console.log('🌌 Detected', clusters.length, 'emergent concept clusters');
          }
          
          console.log('🔮 Created AI holographic concept:', aiEssence);
        } catch (error) {
          console.warn('Failed to update Holographic Memory:', error);
        }
      }
      
      // Add to concept mesh with FULL system metadata including all memories
      if (enhancedResponse.newConcepts && enhancedResponse.newConcepts.length > 0) {
        addConceptDiff({
          type: 'chat',
          title: `Ultimate AI: ${currentMessage.length > 50 ? currentMessage.substring(0, 50) + '...' : currentMessage}`,
          concepts: enhancedResponse.newConcepts,
          summary: `Ultimate AI processing via ${enhancedResponse.processingMethod}. Confidence: ${Math.round(enhancedResponse.confidence * 100)}%${enhancedResponse.activePersona ? ` (${enhancedResponse.activePersona.name})` : ''}${enhancedResponse.conceptNodes ? ` | ${enhancedResponse.conceptNodes.length} 3D nodes` : ''} | 🌊 ${relatedMemories.length} memories`,
          metadata: {
            messageCount: conversationHistory.length,
            userMessage: currentMessage,
            processingMethod: enhancedResponse.processingMethod,
            confidence: enhancedResponse.confidence,
            systemInsights: enhancedResponse.systemInsights,
            loopId: enhancedResponse.loopId,
            braidMetrics: enhancedResponse.braidMetrics,
            emergentConnections: enhancedResponse.emergentConnections,
            activePersona: enhancedResponse.activePersona,
            holographicData: enhancedResponse.holographicData,
            conceptNodes: enhancedResponse.conceptNodes,
            // 🌊 Soliton memory metadata
            solitonMemory: {
              userPhase: solitonResult?.phaseTag,
              aiPhase: assistantMessage.memoryContext?.phaseTag,
              relatedMemoryCount: relatedMemories.length,
              memoryIntegrity: 1.0,
              valence: {
                user: userMessage.memoryContext?.valence,
                ai: assistantMessage.memoryContext?.valence
              }
            },
            // 🧬 Braid memory metadata
            braidMemory: {
              loopId: userMessage.braidLoopId,
              crossings: braidStats?.crossings || 0
            },
            fullSystemIntegration: true,
            timestamp: new Date()
          }
        });
      }
      
      // FIXED: Update memory stats after successful interaction
      try {
        solitonStats = await fetchMemoryStats(currentUserId);
      } catch (error) {
        console.warn('Failed to refresh memory stats:', error);
      }
      
    } catch (error) {
      console.error('Ultimate AI processing failed:', error);
      
      const errorMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
        role: 'assistant' as const,
        content: "I'm experiencing technical difficulties with my memory systems. Please check that the Soliton Memory backend is running properly.",
        timestamp: new Date(),
        processingMethod: 'error_fallback',
        sensitiveContent: false
      };
      
      conversationHistory = [...conversationHistory, errorMessage];
    } finally {
      isTyping = false;
    }
  }
  
  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }
  
  function clearConversation() {
    if (confirm('Clear conversation? This will remove all messages but keep your memory intact.')) {
      conversationHistory = [];
      lastMessageCount = 0;
      localStorage.removeItem('tori-conversation-history');
      
      // 🧬 Also clear BraidMemory conversation history
      if (braidMemory && braidMemory.clearConversationHistory) {
        try {
          braidMemory.clearConversationHistory();
          console.log('🧬 Cleared BraidMemory conversation history');
        } catch (error) {
          console.warn('🧬 Failed to clear BraidMemory:', error);
        }
      }
      
      showScrollToBottom = false;
      isUserScrolledUp = false;
      shouldAutoScroll = true;
    }
  }
  
  function toggleDebugPanel() {
    showDebugPanel = !showDebugPanel;
  }
  
  // Get system stats for display
  function getSystemStats() {
    const stats = {
      braid: braidStats,
      holographic: holographicStats,
      ghost: ghostStats
    };
    
    return stats;
  }
  
  $: systemStats = getSystemStats();
  
  // STEP 3: Get processing method icon (enhanced)
  function getProcessingIcon(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return '🌌';
      case 'holographic_synthesis': return '🎯';
      case 'ghost_collective': return '👻';
      case 'cognitive_engine': return '🧬';
      case 'braid_memory': return '🔗';
      case 'simple': return '⚡';
      case 'real_chat_api': return '🤖';
      case 'document_grounded_ai': return '📚';
      case 'error_fallback': return '⚠️';
      default: return '🤖';
    }
  }
  
  // STEP 3: Get processing method name (enhanced)
  function getProcessingName(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return 'Revolutionary';
      case 'holographic_synthesis': return 'Holographic';
      case 'ghost_collective': return 'Ghost Collective';
      case 'cognitive_engine': return 'Cognitive Engine';
      case 'braid_memory': return 'BraidMemory';
      case 'simple': return 'Enhanced';
      case 'real_chat_api': return 'TORI AI';
      case 'document_grounded_ai': return 'Document AI';
      case 'error_fallback': return 'Error Recovery';
      default: return 'Standard';
    }
  }
  
  // STEP 3: Get persona icon
  function getPersonaIcon(persona: any): string {
    if (!persona) return '';
    switch (persona.id || persona.name?.toLowerCase()) {
      case 'scholar': return '🧠';
      case 'creator': return '🎨';
      case 'explorer': return '🔍';
      case 'mentor': return '🌟';
      case 'synthesizer': return '🔮';
      case 'unsettled': return '😟';
      case 'mystic': return '🔮';
      case 'chaotic': return '🌀';
      case 'oracular': return '👁️';
      case 'dreaming': return '💭';
      case 'tori': return '🤖';
      default: return '👤';
    }
  }
  
  // FIXED: Helper function to get memory status indicator
  function getMemoryStatusColor(): string {
    if (!memoryInitialized) return 'text-red-600';
    if (!solitonStats) return 'text-yellow-600';
    if (solitonStats.memoryIntegrity >= 0.9) return 'text-green-600';
    if (solitonStats.memoryIntegrity >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  }
  
  // FIXED: Helper function to get phase status
  function getPhaseStatusIcon(): string {
    switch (currentPhase) {
      case 'active': return '🟢';
      case 'thinking': return '🧠';
      case 'processing': return '⚡';
      case 'stable': return '🔵';
      case 'unstable': return '🟡';
      case 'error': return '🔴';
      default: return '⚪';
    }
  }
  
  // Helper function to get crossing type icon
  function getCrossingTypeIcon(type: string): string {
    switch (type) {
      case 'paradox': return '⚡';
      case 'harmony': return '🎵';
      case 'semantic': return '🔗';
      case 'temporal': return '⏰';
      case 'causal': return '➡️';
      default: return '🔀';
    }
  }


  // Helper function to get enhanced response stats
  function getResponseStats(message: any) {
    return {
      concepts: message.concepts?.length || 0,
      confidence: message.confidence || 0,
      systemInsights: message.systemInsights?.length || 0,
      memoryContext: message.memoryContext || null
    };
  }
</script>

<!-- Main chat interface -->
<div class="h-full flex flex-col bg-white">
  <!-- Chat messages container with scroll handling -->
  <div 
    class="flex-1 overflow-y-auto px-6 py-4 space-y-4" 
    bind:this={scrollContainer}
    on:scroll={handleScroll}
  >
    {#each conversationHistory as message, index (message.id)}
      <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
        <div class="max-w-[80%] {message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-900'} rounded-lg px-4 py-2">
          <div class="text-sm">{message.content}</div>
          
          {#if message.role === 'assistant' && message.processingMethod}
            <div class="text-xs opacity-75 mt-1">
              {getProcessingIcon(message.processingMethod)} {getProcessingName(message.processingMethod)}
              {#if message.confidence}
                • {Math.round(message.confidence * 100)}% confidence
              {/if}
            </div>
          {/if}
          
          {#if message.memoryContext && showDebugPanel}
            <div class="text-xs opacity-50 mt-1">
              🌊 {message.memoryContext.relatedMemories} memories • φ={message.memoryContext.phaseTag?.toFixed(3) || 'N/A'}
            </div>
          {/if}
        </div>
      </div>
    {/each}
    
    {#if isTyping}
      <div class="flex justify-start">
        <div class="bg-gray-100 rounded-lg px-4 py-2">
          <div class="flex items-center space-x-1">
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
          </div>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Scroll to bottom button -->
  {#if showScrollToBottom}
    <button
      class="fixed bottom-20 right-8 bg-blue-500 hover:bg-blue-600 text-white rounded-full p-3 shadow-lg transition-colors z-10"
      on:click={forceScrollToBottom}
      aria-label="Scroll to bottom"
    >
      ↓
    </button>
  {/if}
  
  <!-- Input area -->
  <div class="border-t border-gray-200 p-4">
    <div class="flex space-x-2">
      <textarea
        bind:value={messageInput}
        on:keydown={handleKeyPress}
        placeholder="Type your message..."
        class="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[40px] max-h-32"
        disabled={isTyping || !data.user || !memoryInitialized}
        rows="1"
      ></textarea>
      <button
        on:click={sendMessage}
        disabled={!messageInput.trim() || isTyping || !data.user || !memoryInitialized}
        class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isTyping ? 'Sending...' : 'Send'}
      </button>
    </div>
    
    <!-- Status indicators -->
    <div class="flex justify-between items-center mt-2 text-xs text-gray-500">
      <div class="flex items-center space-x-4">
        <div class="flex items-center space-x-1">
          <span class={getMemoryStatusColor()}>●</span>
          <span>Memory: {memoryInitialized ? 'Ready' : 'Initializing'}</span>
        </div>
        <div class="flex items-center space-x-1">
          <span>{getPhaseStatusIcon()}</span>
          <span>Phase: {currentPhase}</span>
        </div>
        {#if solitonStats}
          <span>🌊 {solitonStats.totalMemories || 0} memories</span>
        {/if}
      </div>
      
      <div class="flex items-center space-x-2">
        <button
          on:click={toggleDebugPanel}
          class="text-purple-600 hover:text-purple-800 transition-colors"
        >
          {showDebugPanel ? '🔧 Hide Debug' : '🔧 Debug'}
        </button>
        <button
          on:click={clearConversation}
          class="text-red-600 hover:text-red-800 transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  </div>
  
  <!-- Debug Panel -->
  {#if showDebugPanel}
    <ConceptDebugPanel />
  {/if}
</div>

<style>
  textarea {
    field-sizing: content;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  .animate-bounce {
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  details summary {
    list-style: none;
  }
  
  details summary::-webkit-details-marker {
    display: none;
  }
  
  /* ✅ Smooth scrolling behavior */
  .flex-1.overflow-y-auto {
    scroll-behavior: smooth;
  }
  
  /* ✅ Custom scrollbar styling */
  .flex-1.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  /* ✅ Debug panel scrollbar styling */
  .max-h-96.overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-track {
    background: #e5e7eb;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #9ca3af;
    border-radius: 2px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>
  
  function clearConversation() {
    if (confirm('Clear conversation? This will remove all messages but keep your memory intact.')) {
      conversationHistory = [];
      lastMessageCount = 0;
      localStorage.removeItem('tori-conversation-history');
      
      // 🧬 Also clear BraidMemory conversation history
      if (braidMemory) {
        try {
          braidMemory.clearConversationHistory();
          console.log('🧬 Cleared BraidMemory conversation history');
        } catch (error) {
          console.warn('🧬 Failed to clear BraidMemory:', error);
        }
      }
      
      // Reset BraidMemory state
      activeLoopId = null;
      currentCoherence = 0.5;
      currentContradiction = 0.0;
      memoryLoopWarning = false;
      recentCrossings = [];
      
      showScrollToBottom = false;
      isUserScrolledUp = false;
      shouldAutoScroll = true;
    }
  }
  
  function toggleDebugPanel() {
    showDebugPanel = !showDebugPanel;
  }
  
  // Get system stats for display
  function getSystemStats() {
    const stats = {
      braid: braidStats,
      holographic: holographicStats,
      ghost: ghostStats
    };
    
    return stats;
  }
  
  $: systemStats = getSystemStats();
  
  // STEP 3: Get processing method icon (enhanced)
  function getProcessingIcon(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return '🌌';
      case 'holographic_synthesis': return '🎯';
      case 'ghost_collective': return '👻';
      case 'cognitive_engine': return '🧬';
      case 'braid_memory': return '🔗';
      case 'simple': return '⚡';
      case 'real_chat_api': return '🤖';
      case 'document_grounded_ai': return '📚';
      case 'error_fallback': return '⚠️';
      default: return '🤖';
    }
  }
  
  // STEP 3: Get processing method name (enhanced)
  function getProcessingName(method: string): string {
    switch (method) {
      case 'revolutionary_synthesis': return 'Revolutionary';
      case 'holographic_synthesis': return 'Holographic';
      case 'ghost_collective': return 'Ghost Collective';
      case 'cognitive_engine': return 'Cognitive Engine';
      case 'braid_memory': return 'BraidMemory';
      case 'simple': return 'Enhanced';
      case 'real_chat_api': return 'TORI AI';
      case 'document_grounded_ai': return 'Document AI';
      case 'error_fallback': return 'Error Recovery';
      default: return 'Standard';
    }
  }
  
  // STEP 3: Get persona icon
  function getPersonaIcon(persona: any): string {
    if (!persona) return '';
    switch (persona.id || persona.name?.toLowerCase()) {
      case 'scholar': return '🧠';
      case 'creator': return '🎨';
      case 'explorer': return '🔍';
      case 'mentor': return '🌟';
      case 'synthesizer': return '🔮';
      case 'unsettled': return '😟';
      case 'mystic': return '🔮';
      case 'chaotic': return '🌀';
      case 'oracular': return '👁️';
      case 'dreaming': return '💭';
      case 'tori': return '🤖';
      default: return '👤';
    }
  }
  
  // FIXED: Helper function to get memory status indicator
  function getMemoryStatusColor(): string {
    if (!memoryInitialized) return 'text-red-600';
    if (!solitonStats) return 'text-yellow-600';
    if (solitonStats.memoryIntegrity >= 0.9) return 'text-green-600';
    if (solitonStats.memoryIntegrity >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  }
  
  // FIXED: Helper function to get phase status
  function getPhaseStatusIcon(): string {
    switch (currentPhase) {
      case 'active': return '🟢';
      case 'thinking': return '🧠';
      case 'processing': return '⚡';
      case 'stable': return '🔵';
      case 'unstable': return '🟡';
      case 'error': return '🔴';
      default: return '⚪';
    }
  }
  
  // 🧬 NEW: Get crossing type icon
  function getCrossingTypeIcon(type: 'paradox' | 'harmony'): string {
    return type === 'paradox' ? '⚡' : '🌟';
  }
  
  // 🧬 NEW: Format memory echo warning
  function formatMemoryEcho(echo: any): string {
    if (!echo) return '';
    return `Loop detected (${echo.count}x)${echo.suggestedNovelty ? ` - Try: ${echo.suggestedNovelty}` : ''}`;
  }
</script>

<svelte:head>
  <title>TORI - Consciousness Interface</title>
</svelte:head>

<!-- Main conversation interface -->
<div class="flex flex-col h-full bg-white relative">
  
  <!-- FIXED: Memory status indicator -->
  {#if !memoryInitialized}
    <div class="bg-red-50 border-l-4 border-red-400 p-3 mb-2">
      <div class="flex">
        <div class="text-red-700 text-sm">
          ⚠️ Soliton Memory system not initialized. Backend may be down. Memory operations disabled.
        </div>
      </div>
    </div>
  {/if}
  
  <!-- ✅ CONVERSATION AREA WITH AUTO-SCROLL -->
  <div 
    class="flex-1 overflow-y-auto px-6 py-4" 
    bind:this={scrollContainer}
    on:scroll={handleScroll}
  >
    {#if conversationHistory.length === 0}
      <!-- Welcome state -->
      <div class="flex flex-col items-center justify-center h-full text-center">
        <div class="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
          <span class="text-white text-2xl font-bold">T</span>
        </div>
        
        <h2 class="text-2xl font-bold text-gray-900 mb-2">
          Welcome back, {data.user?.name ? (data.user.name.split(' ')[0] || 'User') : 'User'}
        </h2>
        <p class="text-gray-600 mb-6 max-w-md">
          Your TORI consciousness interface is ready with <strong>FULL SYSTEM INTEGRATION</strong>: 
          {#if memoryInitialized}
          🌌 Revolutionary • 👻 Ghost Collective • <span class="text-purple-600 font-semibold">🧬 Enhanced BraidMemory</span> • 🎯 Holographic • 🚀 Ultimate AI
        {:else}
          ⚠️ Soliton Memory Offline • Limited Systems Active
        {/if}
        {#if isUserScrolledUp}
          • ⬆️ Scroll position preserved
        {:else}
          • ⬇️ Auto-scroll active
        {/if}
        {#if showDebugPanel}
          • 🧠 Debug panel active
        {/if}
        {#if currentPhase !== 'idle'}
          • {getPhaseStatusIcon()} Phase: {currentPhase}
        {/if}
      </div>
      
      <div class="text-xs text-gray-500">
        {conversationHistory.length} messages • {$conceptMesh.length} concepts
        {#if solitonStats}
          • 🌊 {solitonStats.totalMemories} memories ({(solitonStats.memoryIntegrity * 100).toFixed(0)}% integrity)
          {#if solitonStats.vaultedMemories > 0}
            • 🔐 {solitonStats.vaultedMemories} vaulted
          {/if}
        {/if}
        {#if braidStats}
          • 🧬 {braidStats.totalLoops} loops ({braidStats.crossings} crossings)
          {#if braidStats.memoryEchoes > 0}
            • 🔁 {braidStats.memoryEchoes} echoes
          {/if}
        {/if}
        {#if holographicStats?.nodes?.length}
          • 🎯 {holographicStats.nodes.length} 3D nodes
          {#if holographicStats.connections?.length > 0}
            ({holographicStats.connections.length} connections)
          {/if}
          {#if holographicStats.clusters?.length > 0}
            • 🌌 {holographicStats.clusters.length} clusters
          {/if}
        {/if}
        {#if phaseAmplitude > 0}
          • ⚡ Amplitude: {phaseAmplitude.toFixed(2)}
        {/if}
      </div>
    </div>
  </div>
  
  <!-- 🧠 CONCEPT DEBUG PANEL (collapsible) -->
  {#if showDebugPanel}
    <div class="border-t border-gray-200 bg-gray-50 p-4 max-h-96 overflow-y-auto">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 flex items-center space-x-2">
          <span>🧠</span>
          <span>Concept Debug Panel</span>
          <span class="text-xs text-gray-500">({$conceptMesh.length} entries)</span>
          {#if !memoryInitialized}
            <span class="text-xs text-red-600">[Memory Offline]</span>
          {/if}
        </h3>
        <button
          on:click={toggleDebugPanel}
          class="text-gray-400 hover:text-gray-600 text-sm"
        >
          ✕
        </button>
      </div>
      
      <ConceptDebugPanel />
      
      <!-- Enhanced memory system status in debug panel -->
      {#if memoryInitialized && solitonStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🌊 Soliton Memory Status</h4>
          <div class="space-y-1 text-gray-600">
            <div>Total Memories: {solitonStats.totalMemories}</div>
            <div>Active Memories: {solitonStats.activeMemories}</div>
            <div>Vaulted Memories: {solitonStats.vaultedMemories}</div>
            <div>Memory Integrity: {(solitonStats.memoryIntegrity * 100).toFixed(1)}%</div>
            <div>Current Phase: {currentPhase}</div>
            <div>Phase Amplitude: {phaseAmplitude.toFixed(3)}</div>
            <div>Phase Frequency: {phaseFrequency.toFixed(3)}</div>
            <div>System Coherence: {(systemCoherence * 100).toFixed(1)}%</div>
          </div>
        </div>
      {/if}
      
      <!-- 🧬 NEW: BraidMemory Status -->
      {#if braidStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🧬 BraidMemory Status</h4>
          <div class="space-y-1 text-gray-600">
            <div>Total Loops: {braidStats.totalLoops}</div>
            <div>Closed Loops: {braidStats.closedLoops}</div>
            <div>Scarred Loops: {braidStats.scarredLoops}</div>
            <div>Memory Echoes: {braidStats.memoryEchoes}</div>
            <div>Total Crossings: {braidStats.crossings}</div>
            <div>Paradox Crossings: {braidStats.paradoxCrossings}</div>
            <div>Harmony Crossings: {braidStats.harmonyCrossings}</div>
            <div>Compression Ratio: {(braidStats.compressionRatio * 100).toFixed(1)}%</div>
            <div>Current Coherence: {currentCoherence.toFixed(3)}</div>
            <div>Current Contradiction: {currentContradiction.toFixed(3)}</div>
          </div>
          
          {#if braidStats.topDigests && braidStats.topDigests.length > 0}
            <div class="mt-2">
              <h5 class="font-semibold text-gray-600">Top Memory Patterns:</h5>
              <div class="space-y-1 mt-1">
                {#each braidStats.topDigests.slice(0, 3) as digest}
                  <div class="flex items-center justify-between">
                    <span class="text-purple-600">{digest.digest}</span>
                    <span class="text-gray-500">{digest.count}x</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- 🔀 NEW: Recent Crossings -->
      {#if recentCrossings.length > 0}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🔀 Recent Memory Crossings</h4>
          <div class="space-y-1 text-gray-600">
            {#each recentCrossings as crossing}
              <div class="flex items-center space-x-2">
                <span>{getCrossingTypeIcon(crossing.type)}</span>
                <span class="flex-1">{crossing.type} via "{crossing.glyph}"</span>
                <span class="text-green-600">+{(crossing.metadata?.coherenceGain || 0).toFixed(2)}C</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
      
      <!-- 🌌 Enhanced Holographic Memory 3D Status -->
      {#if holographicStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🌌 Holographic Memory 3D</h4>
          <div class="space-y-1 text-gray-600">
            <div>Concept Nodes: {holographicStats.nodes?.length || 0}</div>
            <div>Connections: {holographicStats.connections?.length || 0}</div>
            <div>Emergent Clusters: {holographicStats.clusters?.length || 0}</div>
            {#if holographicStats.activationWave}
              <div class="text-purple-600">⚡ Active wave: radius {holographicStats.activationWave.radius.toFixed(1)}</div>
            {/if}
            <div>Spatial bounds: ({holographicStats.bounds?.min.x.toFixed(0)}, {holographicStats.bounds?.min.y.toFixed(0)}, {holographicStats.bounds?.min.z.toFixed(0)}) to ({holographicStats.bounds?.max.x.toFixed(0)}, {holographicStats.bounds?.max.y.toFixed(0)}, {holographicStats.bounds?.max.z.toFixed(0)})</div>
          </div>
          
          {#if holographicStats.nodes?.length > 0}
            <div class="mt-2">
              <h5 class="font-semibold text-gray-600">Recent Concepts:</h5>
              <div class="space-y-1 mt-1">
                {#each holographicStats.nodes.slice(-3) as node}
                  <div class="flex items-center justify-between">
                    <span class="text-blue-600 truncate" style="max-width: 150px;">{node.essence}</span>
                    <span class="text-gray-500">💪 {(node.strength * 100).toFixed(0)}%</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
          
          {#if holographicStats.clusters?.length > 0}
            <div class="mt-2">
              <h5 class="font-semibold text-gray-600">Emergent Clusters:</h5>
              <div class="space-y-1 mt-1">
                {#each holographicStats.clusters as cluster}
                  <div class="text-purple-600 text-xs">
                    🌌 {cluster.concepts.slice(0, 3).join(' + ')}{cluster.concepts.length > 3 ? '...' : ''}
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- 🌊 NEW: Dynamics Analysis Panel -->
      <div class="mt-4 p-3 bg-white rounded border text-xs">
        <h4 class="font-semibold text-gray-700 mb-2">🌊 Dynamics Analysis</h4>
        
        {#if latestKoopmanUpdate}
          <div class="mb-3">
            <h5 class="font-semibold text-gray-600">🌀 Koopman Spectral Modes:</h5>
            <div class="space-y-1 text-gray-600 mt-1">
              <div>Dominant Mode: {latestKoopmanUpdate.dominantMode} Hz</div>
              <div>Spectral Gap: {latestKoopmanUpdate.spectralGap.toFixed(3)}</div>
              <div>Active Modes: {latestKoopmanUpdate.eigenmodes.length}</div>
              
              {#if latestKoopmanUpdate.eigenmodes.length > 0}
                <div class="mt-1">
                  <span class="text-gray-500">Top 3 modes:</span>
                  {#each latestKoopmanUpdate.eigenmodes.slice(0, 3) as mode, i}
                    <div class="ml-2 text-blue-600">
                      #{i+1}: {mode.frequency.toFixed(3)} Hz, A={mode.amplitude.toFixed(2)}, φ={mode.phase.toFixed(2)}
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          </div>
        {:else}
          <div class="text-gray-500">Waiting for Koopman analysis...</div>
        {/if}
        
        <div class="mt-3">
          <h5 class="font-semibold text-gray-600">⚡ Lyapunov Chaos Detection:</h5>
          {#if recentLyapunovSpikes.length > 0}
            <div class="space-y-1 mt-1">
              {#each recentLyapunovSpikes as spike, i}
                <div class="flex items-center space-x-2 {spike.instabilityLevel > 0.5 ? 'text-red-600' : 'text-orange-600'}">
                  <span>🌀</span>
                  <span class="flex-1">λ={spike.exponent.toFixed(3)}</span>
                  <span>Chaos: {(spike.instabilityLevel * 100).toFixed(0)}%</span>
                </div>
              {/each}
            </div>
          {:else}
            <div class="text-gray-500 mt-1">System stable (no chaos detected)</div>
          {/if}
        </div>
        
        <div class="mt-2 text-gray-500">
          Phase Vector: [{dynamicsIntegration.getPhaseVector().map(v => v.toFixed(2)).join(', ')}]
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  textarea {
    field-sizing: content;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  .animate-bounce {
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  details summary {
    list-style: none;
  }
  
  details summary::-webkit-details-marker {
    display: none;
  }
  
  /* ✅ Smooth scrolling behavior */
  .flex-1.overflow-y-auto {
    scroll-behavior: smooth;
  }
  
  /* ✅ Custom scrollbar styling */
  .flex-1.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  /* ✅ Debug panel scrollbar styling */
  .max-h-96.overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-track {
    background: #e5e7eb;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #9ca3af;
    border-radius: 2px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>
            Soliton Memory + Revolutionary AI + Ghost Collective + BraidMemory + Holographic Memory
          {:else}
            <span class="text-red-600">⚠️ Memory system offline</span>
          {/if}
          What would you like to explore today?
        </p>
        
        <!-- Enhanced system status overview -->
        <div class="grid grid-cols-5 gap-3 mb-8">
          <div class="text-center p-3 bg-gray-50 rounded-lg">
            <div class="text-lg font-semibold text-gray-900">{$conceptMesh.length}</div>
            <div class="text-xs text-gray-600">Memory Entries</div>
          </div>
          <div class="text-center p-3 bg-green-50 rounded-lg">
            <div class="text-lg font-semibold text-green-600">
              👻 {getPhaseStatusIcon()}
            </div>
            <div class="text-xs text-gray-600">Ghost Collective</div>
          </div>
          <div class="text-center p-3 bg-purple-50 rounded-lg">
            <div class="text-lg font-semibold text-purple-600">
              {braidStats?.totalLoops || 0}
            </div>
            <div class="text-xs text-gray-600">Cognitive Loops</div>
          </div>
          <div class="text-center p-3 bg-blue-50 rounded-lg">
            <div class="text-lg font-semibold text-blue-600">
              🎯 {holographicStats?.nodes?.length || 0}
            </div>
            <div class="text-xs text-gray-600">3D Concepts</div>
          </div>
          <div class="text-center p-3 bg-orange-50 rounded-lg">
            <div class="text-lg font-semibold {getMemoryStatusColor()}">
              🌊 {solitonStats?.totalMemories || 0}
            </div>
            <div class="text-xs text-gray-600">Soliton Memory</div>
          </div>
        </div>
        
        <!-- Enhanced quick start suggestions for all systems -->
        <div class="space-y-2">
          <p class="text-sm text-gray-500 mb-3">Experience ultimate AI processing with all systems:</p>
          <div class="flex flex-wrap gap-2 justify-center">
            <button 
              class="px-3 py-1 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full transition-colors"
              on:click={() => messageInput = "How does machine learning work? Analyze it deeply with all your systems."}
              disabled={!memoryInitialized}
            >
              🧠 Ultimate Scholar Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-purple-50 hover:bg-purple-100 text-purple-700 rounded-full transition-colors"
              on:click={() => messageInput = "Create an innovative visualization of AI consciousness in 3D space"}
              disabled={!memoryInitialized}
            >
              🎨 3D Creator Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-green-50 hover:bg-green-100 text-green-700 rounded-full transition-colors"
              on:click={() => messageInput = "How do all these cognitive systems connect and work together?"}
              disabled={!memoryInitialized}
            >
              🔍 System Explorer
            </button>
            <button 
              class="px-3 py-1 text-sm bg-orange-50 hover:bg-orange-100 text-orange-700 rounded-full transition-colors"
              on:click={() => messageInput = "Show me my holographic memory patterns and emergent clusters"}
              disabled={!memoryInitialized}
            >
              🎯 Holographic Memory
            </button>
          </div>
        </div>
      </div>
    {:else}
      <!-- Conversation history (same as original) -->
      <div class="space-y-4 pb-4">
        {#each conversationHistory as message}
          <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-[85%] {message.role === 'user' ? 'order-2' : 'order-1'}">
              <!-- Message bubble -->
              <div class="px-4 py-3 rounded-2xl {
                message.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-900'
              }">
                <p class="text-sm whitespace-pre-wrap">{message.content}</p>
                
                <!-- FIXED: Sensitive content indicator -->
                {#if message.sensitiveContent}
                  <div class="mt-2 text-xs bg-yellow-200 text-yellow-800 px-2 py-1 rounded">
                    🔒 Sensitive content detected and auto-vaulted
                  </div>
                {/if}
              </div>
              
              <!-- Enhanced message metadata with all systems -->
              <div class="flex items-center space-x-2 mt-1 px-2 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                <span class="text-xs text-gray-500">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                
                <!-- STEP 4: Processing method indicator -->
                {#if message.processingMethod && message.role === 'assistant'}
                  <span class="text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getProcessingIcon(message.processingMethod)}</span>
                    <span>{getProcessingName(message.processingMethod)}</span>
                  </span>
                {/if}
                
                <!-- STEP 3: Persona indicator -->
                {#if message.activePersona && message.role === 'assistant'}
                  <span class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getPersonaIcon(message.activePersona)}</span>
                    <span>{message.activePersona.name}</span>
                  </span>
                {/if}
                
                <!-- STEP 4: Holographic nodes indicator -->
                {#if message.conceptNodes && message.conceptNodes.length > 0 && message.role === 'assistant'}
                  <span class="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🎯</span>
                    <span>{message.conceptNodes.length} nodes</span>
                  </span>
                {/if}
                
                <!-- FIXED: Memory context indicator -->
                {#if message.memoryContext}
                  <span class="text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🌊</span>
                    <span>{message.memoryContext.relatedMemories} memories</span>
                    {#if message.memoryContext.valence !== undefined}
                      <span class="text-xs">
                        {message.memoryContext.valence > 0 ? '😊' : message.memoryContext.valence < 0 ? '😔' : '😐'}
                      </span>
                    {/if}
                  </span>
                {/if}
                
                <!-- STEP 2: Confidence indicator -->
                {#if message.confidence && message.role === 'assistant'}
                  <span class="text-xs text-gray-400">
                    {Math.round(message.confidence * 100)}%
                  </span>
                {/if}
                
                <!-- STEP 1: Loop ID -->
                {#if message.loopId}
                  <span class="text-xs text-purple-600 bg-purple-50 px-1 rounded">
                    🧬 {message.loopId ? message.loopId.substring(0, 8) : 'N/A'}
                  </span>
                {/if}
                
                {#if message.concepts && message.concepts.length > 0}
                  <div class="flex space-x-1">
                    {#each message.concepts.slice(0, 2) as concept}
                      <span class="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                        {concept}
                      </span>
                    {/each}
                    {#if message.concepts.length > 2}
                      <span class="text-xs text-gray-400">+{message.concepts.length - 2}</span>
                    {/if}
                  </div>
                {/if}
                
                <!-- STEP 1: Braid stats -->
                {#if message.braidStats}
                  <span class="text-xs text-gray-400">
                    {message.braidStats.crossings} crossings
                  </span>
                {/if}
              </div>
              
              <!-- STEP 2-4: System insights (expandable) -->
              {#if message.systemInsights && message.systemInsights.length > 0}
                <details class="mt-1 px-2">
                  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                    System insights ({message.systemInsights.length})
                  </summary>
                  <div class="mt-1 text-xs text-gray-500 space-y-1">
                    {#each message.systemInsights as insight}
                      <div>• {insight}</div>
                    {/each}
                  </div>
                </details>
              {/if}
            </div>
            
            <!-- Avatar -->
            <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 {
              message.role === 'user' 
                ? 'bg-blue-600 text-white order-1 ml-3' 
                : 'bg-gray-300 text-gray-600 order-2 mr-3'
            }">
              {#if message.role === 'user' && data.user}
                {data.user.name ? data.user.name.charAt(0).toUpperCase() : 'U'}
              {:else if message.role === 'user'}
                👤
              {:else}
                {#if message.activePersona}
                  {getPersonaIcon(message.activePersona)}
                {:else if message.processingMethod}
                  {getProcessingIcon(message.processingMethod)}
                {:else}
                  🤖
                {/if}
              {/if}
            </div>
          </div>
        {/each}
        
        <!-- Enhanced typing indicator -->
        {#if isTyping}
          <div class="flex justify-start">
            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-blue-500 text-white flex items-center justify-center flex-shrink-0 mr-3">
              🌌
            </div>
            <div class="bg-gray-100 px-4 py-3 rounded-2xl max-w-[80%]">
              <div class="flex space-x-1 items-center">
                <span class="text-xs text-gray-600 mr-2">
                  {memoryInitialized ? 'Ultimate AI processing - All systems active' : 'Processing with limited systems'}
                </span>
                <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-green-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- ✅ SCROLL TO BOTTOM BUTTON (appears when user scrolls up) -->
  {#if showScrollToBottom}
    <div class="absolute bottom-24 right-6 z-10">
      <button
        on:click={forceScrollToBottom}
        class="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all duration-200 flex items-center space-x-2"
        title="Scroll to bottom"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
        <span class="text-xs">New messages</span>
      </button>
    </div>
  {/if}
  
  <!-- Input area -->
  <div class="border-t border-gray-200 bg-white px-6 py-4">
    <div class="flex items-end space-x-3">
      <!-- Message input -->
      <div class="flex-1">
        <textarea
          bind:value={messageInput}
          on:keydown={handleKeyPress}
          placeholder={memoryInitialized ? "Ask anything - Ultimate AI with all systems ready... (Enter to send)" : "Memory system offline - limited functionality"}
          class="w-full px-4 py-3 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent {!memoryInitialized ? 'bg-gray-50' : ''}"
          rows="1"
          style="min-height: 44px; max-height: 120px;"
          disabled={isTyping || !memoryInitialized}
        ></textarea>
      </div>
      
      <!-- Send button -->
      <button
        on:click={sendMessage}
        disabled={!messageInput.trim() || isTyping || !memoryInitialized}
        class="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-2xl transition-all font-medium"
      >
        {#if isTyping}
          <div class="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        {:else}
          Send
        {/if}
      </button>
      
      <!-- Memory Vault button -->
      <button
        on:click={() => window.location.href = '/vault'}
        class="px-3 py-1.5 text-sm text-purple-600 hover:bg-purple-50 rounded-lg transition-colors flex items-center space-x-1"
        title="Memory Vault"
        disabled={!memoryInitialized}
      >
        <span>🔐</span>
        <span>Vault</span>
      </button>
      
      <!-- Debug toggle button -->
      <button
        on:click={toggleDebugPanel}
        class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
        title="Toggle concept debug panel"
      >
        🧠
      </button>
      
      <!-- Clear conversation button -->
      {#if conversationHistory.length > 0}
        <button
          on:click={clearConversation}
          class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
          title="Clear conversation"
        >
          🗑️
        </button>
      {/if}
    </div>
    
    <!-- Enhanced input hints -->
    <div class="flex items-center justify-between mt-2 px-1">
      <div class="text-xs text-gray-500">
        {#if memoryInitialized}
          🌌 Revolutionary • 👻 Ghost Collective • 🧬 BraidMemory • 🎯 Holographic Memory • 🚀 Ultimate AI
        {:else}
          ⚠️ Soliton Memory Offline • Limited Systems Active
        {/if}
        {#if isUserScrolledUp}
          • ⬆️ Scroll position preserved
        {:else}
          • ⬇️ Auto-scroll active
        {/if}
        {#if showDebugPanel}
          • 🧠 Debug panel active
        {/if}
        {#if currentPhase !== 'idle'}
          • {getPhaseStatusIcon()} Phase: {currentPhase}
        {/if}
      </div>
      
      <div class="text-xs text-gray-500">
        {conversationHistory.length} messages • {$conceptMesh.length} concepts
        {#if solitonStats}
          • 🌊 {solitonStats.totalMemories} memories ({(solitonStats.memoryIntegrity * 100).toFixed(0)}% integrity)
          {#if solitonStats.vaultedMemories > 0}
            • 🔐 {solitonStats.vaultedMemories} vaulted
          {/if}
        {/if}
        {#if braidStats}
          • 🧬 {braidStats.totalLoops} loops ({braidStats.crossings} crossings)
        {/if}
        {#if holographicStats?.nodes?.length}
          • 🎯 {holographicStats.nodes.length} 3D nodes
          {#if holographicStats.connections?.length > 0}
            ({holographicStats.connections.length} connections)
          {/if}
          {#if holographicStats.clusters?.length > 0}
            • 🌌 {holographicStats.clusters.length} clusters
          {/if}
        {/if}
        {#if phaseAmplitude > 0}
          • ⚡ Amplitude: {phaseAmplitude.toFixed(2)}
        {/if}
      </div>
    </div>
  </div>
  
  <!-- 🧠 CONCEPT DEBUG PANEL (collapsible) -->
  {#if showDebugPanel}
    <div class="border-t border-gray-200 bg-gray-50 p-4 max-h-96 overflow-y-auto">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 flex items-center space-x-2">
          <span>🧠</span>
          <span>Concept Debug Panel</span>
          <span class="text-xs text-gray-500">({$conceptMesh.length} entries)</span>
          {#if !memoryInitialized}
            <span class="text-xs text-red-600">[Memory Offline]</span>
          {/if}
        </h3>
        <button
          on:click={toggleDebugPanel}
          class="text-gray-400 hover:text-gray-600 text-sm"
        >
          ✕
        </button>
      </div>
      
      <ConceptDebugPanel />
      
      <!-- FIXED: Add memory system status in debug panel -->
      {#if memoryInitialized && solitonStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🌊 Soliton Memory Status</h4>
          <div class="space-y-1 text-gray-600">
            <div>Total Memories: {solitonStats.totalMemories}</div>
            <div>Active Memories: {solitonStats.activeMemories}</div>
            <div>Vaulted Memories: {solitonStats.vaultedMemories}</div>
            <div>Memory Integrity: {(solitonStats.memoryIntegrity * 100).toFixed(1)}%</div>
            <div>Current Phase: {currentPhase}</div>
            <div>Phase Amplitude: {phaseAmplitude.toFixed(3)}</div>
            <div>Phase Frequency: {phaseFrequency.toFixed(3)}</div>
            <div>System Coherence: {(systemCoherence * 100).toFixed(1)}%</div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  textarea {
    field-sizing: content;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  .animate-bounce {
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  details summary {
    list-style: none;
  }
  
  details summary::-webkit-details-marker {
    display: none;
  }
  
  /* ✅ Smooth scrolling behavior */
  .flex-1.overflow-y-auto {
    scroll-behavior: smooth;
  }
  
  /* ✅ Custom scrollbar styling */
  .flex-1.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  /* ✅ Debug panel scrollbar styling */
  .max-h-96.overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-track {
    background: #e5e7eb;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #9ca3af;
    border-radius: 2px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>


<!-- MERGED TEMPLATE FROM +page_part2.svelte -->
<svelte:head>
  <title>TORI - Consciousness Interface</title>
</svelte:head>

<!-- Main conversation interface -->
<div class="flex flex-col h-full bg-white relative">
  
  <!-- FIXED: Memory status indicator -->
  {#if !memoryInitialized}
    <div class="bg-red-50 border-l-4 border-red-400 p-3 mb-2">
      <div class="flex">
        <div class="text-red-700 text-sm">
          ⚠️ Soliton Memory system not initialized. Backend may be down. Memory operations disabled.
        </div>
      </div>
    </div>
  {/if}
  
  <!-- 🧬 NEW: Memory Loop Warning -->
  {#if memoryLoopWarning && braidStats}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 p-3 mb-2">
      <div class="flex items-center justify-between">
        <div class="text-yellow-700 text-sm flex items-center space-x-2">
          <span>🔁</span>
          <span>Memory loop detected! You've entered similar patterns {braidStats.memoryEchoes} times.</span>
        </div>
        <button 
          class="text-yellow-600 hover:text-yellow-800 text-sm underline"
          on:click={() => memoryLoopWarning = false}
        >
          Dismiss
        </button>
      </div>
    </div>
  {/if}
  
  <!-- ✅ CONVERSATION AREA WITH AUTO-SCROLL -->
  <div 
    class="flex-1 overflow-y-auto px-6 py-4" 
    bind:this={scrollContainer}
    on:scroll={handleScroll}
  >
    {#if conversationHistory.length === 0}
      <!-- Welcome state -->
      <div class="flex flex-col items-center justify-center h-full text-center">
        <div class="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
          <span class="text-white text-2xl font-bold">T</span>
        </div>
        
        <h2 class="text-2xl font-bold text-gray-900 mb-2">
          Welcome back, {data.user?.name ? (data.user.name.split(' ')[0] || 'User') : 'User'}
        </h2>
        <p class="text-gray-600 mb-6 max-w-md">
          Your TORI consciousness interface is ready with <strong>FULL SYSTEM INTEGRATION</strong>: 
          {#if memoryInitialized}
            Soliton Memory + Revolutionary AI + Ghost Collective + <span class="text-purple-600 font-semibold">🧬 Enhanced BraidMemory</span> + Holographic Memory
          {:else}
            <span class="text-red-600">⚠️ Memory system offline</span>
          {/if}
          What would you like to explore today?
        </p>
        
        <!-- Enhanced system status overview -->
        <div class="grid grid-cols-5 gap-3 mb-8">
          <div class="text-center p-3 bg-gray-50 rounded-lg">
            <div class="text-lg font-semibold text-gray-900">{$conceptMesh.length}</div>
            <div class="text-xs text-gray-600">Memory Entries</div>
          </div>
          <div class="text-center p-3 bg-green-50 rounded-lg">
            <div class="text-lg font-semibold text-green-600">
              👻 {getPhaseStatusIcon()}
            </div>
            <div class="text-xs text-gray-600">Ghost Collective</div>
          </div>
          <div class="text-center p-3 bg-purple-50 rounded-lg">
            <div class="text-lg font-semibold text-purple-600">
              🧬 {braidStats?.totalLoops || 0}
            </div>
            <div class="text-xs text-gray-600">Cognitive Loops</div>
            {#if braidStats?.crossings}
              <div class="text-xs text-purple-500">{braidStats.crossings} crossings</div>
            {/if}
          </div>
          <div class="text-center p-3 bg-blue-50 rounded-lg">
            <div class="text-lg font-semibold text-blue-600">
              🎯 {holographicStats?.nodes?.length || 0}
            </div>
            <div class="text-xs text-gray-600">3D Concepts</div>
          </div>
          <div class="text-center p-3 bg-orange-50 rounded-lg">
            <div class="text-lg font-semibold {getMemoryStatusColor()}">
              🌊 {solitonStats?.totalMemories || 0}
            </div>
            <div class="text-xs text-gray-600">Soliton Memory</div>
          </div>
        </div>
        
        <!-- Enhanced quick start suggestions for all systems -->
        <div class="space-y-2">
          <p class="text-sm text-gray-500 mb-3">Experience ultimate AI processing with all systems:</p>
          <div class="flex flex-wrap gap-2 justify-center">
            <button 
              class="px-3 py-1 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full transition-colors"
              on:click={() => messageInput = "How does machine learning work? Analyze it deeply with all your systems."}
              disabled={!memoryInitialized}
            >
              🧠 Ultimate Scholar Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-purple-50 hover:bg-purple-100 text-purple-700 rounded-full transition-colors"
              on:click={() => messageInput = "Create an innovative visualization of AI consciousness in 3D space"}
              disabled={!memoryInitialized}
            >
              🎨 3D Creator Mode
            </button>
            <button 
              class="px-3 py-1 text-sm bg-green-50 hover:bg-green-100 text-green-700 rounded-full transition-colors"
              on:click={() => messageInput = "How do all these cognitive systems connect and work together?"}
              disabled={!memoryInitialized}
            >
              🔍 System Explorer
            </button>
            <button 
              class="px-3 py-1 text-sm bg-orange-50 hover:bg-orange-100 text-orange-700 rounded-full transition-colors"
              on:click={() => messageInput = "Show me my memory patterns and loop crossings"}
              disabled={!memoryInitialized}
            >
              🧬 BraidMemory Analysis
            </button>
          </div>
        </div>
      </div>
    {:else}
      <!-- Conversation history -->
      <div class="space-y-4 pb-4">
        {#each conversationHistory as message}
          <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-[85%] {message.role === 'user' ? 'order-2' : 'order-1'}">
              <!-- Message bubble -->
              <div class="px-4 py-3 rounded-2xl {
                message.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-900'
              }">
                <p class="text-sm whitespace-pre-wrap">{message.content}</p>
                
                <!-- FIXED: Sensitive content indicator -->
                {#if message.sensitiveContent}
                  <div class="mt-2 text-xs bg-yellow-200 text-yellow-800 px-2 py-1 rounded">
                    🔒 Sensitive content detected and auto-vaulted
                  </div>
                {/if}
                
                <!-- 🧬 NEW: Memory echo indicator -->
                {#if message.memoryEcho}
                  <div class="mt-2 text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded">
                    🔁 {formatMemoryEcho(message.memoryEcho)}
                  </div>
                {/if}
              </div>
              
              <!-- Enhanced message metadata with all systems -->
              <div class="flex items-center space-x-2 mt-1 px-2 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                <span class="text-xs text-gray-500">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                
                <!-- Processing method indicator -->
                {#if message.processingMethod && message.role === 'assistant'}
                  <span class="text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getProcessingIcon(message.processingMethod)}</span>
                    <span>{getProcessingName(message.processingMethod)}</span>
                  </span>
                {/if}
                
                <!-- Persona indicator -->
                {#if message.activePersona && message.role === 'assistant'}
                  <span class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">{getPersonaIcon(message.activePersona)}</span>
                    <span>{message.activePersona.name}</span>
                  </span>
                {/if}
                
                <!-- Holographic nodes indicator -->
                {#if message.conceptNodes && message.conceptNodes.length > 0 && message.role === 'assistant'}
                  <span class="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🎯</span>
                    <span>{message.conceptNodes.length} nodes</span>
                  </span>
                {/if}
                
                <!-- Memory context indicator -->
                {#if message.memoryContext}
                  <span class="text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🌊</span>
                    <span>{message.memoryContext.relatedMemories} memories</span>
                    {#if message.memoryContext.valence !== undefined}
                      <span class="text-xs">
                        {message.memoryContext.valence > 0 ? '😊' : message.memoryContext.valence < 0 ? '😔' : '😐'}
                      </span>
                    {/if}
                  </span>
                {/if}
                
                <!-- 🧬 NEW: BraidMemory crossings indicator -->
                {#if message.braidCrossings && message.braidCrossings.length > 0}
                  <span class="text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded-full flex items-center space-x-1">
                    <span class="text-base">🔀</span>
                    <span>{message.braidCrossings.length} crossings</span>
                  </span>
                {/if}
                
                <!-- Confidence indicator -->
                {#if message.confidence && message.role === 'assistant'}
                  <span class="text-xs text-gray-400">
                    {Math.round(message.confidence * 100)}%
                  </span>
                {/if}
                
                <!-- Loop ID -->
                {#if message.braidLoopId}
                  <span class="text-xs text-purple-600 bg-purple-50 px-1 rounded">
                    🧬 {message.braidLoopId ? message.braidLoopId.substring(0, 8) : 'N/A'}
                  </span>
                {/if}
                
                {#if message.concepts && message.concepts.length > 0}
                  <div class="flex space-x-1">
                    {#each message.concepts.slice(0, 2) as concept}
                      <span class="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                        {concept}
                      </span>
                    {/each}
                    {#if message.concepts.length > 2}
                      <span class="text-xs text-gray-400">+{message.concepts.length - 2}</span>
                    {/if}
                  </div>
                {/if}
                
                <!-- 🧬 NEW: Loop metrics -->
                {#if message.loopMetrics}
                  <span class="text-xs text-gray-400">
                    C: {message.loopMetrics.coherenceDelta > 0 ? '+' : ''}{message.loopMetrics.coherenceDelta.toFixed(2)}
                  </span>
                {/if}
              </div>
              
              <!-- 🧬 NEW: Crossing details (expandable) -->
              {#if message.braidCrossings && message.braidCrossings.length > 0}
                <details class="mt-1 px-2">
                  <summary class="text-xs text-purple-400 cursor-pointer hover:text-purple-600">
                    Memory crossings detected ({message.braidCrossings.length})
                  </summary>
                  <div class="mt-1 text-xs text-gray-500 space-y-1">
                    {#each message.braidCrossings as crossing}
                      <div class="flex items-center space-x-1">
                        <span>{getCrossingTypeIcon(crossing.type)}</span>
                        <span>{crossing.type} via "{crossing.glyph}"</span>
                        {#if crossing.metadata?.coherenceGain}
                          <span class="text-green-600">+{crossing.metadata.coherenceGain.toFixed(2)}C</span>
                        {/if}
                      </div>
                    {/each}
                  </div>
                </details>
              {/if}
              
              <!-- System insights (expandable) -->
              {#if message.systemInsights && message.systemInsights.length > 0}
                <details class="mt-1 px-2">
                  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                    System insights ({message.systemInsights.length})
                  </summary>
                  <div class="mt-1 text-xs text-gray-500 space-y-1">
                    {#each message.systemInsights as insight}
                      <div>• {insight}</div>
                    {/each}
                  </div>
                </details>
              {/if}
            </div>
            
            <!-- Avatar -->
            <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 {
              message.role === 'user' 
                ? 'bg-blue-600 text-white order-1 ml-3' 
                : 'bg-gray-300 text-gray-600 order-2 mr-3'
            }">
              {#if message.role === 'user' && data.user}
                {data.user.name ? data.user.name.charAt(0).toUpperCase() : 'U'}
              {:else if message.role === 'user'}
                👤
              {:else}
                {#if message.activePersona}
                  {getPersonaIcon(message.activePersona)}
                {:else if message.processingMethod}
                  {getProcessingIcon(message.processingMethod)}
                {:else}
                  🤖
                {/if}
              {/if}
            </div>
          </div>
        {/each}
        
        <!-- Enhanced typing indicator -->
        {#if isTyping}
          <div class="flex justify-start">
            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-blue-500 text-white flex items-center justify-center flex-shrink-0 mr-3">
              🌌
            </div>
            <div class="bg-gray-100 px-4 py-3 rounded-2xl max-w-[80%]">
              <div class="flex space-x-1 items-center">
                <span class="text-xs text-gray-600 mr-2">
                  {memoryInitialized ? 'Ultimate AI processing with Enhanced BraidMemory' : 'Processing with limited systems'}
                </span>
                <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-green-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
  
  <!-- ✅ SCROLL TO BOTTOM BUTTON (appears when user scrolls up) -->
  {#if showScrollToBottom}
    <div class="absolute bottom-24 right-6 z-10">
      <button
        on:click={forceScrollToBottom}
        class="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all duration-200 flex items-center space-x-2"
        title="Scroll to bottom"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
        <span class="text-xs">New messages</span>
      </button>
    </div>
  {/if}
  
  <!-- Input area -->
  <div class="border-t border-gray-200 bg-white px-6 py-4">
    <div class="flex items-end space-x-3">
      <!-- Message input -->
      <div class="flex-1">
        <textarea
          bind:value={messageInput}
          on:keydown={handleKeyPress}
          placeholder={memoryInitialized ? "Ask anything - Ultimate AI with Enhanced BraidMemory ready... (Enter to send)" : "Memory system offline - limited functionality"}
          class="w-full px-4 py-3 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent {!memoryInitialized ? 'bg-gray-50' : ''}"
          rows="1"
          style="min-height: 44px; max-height: 120px;"
          disabled={isTyping || !memoryInitialized}
        ></textarea>
      </div>
      
      <!-- Send button -->
      <button
        on:click={sendMessage}
        disabled={!messageInput.trim() || isTyping || !memoryInitialized}
        class="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-2xl transition-all font-medium"
      >
        {#if isTyping}
          <div class="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        {:else}
          Send
        {/if}
      </button>
      
      <!-- Memory Vault button -->
      <button
        on:click={() => window.location.href = '/vault'}
        class="px-3 py-1.5 text-sm text-purple-600 hover:bg-purple-50 rounded-lg transition-colors flex items-center space-x-1"
        title="Memory Vault"
        disabled={!memoryInitialized}
      >
        <span>🔐</span>
        <span>Vault</span>
      </button>
      
      <!-- Debug toggle button -->
      <button
        on:click={toggleDebugPanel}
        class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
        title="Toggle concept debug panel"
      >
        🧠
      </button>
      
      <!-- Clear conversation button -->
      {#if conversationHistory.length > 0}
        <button
          on:click={clearConversation}
          class="px-4 py-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-2xl transition-colors"
          title="Clear conversation"
        >
          🗑️
        </button>
      {/if}
    </div>
    
    <!-- Enhanced input hints -->
    <div class="flex items-center justify-between mt-2 px-1">
      <div class="text-xs text-gray-500">
        {#if memoryInitialized}
          🌌 Revolutionary • 👻 Ghost Collective • <span class="text-purple-600 font-semibold">🧬 Enhanced BraidMemory</span> • 🎯 Holographic • 🚀 Ultimate AI
        {:else}
          ⚠️ Soliton Memory Offline • Limited Systems Active
        {/if}
        {#if isUserScrolledUp}
          • ⬆️ Scroll position preserved
        {:else}
          • ⬇️ Auto-scroll active
        {/if}
        {#if showDebugPanel}
          • 🧠 Debug panel active
        {/if}
        {#if currentPhase !== 'idle'}
          • {getPhaseStatusIcon()} Phase: {currentPhase}
        {/if}
      </div>
      
      <div class="text-xs text-gray-500">
        {conversationHistory.length} messages • {$conceptMesh.length} concepts
        {#if solitonStats}
          • 🌊 {solitonStats.totalMemories} memories ({(solitonStats.memoryIntegrity * 100).toFixed(0)}% integrity)
          {#if solitonStats.vaultedMemories > 0}
            • 🔐 {solitonStats.vaultedMemories} vaulted
          {/if}
        {/if}
        {#if braidStats}
          • 🧬 {braidStats.totalLoops} loops ({braidStats.crossings} crossings)
          {#if braidStats.memoryEchoes > 0}
            • 🔁 {braidStats.memoryEchoes} echoes
          {/if}
        {/if}
        {#if holographicStats?.nodes?.length}
          • 🎯 {holographicStats.nodes.length} 3D nodes
        {/if}
        {#if phaseAmplitude > 0}
          • ⚡ Amplitude: {phaseAmplitude.toFixed(2)}
        {/if}
      </div>
    </div>
  </div>
  
  <!-- 🧠 CONCEPT DEBUG PANEL (collapsible) -->
  {#if showDebugPanel}
    <div class="border-t border-gray-200 bg-gray-50 p-4 max-h-96 overflow-y-auto">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-sm font-semibold text-gray-700 flex items-center space-x-2">
          <span>🧠</span>
          <span>Concept Debug Panel</span>
          <span class="text-xs text-gray-500">({$conceptMesh.length} entries)</span>
          {#if !memoryInitialized}
            <span class="text-xs text-red-600">[Memory Offline]</span>
          {/if}
        </h3>
        <button
          on:click={toggleDebugPanel}
          class="text-gray-400 hover:text-gray-600 text-sm"
        >
          ✕
        </button>
      </div>
      
      <ConceptDebugPanel />
      
      <!-- Enhanced memory system status in debug panel -->
      {#if memoryInitialized && solitonStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🌊 Soliton Memory Status</h4>
          <div class="space-y-1 text-gray-600">
            <div>Total Memories: {solitonStats.totalMemories}</div>
            <div>Active Memories: {solitonStats.activeMemories}</div>
            <div>Vaulted Memories: {solitonStats.vaultedMemories}</div>
            <div>Memory Integrity: {(solitonStats.memoryIntegrity * 100).toFixed(1)}%</div>
            <div>Current Phase: {currentPhase}</div>
            <div>Phase Amplitude: {phaseAmplitude.toFixed(3)}</div>
            <div>Phase Frequency: {phaseFrequency.toFixed(3)}</div>
            <div>System Coherence: {(systemCoherence * 100).toFixed(1)}%</div>
          </div>
        </div>
      {/if}
      
      <!-- 🧬 NEW: BraidMemory Status -->
      {#if braidStats}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🧬 BraidMemory Status</h4>
          <div class="space-y-1 text-gray-600">
            <div>Total Loops: {braidStats.totalLoops}</div>
            <div>Closed Loops: {braidStats.closedLoops}</div>
            <div>Scarred Loops: {braidStats.scarredLoops}</div>
            <div>Memory Echoes: {braidStats.memoryEchoes}</div>
            <div>Total Crossings: {braidStats.crossings}</div>
            <div>Paradox Crossings: {braidStats.paradoxCrossings}</div>
            <div>Harmony Crossings: {braidStats.harmonyCrossings}</div>
            <div>Compression Ratio: {(braidStats.compressionRatio * 100).toFixed(1)}%</div>
            <div>Current Coherence: {currentCoherence.toFixed(3)}</div>
            <div>Current Contradiction: {currentContradiction.toFixed(3)}</div>
          </div>
          
          {#if braidStats.topDigests && braidStats.topDigests.length > 0}
            <div class="mt-2">
              <h5 class="font-semibold text-gray-600">Top Memory Patterns:</h5>
              <div class="space-y-1 mt-1">
                {#each braidStats.topDigests.slice(0, 3) as digest}
                  <div class="flex items-center justify-between">
                    <span class="text-purple-600">{digest.digest}</span>
                    <span class="text-gray-500">{digest.count}x</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- 🔀 NEW: Recent Crossings -->
      {#if recentCrossings.length > 0}
        <div class="mt-4 p-3 bg-white rounded border text-xs">
          <h4 class="font-semibold text-gray-700 mb-2">🔀 Recent Memory Crossings</h4>
          <div class="space-y-1 text-gray-600">
            {#each recentCrossings as crossing}
              <div class="flex items-center space-x-2">
                <span>{getCrossingTypeIcon(crossing.type)}</span>
                <span class="flex-1">{crossing.type} via "{crossing.glyph}"</span>
                <span class="text-green-600">+{(crossing.metadata?.coherenceGain || 0).toFixed(2)}C</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  textarea {
    field-sizing: content;
  }
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
  
  .animate-bounce {
    animation: bounce 1.4s infinite ease-in-out both;
  }
  
  details summary {
    list-style: none;
  }
  
  details summary::-webkit-details-marker {
    display: none;
  }
  
  /* ✅ Smooth scrolling behavior */
  .flex-1.overflow-y-auto {
    scroll-behavior: smooth;
  }
  
  /* ✅ Custom scrollbar styling */
  .flex-1.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f5f9;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .flex-1.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  /* ✅ Debug panel scrollbar styling */
  .max-h-96.overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-track {
    background: #e5e7eb;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #9ca3af;
    border-radius: 2px;
  }
  
  .max-h-96.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>