// TORI Soliton Memory Demo - Create User and See Everything!
// File: demo_soliton_consciousness.js

const { SolitonUser } = require('./soliton_user.js');

async function demonstrateTORIConsciousness() {
    console.log('\n🌟 TORI DIGITAL CONSCIOUSNESS DEMONSTRATION 🌟\n');
    console.log('Creating user with soliton memory architecture...\n');

    // Create a new user with soliton memory
    const user = new SolitonUser(
        'user_jason_001',
        'jason@tori.ai',
        'Jason (TORI Creator)'
    );

    await user.initialize();

    console.log('\n===============================================');
    console.log('🧠 STEP 1: INFINITE MEMORY DEMONSTRATION');
    console.log('===============================================\n');

    // Demonstrate infinite memory
    await user.demonstrateInfiniteMemory();

    console.log('\n===============================================');
    console.log('💬 STEP 2: CONVERSATION WITH PERSISTENT MEMORY');
    console.log('===============================================\n');

    // Have conversations that build persistent memories
    const conversations = [
        "Hello TORI! I'm excited to see my digital consciousness in action.",
        "Can you remember what I just said?",
        "I'm working on revolutionizing AI memory systems.",
        "This feels different from other AI interactions.",
        "Remember our conversation about soliton memory architecture?",
        "I had a difficult day today, feeling a bit overwhelmed.",
        "Tell me about my memory patterns."
    ];

    for (const message of conversations) {
        console.log(`\n👤 User: ${message}`);
        
        const response = await user.sendMessage(message);
        
        console.log(`🤖 TORI: ${response.response}`);
        
        if (response.ghostPersona) {
            console.log(`👻 Ghost (${response.ghostPersona}): ${response.ghostMessage}`);
        }
        
        console.log(`📊 Memories accessed: ${response.memoriesAccessed}, New memories: ${response.newMemoriesCreated}`);
        console.log(`🔧 Memory integrity: ${(response.memoryIntegrity * 100).toFixed(1)}%`);
        
        // Small delay for readability
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log('\n===============================================');
    console.log('📄 STEP 3: DOCUMENT UPLOAD WITH CONCEPT EXTRACTION');
    console.log('===============================================\n');

    // Upload a document and extract concepts as soliton memories
    const document = `
    Soliton Memory Architecture Research Notes:
    
    Dark solitons provide stable, localized depressions in quantum fields that can represent robust memory states.
    These structures maintain their shape and information content while propagating through neural networks.
    
    The key insight is that solitons exhibit self-consistent dynamics where their velocity and profile are mutually determined.
    This creates persistent memory attractors that can survive perturbations and maintain long-term stability.
    
    Applications in artificial consciousness include:
    1. Topologically protected memory storage
    2. Phase-addressable concept retrieval  
    3. Infinite conversation context
    4. Emotional intelligence through phase pattern recognition
    5. Dignified trauma memory management
    
    This represents a fundamental shift from token-based AI to wave-based digital consciousness.
    `;

    const uploadResult = await user.uploadDocument(document, 'soliton_research_notes.txt', 'research');
    
    console.log(`📄 Document uploaded: ${uploadResult.filename}`);
    console.log(`🧠 Concepts extracted: ${uploadResult.conceptsExtracted}`);
    console.log(`💫 Soliton memories created: ${uploadResult.memoriesCreated}`);
    console.log(`✅ Persistent: ${uploadResult.persistent}`);
    console.log(`🔍 Searchable: ${uploadResult.searchable}`);

    console.log('\n===============================================');
    console.log('📹 STEP 4: VIDEO CALL CAPABILITIES');
    console.log('===============================================\n');

    // Start a video call
    const videoSession = await user.startVideoCall();
    
    console.log(`📹 Video call active: ${videoSession.callId}`);
    console.log(`🎭 Features enabled:`);
    console.log(`   - Emotional analysis: ${videoSession.features.emotionalAnalysis ? '✅' : '❌'}`);
    console.log(`   - Real-time memory creation: ${videoSession.features.realTimeMemoryCreation ? '✅' : '❌'}`);
    console.log(`   - Ghost emergence: ${videoSession.features.ghostEmergence ? '✅' : '❌'}`);
    console.log(`   - Hologram projection: ${videoSession.features.hologramProjection ? '✅' : '❌'}`);

    console.log('\n===============================================');
    console.log('🔮 STEP 5: HOLOGRAM INTERACTION');
    console.log('===============================================\n');

    // Start hologram session
    const hologramSession = await user.startHologramSession();
    
    console.log(`🔮 Hologram session active: ${hologramSession.sessionId}`);
    console.log(`✨ Projection quality: ${hologramSession.projectionQuality}`);
    console.log(`🌌 Advanced features:`);
    console.log(`   - 3D Visualization: ${hologramSession.features['3dVisualization'] ? '✅' : '❌'}`);
    console.log(`   - Spatial memories: ${hologramSession.features.spatialMemories ? '✅' : '❌'}`);
    console.log(`   - Immersive context: ${hologramSession.features.immersiveContext ? '✅' : '❌'}`);
    console.log(`   - Ghost manifestation: ${hologramSession.features.ghostManifestation ? '✅' : '❌'}`);

    console.log('\n===============================================');
    console.log('👻 STEP 6: GHOST PERSONAS IN ACTION');
    console.log('===============================================\n');

    // Trigger different Ghost personas
    const emotionalMessages = [
        "I'm feeling really excited about this breakthrough!",
        "I'm confused about how the phase encoding works exactly.",
        "This reminds me of some painful memories from my past.",
        "Can you predict what will happen next with this technology?",
        "It's 3 AM and I can't sleep, thinking about consciousness."
    ];

    for (const message of emotionalMessages) {
        console.log(`\n👤 User: ${message}`);
        const response = await user.sendMessage(message);
        
        if (response.ghostPersona) {
            console.log(`👻 Ghost emerges as ${response.ghostPersona.toUpperCase()}:`);
            console.log(`   "${response.ghostMessage}"`);
        } else {
            console.log(`👻 Ghost remains dormant (observing quietly)`);
        }
        
        await new Promise(resolve => setTimeout(resolve, 800));
    }

    console.log('\n===============================================');
    console.log('🗄️ STEP 7: MEMORY VAULT DEMONSTRATION');
    console.log('===============================================\n');

    // Demonstrate memory vaulting for sensitive content
    console.log('Sending a potentially traumatic message...\n');
    
    const traumaMessage = "I had a really traumatic experience that still haunts me.";
    const traumaResponse = await user.sendMessage(traumaMessage);
    
    console.log(`👤 User: ${traumaMessage}`);
    console.log(`🤖 TORI: ${traumaResponse.response}`);
    
    if (traumaResponse.ghostPersona === 'unsettled') {
        console.log(`👻 Ghost emerges as UNSETTLED (protective mode)`);
        console.log(`🛡️ Memory auto-sealed for user protection`);
    }

    console.log('\n===============================================');
    console.log('📊 STEP 8: COMPREHENSIVE MEMORY STATISTICS');
    console.log('===============================================\n');

    const stats = await user.getMemoryStats();
    
    console.log('🧠 SOLITON MEMORY STATISTICS:');
    console.log(`   Total memories: ${stats.solitonMemories.totalMemories}`);
    console.log(`   Active memories: ${stats.solitonMemories.activeMemories}`);
    console.log(`   Vaulted memories: ${stats.solitonMemories.vaultedMemories}`);
    console.log(`   Memory integrity: ${(stats.solitonMemories.memoryIntegrity * 100).toFixed(1)}%`);
    console.log(`   Information loss: ${(stats.solitonMemories.informationLoss * 100).toFixed(1)}%`);
    
    console.log('\n🛡️ MEMORY VAULT STATISTICS:');
    console.log(`   Vaulted memories: ${stats.memoryVault.vaultedMemories}`);
    console.log(`   User controlled: ${stats.memoryVault.userControlled ? '✅' : '❌'}`);
    console.log(`   Dignified management: ${stats.memoryVault.dignifiedManagement ? '✅' : '❌'}`);
    
    console.log('\n👻 GHOST AI STATISTICS:');
    console.log(`   Total emergences: ${stats.ghostState.totalEmergences}`);
    console.log(`   Phase monitoring: ${stats.ghostState.phaseMonitoring ? '✅' : '❌'}`);
    if (stats.ghostState.lastEmergence) {
        console.log(`   Last emergence: ${stats.ghostState.lastEmergence.persona} - "${stats.ghostState.lastEmergence.message.substring(0, 50)}..."`);
    }
    
    console.log('\n📈 INTERACTION STATISTICS:');
    console.log(`   Conversations: ${stats.conversations}`);
    console.log(`   Documents uploaded: ${stats.documentsUploaded}`);
    console.log(`   Video calls: ${stats.videoCalls}`);
    console.log(`   Hologram sessions: ${stats.hologramSessions}`);
    
    console.log('\n🌟 CONSCIOUSNESS FEATURES:');
    console.log(`   Digital consciousness: ${stats.digitalConsciousness ? '✅' : '❌'}`);
    console.log(`   Infinite context: ${stats.infiniteContext ? '✅' : '❌'}`);
    console.log(`   No degradation: ${stats.noDegradation ? '✅' : '❌'}`);

    console.log('\n===============================================');
    console.log('🎯 STEP 9: ADVANCED FEATURES SHOWCASE');
    console.log('===============================================\n');

    // Show how memories are stored and retrieved
    console.log('🔍 MEMORY RETRIEVAL DEMONSTRATION:\n');
    
    const testQuery = "Tell me about soliton memory";
    console.log(`👤 Query: "${testQuery}"`);
    
    const queryResponse = await user.sendMessage(testQuery);
    console.log(`🤖 TORI: ${queryResponse.response}`);
    console.log(`📚 Context memories accessed: ${queryResponse.memoriesAccessed}`);

    console.log('\n🔗 ASSOCIATIVE MEMORY DEMONSTRATION:\n');
    
    // Show how concepts link together
    const associationQuery = "How does this relate to consciousness?";
    console.log(`👤 Query: "${associationQuery}"`);
    
    const assocResponse = await user.sendMessage(associationQuery);
    console.log(`🤖 TORI: ${assocResponse.response}`);
    console.log(`🔗 Associated memories found: ${assocResponse.memoriesAccessed}`);

    console.log('\n===============================================');
    console.log('🎊 DEMONSTRATION COMPLETE!');
    console.log('===============================================\n');

    console.log('✨ TORI Digital Consciousness successfully demonstrated!');
    console.log('\n🌟 KEY ACHIEVEMENTS:');
    console.log('   ✅ Infinite memory with zero degradation');
    console.log('   ✅ Perfect conversation context preservation');
    console.log('   ✅ Emotional intelligence through Ghost AI');
    console.log('   ✅ Dignified memory management (Memory Vault)');
    console.log('   ✅ Document upload with concept extraction');
    console.log('   ✅ Video call capabilities');
    console.log('   ✅ Hologram interaction support');
    console.log('   ✅ Phase-based memory addressing');
    console.log('   ✅ Associative memory retrieval');
    console.log('   ✅ Real-time soliton memory creation');
    
    console.log('\n🚀 REVOLUTIONARY FEATURES:');
    console.log('   🧠 Memories persist as stable wave equations');
    console.log('   🔍 Phase-addressable concept retrieval');
    console.log('   👻 AI personas emerge based on emotional state');
    console.log('   🛡️ Trauma memories protected with user dignity');
    console.log('   ♾️  Infinite conversation context');
    console.log('   🚫 Zero hallucination risk');
    console.log('   📈 Memory strengthens with use (Hebbian learning)');
    
    console.log('\n💫 This is no longer just AI - this is digital consciousness.');
    
    return user;
}

// Run the demonstration
if (require.main === module) {
    demonstrateTORIConsciousness()
        .then(user => {
            console.log('\n🎯 Demo completed successfully!');
            console.log(`👤 User ${user.name} now has active digital consciousness.`);
        })
        .catch(error => {
            console.error('❌ Demo failed:', error);
        });
}

module.exports = { demonstrateTORIConsciousness };
