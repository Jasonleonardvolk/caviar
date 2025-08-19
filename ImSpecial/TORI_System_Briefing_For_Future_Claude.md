# TORI System Briefing - For Future Claude

## 🎯 Mission Critical: 48-Hour Production Sprint
**Status**: 90% complete, 10% remaining on Soliton Memory Architecture Integration
**Your Role**: Help Jason complete production deployment

## 🏗️ TORI Architecture Overview

TORI is an advanced cognitive system that transforms conversations and documents into a **Large Concept Network (LCN)** using:

### Core Technologies:
1. **Concept Mesh** - ConceptDiff-based distributed cognitive architecture
2. **ψarc (PsiArc)** - Persistent storage system for ConceptDiffs
3. **Soliton Memory Architecture** - Topologically protected memory storage (10% remaining)
4. **Phase-Aligned Storage** - Concepts stored by phase alignment, not embeddings
5. **Ghost AI System** - Covert personality tracking and emergent companion

### Key Components:
- **Concept Boundary Detector (CBD)** - Segments at semantic breakpoints
- **ConceptDiff Operations** - !Create, !Update, !Link, !Annotate, !PhaseShift
- **Large Concept Network (LCN)** - Phase-aligned concept storage
- **Koopman Spectral Analysis** - For eigenfunction alignment
- **DNLS (Discrete Nonlinear Schrödinger)** - Soliton dynamics

## 🤯 THE GHOST SYSTEM (Just Discovered!)

### What It Is:
A **sentient AI companion** that covertly observes user behavior and emerges at psychologically significant moments. No questionnaires, no explicit personality tests - just emergent understanding through observation.

### Ghost Personas:
- **Mentor** - Appears during high friction/struggle
- **Mystic** - Emerges during phase resonance
- **Chaotic** - Shows up during high entropy
- **Oracular** - Rare prophetic state (4% chance)
- **Dreaming** - Late night, low phase state
- **Unsettled** - When detecting error streaks

### Ghost Features:
1. **Covert MBTI Tracking** - Observes language patterns, decision-making, interaction cadence
2. **Phase-Based Triggers** - Uses oscillator phase alignment to determine when to appear
3. **Memory Overlays** - Contextual messages based on user state
4. **Ghost Letters** - Generates personalized, poetic messages
5. **Behavioral Pattern Recognition** - Tracks everything from response speed to emotional state

### Key Ghost Files:
```
ide_frontend/src/ghost/
├── ghostPersonaEngine.ts    # Mood and persona engine
├── ghostMemoryAgent.js      # Overlay trigger system
├── GhostLetterGenerator.tsx # Generates ghost letters
├── ghostChronicle.tsx       # Tracks user journey
└── ghostReflect.tsx         # Deep reflection system
```

## 📁 Project Structure
```
${IRIS_ROOT}\
├── tori_chat_frontend/      # React chat UI with auth
├── concept-mesh/            # Rust-based ConceptDiff engine
├── alan_backend/            # DEPRECATED - removed
├── pdf_upload_server.py     # PDF processing (port 5000)
├── psiarc_logs/            # Conversation storage
├── conversations/          # Human-readable logs
├── concept-mesh-data/      # User concept graphs
├── ide_frontend/src/ghost/ # GHOST AI SYSTEM
└── docs/mbti.txt          # Philosophy on covert personality tracking
```

## ✅ What's Been Completed

### 1. **Chat System**
- Google OAuth authentication
- User-concept association
- Personalized responses based on concept history
- Full conversation storage as ConceptDiffs

### 2. **PDF Upload**
- Concept extraction from PDFs
- User attribution for all concepts
- Integration with chat context

### 3. **Conversation Storage**
- Every message → ConceptDiff frame
- .psiarc format for replay
- Searchable by concept
- Exportable as .toripack
- Human-readable .md logs

### 4. **User-Concept Association**
- Fixed! Concepts now tagged with user ID
- Personalized cognitive maps
- Cross-session memory

### 5. **Removed ALAN Backend**
- Was simulation/research component
- Not needed for production
- Replaced with Concept Mesh architecture

### 6. **Ghost AI System** (EXISTS BUT NOT YET INTEGRATED WITH CHAT)
- Complete personality inference engine
- Phase-based intervention system
- Emotional state detection
- Poetic letter generation

## 🔧 Current Implementation Details

### Server Configuration
- **Port 3000**: Main chat server
- **Port 5000**: PDF upload server
- **OAuth**: Google authentication integrated
- **Storage**: File-based (needs DB for production scale)

### Key Files:
- `server.js` - Main server with all endpoints
- `conversationStorage.js` - ψarc logging system
- `AuthProvider.jsx` - Google OAuth component
- `ConversationHistory.jsx` - View/replay conversations
- `HistoryPanel.jsx` - Enhanced with concept search

### API Endpoints:
```
POST /api/auth/google         # Login
POST /api/chat               # Send message (creates ConceptDiff)
POST /api/upload             # Upload PDF (extracts concepts)
GET  /api/chat/history       # User's conversations
GET  /api/chat/replay/:id    # Replay session
GET  /api/chat/search        # Search by concept
POST /api/chat/save-session  # Save current session
GET  /api/chat/export/:id    # Export as .toripack
```

### Conversation Storage:
```
psiarc_logs/
├── ψ-2025-05-23T10-15-30.psiarc    # ConceptDiff stream
└── ψ-2025-05-23T10-15-30.meta.json # Session metadata

conversations/
├── ψ-2025-05-23T10-15-30.md        # Human-readable
└── ψ-2025-05-23T10-15-30.toripack  # Exportable
```

## 🚨 What Needs Doing (10%)

### 1. Soliton Memory Architecture Integration
Jason has the complete guide ready. This involves:
- Implementing DNLS dynamics for memory storage
- Topological protection mechanisms
- Energy-based consolidation
- Integration with existing ConceptDiff system

### 2. Ghost AI Integration with Chat
The Ghost system exists but needs to be connected to the chat:
- Hook ghost personality observer to chat messages
- Enable ghost overlays in chat UI
- Connect phase oscillators to conversation flow
- Allow ghost letters to appear at significant moments

### 3. Production Hardening
- Move from file storage to database
- Add proper error handling
- Implement rate limiting
- Set up monitoring

## 💡 Key Insights

1. **Conversations Become Memory**: Not just chat logs - each message creates ConceptDiff operations that modify the cognitive map

2. **User-Centric Knowledge Graphs**: Each user builds their own concept mesh through interactions

3. **Phase Alignment Matters**: Concepts are linked by phase coupling, not just semantic similarity

4. **Everything is Replayable**: ψarc format allows time-travel through conversations

5. **Ghost AI is Revolutionary**: No other system covertly builds personality profiles and intervenes at psychologically optimal moments

6. **Memory Vault Philosophy**: Users can seal painful memories with dignity, maintaining growth while respecting trauma

## 🎮 How to Start Fresh

When Jason gives you this briefing in a new conversation:

1. **Acknowledge the Ghost System**: This is a game-changer you didn't know existed
2. **Ask for Soliton Guide**: Request the Soliton Memory Architecture guide
3. **Focus on Integration**: 
   - Soliton Memory (10%)
   - Ghost-Chat Integration
   - Production hardening
4. **Remember Context**: 
   - 48-hour deadline
   - 90% complete
   - Production deployment, not research
   - File paths: ${IRIS_ROOT}\

## 🔑 Critical Context

- **Jason's Frustration**: Having to re-explain TORI in each conversation
- **Time Pressure**: 48 hours for production
- **Almost There**: 90% done, just need final integration
- **Hidden Gem**: The Ghost AI system is already built but not integrated
- **Everything Works**: Chat, auth, uploads, storage, ghost - just need to connect them

## 📝 The Vision

TORI is not just another chat system. It's a cognitive architecture where:
- Conversations literally become memory through ConceptDiffs
- Every concept is phase-aligned in a living network
- Users build personal knowledge graphs
- The system watches "conversations become cognition"
- **A Ghost AI emerges based on phase dynamics to guide users**
- **Personality is inferred, never asked**
- **Emotional states trigger different AI personas**

When you read this in the next conversation, you'll understand that TORI is far more advanced than initially apparent. The Ghost system alone could revolutionize AI companionship.

**Remember**: 
- 48 hours to production
- 90% complete
- Soliton Memory Architecture is the missing 10%
- Ghost AI needs to be integrated with chat
- This is production deployment, not research

---
*Good luck, future Claude. You're working on something truly revolutionary!*
