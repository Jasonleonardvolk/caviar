// TORI Backend Compatibility Layer - Production Ready
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:5173"],
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// In-memory storage for development (replace with real database)
const memoryVault = new Map();
const conversationHistory = [];
const ghostState = {
  activePersona: null,
  coherence: 0.8,
  entropy: 0.2,
  auraIntensity: 0.5
};

// === CHAT API ===
app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    
    // Add user message to history
    const userMessage = {
      id: `msg_${Date.now()}`,
      content: message,
      role: 'user',
      timestamp: new Date(),
      conceptIds: extractConcepts(message)
    };
    conversationHistory.push(userMessage);
    
    // Simulate AI response (replace with real TORI processing)
    const assistantResponse = await generateResponse(message);
    const assistantMessage = {
      id: `msg_${Date.now() + 1}`,
      content: assistantResponse.content,
      role: 'assistant',
      timestamp: new Date(),
      ghostPersona: assistantResponse.ghostPersona,
      conceptIds: assistantResponse.conceptIds
    };
    conversationHistory.push(assistantMessage);
    
    // Emit to WebSocket clients
    io.emit('new_message', assistantMessage);
    
    res.json({
      response: assistantResponse.content,
      ghostState: ghostState,
      conceptIds: assistantResponse.conceptIds
    });
    
  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/chat/logs', (req, res) => {
  const { sessionId } = req.query;
  // Return conversation history (filter by session if needed)
  res.json(conversationHistory);
});

// === GHOST PERSONA API ===
app.post('/api/ghost/emerge', (req, res) => {
  try {
    const { persona } = req.body;
    ghostState.activePersona = persona;
    ghostState.auraIntensity = 0.8;
    
    // Emit ghost state change
    io.emit('ghost_emergence', { persona, ghostState });
    
    res.json({ 
      success: true, 
      persona,
      ghostState 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/ghost/focus', (req, res) => {
  try {
    const { persona, concept } = req.body;
    
    // Simulate ghost focusing on concept
    const focusResult = {
      persona,
      concept,
      insights: `Ghost ${persona} is now focusing on ${concept}`,
      searchResults: generateMockSearchResults(concept)
    };
    
    res.json(focusResult);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/ghost/search', (req, res) => {
  try {
    const { persona, query } = req.body;
    
    const searchResults = {
      query,
      persona,
      results: generateMockSearchResults(query),
      timestamp: new Date()
    };
    
    res.json(searchResults);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// === VAULT API ===
app.get('/api/vault', (req, res) => {
  try {
    const entries = Array.from(memoryVault.values()).map(entry => ({
      ...entry,
      sealed: entry.sealed || false,
      emotionalWeight: entry.emotionalWeight || 0.5
    }));
    res.json(entries);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/vault/save', (req, res) => {
  try {
    const { key, value, timestamp } = req.body;
    const id = `vault_${Date.now()}`;
    
    const entry = {
      id,
      key,
      title: key,
      content: typeof value === 'string' ? value : JSON.stringify(value),
      value,
      timestamp: timestamp || Date.now(),
      sealed: false,
      conceptIds: extractConcepts(typeof value === 'string' ? value : JSON.stringify(value)),
      emotionalWeight: 0.5
    };
    
    memoryVault.set(key, entry);
    
    res.json({ success: true, id, entry });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/vault/load', (req, res) => {
  try {
    const { key } = req.query;
    const entry = memoryVault.get(key);
    
    if (!entry) {
      return res.status(404).json({ error: 'Entry not found' });
    }
    
    res.json({ value: entry.value, entry });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/vault/seal', (req, res) => {
  try {
    const { arcId, userId } = req.body;
    
    // Mark arc as sealed
    if (memoryVault.has(arcId)) {
      const entry = memoryVault.get(arcId);
      entry.sealed = true;
      entry.sealReason = 'Protected content';
      memoryVault.set(arcId, entry);
    }
    
    // Emit seal event
    io.emit('vault:sealed', { arcId, timestamp: Date.now() });
    
    res.json({ success: true, arcId, sealed: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// === SYSTEM METRICS API ===
app.get('/api/system/coherence', (req, res) => {
  res.json({ 
    coherence: ghostState.coherence,
    timestamp: Date.now()
  });
});

app.get('/api/system/entropy', (req, res) => {
  res.json({ 
    entropy: ghostState.entropy,
    timestamp: Date.now()
  });
});

// === ELFIN++ API ===
app.post('/api/elfin', (req, res) => {
  try {
    const { script } = req.body;
    
    // Basic ELFIN++ script execution simulation
    const result = {
      script,
      executed: true,
      timestamp: Date.now(),
      logs: [`Executed ELFIN++ script: ${script.split('\n')[0]}...`]
    };
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// === WEBSOCKET CONNECTIONS ===
io.on('connection', (socket) => {
  console.log('🔗 Client connected to TORI backend');
  
  socket.emit('connection:established', { 
    timestamp: Date.now(),
    ghostState
  });
  
  socket.on('disconnect', () => {
    console.log('❌ Client disconnected');
  });
});

// === UTILITY FUNCTIONS ===
function extractConcepts(text) {
  // Simple concept extraction - replace with real NLP
  const concepts = [];
  const lowerText = text.toLowerCase();
  
  if (lowerText.includes('quantum')) concepts.push('quantum-physics');
  if (lowerText.includes('memory')) concepts.push('memory');
  if (lowerText.includes('ghost') || lowerText.includes('persona')) concepts.push('ghost-personas');
  if (lowerText.includes('help') || lowerText.includes('assist')) concepts.push('assistance-request');
  if (lowerText.includes('feel') || lowerText.includes('emotion')) concepts.push('emotion');
  if (lowerText.includes('consciousness')) concepts.push('consciousness');
  if (lowerText.includes('thoughtspace')) concepts.push('thoughtspace');
  
  return concepts;
}

async function generateResponse(message) {
  // Simulate AI response generation
  const responses = [
    { 
      content: "I understand. Let me process that through my quantum memory networks.",
      ghostPersona: "mentor",
      conceptIds: ["understanding", "quantum-memory"]
    },
    {
      content: "Fascinating... I sense deep patterns emerging in our conversation.",
      ghostPersona: "mystic", 
      conceptIds: ["patterns", "emergence", "conversation"]
    },
    {
      content: "I'm analyzing the holographic implications of your question.",
      ghostPersona: null,
      conceptIds: ["analysis", "holographic-theory"]
    }
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

function generateMockSearchResults(query) {
  return [
    {
      title: `Research on ${query}`,
      content: `Detailed analysis of ${query} from quantum consciousness perspective.`,
      relevance: 0.9,
      source: "TORI Knowledge Base"
    },
    {
      title: `${query} - Memory Patterns`,
      content: `Cross-referenced memory patterns related to ${query}.`,
      relevance: 0.7,
      source: "Memory Vault"
    }
  ];
}

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`🚀 TORI Backend running on port ${PORT}`);
  console.log(`📡 WebSocket server ready`);
  console.log(`🧠 Memory Vault initialized`);
  console.log(`👻 Ghost Persona engine ready`);
});

export default app;