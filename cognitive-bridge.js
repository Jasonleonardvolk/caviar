#!/usr/bin/env node
/**
 * Cognitive Engine Bridge - Node.js microservice
 * Wraps the TypeScript cognitive engine for Python FastAPI integration
 */

import express from 'express';
import cors from 'cors';
import path from 'path';

const app = express();
app.use(express.json());
app.use(cors());

// Track if engine is loaded
let cognitiveEngine = null;
let engineLoadError = null;

// Try to load the cognitive engine
async function loadCognitiveEngine() {
    try {
        console.log('🧠 Loading cognitive engine...');
        
        // Import the cognitive engine (adjust path as needed)
        const engineModule = await import('./lib/cognitive/cognitiveEngine.js');
        cognitiveEngine = engineModule.cognitiveEngine;
        
        console.log('✅ Cognitive engine loaded successfully');
        console.log('🔧 Engine stats:', cognitiveEngine.getStats());
        
        return true;
    } catch (error) {
        console.error('❌ Failed to load cognitive engine:', error);
        engineLoadError = error.message;
        return false;
    }
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        engineLoaded: !!cognitiveEngine,
        engineError: engineLoadError,
        timestamp: new Date().toISOString()
    });
});

// Main cognitive processing endpoint
app.post('/api/process', async (req, res) => {
    try {
        if (!cognitiveEngine) {
            return res.status(503).json({
                error: 'Cognitive engine not loaded',
                details: engineLoadError,
                fallback: generateFallbackResponse(req.body.message)
            });
        }

        const { message, glyphs = ['anchor', 'concept-synthesizer', 'return'], metadata = {} } = req.body;

        if (!message) {
            return res.status(400).json({
                error: 'Message is required',
                example: { message: "Hello", glyphs: ["anchor", "return"] }
            });
        }

        console.log(`🧠 Processing: "${message}" with glyphs: [${glyphs.join(', ')}]`);

        // Call the cognitive engine
        const startTime = Date.now();
        const result = await cognitiveEngine.triggerManualProcessing(message, glyphs);
        const processingTime = Date.now() - startTime;

        console.log(`✅ Completed in ${processingTime}ms - Loop ${result.id} (closed: ${result.closed})`);

        // Extract meaningful response from LoopRecord
        const response = {
            answer: generateAnswerFromLoopRecord(result, message),
            loopRecord: {
                id: result.id,
                closed: result.closed,
                scarFlag: result.scarFlag,
                processingTime: result.processingTime,
                glyphPath: result.glyphPath,
                coherenceTrace: result.coherenceTrace,
                contradictionTrace: result.contradictionTrace,
                conceptFootprint: result.metadata?.conceptFootprint || [],
                phaseGateHits: result.metadata?.phaseGateHits || [],
                coherenceGains: result.metadata?.coherenceGains || []
            },
            stats: cognitiveEngine.getStats(),
            processingTimeMs: processingTime
        };

        res.json(response);

    } catch (error) {
        console.error('❌ Processing error:', error);
        res.status(500).json({
            error: 'Processing failed',
            details: error.message,
            fallback: generateFallbackResponse(req.body.message)
        });
    }
});

// Generate meaningful answer from LoopRecord
function generateAnswerFromLoopRecord(loopRecord, originalMessage) {
    const { closed, scarFlag, coherenceTrace, contradictionTrace, metadata } = loopRecord;
    
    // Get final coherence and contradiction levels
    const finalCoherence = coherenceTrace[coherenceTrace.length - 1] || 0;
    const finalContradiction = contradictionTrace[contradictionTrace.length - 1] || 0;
    
    // Base response based on processing success
    let answer;
    
    if (closed && !scarFlag) {
        if (finalCoherence > 0.7) {
            answer = `I've processed your question about "${originalMessage}" through my cognitive reasoning system. Based on my analysis, I have high confidence in providing insights on this topic.`;
        } else if (finalCoherence > 0.4) {
            answer = `I understand your question about "${originalMessage}". My cognitive analysis shows moderate confidence in the available information.`;
        } else {
            answer = `Regarding "${originalMessage}", I can provide some perspective, though my confidence level is moderate based on the current cognitive analysis.`;
        }
    } else if (scarFlag) {
        answer = `I encountered some complexity while processing your question about "${originalMessage}". Let me provide what insights I can gather.`;
    } else {
        answer = `I'm working on understanding your question about "${originalMessage}". The cognitive processing is ongoing.`;
    }
    
    // Add context from concepts if available
    if (metadata?.conceptFootprint && metadata.conceptFootprint.length > 0) {
        const concepts = metadata.conceptFootprint.slice(0, 3).join(', ');
        answer += ` This relates to concepts like: ${concepts}.`;
    }
    
    // Add cognitive insights
    if (metadata?.coherenceGains && metadata.coherenceGains.length > 0) {
        answer += ` My reasoning process showed strong coherence development during analysis.`;
    }
    
    if (finalContradiction > 0.5) {
        answer += ` I notice some complexity and conflicting aspects in this topic that merit careful consideration.`;
    }
    
    return answer;
}

// Fallback response generator
function generateFallbackResponse(message) {
    const messageLower = message?.toLowerCase() || '';
    
    const responses = {
        'darwin': "Charles Darwin's theory of evolution by natural selection fundamentally changed our understanding of life. His observations during the voyage of the Beagle led to insights about how species adapt and change over time through natural selection.",
        'evolution': "Evolution is the process by which species change over time through natural selection. Organisms with beneficial traits are more likely to survive and reproduce, passing these advantageous characteristics to their offspring.",
        'ai': "Artificial Intelligence involves creating systems that can perform tasks typically requiring human intelligence. This includes machine learning, natural language processing, computer vision, and reasoning systems like cognitive engines.",
        'lattice': "Lattice models are mathematical frameworks used to describe regular, repeating structures. They're fundamental in physics, chemistry, and materials science for understanding crystalline structures and phase transitions.",
        'quantum': "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales, where particles exhibit both wave and particle properties. This has led to technologies like quantum computing.",
        'strategy': "Strategic thinking involves long-term planning, analyzing competitive landscapes, and making decisions that align resources with goals. It requires understanding both internal capabilities and external opportunities."
    };
    
    // Find matching response
    for (const [keyword, response] of Object.entries(responses)) {
        if (messageLower.includes(keyword)) {
            return response;
        }
    }
    
    return `I'd be happy to help you explore "${message}". While my cognitive engine is currently unavailable, I can provide general insights on this topic.`;
}

// Start server
const PORT = process.env.PORT || 4321;

async function startServer() {
    console.log('🚀 Starting Cognitive Engine Bridge...');
    
    // Try to load the engine
    await loadCognitiveEngine();
    
    app.listen(PORT, () => {
        console.log(`🌐 Cognitive Engine Bridge running on http://localhost:${PORT}`);
        console.log(`📊 Health check: http://localhost:${PORT}/health`);
        console.log(`🧠 Process endpoint: http://localhost:${PORT}/api/process`);
        console.log(`🔧 Engine loaded: ${!!cognitiveEngine}`);
    });
}

startServer().catch(console.error);
