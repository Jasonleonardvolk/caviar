/**
 * 🧠 TORI Cognitive Engine Microservice
 * Node.js/TypeScript microservice for serving the cognitive engine to external systems
 * Supports FastAPI, Python, and other language integrations
 */

import express from 'express';
import cors from 'cors';
import { cognitiveEngine, cognitive } from '../index';
import type { LoopRecord } from '../loopRecord';

const app = express();
const PORT = process.env.COGNITIVE_PORT || 4321;

// Enhanced middleware
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:8000', 'http://localhost:3000'],
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Enhanced logging middleware
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`🔍 [${timestamp}] ${req.method} ${req.path}`);
  next();
});

// ===== CORE COGNITIVE ENGINE ENDPOINTS =====

/**
 * Main cognitive processing endpoint
 * POST /api/engine
 */
app.post('/api/engine', async (req, res) => {
  try {
    const { message, glyphs, metadata } = req.body;
    
    if (!message || !glyphs) {
      return res.status(400).json({
        error: 'Missing required fields: message, glyphs',
        required: { message: 'string', glyphs: 'string[]' }
      });
    }
    
    console.log(`🧠 Processing cognitive request: "${message}" with ${glyphs.length} glyphs`);
    
    // Trigger manual processing
    const result = await cognitiveEngine.triggerManualProcessing(message, glyphs);
    
    // Enhanced response with full trace
    const response = {
      success: true,
      answer: generateAnswerSummary(result),
      trace: {
        loopId: result.id,
        prompt: result.prompt,
        glyphPath: result.glyphPath,
        closed: result.closed,
        scarFlag: result.scarFlag,
        processingTime: result.processingTime,
        coherenceTrace: result.coherenceTrace,
        contradictionTrace: result.contradictionTrace,
        phaseTrace: result.phaseTrace,
        metadata: result.metadata
      },
      fullLoop: result,
      timestamp: new Date().toISOString(),
      cognitive: {
        engine: await cognitiveEngine.getStats(),
        memory: cognitive.memory?.getStats?.() || null,
        ghosts: cognitive.ghosts?.getDiagnostics?.() || null,
        holographic: cognitive.holographic?.getVisualizationData?.() || null
      }
    };
    
    res.json(response);
    
  } catch (error) {
    console.error('❌ Cognitive processing error:', error);
    res.status(500).json({
      error: 'Cognitive processing failed',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Batch cognitive processing endpoint
 * POST /api/engine/batch
 */
app.post('/api/engine/batch', async (req, res) => {
  try {
    const { requests } = req.body;
    
    if (!Array.isArray(requests)) {
      return res.status(400).json({
        error: 'Batch requests must be an array',
        format: { requests: [{ message: 'string', glyphs: 'string[]' }] }
      });
    }
    
    console.log(`🧠 Processing ${requests.length} batch cognitive requests`);
    
    const results = [];
    
    for (const request of requests) {
      try {
        const { message, glyphs, metadata } = request;
        const result = await cognitiveEngine.triggerManualProcessing(message, glyphs);
        
        results.push({
          success: true,
          answer: generateAnswerSummary(result),
          trace: {
            loopId: result.id,
            closed: result.closed,
            processingTime: result.processingTime
          }
        });
      } catch (error) {
        results.push({
          success: false,
          error: error.message
        });
      }
    }
    
    res.json({
      success: true,
      results,
      processed: results.length,
      successful: results.filter(r => r.success).length,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Batch processing error:', error);
    res.status(500).json({
      error: 'Batch processing failed',
      details: error.message
    });
  }
});

/**
 * Stream cognitive processing with real-time updates
 * POST /api/engine/stream
 */
app.post('/api/engine/stream', async (req, res) => {
  const { message, glyphs } = req.body;
  
  // Set up Server-Sent Events
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*'
  });
  
  try {
    // Send initial status
    res.write(`data: ${JSON.stringify({
      type: 'start',
      message: `Starting cognitive processing: ${message}`,
      glyphs: glyphs.length
    })}\n\n`);
    
    // Mock streaming by processing with updates
    for (let i = 0; i < glyphs.length; i++) {
      res.write(`data: ${JSON.stringify({
        type: 'progress',
        glyph: glyphs[i],
        index: i,
        total: glyphs.length,
        progress: (i + 1) / glyphs.length
      })}\n\n`);
      
      await new Promise(resolve => setTimeout(resolve, 100)); // Simulate processing time
    }
    
    // Execute actual processing
    const result = await cognitiveEngine.triggerManualProcessing(message, glyphs);
    
    // Send final result
    res.write(`data: ${JSON.stringify({
      type: 'complete',
      answer: generateAnswerSummary(result),
      trace: result,
      timestamp: new Date().toISOString()
    })}\n\n`);
    
    res.end();
    
  } catch (error) {
    res.write(`data: ${JSON.stringify({
      type: 'error',
      error: error.message
    })}\n\n`);
    res.end();
  }
});

// ===== SYSTEM STATUS AND DIAGNOSTICS =====

/**
 * Get comprehensive system status
 * GET /api/status
 */
app.get('/api/status', async (req, res) => {
  try {
    const status = {
      service: 'TORI Cognitive Engine Microservice',
      version: '1.0.0',
      status: 'online',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cognitive: {
        engine: await cognitiveEngine.getStats(),
        exportedState: cognitiveEngine.exportCognitiveState()
      },
      integrations: {
        braidMemory: !!cognitive.memory,
        ghostCollective: !!cognitive.ghosts,
        holographicMemory: !!cognitive.holographic
      },
      capabilities: {
        symbolicProcessing: true,
        memoryIntegration: true,
        personaSystem: true,
        holographicMemory: true,
        realTimeStreaming: true,
        batchProcessing: true,
        crossLanguageAPI: true
      }
    };
    
    res.json(status);
  } catch (error) {
    res.status(500).json({
      error: 'Status check failed',
      details: error.message
    });
  }
});

/**
 * Health check endpoint
 * GET /api/health
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

/**
 * Get full system metrics
 * GET /api/metrics
 */
app.get('/api/metrics', async (req, res) => {
  try {
    const metrics = {
      cognitive: await cognitiveEngine.getStats(),
      system: {
        memory: process.memoryUsage(),
        uptime: process.uptime(),
        version: process.version
      },
      integrations: {}
    };
    
    // Add subsystem metrics if available
    if (cognitive.memory?.getStats) {
      metrics.integrations.braidMemory = cognitive.memory.getStats();
    }
    
    if (cognitive.ghosts?.getDiagnostics) {
      metrics.integrations.ghostCollective = cognitive.ghosts.getDiagnostics();
    }
    
    if (cognitive.holographic?.getVisualizationData) {
      metrics.integrations.holographicMemory = cognitive.holographic.getVisualizationData();
    }
    
    res.json(metrics);
  } catch (error) {
    res.status(500).json({
      error: 'Metrics collection failed',
      details: error.message
    });
  }
});

// ===== MEMORY AND CONCEPT ENDPOINTS =====

/**
 * Query braid memory
 * GET /api/memory/query?digest=...&limit=...
 */
app.get('/api/memory/query', (req, res) => {
  try {
    const { digest, limit = 10 } = req.query;
    
    if (!cognitive.memory) {
      return res.status(503).json({ error: 'Braid memory not available' });
    }
    
    // Query memory system
    const results = cognitive.memory.queryByDigest?.(digest as string, parseInt(limit as string)) || [];
    
    res.json({
      success: true,
      results,
      query: { digest, limit },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Memory query failed',
      details: error.message
    });
  }
});

/**
 * Get holographic memory visualization
 * GET /api/memory/holographic
 */
app.get('/api/memory/holographic', (req, res) => {
  try {
    if (!cognitive.holographic) {
      return res.status(503).json({ error: 'Holographic memory not available' });
    }
    
    const data = cognitive.holographic.getVisualizationData();
    
    res.json({
      success: true,
      data,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Holographic memory visualization failed',
      details: error.message
    });
  }
});

/**
 * Create concept in holographic memory
 * POST /api/memory/concept
 */
app.post('/api/memory/concept', (req, res) => {
  try {
    const { essence, activationLevel = 0.5 } = req.body;
    
    if (!cognitive.holographic) {
      return res.status(503).json({ error: 'Holographic memory not available' });
    }
    
    if (!essence) {
      return res.status(400).json({ error: 'Concept essence is required' });
    }
    
    const node = cognitive.holographic.createConceptNode(essence, activationLevel);
    
    res.json({
      success: true,
      node,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Concept creation failed',
      details: error.message
    });
  }
});

// ===== PERSONA AND GHOST COLLECTIVE ENDPOINTS =====

/**
 * Query ghost collective for persona selection
 * POST /api/ghosts/query
 */
app.post('/api/ghosts/query', (req, res) => {
  try {
    const { query } = req.body;
    
    if (!cognitive.ghosts) {
      return res.status(503).json({ error: 'Ghost collective not available' });
    }
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    const persona = cognitive.ghosts.selectPersonaForQuery(query);
    
    res.json({
      success: true,
      persona,
      query,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Ghost collective query failed',
      details: error.message
    });
  }
});

/**
 * Get all available personas
 * GET /api/ghosts/personas
 */
app.get('/api/ghosts/personas', (req, res) => {
  try {
    if (!cognitive.ghosts) {
      return res.status(503).json({ error: 'Ghost collective not available' });
    }
    
    const diagnostics = cognitive.ghosts.getDiagnostics();
    
    res.json({
      success: true,
      personas: diagnostics.personas || [],
      stats: {
        totalPersonas: diagnostics.personas?.length || 0,
        activePersonas: diagnostics.activePersonas || 0
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Persona listing failed',
      details: error.message
    });
  }
});

// ===== UTILITY FUNCTIONS =====

/**
 * Generate a human-readable answer summary from a loop record
 */
function generateAnswerSummary(loop: LoopRecord): string {
  if (loop.scarFlag) {
    return `Processing encountered difficulties while working with: "${loop.prompt}". The system needed to abort processing due to instability.`;
  }
  
  if (!loop.closed) {
    return `Processing of "${loop.prompt}" is still in progress. The cognitive loop has not yet reached closure.`;
  }
  
  const coherenceLevel = loop.coherenceTrace[loop.coherenceTrace.length - 1] || 0;
  const contradictionLevel = loop.contradictionTrace[loop.contradictionTrace.length - 1] || 0;
  
  let summary = `Successfully processed "${loop.prompt}" using ${loop.glyphPath.length} symbolic operations.`;
  
  if (coherenceLevel > 0.8) {
    summary += ' The analysis achieved high coherence and clarity.';
  } else if (coherenceLevel > 0.6) {
    summary += ' The analysis achieved moderate coherence.';
  } else {
    summary += ' The analysis completed with some complexity remaining.';
  }
  
  if (contradictionLevel < 0.1) {
    summary += ' No significant contradictions were detected.';
  } else if (contradictionLevel < 0.3) {
    summary += ' Minor contradictions were resolved during processing.';
  } else {
    summary += ' Some contradictions remain that may require further analysis.';
  }
  
  // Add metadata insights
  if (loop.metadata?.conceptFootprint && loop.metadata.conceptFootprint.length > 0) {
    summary += ` Key concepts involved: ${loop.metadata.conceptFootprint.slice(0, 3).join(', ')}.`;
  }
  
  if (loop.metadata?.coherenceGains && loop.metadata.coherenceGains.length > 0) {
    summary += ` Achieved ${loop.metadata.coherenceGains.length} significant insights during processing.`;
  }
  
  return summary;
}

// ===== ERROR HANDLING AND GRACEFUL SHUTDOWN =====

app.use((err: any, req: any, res: any, next: any) => {
  console.error('❌ Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    details: process.env.NODE_ENV === 'development' ? err.message : 'Service temporarily unavailable'
  });
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Received SIGINT, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

// Start the server
app.listen(PORT, () => {
  console.log(`🧠 TORI Cognitive Engine Microservice listening on port ${PORT}`);
  console.log(`🌐 API endpoints available at http://localhost:${PORT}/api/`);
  console.log(`📊 Status: http://localhost:${PORT}/api/status`);
  console.log(`❤️ Health: http://localhost:${PORT}/api/health`);
  console.log(`📈 Metrics: http://localhost:${PORT}/api/metrics`);
  console.log('✅ Ready for cognitive processing requests!');
});

export default app;
