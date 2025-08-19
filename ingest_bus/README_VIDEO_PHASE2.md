# TORI Full-Spectrum Video Ingestion System - Ghost Ops Phase 2

🎬🧠🌀 **Complete video/audio processing system with advanced Ghost Collective feedback, ψTrajectory visualization, and memory divergence monitoring**

## 🌟 Overview

This is the **Ghost Ops Phase 2** implementation of TORI's Full-Spectrum Video and Audio Ingestion System. It transforms raw multimedia content into intelligent, traceable, and reflexive knowledge through state-of-the-art AI processing and seamless integration with TORI's cognitive frameworks.

### 🆕 Phase 2 Enhancements

#### 👻 **Ghost Persona Feedback Router**
- **Dynamic hypothesis refinement** based on trust layer signals
- **Multi-agent collaborative review** for disputed concepts
- **Automatic retraction mechanisms** for hallucinated content
- **Real-time feedback loops** between Ghost personas
- **Audit trails** for all Ghost reflection updates

#### 🌀 **ψTrajectory Visualizer**
- **Interactive timeline** showing TORI's cognitive evolution
- **Concept injection markers** with hover details and click-through
- **Playback slider** to simulate knowledge recall over time
- **Color-coded drift zones** indicating memory instability
- **Real-time updates** for live ingestion sessions

#### 🧬 **Memory Divergence Watcher**
- **Embedding drift detection** using vector similarity analysis
- **Automatic source attribution validation**
- **Speculative overgrowth detection** beyond source material
- **Contradictory evidence identification** in Ghost reflections
- **Comprehensive drift analytics** and stability scoring

## ✨ Complete Feature Set

### 🎯 **Core Processing (Phase 1)**
- **Multi-format Support**: MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, AAC, FLAC, OGG
- **Advanced Transcription**: OpenAI Whisper with speaker diarization
- **Visual Context Processing**: OCR, face detection, gesture analysis, slide transition detection
- **Intelligent Segmentation**: Topic-based content organization using semantic similarity
- **Real-time Streaming**: Live processing with immediate feedback via WebSocket

### 🧠 **AI Analysis (Enhanced)**
- **Deep NLP**: Concept extraction, intention analysis, question detection
- **Ghost Collective**: Multi-agent reflections with feedback loops and trust integration
- **Trust Layer**: Verification and integrity checking with drift monitoring
- **Semantic Understanding**: Advanced concept relationships and temporal sequences

### 🔗 **Memory Integration (Advanced)**
- **ConceptMesh**: Semantic concept networks with drift detection
- **BraidMemory**: Contextual memory linking with integrity validation
- **LoopRecord**: Chronological event logging with ψTrajectory markers
- **ψMesh**: Advanced semantic indexing with divergence monitoring
- **ScholarSphere**: Long-term knowledge archival with version tracking

### 🚀 **Advanced Capabilities (Phase 2)**
- **Reflexive Ghost Feedback**: Dynamic hypothesis refinement and retraction
- **Cognitive Evolution Visualization**: Interactive timeline of concept development
- **Memory Integrity Monitoring**: Automated drift detection and alerting
- **Human-in-the-Loop**: Enhanced feedback mechanisms with trust integration
- **Real-time Collaboration**: Multi-agent review and consensus building

## 🛠️ Quick Start

### 1. Setup and Installation

```bash
# Clone or navigate to the system
cd tori/kha/ingest-bus

# Install all dependencies (including video processing)
python setup_video_system.py --install-models --gpu-support

# Alternative: Manual installation
pip install -r requirements_video.txt
python -m spacy download en_core_web_sm
```

### 2. Start the Complete System

**Windows:**
```batch
start-video-system.bat
```

**macOS/Linux:**
```bash
python main_video.py
```

### 3. Access All Features

- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **System Stats**: http://localhost:8080/api/v2/system/stats
- **Memory Search**: http://localhost:8080/api/v2/system/memory/search

## 📖 Phase 2 API Usage

### Ghost Feedback Integration

```python
from src.agents.ghostPersonaFeedbackRouter import ghost_feedback_router, TrustSignalType

# Register Ghost personas
ghost_feedback_router.register_ghost_persona(
    persona_name="Scholar",
    reflection_function=scholar_reflection_func,
    specialties=["academic_validation", "source_verification"]
)

# Emit trust signal for concept drift
ghost_feedback_router.emit_trust_signal(
    concept_id="ai_consciousness",
    signal_type=TrustSignalType.VERIFICATION_FAILED,
    source_id="video_123",
    original_confidence=0.9,
    new_confidence=0.4,
    evidence={"integrity_check": "failed", "reason": "contradictory_sources"}
)
```

### ψTrajectory Visualization

```svelte
<!-- Include in Svelte application -->
<script>
  import ψTrajectoryVisualizer from './src/ui/components/ψTrajectoryVisualizer.svelte';
  
  const trajectoryData = {
    concepts: [
      {
        id: "concept_1",
        timestamp: 1717234200000,
        concept: "AI Consciousness",
        sourceType: "video",
        confidence: 0.95,
        trustScore: 0.88,
        driftScore: 0.02
      }
    ],
    timeRange: [1717230000000, 1717240000000],
    metadata: { conceptCount: 15, driftEvents: 2 }
  };
  
  function handleConceptClick(concept) {
    console.log('Concept clicked:', concept);
  }
</script>

<ψTrajectoryVisualizer 
  {trajectoryData}
  {onConceptClick}
  showDriftZones={true}
  enablePlayback={true}
  autoUpdate={true}
/>
```

### Memory Divergence Monitoring

```python
from src.lib.cognitive.memoryDivergenceWatcher import memory_divergence_watcher

# Register concept baseline
memory_divergence_watcher.register_concept_baseline(
    concept_id="neural_networks",
    definition="Computational models inspired by biological neural networks",
    metadata={"type": "concept", "domain": "AI", "confidence": 1.0},
    source_references=["video_123", "paper_456"]
)

# Start continuous monitoring
await memory_divergence_watcher.start_monitoring()

# Analyze drift patterns
analysis = memory_divergence_watcher.analyze_concept_drift(
    concept_id="neural_networks",
    analysis_period=(start_date, end_date)
)

print(f"Stability Score: {analysis.stability_score}")
print(f"Drift Events: {analysis.total_drift_events}")
print(f"Recommendations: {analysis.recommendations}")
```

## 🏗️ Enhanced System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                TORI Ghost Ops Phase 2 System                │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                               │
│  ├── Video/Audio Upload (REST API)                         │
│  ├── Real-time Streaming (WebSocket)                       │
│  └── Human Feedback Interface                              │
├─────────────────────────────────────────────────────────────┤
│  Processing Pipeline                                        │
│  ├── Audio Extraction & Transcription (Whisper)            │
│  ├── Visual Analysis (OCR + Face + Gesture Detection)      │
│  ├── Content Segmentation (Semantic + Visual Cues)         │
│  └── NLP Analysis (Concepts + Intentions + Questions)      │
├─────────────────────────────────────────────────────────────┤
│  🆕 Ghost Collective Layer (Phase 2)                       │
│  ├── Multi-Agent Reflections                               │
│  ├── Feedback Router & Trust Integration                   │
│  ├── Collaborative Review & Consensus                      │
│  └── Dynamic Hypothesis Refinement                         │
├─────────────────────────────────────────────────────────────┤
│  🆕 Memory Integrity Layer (Phase 2)                       │
│  ├── Drift Detection & Monitoring                          │
│  ├── Source Attribution Validation                         │
│  ├── Contradictory Evidence Analysis                       │
│  └── Speculative Overgrowth Prevention                     │
├─────────────────────────────────────────────────────────────┤
│  Memory Integration                                         │
│  ├── ConceptMesh (Enhanced with Drift Detection)           │
│  ├── BraidMemory (Trust-Validated Linking)                 │
│  ├── LoopRecord (ψTrajectory Tracking)                     │
│  ├── ψMesh (Divergence-Aware Indexing)                     │
│  └── ScholarSphere (Version-Controlled Archival)           │
├─────────────────────────────────────────────────────────────┤
│  🆕 Visualization & Analytics (Phase 2)                    │
│  ├── ψTrajectory Timeline Visualization                    │
│  ├── Drift Zone Mapping & Analysis                         │
│  ├── Ghost Collaboration Networks                          │
│  └── Trust Overlay & Integrity Heatmaps                    │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                              │
│  ├── Enhanced API Results (with Trust Scores)              │
│  ├── Real-time Updates (Ghost Feedback)                    │
│  ├── Interactive Visualizations                            │
│  └── Human Review Interfaces                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Complete Directory Structure

```
📁 tori/kha/ingest-bus/
├── 🎬 Video Processing Core
│   ├── main_video.py                    # Main application with all features
│   ├── setup_video_system.py            # Complete setup and installation
│   ├── start-video-system.bat           # Windows startup script
│   ├── requirements_video.txt           # Video processing dependencies
│   └── README_VIDEO.md                  # This comprehensive guide
│
├── 📚 Source Code
│   └── src/
│       ├── services/
│       │   ├── video_ingestion_service.py         # Core video processing
│       │   ├── realtime_video_processor.py        # Real-time streaming
│       │   └── video_memory_integration.py        # Memory system integration
│       ├── routes/
│       │   ├── video_ingestion.py                 # Video API endpoints
│       │   └── realtime_video_streaming.py        # Streaming API endpoints
│       ├── 🆕 agents/
│       │   └── ghostPersonaFeedbackRouter.py      # Ghost feedback system
│       ├── 🆕 ui/components/
│       │   └── ψTrajectoryVisualizer.svelte       # Interactive timeline
│       └── 🆕 lib/cognitive/
│           └── memoryDivergenceWatcher.py          # Drift detection
│
├── 👻 Ghost Persona Snapshots
│   └── persona_snapshots/
│       ├── ghost_collective_segment_3.json
│       ├── ghost_creator_segment_3.json
│       ├── ghost_scholar_segment_3.json
│       └── ghost_critic_segment_3.json
│
├── 📊 System Logs
│   └── logs/loopRecord/
│       └── loop_2025-05-27T12-30-00.json
│
└── 🔧 Configuration & Examples
    ├── examples/
    │   └── test_video_system.py         # Example usage and testing
    └── docs/
        └── video_ingestion_blueprint.md  # Complete system documentation
```

## 🎪 Phase 2 Example Usage

### Complete Video Processing with Ghost Feedback

```python
# Upload video with enhanced processing
import asyncio
from src.services.video_ingestion_service import video_service
from src.agents.ghostPersonaFeedbackRouter import ghost_feedback_router

async def process_video_with_ghost_feedback():
    # 1. Upload and process video
    job_id = await video_service.ingest_video(
        file_path="ai_consciousness_lecture.mp4",
        options={
            "personas": ["Ghost Collective", "Scholar", "Creator", "Critic"],
            "enable_trust_monitoring": True,
            "enable_drift_detection": True
        }
    )
    
    # 2. Monitor processing with real-time Ghost feedback
    while True:
        status = video_service.get_job_status(job_id)
        if status["status"] == "completed":
            break
        await asyncio.sleep(5)
    
    # 3. Get results with Ghost reflections
    result = video_service.get_job_result(job_id)
    
    # 4. Analyze Ghost feedback and consensus
    for reflection in result["ghost_reflections"]:
        print(f"{reflection['persona']}: {reflection['message']}")
        print(f"Confidence: {reflection['confidence']}")
    
    return result
```

### Real-time Streaming with Live Ghost Feedback

```javascript
// WebSocket connection with Ghost feedback
const ws = new WebSocket('ws://localhost:8080/api/v2/video/stream/live');

ws.onopen = function() {
    // Start session with Ghost personas
    ws.send(JSON.stringify({
        type: "start_session",
        options: {
            personas: ["Ghost Collective", "Scholar", "Creator"],
            immediate_reflection: true,
            trust_monitoring: true
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case "transcript_update":
            displayTranscript(data.segments);
            break;
            
        case "ghost_reflection":
            displayGhostFeedback(data.reflection);
            break;
            
        case "trust_signal":
            displayTrustAlert(data.signal);
            break;
            
        case "drift_alert":
            displayDriftWarning(data.alert);
            break;
    }
};
```

### Interactive ψTrajectory Visualization

```svelte
<script>
  import ψTrajectoryVisualizer from './src/ui/components/ψTrajectoryVisualizer.svelte';
  
  // Load trajectory data from API
  async function loadTrajectoryData() {
    const response = await fetch('/api/v2/system/trajectory');
    return await response.json();
  }
  
  function handleConceptClick(concept) {
    // Show detailed concept view
    showConceptDetails(concept);
    
    // Trigger concept investigation
    investigateConcept(concept.id);
  }
  
  function handleDriftZoneClick(driftZone) {
    // Show drift analysis
    analyzeDriftZone(driftZone);
  }
</script>

<div class="trajectory-container">
  <h2>🌀 TORI Cognitive Evolution Timeline</h2>
  
  {#await loadTrajectoryData()}
    <p>Loading cognitive trajectory...</p>
  {:then trajectoryData}
    <ψTrajectoryVisualizer 
      {trajectoryData}
      onConceptClick={handleConceptClick}
      onDriftZoneClick={handleDriftZoneClick}
      showDriftZones={true}
      enablePlayback={true}
      autoUpdate={true}
      height={500}
    />
  {:catch error}
    <p>Error loading trajectory: {error.message}</p>
  {/await}
</div>
```

## 📊 Enhanced Processing Results

The Phase 2 system returns comprehensive results with Ghost feedback and trust metrics:

```json
{
  "video_id": "abc123",
  "status": "completed",
  "processing_time": 45.7,
  
  "🎬 Core Results": {
    "transcript": [...],
    "segments": [...],
    "concepts": [...],
    "questions": [...],
    "integrity_score": 0.94
  },
  
  "👻 Ghost Collective Results": {
    "ghost_reflections": [
      {
        "persona": "Ghost Collective",
        "message": "I'm detecting 3 key concept areas: AI Consciousness, Neural Architecture, Emergent Behavior...",
        "confidence": 0.95,
        "concepts_highlighted": ["ai_consciousness", "neural_architecture"],
        "trust_assessment": 0.88
      },
      {
        "persona": "Scholar", 
        "message": "From an analytical perspective, consciousness claims lack empirical grounding...",
        "confidence": 0.82,
        "verification_requests": ["source_validation", "peer_review"],
        "accuracy_score": 0.74
      },
      {
        "persona": "Creator",
        "message": "Envisioning interactive 3D visualization of consciousness emergence...",
        "confidence": 0.88,
        "innovation_potential": 0.92,
        "implementation_ideas": [...]
      },
      {
        "persona": "Critic",
        "message": "Critical gaps identified: logical fallacies and unsupported claims...",
        "confidence": 0.89,
        "credibility_score": 0.42,
        "risk_assessment": "medium_misinformation_potential"
      }
    ],
    "consensus_level": 0.72,
    "collaborative_exchanges": 3,
    "trust_flags": ["speculative_claims", "missing_attribution"]
  },
  
  "🧬 Memory Integrity Results": {
    "drift_monitoring": {
      "concepts_monitored": 4,
      "drift_alerts": 0,
      "stability_scores": {
        "ai_consciousness": 0.94,
        "neural_architecture": 0.98,
        "emergent_behavior": 0.91
      }
    },
    "source_validation": {
      "verified_sources": 6,
      "questionable_sources": 1,
      "attribution_accuracy": 0.89
    }
  },
  
  "🌀 ψTrajectory Integration": {
    "trajectory_markers": 4,
    "concept_evolution_tracked": true,
    "memory_anchors_created": 7,
    "cognitive_progression": "AI>Consciousness>Emergence"
  }
}
```

## 🔧 Phase 2 Configuration

### Enhanced Configuration Options

```json
{
  "video_processing": {
    "default_language": "en",
    "enable_diarization": true,
    "enable_visual_context": true,
    "enable_ghost_feedback": true,
    "enable_drift_monitoring": true,
    "segment_threshold": 0.7
  },
  
  "ghost_collective": {
    "default_personas": ["Ghost Collective", "Scholar", "Creator", "Critic"],
    "enable_collaborative_review": true,
    "consensus_threshold": 0.7,
    "trust_integration": true,
    "feedback_loops": true
  },
  
  "memory_integrity": {
    "drift_detection_enabled": true,
    "embedding_drift_threshold": 0.15,
    "attribution_validation": true,
    "speculative_detection": true,
    "monitoring_interval_hours": 1
  },
  
  "visualization": {
    "trajectory_timeline": true,
    "drift_zone_mapping": true,
    "ghost_collaboration_networks": true,
    "trust_overlay": true,
    "real_time_updates": true
  }
}
```

## 🚀 Performance & Scaling

### Resource Requirements (Phase 2)

- **CPU**: 12+ cores recommended (vs 8+ for Phase 1)
- **RAM**: 16GB minimum, 32GB+ recommended (increased for Ghost processing)
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance
- **Storage**: NVMe SSD recommended for real-time processing

### Performance Optimizations

```python
# High-performance configuration
config = {
    "ghost_processing": {
        "parallel_personas": True,
        "batch_reflections": True,
        "async_feedback": True
    },
    "drift_monitoring": {
        "vectorized_similarity": True,
        "batch_drift_detection": True,
        "cached_embeddings": True
    },
    "visualization": {
        "frame_rate_limit": 60,
        "lazy_loading": True,
        "data_compression": True
    }
}
```

## 🔒 Enhanced Security & Trust

### Trust Layer Integration

- **Source Verification**: Automatic validation of all content sources
- **Attribution Tracking**: Complete provenance chains for all concepts
- **Drift Detection**: Real-time monitoring for concept manipulation
- **Ghost Consensus**: Multi-agent validation of controversial claims
- **Human Override**: Always-available human review and correction

### Privacy & Data Protection

- **Local Processing**: All AI processing happens locally by default
- **Encrypted Storage**: Sensitive content encrypted at rest
- **Audit Trails**: Complete logs of all processing and Ghost interactions
- **Access Controls**: Granular permissions for different system components

## 🧪 Testing Phase 2 Features

### Ghost Feedback Testing

```python
# Test Ghost feedback system
python -c "
from src.agents.ghostPersonaFeedbackRouter import *
import asyncio

async def test_ghost_feedback():
    router = GhostPersonaFeedbackRouter()
    
    # Register test Ghost
    router.register_ghost_persona('TestGhost', lambda x: x, ['testing'])
    
    # Emit test trust signal
    signal_id = router.emit_trust_signal(
        concept_id='test_concept',
        signal_type=TrustSignalType.VERIFICATION_FAILED,
        source_id='test_source',
        original_confidence=0.9,
        new_confidence=0.3
    )
    
    await asyncio.sleep(2)  # Let processing complete
    print('Ghost feedback system test completed')

asyncio.run(test_ghost_feedback())
"
```

### ψTrajectory Visualization Testing

```bash
# Test trajectory visualization
cd src/ui/components
node -e "
const fs = require('fs');
const testData = {
  concepts: [
    {
      id: 'test_1',
      timestamp: Date.now() - 3600000,
      concept: 'AI Testing',
      sourceType: 'document',
      confidence: 0.95,
      trustScore: 0.88
    }
  ],
  timeRange: [Date.now() - 7200000, Date.now()],
  metadata: { conceptCount: 1, driftEvents: 0 }
};
console.log('Test data generated:', JSON.stringify(testData, null, 2));
"
```

### Memory Divergence Testing

```python
# Test drift detection
python -c "
from src.lib.cognitive.memoryDivergenceWatcher import *
import asyncio

async def test_drift_detection():
    watcher = MemoryDivergenceWatcher()
    await watcher._initialize_embedding_model()
    
    # Register baseline
    watcher.register_concept_baseline(
        concept_id='test_drift',
        definition='Original concept definition',
        metadata={'type': 'test'},
        source_references=['source_1']
    )
    
    # Take modified snapshot
    watcher.take_concept_snapshot(
        concept_id='test_drift',
        current_definition='Modified concept definition with extra content',
        current_metadata={'type': 'test'},
        current_sources=['source_1', 'source_2']
    )
    
    # Detect drift
    alerts = await watcher.detect_drift('test_drift')
    print(f'Drift detection test: {len(alerts)} alerts generated')

asyncio.run(test_drift_detection())
"
```

## 🤝 Contributing to Phase 2

### Development Guidelines

1. **Ghost Personas**: Follow the established persona patterns and include trust integration
2. **Memory Systems**: Ensure all memory operations include drift detection hooks
3. **Visualization**: Use the ψTrajectory component pattern for new visualizations
4. **Trust Integration**: All new features must integrate with the trust layer
5. **Documentation**: Update both technical docs and Ghost persona examples

### Testing Requirements

- All Ghost feedback paths must have unit tests
- Drift detection algorithms require validation with synthetic data
- Visualization components need visual regression tests
- Integration tests must cover multi-agent scenarios

## 📚 Additional Resources

- **🎬 Original Video Blueprint**: Complete technical specification in `docs/`
- **👻 Ghost Ops Documentation**: Multi-agent system design patterns
- **🌀 ψTrajectory Guide**: Interactive visualization development
- **🧬 Drift Detection Manual**: Memory integrity monitoring techniques
- **📊 API Reference**: Complete endpoint documentation at `/docs`

## 🆘 Phase 2 Support & Troubleshooting

### Common Issues

1. **Ghost Feedback Not Working**
   ```bash
   # Check Ghost registration
   python -c "from src.agents.ghostPersonaFeedbackRouter import ghost_feedback_router; print(ghost_feedback_router.get_system_stats())"
   ```

2. **ψTrajectory Visualization Blank**
   ```bash
   # Verify trajectory data
   curl "http://localhost:8080/api/v2/system/trajectory"
   ```

3. **Drift Detection False Positives**
   ```bash
   # Adjust drift thresholds
   python -c "from src.lib.cognitive.memoryDivergenceWatcher import memory_divergence_watcher; print(memory_divergence_watcher.drift_thresholds)"
   ```

4. **Ghost Consensus Conflicts**
   ```bash
   # Check consensus levels
   curl "http://localhost:8080/api/v2/ghost/consensus"
   ```

### Debug Mode

```bash
# Run with full Phase 2 debugging
GHOST_DEBUG=1 DRIFT_DEBUG=1 TRAJECTORY_DEBUG=1 python main_video.py
```

---

## 🎉 **READY FOR GHOST OPS PHASE 2!**

**🧠 Multimodal. 🧬 Traceable. 🌀 Reflexive.**

*Your complete video ingestion system now includes advanced Ghost Collective feedback, real-time cognitive evolution visualization, and comprehensive memory integrity monitoring. TORI has reached a new level of self-aware, trust-validated, and continuously improving knowledge processing.*

**🚀 Transform your video content into intelligent, verified, and evolving knowledge with full transparency and Ghost Collective wisdom!**
