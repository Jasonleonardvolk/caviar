# TORI Full-Spectrum Video and Audio Ingestion System

🎬 **Complete video/audio processing system implementing the TORI Video Ingestion Blueprint**

## 🌟 Overview

This is the production-ready implementation of TORI's Full-Spectrum Video and Audio Ingestion System. It transforms raw multimedia content into rich, contextual knowledge through state-of-the-art AI processing and seamless integration with TORI's cognitive frameworks.

## ✨ Key Features

### 🎯 Core Processing
- **Multi-format Support**: MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, AAC, FLAC, OGG
- **Advanced Transcription**: OpenAI Whisper with speaker diarization
- **Visual Context Processing**: OCR, face detection, gesture analysis, slide transition detection
- **Intelligent Segmentation**: Topic-based content organization using semantic similarity
- **Real-time Streaming**: Live processing with immediate feedback via WebSocket

### 🧠 AI Analysis
- **Deep NLP**: Concept extraction, intention analysis, question detection
- **Ghost Collective**: Multi-agent reflections from different AI personas
- **Trust Layer**: Verification and integrity checking against source material
- **Semantic Understanding**: Advanced concept relationships and temporal sequences

### 🔗 Memory Integration
- **ConceptMesh**: Semantic concept networks with cross-referencing
- **BraidMemory**: Contextual memory linking and content braiding
- **LoopRecord**: Chronological event logging with time anchors
- **ψMesh**: Advanced semantic indexing and trajectory tracking
- **ScholarSphere**: Long-term knowledge archival and retrieval

### 🚀 Advanced Capabilities
- **Human-in-the-Loop**: Feedback mechanisms for continuous improvement
- **Batch Processing**: Efficient handling of multiple files
- **Live Streaming**: Real-time processing with progressive analysis
- **Cross-platform**: Windows, macOS, and Linux support

## 🛠️ Quick Start

### 1. Setup and Installation

```bash
# Clone or download the system
cd tori/kha/ingest-bus

# Run setup script (installs dependencies and models)
python setup_video_system.py --install-models

# Alternative: Manual installation
pip install -r requirements_video.txt
python -m spacy download en_core_web_sm
```

### 2. Start the System

**Windows:**
```batch
start-video-system.bat
```

**macOS/Linux:**
```bash
python main_video.py
```

### 3. Access the System

- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## 📖 API Usage

### Upload and Process Video

```bash
# Upload a video file
curl -X POST "http://localhost:8080/api/v2/video/ingest" \
     -F "file=@your_video.mp4" \
     -F "language=en" \
     -F "enable_diarization=true" \
     -F "enable_visual_context=true" \
     -F "personas=Ghost Collective,Scholar,Creator"

# Response: {"job_id": "abc123", "status": "processing"}
```

### Monitor Processing

```bash
# Check status
curl "http://localhost:8080/api/v2/video/jobs/abc123/status"

# Get complete results
curl "http://localhost:8080/api/v2/video/jobs/abc123/result"
```

### Real-time Streaming

```javascript
// WebSocket connection for live streaming
const ws = new WebSocket('ws://localhost:8080/api/v2/video/stream/live');

// Start session
ws.send(JSON.stringify({
    type: "start_session",
    options: {
        language: "en",
        enable_diarization: true,
        personas: ["Ghost Collective", "Scholar"]
    }
}));

// Send audio chunk
ws.send(JSON.stringify({
    type: "audio_chunk",
    data: base64AudioData,
    timestamp: Date.now()
}));
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TORI Video Ingestion                     │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                               │
│  ├── File Upload (REST API)                                │
│  ├── Real-time Streaming (WebSocket)                       │
│  └── Batch Processing Queue                                │
├─────────────────────────────────────────────────────────────┤
│  Processing Pipeline                                        │
│  ├── Audio Extraction (FFmpeg)                             │
│  ├── Transcription (Whisper + Diarization)                 │
│  ├── Visual Analysis (OpenCV + OCR + Face Detection)       │
│  ├── Content Segmentation (Semantic + Visual Cues)         │
│  └── NLP Analysis (spaCy + Transformers)                   │
├─────────────────────────────────────────────────────────────┤
│  AI Analysis Layer                                         │
│  ├── Concept Extraction                                    │
│  ├── Intention Analysis                                    │
│  ├── Question Detection                                    │
│  ├── Ghost Collective Reflections                          │
│  └── Trust Verification                                    │
├─────────────────────────────────────────────────────────────┤
│  Memory Integration                                         │
│  ├── ConceptMesh (Semantic Networks)                       │
│  ├── BraidMemory (Content Linking)                         │
│  ├── LoopRecord (Event Logging)                            │
│  ├── ψMesh (Advanced Indexing)                             │
│  └── ScholarSphere (Archival)                              │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                              │
│  ├── Structured Results (JSON/API)                         │
│  ├── Real-time Updates (WebSocket)                         │
│  ├── Memory Integration Hooks                              │
│  └── Human Feedback Interface                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
ingest-bus/
├── main_video.py                    # Main application entry point
├── setup_video_system.py            # Setup and installation script
├── start-video-system.bat           # Windows startup script
├── requirements_video.txt           # Video processing dependencies
├── src/
│   ├── services/
│   │   ├── video_ingestion_service.py         # Core video processing
│   │   ├── realtime_video_processor.py        # Real-time streaming
│   │   └── video_memory_integration.py        # Memory system integration
│   └── routes/
│       ├── video_ingestion.py                 # Video API endpoints
│       └── realtime_video_streaming.py        # Streaming API endpoints
├── examples/
│   └── test_video_system.py         # Example usage and testing
└── docs/
    └── video_ingestion_blueprint.md  # Complete system documentation
```

## 🔧 Configuration

### Video Processing Settings

```json
{
  "video_processing": {
    "default_language": "en",
    "enable_diarization": true,
    "enable_visual_context": true,
    "segment_threshold": 0.7,
    "quality": "balanced"
  },
  "ai_models": {
    "whisper_model": "base",
    "use_gpu": false,
    "batch_size": 16
  },
  "memory_integration": {
    "enable_concept_mesh": true,
    "enable_braid_memory": true,
    "enable_psi_mesh": true,
    "enable_scholar_sphere": true
  }
}
```

## 📊 Processing Results

The system returns comprehensive results including:

```json
{
  "video_id": "unique_id",
  "transcript": [
    {
      "id": "segment_1",
      "start_time": 0.0,
      "end_time": 5.2,
      "text": "Welcome to our presentation...",
      "speaker_id": "Speaker_A",
      "confidence": 0.95
    }
  ],
  "segments": [
    {
      "id": "content_segment_1",
      "topic": "Introduction",
      "summary": "Opening remarks and agenda overview",
      "concepts": ["presentation", "agenda", "overview"],
      "speakers": ["Speaker_A"]
    }
  ],
  "concepts": [
    {
      "term": "artificial intelligence",
      "type": "TOPIC",
      "confidence": 0.9,
      "context": "Discussion of AI applications...",
      "timestamp_ranges": [[45.2, 67.8]]
    }
  ],
  "questions": [
    "What are the key challenges in AI development?",
    "How does this approach differ from existing methods?"
  ],
  "ghost_reflections": [
    {
      "persona": "Ghost Collective",
      "message": "I'm detecting rich discussions around AI consciousness with 3 key concept areas emerging...",
      "confidence": 1.0
    }
  ],
  "integrity_score": 0.94,
  "processing_time": 45.2
}
```

## 🚀 Performance Optimization

### Hardware Recommendations

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD recommended for temporary file processing

### Configuration Tips

```python
# GPU acceleration (if available)
options = {
    "whisper_model": "base.en",  # English-only model is faster
    "use_gpu": True,
    "batch_size": 32,
    "quality": "fast"  # vs "balanced" or "accurate"
}

# For real-time streaming
streaming_options = {
    "chunk_size": 3.0,  # seconds
    "immediate_reflection": True,
    "quality": "fast"
}
```

## 🔒 Security and Privacy

- **Data Handling**: All processing is local by default
- **File Security**: Temporary files are automatically cleaned up
- **API Authentication**: Configurable authentication mechanisms
- **Privacy Mode**: Option to disable cloud-based enhancements

## 🧪 Testing

```bash
# Run the example test script
python test_video_system.py

# Test with a sample video
curl -X POST "http://localhost:8080/api/v2/video/ingest" \
     -F "file=@sample_video.mp4"

# Real-time streaming test
python examples/websocket_streaming_test.py
```

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   # Windows: Download from https://ffmpeg.org/
   # macOS: brew install ffmpeg
   # Linux: sudo apt-get install ffmpeg
   ```

2. **Whisper model download fails**
   ```bash
   # Manual download
   python -c "import whisper; whisper.load_model('base')"
   ```

3. **OCR not working**
   ```bash
   # Install Tesseract OCR
   # Windows: Download from GitHub releases
   # macOS: brew install tesseract
   # Linux: sudo apt-get install tesseract-ocr
   ```

4. **GPU acceleration not working**
   ```bash
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH=. python main_video.py --log-level debug

# Check system health
curl http://localhost:8080/health
```

## 📈 Monitoring and Analytics

The system provides comprehensive monitoring:

```bash
# System statistics
curl http://localhost:8080/api/v2/system/stats

# Memory search
curl "http://localhost:8080/api/v2/system/memory/search?query=AI&memory_types=concept,segment"

# Active processing jobs
curl http://localhost:8080/api/v2/video/active
```

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include tests for new functionality
4. Update documentation as needed

## 📚 Additional Resources

- **TORI Video Ingestion Blueprint**: Complete technical specification
- **API Documentation**: http://localhost:8080/docs
- **WebSocket Protocol**: Real-time streaming message formats
- **Memory Integration Guide**: How video content integrates with TORI's memory systems

## 🆘 Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check system health at `/health`
4. Examine logs in `video_ingestion.log`

---

**🎉 Ready to transform your video content into intelligent, searchable knowledge with TORI!**

*This system implements the complete Full-Spectrum Video and Audio Ingestion System Blueprint, providing production-ready video processing capabilities for TORI's cognitive architecture.*
