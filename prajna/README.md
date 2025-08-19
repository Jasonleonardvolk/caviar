# Prajna: TORI's Voice and Language Model

**Prajna is the voice, mouth, and language model of the TORI cognitive system - the only component that speaks.**

## 🧠 What is Prajna?

Prajna is TORI's custom-trained language model that serves as the singular voice for all language output in the system. Unlike general-purpose language models, Prajna is:

- **Completely Private**: Trained from scratch using ONLY data ingested through TORI's pipeline
- **Fully Traceable**: Every answer can be traced back to specific sources in your knowledge mesh
- **Zero External Knowledge**: No pre-trained weights, no internet data, no world knowledge contamination
- **Phase-Integrated**: Uses TORI's phase dynamics for coherent memory and reasoning

### Key Principles

1. **Prajna = The Voice**: All language output comes from Prajna. No exceptions.
2. **Source Transparency**: Every statement is grounded in provided context from TORI's memory
3. **Trust Verification**: Alien overlay system audits every response for unsupported content
4. **Privacy First**: Your data never leaves your system, no external API calls

## 🏗️ Architecture Overview

```
User Query → Context Builder → Prajna (LLM) → Alien Overlay → Response
     ↓             ↓               ↓              ↓            ↓
  Frontend    Soliton Memory   Language      Trust Audit   Verified
    UI        Concept Mesh     Generation    Ghost Check    Answer
```

### Core Components

1. **Prajna API** (`prajna_api.py`) - FastAPI orchestrator with WebSocket streaming
2. **Prajna Language Model** (`prajna_mouth.py`) - The neural network that generates responses
3. **Context Builder** (`context_builder.py`) - Retrieves relevant knowledge from memory systems
4. **Soliton Memory Interface** (`soliton_interface.py`) - Phase-based personal/conversation memory
5. **Concept Mesh API** (`concept_mesh_api.py`) - Knowledge graph from ingested content
6. **Alien Overlay** (`alien_overlay.py`) - Audit system for trust and hallucination detection
7. **Frontend Integration** (`PrajnaChat.svelte`) - Complete chat interface with audit visualization

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn websockets aiohttp numpy scikit-learn torch transformers
```

### 2. Start Prajna in Demo Mode

```bash
cd /path/to/tori/kha/prajna
python start_prajna.py --demo
```

This starts Prajna with:
- Demo language model (no heavy downloads)
- Mock memory systems
- Full audit and streaming capabilities
- API at http://localhost:8001

### 3. Test the API

```bash
curl -X POST http://localhost:8001/api/answer \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is quantum phase dynamics?"}'
```

### 4. Open the Chat Interface

Navigate to your Svelte frontend and include the `PrajnaChat.svelte` component:

```svelte
<script>
  import PrajnaChat from './prajna/frontend/PrajnaChat.svelte';
</script>

<PrajnaChat />
```

## ⚙️ Configuration

### Environment Variables

```bash
# Model Configuration
export PRAJNA_MODEL_TYPE=rwkv          # rwkv, llama, gpt, custom, demo
export PRAJNA_MODEL_PATH=./models/prajna_v1.pth
export PRAJNA_DEVICE=auto              # auto, cuda, cpu, mps
export PRAJNA_TEMPERATURE=0.7

# Memory System Configuration
export SOLITON_REST_ENDPOINT=http://localhost:8002
export SOLITON_FFI_ENABLED=false
export CONCEPT_MESH_SNAPSHOT_PATH=./data/concept_mesh.pkl

# API Configuration
export PRAJNA_API_HOST=0.0.0.0
export PRAJNA_API_PORT=8001
export PRAJNA_DEBUG=false
```

### Configuration File

Create `config/prajna.json`:

```json
{
  "model_type": "rwkv",
  "model_path": "./models/prajna_v1.pth",
  "device": "auto",
  "max_context_length": 2048,
  "temperature": 0.7,
  "api_port": 8001,
  "debug_mode": false,
  "enable_streaming": true,
  "audit_trust_threshold": 0.7
}
```

Load with: `python start_prajna.py --config config/prajna.json`

## 🎯 Usage Examples

### Basic Query

```python
import asyncio
import aiohttp

async def ask_prajna(question):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8001/api/answer',
            json={'user_query': question}
        ) as response:
            result = await response.json()
            print(f"Prajna: {result['answer']}")
            print(f"Trust Score: {result['trust_score']:.2f}")
            print(f"Sources: {', '.join(result['sources'])}")

# Ask Prajna
asyncio.run(ask_prajna("Explain soliton memory dynamics"))
```

### Streaming Response

```python
import asyncio
import websockets
import json

async def stream_prajna():
    uri = "ws://localhost:8001/api/stream"
    
    async with websockets.connect(uri) as websocket:
        # Send query
        await websocket.send(json.dumps({
            "user_query": "What is the concept mesh?",
            "streaming": True
        }))
        
        # Receive streaming response
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'chunk':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'complete':
                print(f"\n\nSources: {data['sources']}")
                break

asyncio.run(stream_prajna())
```

### Using Focus Concepts

```python
response = requests.post('http://localhost:8001/api/answer', json={
    "user_query": "How does phase drift affect memory coherence?",
    "focus_concept": "phase_dynamics",
    "conversation_id": "session_123"
})
```

## 🛡️ Trust and Audit System

Prajna includes sophisticated audit capabilities to ensure response quality:

### Alien Overlay

Detects "alien" content - statements not supported by provided context:

```python
{
  "audit": {
    "trust_score": 0.85,
    "alien_detections": [
      {
        "sentence": "According to general knowledge...",
        "confidence": 0.9,
        "reason": "External knowledge indicators",
        "suggested_fix": "Rephrase using provided sources"
      }
    ],
    "supported_ratio": 0.8,
    "recommendations": ["High trust response"]
  }
}
```

### Ghost Feedback

Identifies reasoning gaps and implicit questions:

```python
{
  "ghost_overlays": {
    "leaps_detected": true,
    "ghost_questions": [
      {
        "question": "Why is phase coherence important?",
        "confidence": 0.7,
        "context_gap": "Implied but not answered"
      }
    ],
    "reasoning_gaps": ["Logical leap detected"],
    "completeness_score": 0.75
  }
}
```

## 🔧 Advanced Configuration

### Custom Model Integration

To use your own trained model:

1. **RWKV Model**:
```python
# In prajna_config.py
model_type = "rwkv"
model_path = "./models/my_prajna_rwkv.pth"
```

2. **HuggingFace Model**:
```python
model_type = "llama"
model_path = "./models/my_prajna_llama"
```

3. **Custom Architecture**:
```python
model_type = "custom"
# Implement in prajna_mouth.py -> _load_custom_model()
```

### Memory System Integration

#### Soliton Memory (REST API)
```python
soliton_rest_endpoint = "http://localhost:8002"
soliton_timeout = 10.0
```

#### Soliton Memory (FFI)
```python
soliton_ffi_enabled = True
soliton_ffi_lib_path = "./lib/libsoliton.so"
```

#### Concept Mesh
```python
concept_mesh_in_memory = True
concept_mesh_snapshot_path = "./data/concept_mesh.pkl"
concept_mesh_max_nodes = 100000
```

### Performance Tuning

```python
# Concurrency
max_concurrent_requests = 10
request_timeout = 30.0

# Context Building
max_context_snippets = 10
context_relevance_threshold = 0.3

# Audit System
audit_trust_threshold = 0.7
audit_similarity_threshold = 0.3
```

## 📊 Monitoring and Logging

### Health Check
```bash
curl http://localhost:8001/api/health
```

Response:
```json
{
  "status": "healthy",
  "prajna_model": true,
  "soliton_memory": true,
  "concept_mesh": true,
  "timestamp": 1699564800.0
}
```

### System Statistics
```bash
curl http://localhost:8001/api/stats
```

### Structured Logging

Prajna provides detailed logging:
```
2025-06-05 10:30:15 - prajna.api - INFO - 🤔 Prajna received query: What is quantum...
2025-06-05 10:30:15 - prajna.context - INFO - 🔍 Building context from TORI's memory systems...
2025-06-05 10:30:16 - prajna.soliton - INFO - 🌊 Soliton Memory returned 3 results in 0.45s
2025-06-05 10:30:16 - prajna.concept_mesh - INFO - 🕸️ Concept Mesh returned 2 results in 0.23s
2025-06-05 10:30:17 - prajna.core - INFO - 🗣️ Prajna generated 156 chars in 0.78s
2025-06-05 10:30:17 - prajna.alien_overlay - INFO - 👽 Audit complete - Trust: 0.92, Aliens: 0, Time: 0.12s
```

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Fails**
```bash
# Use demo mode for testing
python start_prajna.py --demo

# Check model path
export PRAJNA_MODEL_PATH=/correct/path/to/model.pth
```

**2. Memory Systems Unavailable**
```bash
# Check endpoints
curl http://localhost:8002/health  # Soliton Memory
curl http://localhost:8003/health  # Concept Mesh (if separate)

# Use standalone mode
python start_prajna.py --demo  # Uses mock memory systems
```

**3. CUDA/GPU Issues**
```bash
# Force CPU mode
python start_prajna.py --device cpu

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**4. Port Conflicts**
```bash
# Use different port
python start_prajna.py --port 8080

# Check what's using port 8001
netstat -tulpn | grep 8001
```

### Debug Mode

Enable detailed debugging:
```bash
python start_prajna.py --debug --log-level DEBUG
```

This provides:
- Detailed request/response logging
- Memory retrieval details
- Model inference timing
- Audit system breakdown

## 🔗 API Reference

### POST /api/answer

Generate answer from Prajna.

**Request:**
```json
{
  "user_query": "What is quantum phase dynamics?",
  "focus_concept": "quantum",
  "conversation_id": "session_123",
  "streaming": false
}
```

**Response:**
```json
{
  "answer": "Quantum phase dynamics involves...",
  "sources": ["physics_textbook.pdf", "quantum_notes.md"],
  "audit": {
    "trust_score": 0.92,
    "alien_detections": [],
    "phase_analysis": {
      "phase_drift": 0.1,
      "coherence_score": 0.9,
      "stability_index": 0.85
    },
    "supported_ratio": 0.95,
    "confidence_score": 0.88
  },
  "ghost_overlays": {
    "leaps_detected": false,
    "ghost_questions": [],
    "completeness_score": 0.9
  },
  "context_used": "Relevant context from memory...",
  "processing_time": 1.23,
  "trust_score": 0.92
}
```

### WebSocket /api/stream

Real-time streaming responses.

**Connect:** `ws://localhost:8001/api/stream`

**Send:**
```json
{
  "user_query": "Explain soliton memory",
  "streaming": true
}
```

**Receive:**
```json
{"type": "chunk", "content": "Soliton ", "timestamp": 1699564800}
{"type": "chunk", "content": "memory ", "timestamp": 1699564801}
{"type": "complete", "sources": ["memory_docs.pdf"], "audit": {...}}
```

### GET /api/health

System health check.

### GET /api/stats

System performance statistics.

## 🤝 Integration with TORI

Prajna is designed to integrate seamlessly with the broader TORI ecosystem:

### Memory Systems
- **Soliton Memory**: Phase-based personal/conversation memory
- **Concept Mesh**: Knowledge graph from ingested documents

### Data Ingestion
- PDFs, documents, textbooks
- Video transcripts
- Image OCR and descriptions
- Structured data imports

### Frontend Integration
- SvelteKit components
- Real-time streaming
- Trust score visualization
- Source citation display

## 📝 Development

### Project Structure
```
prajna/
├── api/
│   └── prajna_api.py          # FastAPI orchestrator
├── core/
│   └── prajna_mouth.py        # Language model implementation
├── memory/
│   ├── context_builder.py     # Context retrieval
│   ├── soliton_interface.py   # Soliton Memory interface
│   └── concept_mesh_api.py    # Concept Mesh interface
├── audit/
│   └── alien_overlay.py       # Trust and audit system
├── config/
│   └── prajna_config.py       # Configuration management
├── frontend/
│   └── PrajnaChat.svelte      # Chat interface
└── start_prajna.py            # Launch script
```

### Adding New Model Types

1. **Implement in `prajna_mouth.py`**:
```python
async def _load_my_model(self):
    # Your model loading logic
    self.model = MyCustomModel(self.model_path)
```

2. **Add generation method**:
```python
async def _generate_my_model(self, prompt, max_tokens):
    # Your generation logic
    return self.model.generate(prompt, max_tokens)
```

3. **Update configuration**:
```python
# In prajna_config.py
model_type = "my_model"
```

### Custom Audit Rules

Extend the alien overlay system:

```python
# In alien_overlay.py
async def _check_custom_patterns(self, sentence: str):
    # Your custom audit logic
    if "suspicious_pattern" in sentence:
        return True, 0.8, "Custom rule violation"
    return False, 0.0, ""
```

## 🚨 Production Deployment

### Security Considerations

1. **API Access Control**
```python
# Enable rate limiting
enable_rate_limiting = True
rate_limit_requests = 100  # per minute

# Input validation
enable_input_validation = True
max_query_length = 10000
```

2. **CORS Configuration**
```python
api_cors_origins = ["https://yourdomain.com"]  # Restrict origins
```

3. **Logging and Monitoring**
```python
enable_audit_logging = True
enable_performance_logging = True
log_file = "./logs/prajna.log"
```

### Performance Optimization

1. **Model Optimization**
   - Use quantization for smaller models
   - Enable GPU acceleration when available
   - Implement model caching

2. **Context Caching**
```python
enable_caching = True
cache_ttl = 3600  # 1 hour
```

3. **Concurrency Tuning**
```python
max_concurrent_requests = 50  # Based on hardware
request_timeout = 30.0
```

### Backup and Recovery

1. **Model Snapshots**
   - Regular model backups
   - Version control for model weights

2. **Memory System Backups**
   - Soliton Memory snapshots
   - Concept Mesh periodic saves

3. **Configuration Management**
   - Version controlled configs
   - Environment-specific settings

## 📄 License

This code is part of the TORI project. See main project license for details.

## 🙋‍♂️ Support

For questions or issues:

1. Check the troubleshooting section above
2. Review logs with `--debug` mode
3. Test with `--demo` mode to isolate issues
4. Verify memory system connectivity

**Remember: Prajna is TORI's voice. Every word it speaks comes from your knowledge, ensuring complete privacy and traceability.**
