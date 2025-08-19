# TORI Enhanced API Documentation

## Overview

The TORI Enhanced API extends the existing Prajna API with additional endpoints for matrix operations, intent-driven reasoning, and real-time monitoring via WebSockets.

## New Endpoints

### 1. Matrix Multiplication - `/multiply`

Performs hyperbolic matrix multiplication in curved space.

**Method:** `POST`

**Request Body:**
```json
{
  "matrix_a": [[1, 2], [3, 4]],
  "matrix_b": [[5, 6], [7, 8]],
  "curvature": -1.0,
  "precision": 100
}
```

**Response:**
```json
{
  "result": [[19, 22], [43, 50]],
  "computation_time": 0.023,
  "curvature_used": -1.0,
  "eigenvalues": [0.372, 53.628],
  "is_stable": false
}
```

**Features:**
- Hyperbolic matrix multiplication with configurable curvature
- Automatic eigenvalue analysis for stability
- Real-time stability warnings via WebSocket

### 2. Intent-Driven Reasoning - `/intent`

Analyzes user intent and provides reasoned responses.

**Method:** `POST`

**Request Body:**
```json
{
  "query": "How can I improve system stability?",
  "context": {
    "domain": "engineering",
    "user_role": "researcher"
  },
  "max_reasoning_depth": 3,
  "enable_chaos": true
}
```

**Response:**
```json
{
  "intent": "stability_optimization",
  "confidence": 0.85,
  "reasoning_path": [
    "Identified stability concern",
    "Analyzed system dynamics",
    "Generated optimization strategies"
  ],
  "response": "To improve system stability, consider...",
  "processing_time": 0.156,
  "chaos_used": true
}
```

**Features:**
- Deep intent analysis with configurable reasoning depth
- Optional chaos-enhanced reasoning for creative solutions
- Context-aware responses

### 3. WebSocket Channels

#### Stability Monitoring - `/ws/stability`

Real-time stability updates from the eigenvalue monitor.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8002/ws/stability');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Stability update:', data);
};
```

**Message Format:**
```json
{
  "type": "stability_update",
  "max_eigenvalue": 0.982,
  "is_stable": true,
  "stability_score": 0.75,
  "timestamp": "2025-01-18T10:30:45"
}
```

#### Chaos Events - `/ws/chaos`

Real-time chaos computation events.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8002/ws/chaos');
```

**Message Format:**
```json
{
  "type": "chaos_event",
  "mode": "dark_soliton",
  "efficiency_gain": 3.2,
  "timestamp": "2025-01-18T10:30:45"
}
```

### 4. WebSocket Test Page - `/ws/test`

Interactive HTML page for testing WebSocket connections.

**URL:** `http://localhost:8002/ws/test`

### 5. Extended Health Check - `/health/extended`

Comprehensive health check with component status.

**Method:** `GET`

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "core_available": true,
    "eigenvalue_monitor": true,
    "intent_reasoner": true,
    "chaos_controller": true,
    "websocket_manager": true,
    "prajna_integrated": true
  },
  "metrics": {
    "max_eigenvalue": 0.95,
    "stability_score": 0.8,
    "active_websockets": 3,
    "stability_subscribers": 2,
    "chaos_subscribers": 1
  },
  "timestamp": "2025-01-18T10:30:45"
}
```

## Integration with Existing Endpoints

The enhanced API maintains all existing Prajna endpoints:

- `/api/answer` - Prajna language model responses
- `/api/upload` - PDF upload and concept extraction
- `/api/health` - Basic health check
- `/api/soliton/*` - Soliton memory operations

## Example Usage

### Python Client Example

```python
import requests
import json

# Matrix multiplication
response = requests.post('http://localhost:8002/multiply', json={
    'matrix_a': [[1, 2], [3, 4]],
    'matrix_b': [[5, 6], [7, 8]],
    'curvature': -1.0
})
result = response.json()
print(f"Result: {result['result']}")
print(f"Stable: {result['is_stable']}")

# Intent reasoning
response = requests.post('http://localhost:8002/intent', json={
    'query': 'How can I optimize memory usage?',
    'enable_chaos': True
})
intent_result = response.json()
print(f"Intent: {intent_result['intent']}")
print(f"Response: {intent_result['response']}")
```

### JavaScript WebSocket Example

```javascript
// Connect to stability monitoring
const stabilityWs = new WebSocket('ws://localhost:8002/ws/stability');

stabilityWs.onopen = () => {
    console.log('Connected to stability monitor');
};

stabilityWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'stability_update') {
        console.log(`Stability: ${data.is_stable ? 'OK' : 'WARNING'}`);
        console.log(`Max eigenvalue: ${data.max_eigenvalue}`);
        
        if (!data.is_stable) {
            alert('System instability detected!');
        }
    }
};

// Send ping to keep connection alive
setInterval(() => {
    if (stabilityWs.readyState === WebSocket.OPEN) {
        stabilityWs.send('ping');
    }
}, 30000);
```

## Configuration

The enhanced API uses the same configuration as the main TORI system:

- **Port:** 8002 (default, falls back to 8003-8005)
- **Host:** 0.0.0.0 (accessible from network)
- **CORS:** Enabled for all origins
- **WebSocket:** Automatic reconnection recommended

## Error Handling

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid input)
- `404` - Endpoint not found
- `500` - Internal server error

Error responses include detailed messages:

```json
{
  "error": "Matrix dimensions incompatible",
  "detail": "Matrix A shape (2, 3) cannot multiply with Matrix B shape (2, 2)"
}
```

## Performance Considerations

1. **Matrix Operations:** Large matrices may take longer. Consider chunking for matrices > 1000x1000
2. **WebSocket Connections:** Each connection uses resources. Implement reconnection logic
3. **Intent Reasoning:** Deep reasoning (depth > 5) may be slow. Use chaos mode judiciously

## Security

- All endpoints are public by default
- Implement authentication/authorization as needed
- WebSocket connections should use authentication tokens in production
- Rate limiting recommended for production deployment

## Deployment

The enhanced API is automatically started by `enhanced_launcher.py`. For manual startup:

```bash
python main_enhanced.py
```

Or to run just the enhanced endpoints:

```bash
python -m uvicorn api.enhanced_api:app --host 0.0.0.0 --port 8002
```

## GraphQL Support

GraphQL is available as a separate service. To start:

```bash
python python/core/graphql_api.py
```

This runs on port 8080 by default and provides a GraphQL interface to all TORI components.
