from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, List, Optional, Set, Any, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import asyncio
import json
import uuid
# # import redis // FIXED: We use file-based storage only // FIXED: We use file-based storage only
from redis.asyncio import Redis
import numpy as np
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

# Import oscillator system
from audio.spectral_oscillator import BanksyOscillator, create_oscillator_for_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v2/hologram', tags=['hologram'])

# Redis connection for session management (production-ready)
redis_client: Optional[Redis] = None

# Thread pool for compute-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Session oscillator storage
session_oscillators: Dict[str, BanksyOscillator] = {}
oscillator_lock = asyncio.Lock()

async def get_redis() -> Redis:
    global redis_client
    if not redis_client:
        redis_client = await Redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
    return redis_client

# Models with validation
class HologramCalibration(BaseModel):
    pitch: float = Field(default=49.825, ge=0, le=100)
    tilt: float = Field(default=-0.1745, ge=-1, le=1)
    center: float = Field(default=0.04239, ge=-1, le=1)
    subp: float = Field(default=0.013, ge=0, le=1)
    displayAspect: float = Field(default=16/9, ge=0.1, le=10)
    numViews: int = Field(default=45, ge=1, le=100)
    
    @validator('displayAspect')
    def validate_aspect(cls, v):
        if v <= 0:
            raise ValueError('Display aspect must be positive')
        return v

class HologramSession(BaseModel):
    session_id: str
    psi_state: Dict
    hologram_hints: Dict
    active_views: List[Dict]
    calibration: Optional[HologramCalibration] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    client_count: int = 0
    status: str = "active"  # active, paused, stopped
    performance_metrics: Dict = Field(default_factory=dict)

class HologramStartRequest(BaseModel):
    display_type: str = "auto"  # auto, looking_glass, custom, webgpu_only
    calibration: Optional[HologramCalibration] = None
    quality: str = "high"  # low, medium, high, ultra
    session_type: str = "realtime"  # realtime, playback, hybrid
    max_clients: int = Field(default=10, ge=1, le=100)

class PsiStateUpdate(BaseModel):
    psi_phase: Optional[float] = Field(None, ge=0, le=2*np.pi)
    psi_magnitude: Optional[float] = Field(None, ge=0, le=1)
    phase_coherence: Optional[float] = Field(None, ge=0, le=1)
    oscillator_phases: Optional[List[float]] = None
    dominant_frequency: Optional[float] = Field(None, ge=20, le=20000)
    emotional_resonance: Optional[Dict[str, float]] = None
    
    @validator('oscillator_phases')
    def validate_phases(cls, v):
        if v and any(phase < 0 or phase > 2*np.pi for phase in v):
            raise ValueError('Oscillator phases must be between 0 and 2π')
        return v

class WavefieldParameters(BaseModel):
    phase_modulation: float = Field(..., ge=0, le=2*np.pi)
    coherence: float = Field(..., ge=0, le=1)
    oscillator_phases: List[float]
    dominant_freq: float = Field(..., ge=20, le=20000)
    spatial_frequencies: List[Tuple[float, float]]
    
class WavefieldUpdate(BaseModel):
    centroid: float = Field(..., ge=20, le=20000, description="Spectral centroid in Hz")
    emotion_intensity: float = Field(..., ge=0, le=1, description="Emotion strength")
    rms: Optional[float] = Field(None, ge=0, le=1, description="RMS amplitude")
    context: str = Field(default="general", description="Audio context: general, voice, music")
    compute_interference: bool = Field(default=False, description="Compute full interference pattern")
    grid_size: Optional[int] = Field(None, ge=32, le=512, description="Grid size for interference pattern")
    
class WavefieldResponse(BaseModel):
    status: str
    wavefield_params: WavefieldParameters
    interference_pattern: Optional[Dict[str, Any]] = None
    phase_gradients: Optional[Dict[str, Any]] = None
    adaptive_mapping: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float]
    timestamp: datetime

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_info: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, client_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        self.connection_info[websocket] = {
            'session_id': session_id,
            'client_id': client_id,
            'connected_at': datetime.utcnow()
        }
    
    def disconnect(self, websocket: WebSocket):
        info = self.connection_info.get(websocket)
        if info:
            session_id = info['session_id']
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            del self.connection_info[websocket]
    
    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn)
    
    def get_session_client_count(self, session_id: str) -> int:
        return len(self.active_connections.get(session_id, set()))

manager = ConnectionManager()

# Oscillator management functions
async def get_oscillator_for_session(session_id: str, context: str = "general") -> BanksyOscillator:
    """Get or create oscillator for a session"""
    async with oscillator_lock:
        if session_id not in session_oscillators:
            session_oscillators[session_id] = create_oscillator_for_context(context)
            logger.info(f"Created {context} oscillator for session {session_id}")
        return session_oscillators[session_id]

async def cleanup_oscillator(session_id: str):
    """Clean up oscillator when session ends"""
    async with oscillator_lock:
        if session_id in session_oscillators:
            del session_oscillators[session_id]
            logger.info(f"Cleaned up oscillator for session {session_id}")

@lru_cache(maxsize=128)
def compute_wavefield_hash(centroid: float, emotion: float, rms: float) -> str:
    """Generate cache key for wavefield parameters"""
    data = f"{centroid:.2f}_{emotion:.3f}_{rms:.3f}"
    return hashlib.md5(data.encode()).hexdigest()[:8]

async def compute_wavefield_async(oscillator: BanksyOscillator, grid_size: int) -> np.ndarray:
    """Compute wavefield in thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        oscillator.compute_wavefield_interference,
        grid_size
    )

# Session storage with Redis
async def get_session(session_id: str, redis: Redis = Depends(get_redis)) -> Optional[HologramSession]:
    """Retrieve session from Redis"""
    data = await redis.get(f"hologram:session:{session_id}")
    if data:
        session_data = json.loads(data)
        return HologramSession(**session_data)
    return None

async def save_session(session: HologramSession, redis: Redis = Depends(get_redis)):
    """Save session to Redis with TTL"""
    session.updated_at = datetime.utcnow()
    session.client_count = manager.get_session_client_count(session.session_id)
    
    await redis.setex(
        f"hologram:session:{session.session_id}",
        timedelta(hours=24),  # 24 hour TTL
        session.json()
    )

# API Routes
@router.post('/start', response_model=HologramSession)
async def start_hologram_session(
    request: HologramStartRequest, 
    background_tasks: BackgroundTasks,
    redis: Redis = Depends(get_redis)
):
    """Start a new holographic rendering session"""
    session_id = str(uuid.uuid4())
    
    # Initialize default ψ-state with request quality settings
    quality_presets = {
        'low': {'coherence': 0.3, 'oscillators': 4},
        'medium': {'coherence': 0.5, 'oscillators': 8},
        'high': {'coherence': 0.7, 'oscillators': 12},
        'ultra': {'coherence': 0.9, 'oscillators': 16}
    }
    
    preset = quality_presets.get(request.quality, quality_presets['high'])
    
    initial_psi_state = {
        'psi_phase': 0.0,
        'psi_magnitude': 1.0,
        'phase_coherence': preset['coherence'],
        'oscillator_phases': [0.0] * preset['oscillators'],
        'dominant_frequency': 440.0,
        'emotional_resonance': {
            'excitement': 0.0,
            'calmness': 0.5,
            'energy': 0.3,
            'clarity': 0.7
        }
    }
    
    # Get calibration for display type
    if request.calibration:
        calibration = request.calibration
    else:
        calibration = await get_display_calibration_model(request.display_type)
    
    # Generate initial hologram hints
    from core.psiMemory.psiFrames import generate_initial_hologram_hints
    hologram_hints = generate_initial_hologram_hints(initial_psi_state)
    
    # Create session
    session = HologramSession(
        session_id=session_id,
        psi_state=initial_psi_state,
        hologram_hints=hologram_hints,
        active_views=generate_initial_views(request.display_type),
        calibration=calibration,
        performance_metrics={
            'quality': request.quality,
            'session_type': request.session_type,
            'max_clients': request.max_clients
        }
    )
    
    # Save to Redis
    await save_session(session, redis)
    
    # Start background hologram processing
    background_tasks.add_task(
        initialize_hologram_processing, 
        session_id, 
        request.quality,
        redis
    )
    
    # Log session creation
    await redis.zadd(
        "hologram:sessions:active",
        {session_id: datetime.utcnow().timestamp()}
    )
    
    return session

@router.get('/session/{session_id}', response_model=HologramSession)
async def get_hologram_session(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Get current hologram session state"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update client count from active connections
    session.client_count = manager.get_session_client_count(session_id)
    
    return session

@router.post('/session/{session_id}/update_psi')
async def update_psi_state(
    session_id: str, 
    psi_update: PsiStateUpdate,
    redis: Redis = Depends(get_redis)
):
    """Update ψ-state for holographic rendering"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update only provided fields
    update_dict = psi_update.dict(exclude_unset=True)
    for key, value in update_dict.items():
        if key == 'emotional_resonance' and value:
            session.psi_state['emotional_resonance'].update(value)
        else:
            session.psi_state[key] = value
    
    # Regenerate hologram hints based on new ψ-state
    from core.psiMemory.psiFrames import update_hologram_hints
    session.hologram_hints = update_hologram_hints(
        session.psi_state, 
        session.hologram_hints
    )
    
    # Save updated session
    await save_session(session, redis)
    
    # Broadcast update to all connected clients
    await manager.broadcast_to_session(session_id, {
        'type': 'psi_update',
        'psi_state': session.psi_state,
        'hologram_hints': session.hologram_hints,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return {
        "status": "updated", 
        "psi_state": session.psi_state,
        "broadcast_to": session.client_count
    }

@router.post('/session/{session_id}/pause')
async def pause_session(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Pause holographic rendering"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.status = "paused"
    await save_session(session, redis)
    
    await manager.broadcast_to_session(session_id, {
        'type': 'session_paused',
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return {"status": "paused", "session_id": session_id}

@router.post('/session/{session_id}/resume')
async def resume_session(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Resume holographic rendering"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.status = "active"
    await save_session(session, redis)
    
    await manager.broadcast_to_session(session_id, {
        'type': 'session_resumed',
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return {"status": "resumed", "session_id": session_id}

@router.delete('/session/{session_id}')
async def stop_hologram_session(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Stop and cleanup hologram session"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Notify clients before cleanup
    await manager.broadcast_to_session(session_id, {
        'type': 'session_stopping',
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Cleanup session and oscillator
    await cleanup_oscillator(session_id)
    await redis.delete(f"hologram:session:{session_id}")
    await redis.delete(f"hologram:session:{session_id}:wavefield")
    await redis.delete(f"hologram:session:{session_id}:metrics")
    await redis.zrem("hologram:sessions:active", session_id)
    
    # Archive session data
    await redis.setex(
        f"hologram:session:archive:{session_id}",
        timedelta(days=7),  # Keep archive for 7 days
        session.json()
    )
    
    return {"status": "stopped", "session_id": session_id}

@router.get('/sessions/active')
async def list_active_sessions(
    limit: int = 10,
    redis: Redis = Depends(get_redis)
):
    """List active hologram sessions"""
    # Get active sessions from sorted set
    active_ids = await redis.zrevrange(
        "hologram:sessions:active", 
        0, 
        limit - 1,
        withscores=True
    )
    
    sessions = []
    for session_id, score in active_ids:
        session = await get_session(session_id, redis)
        if session:
            sessions.append({
                'session_id': session.session_id,
                'status': session.status,
                'client_count': manager.get_session_client_count(session.session_id),
                'created_at': session.created_at,
                'quality': session.performance_metrics.get('quality', 'unknown')
            })
    
    return {
        'sessions': sessions,
        'total': await redis.zcard("hologram:sessions:active")
    }

@router.get('/calibration/{display_type}')
async def get_display_calibration(display_type: str):
    """Get calibration data for different display types"""
    calibration = await get_display_calibration_model(display_type)
    return calibration.dict()

@router.post('/session/{session_id}/update_wavefield', response_model=WavefieldResponse)
async def update_wavefield(
    session_id: str,
    update: WavefieldUpdate,
    redis: Redis = Depends(get_redis)
):
    """Update holographic wavefield with latest oscillator state"""
    start_time = datetime.utcnow()
    
    # Get session
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get or create oscillator for session
    oscillator = await get_oscillator_for_session(session_id, update.context)
    
    # Map audio parameters to oscillator
    oscillator.map_parameters(
        centroid=update.centroid,
        emotion_intensity=update.emotion_intensity,
        rms=update.rms
    )
    
    # Step oscillator dynamics
    oscillator.step()
    
    # Get wavefield parameters
    wavefield_params = oscillator.get_wavefield_params()
    
    # Compute additional features if requested
    interference_pattern = None
    phase_gradients = None
    adaptive_mapping = None
    
    if update.compute_interference:
        grid_size = update.grid_size or 128
        # Compute asynchronously to avoid blocking
        wavefield = await compute_wavefield_async(oscillator, grid_size)
        interference_pattern = {
            'amplitude': np.abs(wavefield).tolist(),
            'phase': np.angle(wavefield).tolist(),
            'grid_size': grid_size,
            'stats': {
                'max_amplitude': float(np.max(np.abs(wavefield))),
                'mean_amplitude': float(np.mean(np.abs(wavefield))),
                'phase_variance': float(np.var(np.angle(wavefield)))
            }
        }
    
    # Always compute phase gradients for smooth visualization
    phase_gradients = oscillator.get_phase_gradients()
    
    # Get adaptive spatial mapping
    audio_bandwidth = update.centroid * 0.5  # Estimate bandwidth from centroid
    adaptive_mapping = oscillator.get_adaptive_spatial_mapping(audio_bandwidth)
    
    # Update session with wavefield data
    session_update = {
        'wavefield_params': wavefield_params,
        'last_wavefield_update': datetime.utcnow().isoformat(),
        'wavefield_context': update.context
    }
    
    # Store in Redis with short TTL for real-time data
    await redis.setex(
        f"hologram:session:{session_id}:wavefield",
        timedelta(seconds=60),
        json.dumps(wavefield_params)
    )
    
    # Broadcast to connected clients
    await manager.broadcast_to_session(session_id, {
        'type': 'wavefield_update',
        'wavefield_params': wavefield_params,
        'phase_gradients': phase_gradients,
        'adaptive_mapping': adaptive_mapping,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Calculate performance metrics
    end_time = datetime.utcnow()
    compute_time = (end_time - start_time).total_seconds() * 1000  # ms
    
    return WavefieldResponse(
        status="updated",
        wavefield_params=WavefieldParameters(**wavefield_params),
        interference_pattern=interference_pattern,
        phase_gradients=phase_gradients,
        adaptive_mapping=adaptive_mapping,
        performance_metrics={
            'compute_time_ms': compute_time,
            'oscillator_count': len(wavefield_params['oscillator_phases']),
            'broadcast_clients': session.client_count
        },
        timestamp=datetime.utcnow()
    )

@router.get('/session/{session_id}/wavefield/current')
async def get_current_wavefield(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Get current wavefield parameters without updating"""
    # Check session exists
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get cached wavefield data
    wavefield_data = await redis.get(f"hologram:session:{session_id}:wavefield")
    if not wavefield_data:
        # Return default if no wavefield data yet
        oscillator = await get_oscillator_for_session(session_id)
        return {
            'status': 'no_updates_yet',
            'wavefield_params': oscillator.get_wavefield_params()
        }
    
    return {
        'status': 'current',
        'wavefield_params': json.loads(wavefield_data),
        'timestamp': datetime.utcnow().isoformat()
    }

@router.post('/session/{session_id}/wavefield/interpolate')
async def interpolate_wavefield(
    session_id: str,
    alpha: float = Query(..., ge=0, le=1, description="Interpolation factor"),
    redis: Redis = Depends(get_redis)
):
    """Get interpolated wavefield parameters for smooth transitions"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    oscillator = await get_oscillator_for_session(session_id)
    interpolated_state = oscillator.interpolate_parameters(alpha)
    
    return {
        'status': 'interpolated',
        'alpha': alpha,
        'state': interpolated_state,
        'timestamp': datetime.utcnow().isoformat()
    }

@router.post('/session/{session_id}/wavefield/multiscale')
async def compute_multiscale_wavefield(
    session_id: str,
    scales: Optional[List[int]] = Query(None, description="Grid sizes for multiscale computation"),
    redis: Redis = Depends(get_redis)
):
    """Compute wavefield at multiple spatial scales"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    oscillator = await get_oscillator_for_session(session_id)
    
    # Compute multiscale wavefields asynchronously
    start_time = datetime.utcnow()
    loop = asyncio.get_event_loop()
    wavefields = await loop.run_in_executor(
        executor,
        oscillator.compute_multiscale_wavefield,
        scales
    )
    
    # Convert complex arrays to magnitude and phase
    result = {}
    for scale_key, wavefield in wavefields.items():
        result[scale_key] = {
            'amplitude': np.abs(wavefield).tolist(),
            'phase': np.angle(wavefield).tolist(),
            'shape': wavefield.shape
        }
    
    compute_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    return {
        'status': 'computed',
        'wavefields': result,
        'performance': {
            'compute_time_ms': compute_time,
            'scales_computed': len(result)
        },
        'timestamp': datetime.utcnow().isoformat()
    }

@router.get('/session/{session_id}/wavefield/velocity_field')
async def get_phase_velocity_field(
    session_id: str,
    grid_size: int = Query(64, ge=32, le=256),
    redis: Redis = Depends(get_redis)
):
    """Compute instantaneous phase velocity field for flow visualization"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    oscillator = await get_oscillator_for_session(session_id)
    
    # Compute velocity field asynchronously
    loop = asyncio.get_event_loop()
    velocity_field = await loop.run_in_executor(
        executor,
        oscillator.compute_phase_velocity_field,
        grid_size
    )
    
    # Convert to list format
    vx, vy = velocity_field
    
    return {
        'status': 'computed',
        'velocity_field': {
            'vx': vx.tolist(),
            'vy': vy.tolist(),
            'magnitude': np.sqrt(vx**2 + vy**2).tolist(),
            'grid_size': grid_size
        },
        'timestamp': datetime.utcnow().isoformat()
    }

async def get_display_calibration_model(display_type: str) -> HologramCalibration:
    """Get calibration model for display type"""
    calibrations = {
        "looking_glass_portrait": HologramCalibration(
            pitch=49.825,
            tilt=-0.1745,
            center=0.04239,
            subp=0.013,
            displayAspect=3/4,
            numViews=45
        ),
        "looking_glass_32": HologramCalibration(
            pitch=52.56,
            tilt=-0.1745,
            center=0.0,
            subp=0.013,
            displayAspect=16/9,
            numViews=45
        ),
        "looking_glass_65": HologramCalibration(
            pitch=63.42,
            tilt=-0.1745,
            center=0.0,
            subp=0.013,
            displayAspect=16/9,
            numViews=100
        ),
        "webgpu_only": HologramCalibration(
            pitch=50.0,
            tilt=0.0,
            center=0.0,
            subp=0.013,
            displayAspect=16/9,
            numViews=25
        ),
        "custom": HologramCalibration()  # Default values
    }
    
    if display_type == "auto":
        # Auto-detect would go here - for now return webgpu_only
        return calibrations["webgpu_only"]
    
    if display_type not in calibrations:
        raise HTTPException(
            status_code=400, 
            detail=f"Display type '{display_type}' not supported. Available: {list(calibrations.keys())}"
        )
    
    return calibrations[display_type]

def generate_initial_views(display_type: str) -> List[Dict]:
    """Generate initial view configurations based on display type"""
    if display_type == "looking_glass_65":
        # More views for larger display
        return [
            {
                'name': f'view_{i}',
                'azimuth': i * (360 / 100),
                'elevation': 0,
                'distance': 1.0,
                'weight': 1.0
            }
            for i in range(5)  # Sample subset
        ]
    else:
        # Standard configuration
        return [
            {'name': 'primary', 'azimuth': 0, 'elevation': 0, 'distance': 1.0, 'weight': 1.0},
            {'name': 'left', 'azimuth': -30, 'elevation': 0, 'distance': 1.0, 'weight': 0.8},
            {'name': 'right', 'azimuth': 30, 'elevation': 0, 'distance': 1.0, 'weight': 0.8}
        ]

# WebSocket endpoint for real-time updates
@router.websocket('/ws/{session_id}')
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    client_id: Optional[str] = None,
    redis: Redis = Depends(get_redis)
):
    """WebSocket connection for real-time hologram updates"""
    # Verify session exists
    session = await get_session(session_id, redis)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    # Check if session accepts more clients
    current_clients = manager.get_session_client_count(session_id)
    max_clients = session.performance_metrics.get('max_clients', 10)
    if current_clients >= max_clients:
        await websocket.close(code=4003, reason="Session full")
        return
    
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid.uuid4())
    
    await manager.connect(websocket, session_id, client_id)
    
    try:
        # Send initial state
        await websocket.send_json({
            'type': 'connection_established',
            'client_id': client_id,
            'session': session.dict(),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Notify other clients
        await manager.broadcast_to_session(session_id, {
            'type': 'client_joined',
            'client_id': client_id,
            'client_count': manager.get_session_client_count(session_id),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            # Process different message types
            if data.get('type') == 'psi_update':
                # Validate and update psi state
                try:
                    psi_update = PsiStateUpdate(**data.get('psi_state', {}))
                    await update_psi_state(session_id, psi_update, redis)
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': str(e)
                    })
            
            elif data.get('type') == 'request_highlights':
                # Send holographic highlights
                from core.psiMemory.psiFrames import getHolographicHighlights
                highlights = getHolographicHighlights(limit=data.get('limit', 10))
                await websocket.send_json({
                    'type': 'highlights',
                    'data': highlights
                })
            
            elif data.get('type') == 'performance_report':
                # Store client performance metrics
                await redis.hset(
                    f"hologram:session:{session_id}:metrics",
                    client_id,
                    json.dumps({
                        'fps': data.get('fps'),
                        'latency': data.get('latency'),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                )
            
            elif data.get('type') == 'wavefield_update':
                # Handle wavefield update request
                try:
                    wavefield_update = WavefieldUpdate(**data.get('update', {}))
                    
                    # Process update
                    response = await update_wavefield(
                        session_id=session_id,
                        update=wavefield_update,
                        redis=redis
                    )
                    
                    # Send response to requesting client
                    await websocket.send_json({
                        'type': 'wavefield_update_response',
                        'status': 'success',
                        'data': response.dict()
                    })
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': f"Wavefield update error: {str(e)}"
                    })
            
            elif data.get('type') == 'request_velocity_field':
                # Request phase velocity field
                grid_size = data.get('grid_size', 64)
                oscillator = await get_oscillator_for_session(session_id)
                
                # Compute asynchronously
                loop = asyncio.get_event_loop()
                velocity_field = await loop.run_in_executor(
                    executor,
                    oscillator.compute_phase_velocity_field,
                    grid_size
                )
                
                vx, vy = velocity_field
                await websocket.send_json({
                    'type': 'velocity_field',
                    'data': {
                        'vx': vx.tolist(),
                        'vy': vy.tolist(),
                        'grid_size': grid_size
                    }
                })
            
            elif data.get('type') == 'ping':
                # Respond to ping
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        
        # Notify remaining clients
        await manager.broadcast_to_session(session_id, {
            'type': 'client_left',
            'client_id': client_id,
            'client_count': manager.get_session_client_count(session_id),
            'timestamp': datetime.utcnow().isoformat()
        })

# Background tasks
async def initialize_hologram_processing(
    session_id: str, 
    quality: str,
    redis: Redis
):
    """Background task to initialize hologram processing pipeline"""
    try:
        # Simulate initialization based on quality
        init_times = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0,
            'ultra': 3.0
        }
        
        await asyncio.sleep(init_times.get(quality, 1.0))
        
        # Mark session as fully initialized
        await redis.hset(
            f"hologram:session:{session_id}:status",
            "initialized",
            "true"
        )
        
        # Notify clients
        await manager.broadcast_to_session(session_id, {
            'type': 'initialization_complete',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Initialization error for session {session_id}: {e}")

# Performance monitoring endpoint
@router.get('/session/{session_id}/metrics')
async def get_session_metrics(
    session_id: str,
    redis: Redis = Depends(get_redis)
):
    """Get performance metrics for a session"""
    session = await get_session(session_id, redis)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all client metrics
    metrics = await redis.hgetall(f"hologram:session:{session_id}:metrics")
    
    client_metrics = {}
    for client_id, data in metrics.items():
        client_metrics[client_id] = json.loads(data)
    
    # Calculate aggregates
    fps_values = [m['fps'] for m in client_metrics.values() if m.get('fps')]
    latency_values = [m['latency'] for m in client_metrics.values() if m.get('latency')]
    
    return {
        'session_id': session_id,
        'client_metrics': client_metrics,
        'aggregates': {
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'min_fps': np.min(fps_values) if fps_values else 0,
            'max_fps': np.max(fps_values) if fps_values else 0,
            'avg_latency': np.mean(latency_values) if latency_values else 0,
            'active_clients': len(client_metrics)
        }
    }

# Health check endpoint
@router.get('/health')
async def health_check(redis: Redis = Depends(get_redis)):
    """Check hologram service health"""
    try:
        # Check Redis connection
        await redis.ping()
        
        # Get service stats
        active_sessions = await redis.zcard("hologram:sessions:active")
        total_connections = sum(
            len(conns) for conns in manager.active_connections.values()
        )
        
        return {
            'status': 'healthy',
            'active_sessions': active_sessions,
            'total_connections': total_connections,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )