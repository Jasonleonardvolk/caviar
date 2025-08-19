"""
Prosody Engine API Integration
==============================

FastAPI routes for prosody analysis.
"""

from fastapi import APIRouter, UploadFile, File, Query, Header, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any
import numpy as np
import httpx

from .core import get_prosody_engine, ProsodyResult

# Import TORI systems
try:
    from hott_integration.psi_morphon import PsiMorphon, ModalityType
except ImportError:
    # Fallback for development
    class PsiMorphon:
        pass
    class ModalityType:
        AUDIO = "audio"

logger = logging.getLogger(__name__)

# Create API router
prosody_router = APIRouter(prefix="/api/prosody", tags=["prosody"])

@prosody_router.post("/analyze")
async def analyze_prosody(
    file: UploadFile = File(...),
    emotion_categories: int = Query(2000, description="Number of emotion categories to consider"),
    cultural_context: Optional[str] = Query(None, description="Cultural context for adaptation"),
    include_trajectory: bool = Query(True, description="Include emotional trajectory analysis"),
    tenant_id: str = Header(..., alias="X-Tenant-ID"),
):
    """
    Complete prosody analysis endpoint.
    
    Analyzes uploaded audio for:
    - 2000+ emotion categories
    - Voice quality (breathiness, roughness, strain, clarity, warmth)
    - Prosody patterns (15 categories from research)
    - Cultural adaptation
    - Emotional trajectory
    - Netflix-killer features (subtitle colors, sarcasm detection)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio format")
        
        # Read audio data
        audio_data = await file.read()
        
        # Get prosody engine
        engine = get_prosody_engine()
        
        # Analyze prosody
        result = await engine.analyze_complete(
            audio_data=audio_data,
            options={
                'emotion_categories': emotion_categories,
                'cultural_context': cultural_context,
                'include_trajectory': include_trajectory
            }
        )
        
        # Create morphon for concept mesh integration
        morphon_data = {
            'modality': 'audio',
            'content': f"prosody_analysis_{file.filename}",
            'embedding': result['emotion_vector'][:512] if len(result['emotion_vector']) > 512 else result['emotion_vector'],
            'metadata': {
                'primary_emotion': result['primary_emotion'],
                'voice_quality': result['voice_quality'],
                'prosody_patterns': result['prosody_patterns'],
                'cultural_markers': result.get('cultural_context'),
                'psi_phase': result['psi_phase'],
                'processing_latency': result['processing_latency']
            },
            'salience': result['emotional_intensity'],
            'tenant_scope': 'user',
            'tenant_id': tenant_id
        }
        
        # Send to concept mesh (async, don't wait)
        asyncio.create_task(_store_in_concept_mesh(morphon_data, tenant_id))
        
        # Log performance
        if result['processing_latency'] <= 35:
            logger.info(f"✅ Prosody analysis completed in {result['processing_latency']:.1f}ms")
        else:
            logger.warning(f"⚠️ Prosody analysis took {result['processing_latency']:.1f}ms (target: 35ms)")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prosody analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@prosody_router.websocket("/stream")
async def prosody_stream(websocket: WebSocket, tenant_id: str = Query(...)):
    """
    Real-time prosody streaming endpoint.
    
    WebSocket endpoint for live prosody analysis with 35ms target latency.
    Perfect for:
    - Live streaming emotion detection
    - Real-time subtitle generation
    - Interactive applications
    """
    await websocket.accept()
    
    engine = get_prosody_engine()
    
    try:
        logger.info(f"🎤 Prosody stream started for tenant {tenant_id}")
        
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Real-time analysis
            result = await engine.analyze_streaming(data)
            
            # Send results immediately
            await websocket.send_json(result)
            
            # Log ultra-low latency achievements
            if result['latency_ms'] < 20:
                logger.debug(f"🚀 Ultra-low latency: {result['latency_ms']:.1f}ms")
                
    except WebSocketDisconnect:
        logger.info(f"Prosody stream disconnected for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Prosody stream error: {e}")
        await websocket.close()

@prosody_router.post("/scene/analyze")
async def analyze_scene(
    file: UploadFile = File(...),
    tenant_id: str = Header(..., alias="X-Tenant-ID"),
):
    """
    Analyze emotional arc of entire scene.
    
    Perfect for:
    - Movie/TV scene analysis
    - Detecting dramatic moments
    - Understanding emotional flow
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Analyze scene
            engine = get_prosody_engine()
            result = await engine.analyze_scene_emotions(tmp_path)
            
            # Add Netflix-killer insights
            result['insights'] = {
                'most_dramatic_moment': max(result['turning_points'], 
                                          key=lambda x: x['intensity_change'])['time'] if result['turning_points'] else None,
                'emotional_variety': len(set(e['emotion'].split('_')[0] for e in result['emotions'])),
                'recommended_music_style': _get_music_recommendation(result['arc_type']),
                'subtitle_timing_adjustments': _calculate_subtitle_timing(result['turning_points'])
            }
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Scene analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@prosody_router.post("/subtitle/generate")
async def generate_subtitle(
    text: str = Query(..., description="Text to generate subtitle for"),
    audio_file: Optional[UploadFile] = File(None, description="Optional audio for prosody matching"),
    style: str = Query("auto", description="Subtitle style: auto, dramatic, subtle, accessible"),
    tenant_id: str = Header(..., alias="X-Tenant-ID"),
):
    """
    Generate emotionally-aware subtitles.
    
    Creates subtitles with:
    - Emotional coloring
    - Animation suggestions
    - Sarcasm indicators
    - Accessibility enhancements
    """
    try:
        engine = get_prosody_engine()
        
        # If audio provided, analyze it
        if audio_file:
            audio_data = await audio_file.read()
            prosody_result = await engine.analyze_complete(
                audio_data=audio_data,
                options={'include_trajectory': False}
            )
            
            # Convert dict back to ProsodyResult for markup generation
            # This is a simplified version - in production, properly reconstruct
            from .core import ProsodyResult
            prosody = ProsodyResult(
                primary_emotion=prosody_result['primary_emotion'],
                emotion_vector=np.array(prosody_result['emotion_vector']),
                emotion_confidence=prosody_result['emotion_confidence'],
                voice_quality=prosody_result['voice_quality'],
                prosody_patterns=prosody_result['prosody_patterns'],
                subtitle_color=prosody_result['subtitle_color'],
                subtitle_animation=prosody_result['subtitle_animation'],
                emotional_intensity=prosody_result['emotional_intensity'],
                sarcasm_detected=prosody_result['sarcasm_detected']
            )
        else:
            # Generate neutral prosody
            prosody = ProsodyResult(
                primary_emotion="neutral_moderate_genuine",
                emotion_vector=np.zeros(2000),
                emotion_confidence=0.5,
                voice_quality={'breathiness': 0.2, 'roughness': 0.2, 'strain': 0.2, 'clarity': 0.7, 'warmth': 0.5},
                prosody_patterns=[],
                subtitle_color="#FFFFFF",
                subtitle_animation="none",
                emotional_intensity=0.5,
                sarcasm_detected=False
            )
        
        # Generate subtitle markup
        markup = engine.generate_subtitle_markup(text, prosody)
        
        # Apply style modifications
        if style == "dramatic":
            markup = _enhance_dramatic_subtitle(markup, prosody)
        elif style == "accessible":
            markup = _enhance_accessible_subtitle(markup, prosody)
        elif style == "subtle":
            markup = _enhance_subtle_subtitle(markup, prosody)
        
        return JSONResponse(content={
            'text': text,
            'markup': markup,
            'prosody': {
                'emotion': prosody.primary_emotion,
                'intensity': prosody.emotional_intensity,
                'voice_quality': prosody.voice_quality,
                'sarcasm': prosody.sarcasm_detected
            },
            'style': style
        })
        
    except Exception as e:
        logger.error(f"Subtitle generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@prosody_router.get("/emotions/list")
async def list_emotions(
    category: Optional[str] = Query(None, description="Filter by base emotion category"),
    limit: int = Query(100, description="Maximum number of emotions to return")
):
    """
    List available emotion categories.
    
    Returns the 2000+ fine-grained emotions available for detection.
    """
    engine = get_prosody_engine()
    
    emotions = list(engine.emotion_categories.keys())
    
    # Filter if category specified
    if category:
        emotions = [e for e in emotions if e.startswith(category)]
    
    # Limit results
    emotions = emotions[:limit]
    
    return JSONResponse(content={
        'total': len(engine.emotion_categories),
        'returned': len(emotions),
        'emotions': emotions,
        'base_categories': [
            'excitement', 'delight', 'sorrow', 'anger', 'aversion',
            'hesitation', 'depression', 'helplessness', 'confusion',
            'admiration', 'anxious', 'bitter_and_aggrieved'
        ]
    })

@prosody_router.get("/health")
async def health_check():
    """
    Health check endpoint for prosody engine.
    """
    engine = get_prosody_engine()
    
    # Quick performance test
    test_audio = np.random.randn(16000).astype(np.float32)  # 1 second of noise
    
    start = time.time()
    features = engine._fast_spectral_features(test_audio)
    latency = (time.time() - start) * 1000
    
    return JSONResponse(content={
        'status': 'healthy',
        'engine': 'netflix-killer-prosody',
        'version': '1.0.0',
        'emotion_categories': len(engine.emotion_categories),
        'prosody_patterns': len(engine.prosody_patterns),
        'target_latency_ms': engine.target_latency,
        'test_latency_ms': round(latency, 2),
        'performance': 'optimal' if latency < 35 else 'degraded'
    })

# Helper functions

async def _store_in_concept_mesh(morphon_data: Dict, tenant_id: str):
    """
    Store prosody analysis in concept mesh.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/morphons",
                json=morphon_data,
                headers={"X-Tenant-ID": tenant_id}
            )
            if response.status_code == 200:
                logger.info("✅ Prosody analysis stored in concept mesh")
            else:
                logger.warning(f"Failed to store in concept mesh: {response.status_code}")
    except Exception as e:
        logger.error(f"Concept mesh storage error: {e}")

def _get_music_recommendation(arc_type: str) -> str:
    """
    Recommend music style based on emotional arc.
    """
    recommendations = {
        'escalating': 'building orchestral',
        'declining': 'melancholic strings',
        'volatile': 'dissonant electronic',
        'stable': 'ambient pads',
        'flat': 'minimal piano'
    }
    return recommendations.get(arc_type, 'adaptive')

def _calculate_subtitle_timing(turning_points: list) -> list:
    """
    Calculate subtitle timing adjustments for dramatic moments.
    """
    adjustments = []
    
    for point in turning_points:
        if point['intensity_change'] > 0.5:
            adjustments.append({
                'time': point['time'],
                'adjustment': 'extend_display',
                'duration_multiplier': 1.5
            })
        elif point['intensity_change'] > 0.3:
            adjustments.append({
                'time': point['time'],
                'adjustment': 'fade_transition',
                'duration_multiplier': 1.2
            })
    
    return adjustments

def _enhance_dramatic_subtitle(markup: str, prosody: Any) -> str:
    """
    Enhance subtitle for dramatic effect.
    """
    if prosody.emotional_intensity > 0.7:
        markup = markup.replace('animation="', 'animation="dramatic-')
    if prosody.sarcasm_detected:
        markup = markup.replace('<subtitle', '<subtitle effect="sarcasm"')
    return markup

def _enhance_accessible_subtitle(markup: str, prosody: Any) -> str:
    """
    Enhance subtitle for accessibility.
    """
    # Add emotion description
    emotion_desc = f' emotion-description="{prosody.primary_emotion.replace("_", " ")}"'
    markup = markup.replace('<subtitle', f'<subtitle{emotion_desc}')
    
    # Add voice quality indicators
    if prosody.voice_quality['strain'] > 0.7:
        markup += ' [strained voice]'
    if prosody.voice_quality['breathiness'] > 0.7:
        markup += ' [breathy voice]'
    
    return markup

def _enhance_subtle_subtitle(markup: str, prosody: Any) -> str:
    """
    Make subtitle more subtle.
    """
    # Reduce color intensity
    if prosody.subtitle_color != '#FFFFFF':
        # Mix with white for subtlety
        markup = markup.replace(prosody.subtitle_color, '#FFEEEE')
    
    # Remove strong animations
    markup = markup.replace('animation="pulse"', 'animation="fade"')
    
    return markup