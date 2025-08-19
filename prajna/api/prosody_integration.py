"""
Prosody Engine Integration for Prajna API
========================================

Wires the Netflix-Killer Prosody Engine into Prajna.
"""

import logging
from pathlib import Path
import sys

# Add prosody engine to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from prosody_engine.api import prosody_router
    from prosody_engine.core import get_prosody_engine
    PROSODY_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Prosody Engine not available: {e}")
    prosody_router = None
    PROSODY_ENGINE_AVAILABLE = False

def integrate_prosody_with_prajna(app):
    """
    Integrate prosody engine with Prajna FastAPI app.
    
    This adds all prosody endpoints to the Prajna API:
    - /api/prosody/analyze - Full prosody analysis
    - /api/prosody/stream - Real-time WebSocket streaming
    - /api/prosody/scene/analyze - Scene emotion analysis
    - /api/prosody/subtitle/generate - Emotional subtitle generation
    - /api/prosody/emotions/list - List 2000+ emotions
    - /api/prosody/health - Health check
    """
    if not PROSODY_ENGINE_AVAILABLE:
        logging.error("Cannot integrate prosody engine - not available")
        return False
    
    try:
        # Include the prosody router
        app.include_router(prosody_router)
        
        # Add to startup
        @app.on_event("startup")
        async def init_prosody_engine():
            """Initialize prosody engine on startup"""
            engine = get_prosody_engine()
            logging.info(f"🎭 Prosody Engine initialized with {len(engine.emotion_categories)} emotions")
            
            # Store in app state for easy access
            app.state.prosody_engine = engine
        
        logging.info("✅ Prosody Engine integrated with Prajna API")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate prosody engine: {e}")
        return False

# Enhanced Prajna endpoints that use prosody

def create_enhanced_endpoints(app):
    """
    Create enhanced Prajna endpoints that leverage prosody.
    """
    from fastapi import UploadFile, File, Query, Header
    from fastapi.responses import JSONResponse
    from typing import Optional
    import numpy as np
    
    @app.post("/api/prajna/answer_with_emotion")
    async def answer_with_emotion(
        query: str = Query(..., description="User's question"),
        audio: Optional[UploadFile] = File(None, description="Optional audio of question"),
        tenant_id: str = Header(..., alias="X-Tenant-ID"),
    ):
        """
        Answer query with emotional understanding.
        
        If audio is provided, detects emotion and adapts response style.
        """
        # Get prosody engine
        engine = app.state.prosody_engine
        
        # Analyze emotion if audio provided
        emotion_context = None
        if audio:
            audio_data = await audio.read()
            prosody_result = await engine.analyze_complete(
                audio_data=audio_data,
                options={'include_trajectory': False}
            )
            
            emotion_context = {
                'emotion': prosody_result['primary_emotion'],
                'intensity': prosody_result['emotional_intensity'],
                'voice_quality': prosody_result['voice_quality'],
                'sarcasm': prosody_result['sarcasm_detected']
            }
        
        # Generate response with Prajna (placeholder for now)
        # In real implementation, this would use Saigon when ready
        response_text = f"I understand your question about '{query}'."
        
        # Adapt response based on emotion
        if emotion_context:
            if emotion_context['emotion'].startswith('anxious'):
                response_text = f"I can sense this is concerning. Let me help with {query}..."
            elif emotion_context['emotion'].startswith('excitement'):
                response_text = f"Great question! I'm excited to help with {query}!"
            elif emotion_context['sarcasm']:
                response_text = f"I detect some skepticism. Let me address {query} directly..."
        
        # Generate response audio with matching prosody
        if emotion_context:
            # TODO: Use TTS with prosody matching
            response_audio_url = None
        else:
            response_audio_url = None
        
        return JSONResponse(content={
            'query': query,
            'response': response_text,
            'emotion_context': emotion_context,
            'response_audio': response_audio_url,
            'processing_mode': 'emotional' if emotion_context else 'standard'
        })
    
    @app.post("/api/prajna/analyze_conversation")
    async def analyze_conversation(
        conversation_id: str = Query(..., description="Conversation ID to analyze"),
        tenant_id: str = Header(..., alias="X-Tenant-ID"),
    ):
        """
        Analyze emotional trajectory of entire conversation.
        
        Retrieves conversation from concept mesh and analyzes emotional flow.
        """
        # TODO: Retrieve conversation audio from concept mesh
        # For now, return placeholder
        
        return JSONResponse(content={
            'conversation_id': conversation_id,
            'emotional_arc': 'escalating',
            'key_emotions': ['curiosity', 'frustration', 'understanding', 'satisfaction'],
            'turning_points': [
                {'time': 45.2, 'from': 'curiosity', 'to': 'frustration'},
                {'time': 132.7, 'from': 'frustration', 'to': 'understanding'}
            ],
            'recommendation': 'User showed initial frustration but reached understanding. Follow up may be beneficial.'
        })

# Main integration function
def integrate_prosody_complete(prajna_app):
    """
    Complete prosody integration with Prajna.
    """
    # Add prosody routes
    success = integrate_prosody_with_prajna(prajna_app)
    
    if success:
        # Add enhanced endpoints
        create_enhanced_endpoints(prajna_app)
        logging.info("🎭 Prosody Engine fully integrated with enhanced endpoints")
    
    return success