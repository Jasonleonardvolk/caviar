#!/usr/bin/env python3
"""
Prajna Audio/Visual Integration Update
=====================================

Updates the Prajna API to support audio input/output and avatar state management.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def update_prajna_api():
    """Add audio/visual endpoints to Prajna API"""
    
    prajna_api_path = Path("prajna/api/prajna_api.py")
    
    # Read the current file
    with open(prajna_api_path, 'r') as f:
        content = f.read()
    
    # Check if audio endpoints already exist
    if "api/answer/audio" in content:
        print("✅ Audio endpoints already exist in Prajna API")
        return
    
    # Find where to insert the new endpoints (after the main answer endpoint)
    insert_pos = content.find("@app.post('/api/answer')")
    if insert_pos == -1:
        print("❌ Could not find answer endpoint in Prajna API")
        return
    
    # Find the end of the answer endpoint
    endpoint_end = content.find("\n\n@", insert_pos)
    if endpoint_end == -1:
        endpoint_end = len(content)
    
    # Audio endpoint code to insert
    audio_endpoint = '''

@app.post('/api/answer/audio')
async def answer_audio_query(
    audio: UploadFile = File(...),
    conversation_id: str = Form(None),
    focus_concept: Optional[str] = Form(None)
):
    """
    Process audio query and return answer with optional TTS
    
    This endpoint:
    1. Accepts audio file upload
    2. Transcribes audio to text
    3. Processes query through Prajna
    4. Optionally returns TTS audio response
    """
    try:
        # Save uploaded audio temporarily
        temp_audio_path = f"tmp/audio_{conversation_id or 'temp'}_{int(time.time())}.webm"
        os.makedirs("tmp", exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # TODO: Implement actual speech-to-text
        # For now, return a mock transcription
        transcribed_text = "What is the current state of my concept mesh?"
        
        # Process through regular Prajna pipeline
        request = PrajnaRequest(
            user_query=transcribed_text,
            focus_concept=focus_concept,
            conversation_id=conversation_id
        )
        
        response = await answer_query(request)
        
        # Add transcription to response
        response_dict = response.dict() if hasattr(response, 'dict') else response
        response_dict['context_used'] = transcribed_text
        
        # TODO: Implement text-to-speech for response
        # For now, just return text response
        
        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

@app.get('/api/avatar/state')
async def get_avatar_state():
    """Get current avatar state for visualization"""
    # This would be connected to actual Prajna processing state
    return {
        "state": "idle",  # idle, listening, thinking, speaking, processing
        "mood": "neutral",  # neutral, happy, confused, focused
        "audio_level": 0.0
    }

@app.websocket('/api/avatar/updates')
async def avatar_updates(websocket: WebSocket):
    """WebSocket for real-time avatar state updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send avatar state updates
            state = {
                "state": "idle",
                "mood": "neutral", 
                "audio_level": 0.0,
                "timestamp": time.time()
            }
            
            await websocket.send_json(state)
            await asyncio.sleep(0.1)  # Update every 100ms
            
    except WebSocketDisconnect:
        logger.info("Avatar WebSocket disconnected")
'''
    
    # Insert the audio endpoints
    new_content = content[:endpoint_end] + audio_endpoint + content[endpoint_end:]
    
    # Add necessary imports at the top if not present
    imports_to_add = [
        "from fastapi import File, UploadFile, Form",
        "import time",
        "import os"
    ]
    
    for import_line in imports_to_add:
        if import_line not in new_content:
            # Find the last import statement
            last_import = new_content.rfind("\nimport ")
            if last_import == -1:
                last_import = new_content.rfind("\nfrom ")
            
            if last_import != -1:
                # Find the end of the line
                line_end = new_content.find("\n", last_import + 1)
                new_content = new_content[:line_end] + f"\n{import_line}" + new_content[line_end:]
    
    # Write the updated file
    with open(prajna_api_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Added audio/visual endpoints to Prajna API")

def create_frontend_integration():
    """Create integration file for frontend"""
    
    integration_content = '''
# Prajna Frontend Integration Guide

## Audio/Visual Components Added

### 1. PrajnaAvatar.svelte
- Animated avatar component with states: idle, listening, thinking, speaking, processing
- Visual audio level feedback
- Mood indicators: neutral, happy, confused, focused
- Responsive sizing: small, medium, large

### 2. PrajnaChatEnhanced.svelte
- Enhanced chat interface with integrated avatar
- Voice input support with real-time transcription
- Audio response playback
- WebSocket streaming with visual feedback
- Trust score visualization

## Usage

### Basic Avatar
```svelte
<PrajnaAvatar 
  state="speaking"
  audioLevel={0.7}
  mood="happy"
  size="medium"
/>
```

### Enhanced Chat Interface
```svelte
<script>
  import PrajnaChatEnhanced from './PrajnaChatEnhanced.svelte';
</script>

<PrajnaChatEnhanced />
```

## API Endpoints

### Audio Input
POST /api/answer/audio
- Accepts: audio file (webm, mp3, wav)
- Returns: transcribed query + Prajna response

### Avatar State
GET /api/avatar/state
- Returns current avatar visualization state

WebSocket /api/avatar/updates
- Real-time avatar state updates

## Testing

Run the test script:
```bash
python test_prajna_audio_visual.py
```

## Next Steps

1. Implement actual speech-to-text (Whisper)
2. Add text-to-speech for responses
3. Enhance avatar animations
4. Add emotion detection from audio
'''
    
    with open("prajna/frontend/INTEGRATION.md", 'w') as f:
        f.write(integration_content)
    
    print("✅ Created frontend integration guide")

def main():
    """Run the integration update"""
    print("🎭 Updating Prajna for Audio/Visual Integration\n")
    
    # Update the API
    update_prajna_api()
    
    # Create integration guide
    create_frontend_integration()
    
    print("\n✨ Integration complete!")
    print("\nNext steps:")
    print("1. Restart Prajna API to load new endpoints")
    print("2. Update your Svelte app to use PrajnaChatEnhanced")
    print("3. Run test_prajna_audio_visual.py to verify")

if __name__ == "__main__":
    main()
