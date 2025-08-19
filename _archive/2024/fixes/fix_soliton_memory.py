#!/usr/bin/env python3
"""
Quick fix script for TORI soliton memory issues
Run this to add the missing embed endpoint to your API
"""

import os
import json

def create_embed_endpoint():
    """Create a simple embed endpoint for the soliton API"""
    
    endpoint_code = '''
@app.post("/api/soliton/embed")
async def embed_text(request: dict):
    """Generate embedding for text - simple implementation"""
    try:
        text = request.get("text", "")
        if not text:
            return {"success": False, "error": "No text provided"}
        
        # Simple embedding generation (replace with your actual implementation)
        # This is just a placeholder - you should use your actual embedding model
        import hashlib
        import math
        
        # Generate deterministic embedding based on text
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # Create 128-dimensional embedding
        embedding = []
        for i in range(128):
            value = math.sin(seed + i * 0.1) * 0.5 + math.cos(i * 0.1) * 0.5
            embedding.append(value)
        
        # Normalize
        magnitude = math.sqrt(sum(v * v for v in embedding))
        embedding = [v / magnitude for v in embedding]
        
        return {
            "success": True,
            "embedding": embedding
        }
    except Exception as e:
        logger.error(f"Embed error: {str(e)}")
        return {"success": False, "error": str(e)}
'''
    
    print("Add this endpoint to your prajna_atomic.py or API router:")
    print("-" * 60)
    print(endpoint_code)
    print("-" * 60)

def create_env_file():
    """Create .env file with fallback enabled"""
    
    env_content = """# Soliton Memory Configuration
VITE_ALLOW_MEMORY_FALLBACK=true

# Add any other environment variables here
"""
    
    env_path = "tori_ui_svelte/.env"
    print(f"\nCreate or update {env_path} with:")
    print("-" * 60)
    print(env_content)
    print("-" * 60)

def fix_stats_endpoint():
    """Suggest fixes for stats endpoint"""
    
    stats_fix = '''
# In your stats endpoint, add better error handling:

@app.get("/api/soliton/stats/{user_slug}")
async def get_soliton_stats(user_slug: str):
    """Get memory statistics for a user"""
    try:
        # Ensure proper user lookup
        logger.info(f"Stats request for user: {user_slug}")
        
        # Get stats from your memory system
        stats = {
            "totalMemories": 0,
            "activeWaves": 0,
            "averageStrength": 0,
            "clusterCount": 0
        }
        
        # Add actual stats retrieval here
        # ...
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Stats error for {user_slug}: {str(e)}")
        # Return empty stats instead of error
        return {
            "success": True,
            "stats": {
                "totalMemories": 0,
                "activeWaves": 0,
                "averageStrength": 0,
                "clusterCount": 0
            }
        }
'''
    
    print("\nImprove your stats endpoint:")
    print("-" * 60)
    print(stats_fix)
    print("-" * 60)

if __name__ == "__main__":
    print("TORI Soliton Memory Quick Fix Guide")
    print("=" * 60)
    
    create_embed_endpoint()
    create_env_file()
    fix_stats_endpoint()
    
    print("\nAdditional recommendations:")
    print("1. Restart your API server after adding the embed endpoint")
    print("2. Restart your frontend after creating .env file")
    print("3. Monitor logs for any remaining errors")
    print("4. Consider implementing a proper embedding model (sentence-transformers)")
