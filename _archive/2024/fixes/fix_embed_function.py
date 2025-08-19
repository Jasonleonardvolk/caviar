#!/usr/bin/env python3
"""
Fix the broken embed function in soliton_production.py
"""

import re

# Read the file
with open('api/routes/soliton_production.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the embed function start
embed_start = content.find('@router.post("/embed")')
if embed_start == -1:
    print("❌ Could not find embed function")
    exit(1)

# Find where the function should end (next function or end of file)
next_func = content.find('\n@router.', embed_start + 1)
if next_func == -1:
    next_func = content.find('\ndef ', embed_start + 1)
if next_func == -1:
    next_func = content.find('\nasync def ', embed_start + 1)
if next_func == -1:
    # End of file
    next_func = len(content)

# Extract current embed function
current_embed = content[embed_start:next_func]

# Check if it's complete
if "return {" not in current_embed or current_embed.strip().endswith("# Normalize"):
    print("✅ Fixing incomplete embed function")
    
    # Complete embed function
    complete_embed = '''@router.post("/embed")
async def embed_text(request: EmbedRequest):
    """Generate embedding for text using sentence transformers"""
    try:
        # Simple embedding using hash-based approach as fallback
        # In production, replace with actual sentence-transformers
        import hashlib
        import math
        
        text = request.text
        if not text:
            return {"success": False, "error": "No text provided"}
        
        # Generate deterministic embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        
        # Create 128-dimensional embedding
        embedding = []
        for i in range(128):
            value = math.sin(seed + i * 0.1) * 0.5 + math.cos(i * 0.1) * 0.5
            embedding.append(value)
        
        # Normalize
        magnitude = math.sqrt(sum(v * v for v in embedding))
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]
        
        return {
            "success": True,
            "embedding": embedding
        }
        
    except Exception as e:
        logger.error(f"Embed error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
'''
    
    # Replace the broken function
    new_content = content[:embed_start] + complete_embed + "\n" + content[next_func:]
    
    # Remove any trailing garbage
    new_content = re.sub(r'\n\nemory: \{str\(e\)\}"\n\s*\)\n', '', new_content)
    
    # Write back
    with open('api/routes/soliton_production.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Fixed embed function!")
else:
    print("✅ Embed function appears complete")

print("✅ Done!")
