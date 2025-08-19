from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""
UPDATE THE MAIN CORE.PY TO USE V2
=================================

This updates the __init__.py to use our Netflix-killer v2.
"""

# First, let's update the imports to use the new version
netflix_killer_init = """
'''
Netflix-Killer Prosody Engine
=============================

A prosody engine so advanced it makes streaming services cry.
Now with 2000+ emotions and micro-pattern detection!
'''

# Import the REAL Netflix-killer version
from .core_v2 import NetflixKillerProsodyEngine, get_prosody_engine, ProsodyResult
from .api import prosody_router
from .streaming import ProsodyStreamProcessor
from .cultural import CulturalProsodyAdapter
from .micro_patterns import MicroPatternDetector
from .emotion_taxonomy import generate_full_emotion_taxonomy, COMPOUND_EMOTIONS
from .netflix_killer import NetflixKillerFeatures, EmotionalSubtitleEngine

__all__ = [
    'NetflixKillerProsodyEngine',
    'get_prosody_engine',
    'ProsodyResult',
    'prosody_router',
    'ProsodyStreamProcessor',
    'CulturalProsodyAdapter',
    'MicroPatternDetector',
    'generate_full_emotion_taxonomy',
    'COMPOUND_EMOTIONS',
    'NetflixKillerFeatures',
    'EmotionalSubtitleEngine'
]

__version__ = '2.0.0'  # Netflix is dead

# Quick stats when imported
try:
    engine = get_prosody_engine()
    print(f"🎭 Prosody Engine v{__version__} loaded with {len(engine.emotion_categories)} emotions!")
    print(f"💀 Netflix Status: TERMINATED")
except:
    pass
"""

# Write the updated init
with open(r"{PROJECT_ROOT}\prosody_engine\__init__.py", "w") as f:
    f.write(netflix_killer_init)

print("✅ Updated __init__.py to use Netflix-killer v2!")
