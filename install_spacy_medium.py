#!/usr/bin/env python3
"""
Install the medium spaCy model with word vectors for better similarity
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_spacy_medium():
    """Install the medium spaCy model with word vectors"""
    try:
        logger.info("📦 Installing en_core_web_md (medium model with word vectors)...")
        logger.info("This may take a minute...")
        
        # Install the model
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_md"
        ])
        
        logger.info("✅ Successfully installed en_core_web_md")
        logger.info("This model includes:")
        logger.info("  - Word vectors (GloVe)")
        logger.info("  - Better similarity calculations")
        logger.info("  - All standard NLP components")
        
        # Test import
        import spacy
        nlp = spacy.load("en_core_web_md")
        logger.info(f"✅ Model loaded successfully with {len(nlp.vocab.vectors)} word vectors")
        
    except Exception as e:
        logger.error(f"❌ Failed to install: {e}")
        logger.info("You can manually install with:")
        logger.info("  python -m spacy download en_core_web_md")

if __name__ == "__main__":
    install_spacy_medium()
