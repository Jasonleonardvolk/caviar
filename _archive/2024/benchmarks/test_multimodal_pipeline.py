#!/usr/bin/env python3
"""
Test script for multimodal pipeline integration.
Tests PDF, Image, Audio, and Video processing through the unified router.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_multimodal")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the router
from ingest_pdf.pipeline.router import ingest_file

async def test_file(file_path: str, expected_type: str):
    """Test a single file through the pipeline"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {expected_type}: {file_path}")
    logger.info(f"{'='*60}")
    
    try:
        # Process the file
        result = await ingest_file(
            file_path,
            doc_id=f"test_{expected_type}",
            admin_mode=False,
            progress_callback=lambda stage, percent, msg: logger.info(f"[{percent}%] {stage}: {msg}")
        )
        
        # Display results
        logger.info(f"\nResults for {expected_type}:")
        logger.info(f"- Status: {result.get('status', 'success')}")
        logger.info(f"- Media Type: {result.get('media_type', 'unknown')}")
        logger.info(f"- Concept Count: {result.get('concept_count', 0)}")
        logger.info(f"- File Size: {result.get('file_size_mb', 0):.2f} MB")
        
        if result.get('transcript'):
            logger.info(f"- Transcript Length: {len(result['transcript'])} chars")
            logger.info(f"- Transcript Preview: {result['transcript'][:100]}...")
        
        if result.get('concepts'):
            logger.info(f"\nTop Concepts:")
            for concept in result['concepts'][:5]:
                logger.info(f"  - {concept.get('name', 'unknown')}: {concept.get('score', 0):.3f}")
        
        if result.get('psi_state'):
            psi = result['psi_state']
            logger.info(f"\nΨ-State:")
            logger.info(f"  - Phase: {psi.get('psi_phase', 0):.3f}")
            logger.info(f"  - Coherence: {psi.get('phase_coherence', 0):.3f}")
        
        if result.get('error_message'):
            logger.error(f"Error: {result['error_message']}")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {expected_type}: {e}", exc_info=True)
        return None

async def create_test_files():
    """Create simple test files if they don't exist"""
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create test text file
    test_txt = test_dir / "test.txt"
    if not test_txt.exists():
        test_txt.write_text("""
        This is a test document for the multimodal pipeline.
        It contains information about artificial intelligence, machine learning,
        quantum computing, and holographic displays.
        
        The pipeline should extract concepts from this text and generate
        a psi-state representation for holographic visualization.
        """)
        logger.info(f"Created {test_txt}")
    
    # Create test HTML file
    test_html = test_dir / "test.html"
    if not test_html.exists():
        test_html.write_text("""
        <html>
        <head><title>Test HTML Document</title></head>
        <body>
            <h1>Multimodal Pipeline Test</h1>
            <p>This HTML document tests the pipeline's ability to extract concepts
            from structured web content.</p>
            <ul>
                <li>Natural Language Processing</li>
                <li>Computer Vision</li>
                <li>Audio Processing</li>
                <li>Holographic Visualization</li>
            </ul>
        </body>
        </html>
        """)
        logger.info(f"Created {test_html}")
    
    # Create test Markdown file
    test_md = test_dir / "test.md"
    if not test_md.exists():
        test_md.write_text("""
# Multimodal Pipeline Test

## Introduction
This markdown document demonstrates the pipeline's markdown processing capabilities.

## Key Concepts
- **Deep Learning**: Neural networks with multiple layers
- **Reinforcement Learning**: Learning through interaction
- **Transformers**: Attention-based architectures
- **Quantum Computing**: Leveraging quantum mechanics

## Conclusion
The multimodal pipeline should successfully extract these concepts.
        """)
        logger.info(f"Created {test_md}")
    
    return test_dir

async def main():
    """Run multimodal pipeline tests"""
    logger.info("Starting Multimodal Pipeline Tests")
    
    # Create test files
    test_dir = await create_test_files()
    
    # Define test cases
    test_cases = [
        (test_dir / "test.txt", "text"),
        (test_dir / "test.html", "html"),
        (test_dir / "test.md", "markdown"),
    ]
    
    # Add PDF test if available
    pdf_files = list(Path(".").glob("*.pdf"))
    if pdf_files:
        test_cases.append((pdf_files[0], "pdf"))
        logger.info(f"Found PDF for testing: {pdf_files[0]}")
    
    # Add image test if available
    for ext in ["jpg", "jpeg", "png"]:
        img_files = list(Path(".").glob(f"*.{ext}"))
        if img_files:
            test_cases.append((img_files[0], "image"))
            logger.info(f"Found image for testing: {img_files[0]}")
            break
    
    # Add audio test if available
    for ext in ["mp3", "wav"]:
        audio_files = list(Path(".").glob(f"*.{ext}"))
        if audio_files:
            test_cases.append((audio_files[0], "audio"))
            logger.info(f"Found audio for testing: {audio_files[0]}")
            break
    
    # Add video test if available
    for ext in ["mp4", "mkv"]:
        video_files = list(Path(".").glob(f"*.{ext}"))
        if video_files:
            test_cases.append((video_files[0], "video"))
            logger.info(f"Found video for testing: {video_files[0]}")
            break
    
    # Run tests
    results = []
    for file_path, file_type in test_cases:
        if file_path.exists():
            result = await test_file(str(file_path), file_type)
            results.append((file_type, result))
        else:
            logger.warning(f"Skipping {file_type}: file not found")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    success_count = 0
    for file_type, result in results:
        if result and result.get('status') != 'error':
            logger.info(f"✓ {file_type.upper()}: SUCCESS - {result.get('concept_count', 0)} concepts")
            success_count += 1
        else:
            logger.error(f"✗ {file_type.upper()}: FAILED")
    
    logger.info(f"\nTotal: {success_count}/{len(results)} tests passed")
    
    # Display holographic visualization hint
    if success_count > 0:
        logger.info("\n🎭 Holographic Visualization Ready!")
        logger.info("The extracted concepts and ψ-states are ready for 3D holographic display")
        logger.info("Connect to http://localhost:8000 to see the real-time visualization")

if __name__ == "__main__":
    asyncio.run(main())
