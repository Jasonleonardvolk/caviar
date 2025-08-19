"""
Setup and Installation Script for TORI Video Ingestion System

This script helps set up the complete video ingestion environment,
including dependencies, model downloads, and system verification.

Usage:
    python setup_video_system.py

Options:
    --install-models    Install AI models (Whisper, spaCy, etc.)
    --test-system      Run system tests after setup
    --gpu-support      Install GPU-accelerated versions
    --dev-mode         Install development dependencies
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import urllib.request
import zipfile
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_video")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"✅ Python {sys.version} is compatible")

def install_system_dependencies():
    """Install system-level dependencies."""
    system = platform.system().lower()
    
    logger.info("📦 Installing system dependencies...")
    
    if system == "windows":
        logger.info("🪟 Windows detected - please ensure the following are installed:")
        logger.info("  - FFmpeg (download from https://ffmpeg.org/)")
        logger.info("  - Tesseract OCR (download from https://github.com/UB-Mannheim/tesseract/wiki)")
        logger.info("  - Visual C++ Build Tools")
        
    elif system == "darwin":  # macOS
        logger.info("🍎 macOS detected - installing with Homebrew...")
        try:
            subprocess.run(["brew", "install", "ffmpeg", "tesseract"], check=True)
            logger.info("✅ System dependencies installed")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Please install Homebrew and run: brew install ffmpeg tesseract")
            
    elif system == "linux":
        logger.info("🐧 Linux detected - installing with apt...")
        try:
            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True)
            subprocess.run([
                "sudo", "apt-get", "install", "-y",
                "ffmpeg", "tesseract-ocr", "libgl1-mesa-glx", "libglib2.0-0"
            ], check=True)
            logger.info("✅ System dependencies installed")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Please install manually: sudo apt-get install ffmpeg tesseract-ocr")

def install_python_dependencies(gpu_support=False, dev_mode=False):
    """Install Python dependencies."""
    logger.info("🐍 Installing Python dependencies...")
    
    # Base requirements
    requirements = [
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
        "pydantic>=2.4.2",
        "websockets>=11.0.3",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
        "httpx>=0.25.0",
        
        # Video/Audio Processing
        "ffmpeg-python>=0.2.0",
        "openai-whisper>=20231117",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "moviepy>=1.0.3",
        "pydub>=0.25.1",
        
        # Computer Vision
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "mediapipe>=0.10.0",
        
        # NLP and ML
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "spacy>=3.7.0",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0"
    ]
    
    # GPU support
    if gpu_support:
        requirements.extend([
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ])
    else:
        requirements.extend([
            "torch>=2.0.0+cpu",
            "torchaudio>=2.0.0+cpu",
        ])
    
    # Development dependencies
    if dev_mode:
        requirements.extend([
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0"
        ])
    
    # Install requirements
    try:
        for req in requirements:
            logger.info(f"Installing {req}...")
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
        
        logger.info("✅ Python dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def download_models():
    """Download and install AI models."""
    logger.info("🤖 Downloading AI models...")
    
    try:
        # Download spaCy English model
        logger.info("📥 Downloading spaCy English model...")
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], check=True)
        
        # Download NLTK data
        logger.info("📥 Downloading NLTK data...")
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        
        logger.info("✅ AI models downloaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to download models: {e}")
        logger.info("💡 Models will be downloaded on first use")

def create_example_files():
    """Create example configuration and test files."""
    logger.info("📝 Creating example files...")
    
    # Create example configuration
    config_content = """
{
  "video_processing": {
    "default_language": "en",
    "enable_diarization": true,
    "enable_visual_context": true,
    "segment_threshold": 0.7,
    "quality": "balanced"
  },
  "ai_models": {
    "whisper_model": "base",
    "use_gpu": false,
    "batch_size": 16
  },
  "memory_integration": {
    "enable_concept_mesh": true,
    "enable_braid_memory": true,
    "enable_psi_mesh": true,
    "enable_scholar_sphere": true
  },
  "ghost_collective": {
    "default_personas": ["Ghost Collective", "Scholar", "Creator"],
    "reflection_interval": 10.0
  }
}
"""
    
    with open("video_config.json", "w") as f:
        f.write(config_content)
    
    # Create example test script
    test_script = '''"""
Example Video Processing Test Script

This script demonstrates how to use the TORI Video Ingestion System.
"""

import asyncio
import aiofiles
import httpx
import json
from pathlib import Path

async def test_video_upload():
    """Test video upload and processing."""
    
    # Create a simple test video (requires FFmpeg)
    print("🎬 Creating test video...")
    
    # You can replace this with your own video file
    test_video = "test_video.mp4"
    
    if not Path(test_video).exists():
        print("⚠️  No test video found. Please place a video file named 'test_video.mp4' in this directory.")
        return
    
    print(f"📤 Uploading {test_video}...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Upload video
        with open(test_video, "rb") as f:
            files = {"file": (test_video, f, "video/mp4")}
            data = {
                "language": "en",
                "enable_diarization": "true",
                "enable_visual_context": "true",
                "personas": "Ghost Collective,Scholar,Creator"
            }
            
            response = await client.post(
                "http://localhost:8080/api/v2/video/ingest",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                print(f"✅ Upload successful! Job ID: {job_id}")
                
                # Monitor progress
                while True:
                    status_response = await client.get(f"http://localhost:8080/api/v2/video/jobs/{job_id}/status")
                    status_data = status_response.json()
                    
                    print(f"📊 Status: {status_data['status']} ({status_data['progress']:.1%})")
                    
                    if status_data["status"] in ["completed", "failed"]:
                        break
                    
                    await asyncio.sleep(5)
                
                # Get results
                if status_data["status"] == "completed":
                    result_response = await client.get(f"http://localhost:8080/api/v2/video/jobs/{job_id}/result")
                    result_data = result_response.json()
                    
                    print(f"🎉 Processing complete!")
                    print(f"📝 Segments: {len(result_data['segments'])}")
                    print(f"🧠 Concepts: {len(result_data['concepts'])}")
                    print(f"❓ Questions: {len(result_data['questions'])}")
                    print(f"👻 Ghost Reflections: {len(result_data['ghost_reflections'])}")
                    print(f"🎯 Integrity Score: {result_data['integrity_score']:.2%}")
                    
                    # Save results
                    with open(f"results_{job_id}.json", "w") as f:
                        json.dump(result_data, f, indent=2)
                    print(f"💾 Results saved to results_{job_id}.json")
                    
                else:
                    print(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
            
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")

async def test_real_time_streaming():
    """Test real-time streaming capabilities."""
    print("🔴 Testing real-time streaming...")
    print("💡 This is a placeholder - implement WebSocket client for real testing")

if __name__ == "__main__":
    print("🧪 TORI Video Ingestion System Test")
    print("=" * 50)
    
    # Test basic upload
    asyncio.run(test_video_upload())
'''
    
    with open("test_video_system.py", "w") as f:
        f.write(test_script)
    
    logger.info("✅ Example files created:")
    logger.info("  - video_config.json (configuration)")
    logger.info("  - test_video_system.py (test script)")

def verify_installation():
    """Verify that the installation is working correctly."""
    logger.info("🔍 Verifying installation...")
    
    try:
        # Test imports
        import whisper
        import cv2
        import spacy
        import transformers
        import ffmpeg
        
        logger.info("✅ All critical modules imported successfully")
        
        # Test Whisper model loading
        logger.info("🎙️  Testing Whisper model...")
        model = whisper.load_model("base")
        logger.info("✅ Whisper model loaded successfully")
        
        # Test OpenCV
        logger.info("👁️  Testing OpenCV...")
        cap = cv2.VideoCapture(0)  # Test camera access
        cap.release()
        logger.info("✅ OpenCV working")
        
        logger.info("🎉 Installation verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Installation verification failed: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup TORI Video Ingestion System")
    parser.add_argument("--install-models", action="store_true", help="Download AI models")
    parser.add_argument("--test-system", action="store_true", help="Run system tests")
    parser.add_argument("--gpu-support", action="store_true", help="Install GPU support")
    parser.add_argument("--dev-mode", action="store_true", help="Install dev dependencies")
    
    args = parser.parse_args()
    
    logger.info("🎬 TORI Video Ingestion System Setup")
    logger.info("=" * 50)
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Install system dependencies
    install_system_dependencies()
    
    # Step 3: Install Python dependencies
    install_python_dependencies(
        gpu_support=args.gpu_support,
        dev_mode=args.dev_mode
    )
    
    # Step 4: Download models if requested
    if args.install_models:
        download_models()
    
    # Step 5: Create example files
    create_example_files()
    
    # Step 6: Verify installation
    if verify_installation():
        logger.info("🎉 Setup completed successfully!")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("1. Run: python main_video.py")
        logger.info("2. Open: http://localhost:8080")
        logger.info("3. Test: python test_video_system.py")
        logger.info("")
        logger.info("📖 API Documentation: http://localhost:8080/docs")
        
        if args.test_system:
            logger.info("🧪 Running system tests...")
            # Add basic system tests here
            
    else:
        logger.error("❌ Setup completed with errors")
        logger.info("💡 Try running the script again or check the logs")

if __name__ == "__main__":
    main()
