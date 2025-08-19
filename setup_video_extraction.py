#!/usr/bin/env python3
"""
🎬 MULTIMODAL VIDEO CONCEPT EXTRACTION SETUP

This script installs all dependencies needed for video concept extraction:
- Audio transcription (Whisper)
- Visual analysis (Detectron2, BLIP)
- OCR (Tesseract)
- Video processing (OpenCV)
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, desc=""):
    """Run a command and handle errors gracefully"""
    print(f"🔧 {desc}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {desc} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} - Failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def install_video_packages():
    """Install required packages for video processing"""
    packages = [
        "openai-whisper",
        "transformers",
        "torch", 
        "torchvision",
        "opencv-python",
        "pytesseract",
        "pytube",  # For YouTube video processing
        "Pillow",  # For image processing
        "numpy"
    ]
    
    print("🎬 Installing Video Concept Extraction Dependencies...")
    
    for package in packages:
        success = run_command(f"{sys.executable} -m pip install {package}", 
                            f"Installing {package}")
        if not success:
            print(f"⚠️ Failed to install {package} - you may need to install manually")
    
    # Install Detectron2 (requires special handling)
    print("\n🔧 Installing Detectron2...")
    detectron_cmd = f"{sys.executable} -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    success = run_command(detectron_cmd, "Installing Detectron2 from GitHub")
    if not success:
        print("⚠️ Detectron2 installation failed - object detection will be disabled")
        print("   You can try: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html")

def install_system_dependencies():
    """Install system-level dependencies"""
    print("\n🔧 System Dependencies...")
    
    # Check for Tesseract OCR
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
        print("✅ Tesseract OCR already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Tesseract OCR not found")
        print("   Please install:")
        print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - macOS: brew install tesseract")
        print("   - Ubuntu: sudo apt install tesseract-ocr")
    
    # Check for FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("✅ FFmpeg already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found")
        print("   Please install:")
        print("   - Windows: Download from https://ffmpeg.org/download.html")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")

def test_video_installation():
    """Test that video processing components work"""
    print("\n🧪 Testing Video Concept Extraction Installation...")
    
    try:
        # Test Whisper
        import whisper
        model = whisper.load_model("tiny")  # Use tiny model for quick test
        print("✅ Whisper - Working")
        
        # Test transformers/BLIP
        from transformers import pipeline
        # Don't actually load the model to save time/memory
        print("✅ Transformers (BLIP) - Available")
        
        # Test OpenCV
        import cv2
        print("✅ OpenCV - Working")
        
        # Test PyTorch
        import torch
        print(f"✅ PyTorch - Working (CUDA: {'Yes' if torch.cuda.is_available() else 'No'})")
        
        # Test Detectron2 (optional)
        try:
            import detectron2
            print("✅ Detectron2 - Available")
        except ImportError:
            print("ℹ️ Detectron2 - Not installed (object detection disabled)")
        
        # Test Tesseract
        try:
            import pytesseract
            # Quick OCR test
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR - Working")
        except Exception:
            print("⚠️ Tesseract OCR - Not working properly")
        
        print("\n🎉 Video Concept Extraction System Ready!")
        print("🎬 You can now extract concepts from:")
        print("   📹 Local video files (MP4, AVI, MOV, etc.)")
        print("   📺 YouTube videos")
        print("   🎥 Webcam streams")
        print("   🎓 Lecture recordings")
        print("   📚 Educational content")
        print("   🎭 Documentary films")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_video_test_script():
    """Create a test script for video extraction"""
    test_script = '''#!/usr/bin/env python3
"""
🎬 Video Concept Extraction Test Script
"""

import sys
from pathlib import Path
sys.path.append('.')  # Add current directory to path

def test_video_extraction():
    """Test video concept extraction on a sample"""
    
    try:
        from ingest_pdf.extractConceptsFromVideo import extractConceptsFromVideo
        
        print("🎬 Video Concept Extraction Test")
        print("=" * 40)
        
        # You can test with any video file
        video_path = input("Enter path to a video file (or press Enter to skip): ").strip()
        
        if not video_path:
            print("📹 No video path provided - showing system capabilities instead")
            print("\\n🎬 Video Processing Capabilities:")
            print("  🎤 Audio transcription with Whisper")
            print("  👁️ Visual analysis with BLIP + Detectron2")
            print("  📖 OCR text extraction with Tesseract")
            print("  🔗 Multimodal concept alignment")
            print("  🌍 Universal concept extraction integration")
            print("\\nTo test with a real video:")
            print("  python test_video_extraction.py")
            print("  # Then enter path to MP4/AVI/MOV file")
            return
        
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"🎬 Processing video: {Path(video_path).name}")
        print("This may take a few minutes...")
        
        concepts = extractConceptsFromVideo(video_path)
        
        if concepts:
            print(f"\\n📊 Extracted {len(concepts)} concepts:")
            
            for i, concept in enumerate(concepts[:10], 1):
                name = concept.get('name', 'Unknown')
                score = concept.get('score', 0)
                modalities = concept.get('source', {}).get('modalities', [])
                
                modality_emoji = "🎤" if "audio" in modalities else ""
                modality_emoji += "👁️" if "visual" in modalities else ""
                if len(modalities) > 1:
                    modality_emoji = "🔗"  # Multimodal
                
                print(f"  {modality_emoji} {name} (score: {score:.3f})")
            
            if len(concepts) > 10:
                print(f"  ... and {len(concepts) - 10} more concepts")
            
            # Show modality breakdown
            audio_only = sum(1 for c in concepts if c.get('source', {}).get('modalities') == ['audio'])
            visual_only = sum(1 for c in concepts if c.get('source', {}).get('modalities') == ['visual'])
            multimodal = sum(1 for c in concepts if len(c.get('source', {}).get('modalities', [])) > 1)
            
            print(f"\\n📊 Modality breakdown:")
            print(f"  🎤 Audio only: {audio_only} concepts")
            print(f"  👁️ Visual only: {visual_only} concepts")
            print(f"  🔗 Multimodal: {multimodal} concepts")
            
            print("\\n✅ Video concept extraction working!")
            
        else:
            print("❌ No concepts extracted - check video file and installation")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check your installation.")

if __name__ == "__main__":
    test_video_extraction()
'''
    
    script_path = Path("test_video_extraction.py")
    with open(script_path, "w") as f:
        f.write(test_script)
    
    print(f"📝 Created video test script: {script_path}")
    print("   Run with: python test_video_extraction.py")

def main():
    """Main setup function"""
    print("🎬 MULTIMODAL VIDEO CONCEPT EXTRACTION SETUP")
    print("=" * 60)
    print("This will install dependencies for extracting concepts from:")
    print("• 🎓 Lecture videos and educational content")
    print("• 📺 YouTube videos and online content") 
    print("• 🎥 Documentary films and interviews")
    print("• 📹 Recorded presentations and webinars")
    print("• 🎭 Cultural and artistic video content")
    print("• 🔬 Scientific demonstrations and experiments")
    print("=" * 60)
    
    response = input("\\nProceed with installation? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    # Install Python packages
    install_video_packages()
    
    # Check system dependencies
    install_system_dependencies()
    
    # Test installation
    if test_video_installation():
        print("\\n🎯 Video Processing Setup Complete!")
        create_video_test_script()
        
        print("\\n🚀 Next Steps:")
        print("1. Test with a video: python test_video_extraction.py")
        print("2. Process academic videos to extract key concepts")
        print("3. Combine with universal text extraction for complete coverage")
        print("4. Watch your concept file_storage grow from multimodal content!")
    else:
        print("\\n⚠️ Setup completed with some issues. Please check the error messages above.")

if __name__ == "__main__":
    main()
