"""
🎬 MULTIMODAL VIDEO CONCEPT EXTRACTION

Complete pipeline for extracting concepts from academic videos using:
- Audio transcription (Whisper/Vosk)
- Visual analysis (Detectron2, BLIP, OCR)
- Temporal alignment and fusion
- Integration with universal text extraction system
"""

import logging
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import subprocess
import tempfile
import os

# Configure logging
logger = logging.getLogger(__name__)

# Global model instances (loaded once for performance)
_whisper_model = None
_blip_captioner = None
_detectron_predictor = None
_ocr_initialized = False

def _initialize_video_models():
    """Initialize all video processing models once at module load time"""
    global _whisper_model, _blip_captioner, _detectron_predictor, _ocr_initialized
    
    if _whisper_model is not None:
        return  # Already initialized
    
    logger.info("🎬 INITIALIZING MULTIMODAL VIDEO EXTRACTION MODELS...")
    
    try:
        # Whisper for speech-to-text
        import whisper
        _whisper_model = whisper.load_model("base")  # Use "large" for better accuracy, "base" for speed
        logger.info("✅ Whisper ASR model initialized")
        
        # BLIP for image captioning
        from transformers import pipeline
        _blip_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        logger.info("✅ BLIP image captioning initialized")
        
        # Detectron2 for object detection
        try:
            import torch
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detections
            
            _detectron_predictor = DefaultPredictor(cfg)
            logger.info("✅ Detectron2 object detection initialized")
            
        except ImportError:
            logger.warning("⚠️ Detectron2 not available - object detection disabled")
            _detectron_predictor = None
        
        # OCR initialization check
        try:
            import pytesseract
            _ocr_initialized = True
            logger.info("✅ Tesseract OCR initialized")
        except ImportError:
            logger.warning("⚠️ pytesseract not available - OCR disabled")
            _ocr_initialized = False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import required video libraries: {e}")
        logger.error("Please install: pip install openai-whisper transformers torch torchvision detectron2 pytesseract opencv-python")
        raise

def extract_video_frames(video_path: str, interval_seconds: int = 2) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from video at regular intervals
    
    Args:
        video_path: Path to video file
        interval_seconds: Interval between extracted frames
        
    Returns:
        List of (timestamp, frame) tuples
    """
    logger.info(f"🎬 Extracting frames from {Path(video_path).name} (interval: {interval_seconds}s)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Could not open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append((timestamp, frame))
            
        frame_count += 1
    
    cap.release()
    logger.info(f"📸 Extracted {len(frames)} frames from video")
    return frames

def transcribe_video_audio(video_path: str) -> Dict[str, Any]:
    """
    Transcribe video audio using Whisper
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with transcript text and segments
    """
    _initialize_video_models()
    
    logger.info(f"🎤 Transcribing audio from {Path(video_path).name}")
    
    try:
        result = _whisper_model.transcribe(video_path)
        
        logger.info(f"🎤 Transcription complete: {len(result.get('segments', []))} segments")
        logger.info(f"📝 Total transcript length: {len(result['text'])} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Audio transcription failed: {e}")
        return {"text": "", "segments": []}

def analyze_video_frame(frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
    """
    Analyze a single video frame using computer vision
    
    Args:
        frame: OpenCV frame (numpy array)
        timestamp: Frame timestamp in seconds
        
    Returns:
        Dictionary with detected objects, caption, and OCR text
    """
    frame_analysis = {
        "timestamp": timestamp,
        "objects": [],
        "caption": "",
        "ocr_text": "",
        "concepts": []
    }
    
    try:
        # Object detection with Detectron2
        if _detectron_predictor is not None:
            outputs = _detectron_predictor(frame)
            instances = outputs["instances"]
            
            if len(instances) > 0:
                classes = instances.pred_classes.tolist()
                scores = instances.scores.tolist()
                
                # Get class names from COCO dataset
                class_names = _detectron_predictor.metadata.get("thing_classes", [])
                
                detected_objects = []
                for class_id, score in zip(classes, scores):
                    if class_id < len(class_names):
                        obj_name = class_names[class_id]
                        detected_objects.append({
                            "name": obj_name,
                            "confidence": float(score)
                        })
                
                frame_analysis["objects"] = detected_objects
                logger.debug(f"🔍 Frame {timestamp:.1f}s: Detected {len(detected_objects)} objects")
        
        # Image captioning with BLIP
        if _blip_captioner is not None:
            # Convert BGR to RGB for BLIP
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            caption_result = _blip_captioner(rgb_frame)
            if caption_result and len(caption_result) > 0:
                frame_analysis["caption"] = caption_result[0].get('generated_text', '')
                logger.debug(f"📝 Frame {timestamp:.1f}s caption: {frame_analysis['caption'][:50]}...")
        
        # OCR text extraction
        if _ocr_initialized:
            import pytesseract
            
            # Convert to grayscale for better OCR
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply some preprocessing to improve OCR
            gray_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            ocr_text = pytesseract.image_to_string(gray_frame).strip()
            if ocr_text:
                frame_analysis["ocr_text"] = ocr_text
                logger.debug(f"📖 Frame {timestamp:.1f}s OCR: {len(ocr_text)} characters")
        
    except Exception as e:
        logger.warning(f"⚠️ Frame analysis failed for timestamp {timestamp:.1f}s: {e}")
    
    return frame_analysis

def extract_video_captions(video_path: str) -> Optional[str]:
    """
    Extract existing captions/subtitles from video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Caption text if available, None otherwise
    """
    try:
        # Try to extract subtitles using ffmpeg
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.srt', delete=False) as temp_srt:
            cmd = [
                'ffmpeg', '-i', video_path, '-map', '0:s:0', 
                '-c:s', 'srt', temp_srt.name, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read the SRT file
                with open(temp_srt.name, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                # Simple SRT parsing to extract just the text
                import re
                text_lines = []
                for line in srt_content.split('\n'):
                    line = line.strip()
                    # Skip sequence numbers and timestamps
                    if not re.match(r'^\d+$', line) and not re.match(r'\d{2}:\d{2}:\d{2}', line) and line:
                        text_lines.append(line)
                
                caption_text = ' '.join(text_lines)
                logger.info(f"📺 Extracted captions: {len(caption_text)} characters")
                
                # Clean up temp file
                os.unlink(temp_srt.name)
                
                return caption_text
            else:
                os.unlink(temp_srt.name)
                return None
                
    except Exception as e:
        logger.debug(f"📺 Caption extraction failed (this is normal for most videos): {e}")
        return None

def align_and_fuse_video_concepts(
    audio_concepts: List[Tuple[float, str]], 
    visual_concepts: List[Tuple[float, str]],
    alignment_window: float = 30.0
) -> Dict[str, Any]:
    """
    Align and fuse concepts from audio and visual modalities
    
    Args:
        audio_concepts: List of (timestamp, concept) from audio
        visual_concepts: List of (timestamp, concept) from visuals
        alignment_window: Time window for considering concepts as aligned
        
    Returns:
        Dictionary of fused concepts with metadata
    """
    logger.info(f"🔗 Aligning {len(audio_concepts)} audio + {len(visual_concepts)} visual concepts")
    
    concept_timeline = {}
    
    # Process audio concepts
    for timestamp, concept in audio_concepts:
        concept_key = concept.lower().strip()
        if not concept_key:
            continue
            
        if concept_key not in concept_timeline:
            concept_timeline[concept_key] = {
                "name": concept,
                "timestamps": [],
                "modalities": set(),
                "contexts": []
            }
        
        concept_timeline[concept_key]["timestamps"].append(timestamp)
        concept_timeline[concept_key]["modalities"].add("audio")
        concept_timeline[concept_key]["contexts"].append(f"Audio at {timestamp:.1f}s")
    
    # Process visual concepts with alignment check
    for timestamp, concept in visual_concepts:
        concept_key = concept.lower().strip()
        if not concept_key:
            continue
        
        # Check if this concept aligns with any audio concept
        aligned = False
        for audio_time, audio_concept in audio_concepts:
            if (abs(timestamp - audio_time) <= alignment_window and 
                concept_key == audio_concept.lower().strip()):
                # Found alignment - add to existing concept
                concept_timeline[concept_key]["timestamps"].append(timestamp)
                concept_timeline[concept_key]["modalities"].add("visual")
                concept_timeline[concept_key]["contexts"].append(f"Visual at {timestamp:.1f}s")
                aligned = True
                break
        
        if not aligned:
            # New visual-only concept
            if concept_key not in concept_timeline:
                concept_timeline[concept_key] = {
                    "name": concept,
                    "timestamps": [],
                    "modalities": set(),
                    "contexts": []
                }
            
            concept_timeline[concept_key]["timestamps"].append(timestamp)
            concept_timeline[concept_key]["modalities"].add("visual")
            concept_timeline[concept_key]["contexts"].append(f"Visual at {timestamp:.1f}s")
    
    # Calculate scores for each concept
    scored_concepts = []
    for concept_key, data in concept_timeline.items():
        # Score based on frequency and multimodal presence
        frequency_score = min(len(data["timestamps"]) / 10.0, 1.0)  # Normalize frequency
        multimodal_bonus = 0.3 if len(data["modalities"]) > 1 else 0.0
        
        final_score = min(frequency_score + multimodal_bonus, 1.0)
        
        scored_concepts.append({
            "name": data["name"],
            "score": final_score,
            "method": f"video_multimodal_{'_'.join(sorted(data['modalities']))}",
            "source": {
                "video_extraction": True,
                "modalities": list(data["modalities"]),
                "frequency": len(data["timestamps"]),
                "multimodal": len(data["modalities"]) > 1
            },
            "context": f"Video concept from {', '.join(data['modalities'])}",
            "metadata": {
                "timestamps": data["timestamps"],
                "modalities": list(data["modalities"]),
                "contexts": data["contexts"],
                "extraction_method": "multimodal_video_pipeline",
                "alignment_window": alignment_window
            }
        })
    
    # Sort by score
    scored_concepts.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"🎯 Fused concepts: {len(scored_concepts)} unique concepts")
    multimodal_count = sum(1 for c in scored_concepts if c["source"]["multimodal"])
    logger.info(f"🔗 Multimodal concepts: {multimodal_count} concepts found in both audio and visual")
    
    return scored_concepts

def extractConceptsFromVideo(video_path: str, frame_interval: int = 2, alignment_window: float = 30.0) -> List[Dict[str, Any]]:
    """
    🎬 MAIN VIDEO CONCEPT EXTRACTION FUNCTION
    
    Extract concepts from video using multimodal analysis:
    - Audio transcription and concept extraction
    - Visual frame analysis and concept extraction
    - Temporal alignment and fusion
    
    Args:
        video_path: Path to video file
        frame_interval: Seconds between extracted frames
        alignment_window: Time window for aligning concepts across modalities
        
    Returns:
        List of concept dictionaries with scores and metadata
    """
    logger.info(f"🎬 🧬 MULTIMODAL VIDEO CONCEPT EXTRACTION: {Path(video_path).name}")
    
    start_time = datetime.now()
    
    # Initialize models
    _initialize_video_models()
    
    # Import universal text extraction
    try:
        from extractConceptsFromDocument import extractConceptsFromDocument
        logger.info("✅ Universal text extraction available for video content")
    except ImportError:
        logger.warning("⚠️ Universal text extraction not available - using basic extraction")
        extractConceptsFromDocument = None
    
    # Step 1: Audio transcription
    logger.info("🎤 STEP 1: Audio transcription...")
    transcript_data = transcribe_video_audio(video_path)
    transcript_text = transcript_data.get("text", "")
    segments = transcript_data.get("segments", [])
    
    # Step 2: Caption extraction (if available)
    logger.info("📺 STEP 2: Caption extraction...")
    caption_text = extract_video_captions(video_path)
    if caption_text:
        logger.info(f"📺 Using video captions ({len(caption_text)} chars)")
        # Combine or prefer captions over transcript
        full_transcript = f"{transcript_text} {caption_text}"
    else:
        full_transcript = transcript_text
    
    # Step 3: Frame extraction and analysis
    logger.info("🖼️ STEP 3: Visual frame analysis...")
    frames = extract_video_frames(video_path, frame_interval)
    
    visual_analyses = []
    for timestamp, frame in frames:
        frame_data = analyze_video_frame(frame, timestamp)
        visual_analyses.append(frame_data)
    
    # Step 4: Extract concepts from transcript using universal extraction
    logger.info("🧠 STEP 4: Audio concept extraction...")
    audio_concepts = []
    
    if extractConceptsFromDocument and full_transcript.strip():
        # Use universal text extraction on full transcript
        transcript_concepts = extractConceptsFromDocument(full_transcript)
        
        # Assign timestamps to concepts based on segments
        for concept_data in transcript_concepts:
            concept_name = concept_data.get("name", "")
            
            # Find which segment(s) contain this concept
            for segment in segments:
                segment_text = segment.get("text", "").lower()
                if concept_name.lower() in segment_text:
                    timestamp = segment.get("start", 0)
                    audio_concepts.append((timestamp, concept_name))
                    break
            else:
                # If not found in segments, assign to middle of video
                if segments:
                    mid_time = segments[len(segments)//2].get("start", 0)
                    audio_concepts.append((mid_time, concept_name))
    
    # Step 5: Extract concepts from visual content
    logger.info("👁️ STEP 5: Visual concept extraction...")
    visual_concepts = []
    
    for frame_data in visual_analyses:
        timestamp = frame_data["timestamp"]
        
        # Concepts from detected objects
        for obj in frame_data.get("objects", []):
            if obj["confidence"] > 0.7:  # High confidence objects only
                visual_concepts.append((timestamp, obj["name"]))
        
        # Concepts from image captions
        caption = frame_data.get("caption", "")
        if caption and extractConceptsFromDocument:
            caption_concepts = extractConceptsFromDocument(caption)
            for concept_data in caption_concepts[:3]:  # Top 3 concepts per caption
                concept_name = concept_data.get("name", "")
                if concept_name:
                    visual_concepts.append((timestamp, concept_name))
        
        # Concepts from OCR text
        ocr_text = frame_data.get("ocr_text", "")
        if ocr_text and extractConceptsFromDocument:
            ocr_concepts = extractConceptsFromDocument(ocr_text)
            for concept_data in ocr_concepts[:5]:  # Top 5 concepts per OCR
                concept_name = concept_data.get("name", "")
                if concept_name:
                    visual_concepts.append((timestamp, concept_name))
    
    # Step 6: Align and fuse concepts
    logger.info("🔗 STEP 6: Multimodal alignment and fusion...")
    fused_concepts = align_and_fuse_video_concepts(audio_concepts, visual_concepts, alignment_window)
    
    # Processing summary
    processing_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"✅ 🎬 MULTIMODAL VIDEO EXTRACTION COMPLETE: {Path(video_path).name}")
    logger.info(f"📊 Results: {len(fused_concepts)} concepts extracted")
    logger.info(f"   🎤 Audio concepts: {len(audio_concepts)}")
    logger.info(f"   👁️ Visual concepts: {len(visual_concepts)}")
    logger.info(f"   🔗 Fused concepts: {len(fused_concepts)}")
    logger.info(f"⏱️ Processing time: {processing_time:.1f}s")
    
    # Log top concepts
    if fused_concepts:
        logger.info("🏆 Top video concepts:")
        for i, concept in enumerate(fused_concepts[:5], 1):
            name = concept.get("name", "")
            score = concept.get("score", 0)
            modalities = concept.get("source", {}).get("modalities", [])
            logger.info(f"  {i}. {name} (score: {score:.3f}, modalities: {', '.join(modalities)})")
    
    return fused_concepts

# Convenience functions for different video sources
def extractConceptsFromYouTubeVideo(youtube_url: str, **kwargs) -> List[Dict[str, Any]]:
    """Extract concepts from YouTube video"""
    try:
        from pytube import YouTube
        
        yt = YouTube(youtube_url)
        
        # Download video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            if stream:
                stream.download(filename=temp_file.name)
                logger.info(f"📺 Downloaded YouTube video: {yt.title}")
                
                concepts = extractConceptsFromVideo(temp_file.name, **kwargs)
                
                # Clean up
                os.unlink(temp_file.name)
                
                return concepts
            else:
                logger.error("❌ No suitable video stream found")
                return []
                
    except ImportError:
        logger.error("❌ pytube not installed - cannot process YouTube videos")
        logger.error("Install with: pip install pytube")
        return []
    except Exception as e:
        logger.error(f"❌ YouTube video processing failed: {e}")
        return []

def extractConceptsFromWebcamStream(duration_seconds: int = 60, **kwargs) -> List[Dict[str, Any]]:
    """Extract concepts from webcam stream (for live lectures, etc.)"""
    logger.info(f"📹 Recording {duration_seconds}s from webcam...")
    
    # Record from webcam to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        
        # Get webcam resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration_seconds:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
        cap.release()
        out.release()
        
        logger.info(f"📹 Webcam recording complete: {temp_file.name}")
        
        # Extract concepts from recorded video
        concepts = extractConceptsFromVideo(temp_file.name, **kwargs)
        
        # Clean up
        os.unlink(temp_file.name)
        
        return concepts

# Integration with existing pipeline
def video_to_concept_file_storage(video_path: str, auto_prefill: bool = True) -> Dict[str, Any]:
    """
    Process video and optionally add concepts to file_storage
    
    Args:
        video_path: Path to video file
        auto_prefill: Whether to auto-add high-quality concepts to file_storage
        
    Returns:
        Processing results with concept statistics
    """
    logger.info(f"🎬 Processing video for concept file_storage: {Path(video_path).name}")
    
    # Extract concepts from video
    concepts = extractConceptsFromVideo(video_path)
    
    if auto_prefill:
        try:
            # Import auto-prefill function
            from pipeline import auto_prefill_concept_db
            
            # Add high-quality video concepts to file_storage
            prefilled_count = auto_prefill_concept_db(concepts, Path(video_path).name)
            
            logger.info(f"📥 Auto-prefilled {prefilled_count} video concepts to file_storage")
            
        except ImportError:
            logger.warning("⚠️ Auto-prefill not available - concepts not added to file_storage")
            prefilled_count = 0
    else:
        prefilled_count = 0
    
    return {
        "filename": Path(video_path).name,
        "concept_count": len(concepts),
        "concepts": concepts,
        "auto_prefilled_concepts": prefilled_count,
        "extraction_method": "multimodal_video_pipeline",
        "status": "success"
    }

logger.info("🎬 🧬 MULTIMODAL VIDEO CONCEPT EXTRACTION MODULE LOADED")
logger.info("✅ Ready for video analysis: lectures, documentaries, presentations, tutorials!")
