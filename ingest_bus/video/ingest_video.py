import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import deque
import json
import os
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU acceleration
CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
if CUDA_AVAILABLE:
    logger.info("CUDA acceleration available for OpenCV")

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

@dataclass
class FrameAnalysis:
    """Container for frame analysis results"""
    frame_number: int
    timestamp: float
    motion_vector: List[float]
    motion_magnitude: float
    dominant_hue: int
    color_histogram: np.ndarray
    brightness: float
    contrast: float
    sharpness: float
    face_detected: bool
    face_positions: List[Dict]
    scene_change: bool
    quality_score: float

@dataclass
class VideoMetadata:
    """Enhanced video metadata"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    bitrate: float
    has_audio: bool
    creation_time: Optional[str]

class OpticalFlowTracker:
    """Enhanced optical flow tracking with GPU support"""
    
    def __init__(self, use_cuda: bool = CUDA_AVAILABLE):
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        if self.use_cuda:
            # Initialize CUDA optical flow
            self.flow_calculator = cv2.cuda.FarnebackOpticalFlow_create(
                numLevels=5,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=13,
                numIters=10,
                polyN=5,
                polySigma=1.1,
                flags=0
            )
        else:
            # CPU parameters
            self.flow_params = dict(
                pyr_scale=0.5,
                levels=5,
                winsize=13,
                iterations=10,
                poly_n=5,
                poly_sigma=1.1,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
    
    def calculate_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Calculate optical flow between frames"""
        if self.use_cuda:
            # Upload to GPU
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            gpu_flow = cv2.cuda_GpuMat()
            
            gpu_prev.upload(prev_gray)
            gpu_curr.upload(curr_gray)
            
            # Calculate flow on GPU
            gpu_flow = self.flow_calculator.calc(gpu_prev, gpu_curr, None)
            
            # Download result
            flow = gpu_flow.download()
        else:
            # CPU calculation
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, **self.flow_params
            )
        
        return flow

class SceneDetector:
    """Advanced scene change detection"""
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        self.prev_histogram = None
        self.scene_history = deque(maxlen=10)
    
    def detect_scene_change(self, frame: np.ndarray) -> bool:
        """Detect if current frame is a scene change"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if self.prev_histogram is None:
            self.prev_histogram = hist
            return True
        
        # Calculate histogram difference
        diff = cv2.compareHist(self.prev_histogram, hist, cv2.HISTCMP_CHISQR)
        
        # Update history
        self.scene_history.append(diff)
        
        # Adaptive threshold based on recent history
        if len(self.scene_history) > 5:
            mean_diff = np.mean(self.scene_history)
            std_diff = np.std(self.scene_history)
            adaptive_threshold = mean_diff + 2 * std_diff
        else:
            adaptive_threshold = self.threshold
        
        is_scene_change = diff > adaptive_threshold
        
        if is_scene_change:
            self.prev_histogram = hist
        
        return is_scene_change

class FaceDetector:
    """Face detection with tracking"""
    
    def __init__(self):
        # Try to load DNN-based face detector for better accuracy
        self.use_dnn = False
        try:
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.use_dnn = True
                logger.info("Using DNN face detector")
        except:
            pass
        
        if not self.use_dnn:
            # Fallback to Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Using Haar Cascade face detector")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame"""
        faces = []
        
        if self.use_dnn:
            # DNN detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    
                    faces.append({
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': float(confidence)
                    })
        else:
            # Haar Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected:
                faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 0.8  # Default confidence for Haar
                })
        
        return faces

class VideoQualityAnalyzer:
    """Analyze video quality metrics"""
    
    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate frame sharpness using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """Calculate average brightness"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean() / 255.0)
    
    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate RMS contrast"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(gray.std() / 255.0)
    
    @staticmethod
    def calculate_quality_score(sharpness: float, brightness: float, contrast: float) -> float:
        """Combine metrics into overall quality score"""
        # Normalize sharpness (typical range 0-5000)
        norm_sharpness = np.tanh(sharpness / 1000)
        
        # Ideal brightness is around 0.5
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Higher contrast is generally better (up to a point)
        contrast_score = np.tanh(contrast * 3)
        
        # Weighted combination
        quality = (norm_sharpness * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return float(np.clip(quality, 0, 1))

def extract_video_metadata(video_path: str) -> VideoMetadata:
    """Extract comprehensive video metadata"""
    cap = cv2.VideoCapture(video_path)
    
    # Basic properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration
    duration = total_frames / fps if fps > 0 else 0
    
    # Get codec
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    # Estimate bitrate
    file_size = os.path.getsize(video_path)
    bitrate = (file_size * 8) / duration if duration > 0 else 0
    
    # Check for audio (OpenCV doesn't directly support this)
    # In production, use ffprobe or similar
    has_audio = False  # Placeholder
    
    # Get creation time if available
    creation_time = None
    try:
        stat = os.stat(video_path)
        creation_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
    except:
        pass
    
    cap.release()
    
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        codec=codec,
        bitrate=bitrate,
        has_audio=has_audio,
        creation_time=creation_time
    )

def process_frame_batch(
    frames: List[Tuple[int, np.ndarray]], 
    prev_gray: Optional[np.ndarray],
    flow_tracker: OpticalFlowTracker,
    scene_detector: SceneDetector,
    face_detector: FaceDetector,
    base_timestamp: float,
    fps: float
) -> List[FrameAnalysis]:
    """Process a batch of frames in parallel"""
    results = []
    
    for frame_num, frame in frames:
        timestamp = base_timestamp + (frame_num / fps)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Motion analysis
        motion_vector = [0.0, 0.0]
        motion_magnitude = 0.0
        
        if prev_gray is not None:
            flow = flow_tracker.calculate_flow(prev_gray, gray)
            dx, dy = float(flow[..., 0].mean()), float(flow[..., 1].mean())
            motion_vector = [dx, dy]
            motion_magnitude = float(np.sqrt(dx*dx + dy*dy))
        
        # Color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hue_hist))
        
        # Full color histogram for holographic hints
        color_hist = cv2.calcHist([hsv], [0, 1, 2], None, [30, 32, 32], [0, 180, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        
        # Quality metrics
        sharpness = VideoQualityAnalyzer.calculate_sharpness(frame)
        brightness = VideoQualityAnalyzer.calculate_brightness(frame)
        contrast = VideoQualityAnalyzer.calculate_contrast(frame)
        quality_score = VideoQualityAnalyzer.calculate_quality_score(sharpness, brightness, contrast)
        
        # Scene detection
        scene_change = scene_detector.detect_scene_change(frame)
        
        # Face detection
        faces = face_detector.detect_faces(frame)
        face_detected = len(faces) > 0
        
        # Create frame analysis
        analysis = FrameAnalysis(
            frame_number=frame_num,
            timestamp=timestamp,
            motion_vector=motion_vector,
            motion_magnitude=motion_magnitude,
            dominant_hue=dominant_hue,
            color_histogram=color_hist,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            face_detected=face_detected,
            face_positions=faces,
            scene_change=scene_change,
            quality_score=quality_score
        )
        
        results.append(analysis)
        prev_gray = gray
    
    return results

def ingest_video(video_path: str, sample_fps: int = 1, 
                advanced_analysis: bool = True,
                max_frames: Optional[int] = None) -> dict:
    """
    Enhanced video processing with holographic motion analysis.
    
    Args:
        video_path: Path to video file
        sample_fps: Target sampling rate (frames per second)
        advanced_analysis: Enable face detection and quality analysis
        max_frames: Maximum number of frames to process
    
    Returns:
        Comprehensive video analysis with holographic hints
    """
    logger.info(f"Processing video: {video_path}")
    
    # Extract metadata
    metadata = extract_video_metadata(video_path)
    logger.info(f"Video metadata: {metadata}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Initialize components
    flow_tracker = OpticalFlowTracker()
    scene_detector = SceneDetector()
    face_detector = FaceDetector() if advanced_analysis else None
    
    # Calculate frame sampling
    fps = metadata.fps
    step = max(1, round(fps / sample_fps))
    
    # Storage for results
    frame_analyses = []
    motions = []
    color_hues = []
    motion_magnitudes = []
    quality_scores = []
    scene_changes = []
    face_timeline = []
    
    # Process video in batches
    batch_size = 10
    frame_batch = []
    frame_count = 0
    processed_count = 0
    prev_gray = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check max frames limit
            if max_frames and processed_count >= max_frames:
                break
            
            # Sample frames according to step
            if frame_count % step == 0:
                frame_batch.append((frame_count, frame))
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    batch_results = process_frame_batch(
                        frame_batch,
                        prev_gray,
                        flow_tracker,
                        scene_detector,
                        face_detector if advanced_analysis else None,
                        0,  # Base timestamp
                        fps
                    )
                    
                    # Extract results
                    for analysis in batch_results:
                        frame_analyses.append(analysis)
                        motions.append(analysis.motion_vector)
                        color_hues.append(analysis.dominant_hue)
                        motion_magnitudes.append(analysis.motion_magnitude)
                        quality_scores.append(analysis.quality_score)
                        scene_changes.append(analysis.scene_change)
                        
                        if analysis.face_detected:
                            face_timeline.append({
                                'timestamp': analysis.timestamp,
                                'faces': analysis.face_positions
                            })
                    
                    # Update prev_gray
                    if batch_results:
                        last_frame = frame_batch[-1][1]
                        prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                    
                    frame_batch = []
                    processed_count += batch_size
            
            frame_count += 1
            
            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / metadata.total_frames) * 100
                logger.info(f"Processing progress: {progress:.1f}%")
    
    finally:
        # Process remaining frames
        if frame_batch:
            batch_results = process_frame_batch(
                frame_batch,
                prev_gray,
                flow_tracker,
                scene_detector,
                face_detector if advanced_analysis else None,
                0,
                fps
            )
            
            for analysis in batch_results:
                frame_analyses.append(analysis)
                motions.append(analysis.motion_vector)
                color_hues.append(analysis.dominant_hue)
                motion_magnitudes.append(analysis.motion_magnitude)
                quality_scores.append(analysis.quality_score)
                scene_changes.append(analysis.scene_change)
                
                if analysis.face_detected:
                    face_timeline.append({
                        'timestamp': analysis.timestamp,
                        'faces': analysis.face_positions
                    })
        
        cap.release()
    
    # Calculate temporal coherence metrics
    temporal_coherence = calculate_temporal_coherence(
        motion_magnitudes, 
        color_hues,
        quality_scores,
        scene_changes
    )
    
    # Generate holographic hints
    holographic_hints = generate_holographic_hints(
        frame_analyses,
        metadata,
        temporal_coherence
    )
    
    # Prepare results
    result = {
        'motions': motions,
        'color_hues': color_hues,
        'motion_magnitudes': motion_magnitudes,
        'temporal_coherence': temporal_coherence,
        'frame_count': len(motions),
        'estimated_duration': len(motions) / sample_fps,
        'metadata': metadata.__dict__,
        'quality_metrics': {
            'average_quality': float(np.mean(quality_scores)) if quality_scores else 0,
            'quality_std': float(np.std(quality_scores)) if quality_scores else 0,
            'min_quality': float(np.min(quality_scores)) if quality_scores else 0,
            'max_quality': float(np.max(quality_scores)) if quality_scores else 0
        },
        'scene_analysis': {
            'total_scenes': sum(scene_changes),
            'scene_change_rate': sum(scene_changes) / len(scene_changes) if scene_changes else 0,
            'scene_boundaries': [i for i, change in enumerate(scene_changes) if change]
        },
        'holographic_hints': holographic_hints
    }
    
    # Add face timeline if available
    if face_timeline:
        result['face_timeline'] = face_timeline
    
    return result

def calculate_temporal_coherence(
    motion_magnitudes: List[float], 
    color_hues: List[float],
    quality_scores: List[float],
    scene_changes: List[bool]
) -> Dict:
    """Enhanced temporal coherence calculation"""
    if not motion_magnitudes or not color_hues:
        return {
            'motion_stability': 0.0,
            'color_stability': 0.0,
            'overall_coherence': 0.0,
            'beat_frequency': 0.0,
            'motion_energy': 0.0,
            'color_energy': 0.0,
            'quality_consistency': 0.0,
            'scene_stability': 0.0
        }
    
    # Motion stability (lower variance = higher stability)
    motion_array = np.array(motion_magnitudes)
    motion_variance = np.var(motion_array)
    motion_stability = 1.0 / (1.0 + motion_variance)
    
    # Color stability using circular statistics for hues
    hue_radians = np.array(color_hues) * np.pi / 180
    mean_vector = np.mean(np.exp(1j * hue_radians))
    color_stability = float(np.abs(mean_vector))
    
    # Quality consistency
    quality_array = np.array(quality_scores)
    quality_consistency = 1.0 - np.std(quality_array) if len(quality_array) > 1 else 1.0
    
    # Scene stability (fewer scene changes = more stable)
    scene_change_rate = sum(scene_changes) / len(scene_changes) if scene_changes else 0
    scene_stability = 1.0 - scene_change_rate
    
    # Beat frequency estimation from motion
    beat_frequency = estimate_visual_beat_frequency(motion_magnitudes)
    
    # Motion and color energy
    motion_energy = float(np.mean(motion_magnitudes))
    
    # Color energy from hue transitions
    if len(color_hues) > 1:
        hue_diffs = np.diff(color_hues)
        # Handle circular nature of hues
        hue_diffs = np.minimum(np.abs(hue_diffs), 360 - np.abs(hue_diffs))
        color_energy = float(np.mean(hue_diffs) / 180)  # Normalize to [0,1]
    else:
        color_energy = 0.0
    
    # Overall coherence combining all factors
    overall_coherence = (
        motion_stability * 0.3 +
        color_stability * 0.3 +
        quality_consistency * 0.2 +
        scene_stability * 0.2
    )
    
    return {
        'motion_stability': float(motion_stability),
        'color_stability': float(color_stability),
        'overall_coherence': float(overall_coherence),
        'beat_frequency': float(beat_frequency),
        'motion_energy': float(motion_energy),
        'color_energy': float(color_energy),
        'quality_consistency': float(quality_consistency),
        'scene_stability': float(scene_stability)
    }

def estimate_visual_beat_frequency(motion_magnitudes: List[float]) -> float:
    """Enhanced beat frequency estimation using autocorrelation and FFT"""
    if len(motion_magnitudes) < 8:
        return 0.0
    
    # Normalize motion magnitudes
    motion_array = np.array(motion_magnitudes)
    motion_array = (motion_array - np.mean(motion_array)) / (np.std(motion_array) + 1e-6)
    
    # Method 1: Autocorrelation
    autocorr = np.correlate(motion_array, motion_array, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
            peaks.append(i)
    
    if peaks and peaks[0] > 0:
        # Convert to frequency (assuming 1 FPS sampling)
        beat_frequency_autocorr = 1.0 / peaks[0]
    else:
        beat_frequency_autocorr = 0.0
    
    # Method 2: FFT
    fft = np.fft.rfft(motion_array)
    freqs = np.fft.rfftfreq(len(motion_array))
    
    # Find dominant frequency (excluding DC component)
    magnitude = np.abs(fft[1:])
    if len(magnitude) > 0:
        dominant_freq_idx = np.argmax(magnitude) + 1
        beat_frequency_fft = freqs[dominant_freq_idx]
    else:
        beat_frequency_fft = 0.0
    
    # Combine both methods
    if beat_frequency_autocorr > 0 and beat_frequency_fft > 0:
        # Average if both methods found a frequency
        beat_frequency = (beat_frequency_autocorr + beat_frequency_fft) / 2
    else:
        # Use whichever is non-zero
        beat_frequency = max(beat_frequency_autocorr, beat_frequency_fft)
    
    return beat_frequency

def generate_holographic_hints(
    frame_analyses: List[FrameAnalysis],
    metadata: VideoMetadata,
    temporal_coherence: Dict
) -> Dict:
    """Generate comprehensive holographic hints from video analysis"""
    
    if not frame_analyses:
        return {}
    
    # Motion trajectory analysis
    motion_vectors = [fa.motion_vector for fa in frame_analyses]
    motion_trajectory = analyze_motion_trajectory(motion_vectors)
    
    # Color evolution analysis
    color_evolution = analyze_color_evolution(frame_analyses)
    
    # Quality-based recommendations
    quality_scores = [fa.quality_score for fa in frame_analyses]
    quality_recommendations = generate_quality_recommendations(quality_scores)
    
    # Face tracking summary
    face_summary = summarize_face_tracking(frame_analyses)
    
    # Scene-based segmentation
    scene_segments = extract_scene_segments(frame_analyses)
    
    return {
        'motion_trajectory': motion_trajectory,
        'color_evolution': color_evolution,
        'quality_recommendations': quality_recommendations,
        'face_summary': face_summary,
        'scene_segments': scene_segments,
        'temporal_summary': {
            'total_motion': float(sum(fa.motion_magnitude for fa in frame_analyses)),
            'average_motion': float(np.mean([fa.motion_magnitude for fa in frame_analyses])),
            'motion_peaks': find_motion_peaks(frame_analyses),
            'color_transitions': find_color_transitions(frame_analyses),
            'quality_profile': quality_recommendations
        },
        'rendering_hints': {
            'recommended_fps': calculate_recommended_fps(temporal_coherence, metadata),
            'motion_blur': temporal_coherence['motion_energy'] > 0.5,
            'depth_layers': suggest_depth_layers(motion_trajectory),
            'particle_density': calculate_particle_density(temporal_coherence),
            'volumetric_complexity': estimate_volumetric_complexity(frame_analyses)
        }
    }

def analyze_motion_trajectory(motion_vectors: List[List[float]]) -> Dict:
    """Analyze motion patterns for holographic rendering"""
    if not motion_vectors:
        return {'type': 'static', 'parameters': {}}
    
    motion_array = np.array(motion_vectors)
    
    # Calculate trajectory statistics
    mean_motion = np.mean(motion_array, axis=0)
    motion_std = np.std(motion_array, axis=0)
    
    # Classify motion type
    total_motion = np.sum(np.linalg.norm(motion_array, axis=1))
    
    if total_motion < 0.1 * len(motion_vectors):
        motion_type = 'static'
    elif motion_std[0] < 0.5 and motion_std[1] < 0.5:
        motion_type = 'linear'
    elif np.abs(mean_motion[0]) > np.abs(mean_motion[1]) * 2:
        motion_type = 'horizontal'
    elif np.abs(mean_motion[1]) > np.abs(mean_motion[0]) * 2:
        motion_type = 'vertical'
    else:
        motion_type = 'complex'
    
    # Calculate motion path curvature
    if len(motion_vectors) > 2:
        curvature = calculate_path_curvature(motion_array)
    else:
        curvature = 0.0
    
    return {
        'type': motion_type,
        'parameters': {
            'mean_direction': mean_motion.tolist(),
            'variability': motion_std.tolist(),
            'total_displacement': float(total_motion),
            'curvature': float(curvature),
            'smoothness': calculate_motion_smoothness(motion_array)
        }
    }

def calculate_path_curvature(motion_array: np.ndarray) -> float:
    """Calculate curvature of motion path"""
    if len(motion_array) < 3:
        return 0.0
    
    # Calculate cumulative path
    path = np.cumsum(motion_array, axis=0)
    
    # Calculate curvature using three-point method
    curvatures = []
    for i in range(1, len(path) - 1):
        p1, p2, p3 = path[i-1], path[i], path[i+1]
        
        # Calculate curvature
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            curvatures.append(angle)
    
    return float(np.mean(curvatures)) if curvatures else 0.0

def calculate_motion_smoothness(motion_array: np.ndarray) -> float:
    """Calculate smoothness of motion (0-1, higher is smoother)"""
    if len(motion_array) < 2:
        return 1.0
    
    # Calculate acceleration (second derivative)
    velocity = np.diff(motion_array, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    # Smoothness is inverse of acceleration variance
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    smoothness = 1.0 / (1.0 + np.var(acc_magnitude))
    
    return float(smoothness)

def analyze_color_evolution(frame_analyses: List[FrameAnalysis]) -> Dict:
    """Analyze how colors evolve throughout the video"""
    hues = [fa.dominant_hue for fa in frame_analyses]
    
    # Color palette extraction
    unique_hues, counts = np.unique(hues, return_counts=True)
    top_colors = unique_hues[np.argsort(counts)[-5:]][::-1]  # Top 5 colors
    
    # Color transition matrix
    transition_matrix = np.zeros((180, 180))
    for i in range(1, len(hues)):
        transition_matrix[hues[i-1], hues[i]] += 1
    
    # Find most common transitions
    top_transitions = []
    for i in range(5):
        idx = np.unravel_index(np.argmax(transition_matrix), transition_matrix.shape)
        if transition_matrix[idx] > 0:
            top_transitions.append({
                'from_hue': int(idx[0]),
                'to_hue': int(idx[1]),
                'count': int(transition_matrix[idx])
            })
            transition_matrix[idx] = 0
    
    return {
        'dominant_palette': top_colors.tolist(),
        'color_diversity': float(len(unique_hues) / 180),  # Normalized diversity
        'top_transitions': top_transitions,
        'color_temperature': estimate_color_temperature(hues),
        'mood': classify_color_mood(top_colors)
    }

def estimate_color_temperature(hues: List[int]) -> str:
    """Estimate overall color temperature"""
    if not hues:
        return 'neutral'
    
    avg_hue = np.mean(hues)
    
    # Classify based on hue ranges
    if 0 <= avg_hue <= 30 or 330 <= avg_hue <= 360:  # Reds
        return 'warm'
    elif 30 < avg_hue <= 90:  # Yellows/Greens
        return 'warm-neutral'
    elif 90 < avg_hue <= 210:  # Blues/Cyans
        return 'cool'
    elif 210 < avg_hue <= 270:  # Purples
        return 'cool-neutral'
    else:  # Magentas
        return 'warm'

def classify_color_mood(top_colors: np.ndarray) -> str:
    """Classify mood based on dominant colors"""
    if len(top_colors) == 0:
        return 'neutral'
    
    # Simple mood classification based on hue ranges
    moods = []
    for hue in top_colors:
        if 0 <= hue <= 10 or 340 <= hue <= 360:  # Red
            moods.append('energetic')
        elif 20 <= hue <= 60:  # Orange/Yellow
            moods.append('cheerful')
        elif 60 <= hue <= 140:  # Green
            moods.append('calm')
        elif 180 <= hue <= 240:  # Blue
            moods.append('serene')
        elif 270 <= hue <= 320:  # Purple
            moods.append('mysterious')
    
    # Return most common mood
    if moods:
        return max(set(moods), key=moods.count)
    return 'neutral'

def generate_quality_recommendations(quality_scores: List[float]) -> Dict:
    """Generate recommendations based on quality analysis"""
    if not quality_scores:
        return {'overall': 'unknown', 'recommendations': []}
    
    avg_quality = np.mean(quality_scores)
    min_quality = np.min(quality_scores)
    quality_variance = np.var(quality_scores)
    
    # Overall assessment
    if avg_quality >= 0.8:
        overall = 'excellent'
    elif avg_quality >= 0.6:
        overall = 'good'
    elif avg_quality >= 0.4:
        overall = 'fair'
    else:
        overall = 'poor'
    
    # Specific recommendations
    recommendations = []
    
    if min_quality < 0.3:
        recommendations.append('Consider quality enhancement for low-quality segments')
    
    if quality_variance > 0.1:
        recommendations.append('Quality varies significantly - consider stabilization')
    
    if avg_quality < 0.5:
        recommendations.append('Overall quality is low - upscaling recommended')
    
    return {
        'overall': overall,
        'average_score': float(avg_quality),
        'minimum_score': float(min_quality),
        'variance': float(quality_variance),
        'recommendations': recommendations
    }

def summarize_face_tracking(frame_analyses: List[FrameAnalysis]) -> Dict:
    """Summarize face detection results"""
    face_frames = [fa for fa in frame_analyses if fa.face_detected]
    
    if not face_frames:
        return {'faces_detected': False, 'coverage': 0.0}
    
    # Calculate face coverage
    coverage = len(face_frames) / len(frame_analyses) if frame_analyses else 0
    
    # Track face movements
    face_positions = []
    for fa in face_frames:
        if fa.face_positions:
            # Use first face for simplicity
            face = fa.face_positions[0]
            center_x = face['x'] + face['width'] / 2
            center_y = face['y'] + face['height'] / 2
            face_positions.append([center_x, center_y])
    
    # Analyze face movement
    face_motion = {}
    if len(face_positions) > 1:
        positions_array = np.array(face_positions)
        face_motion = {
            'average_position': np.mean(positions_array, axis=0).tolist(),
            'movement_range': (np.max(positions_array, axis=0) - np.min(positions_array, axis=0)).tolist(),
            'stability': float(1.0 / (1.0 + np.var(positions_array)))
        }
    
    return {
        'faces_detected': True,
        'coverage': float(coverage),
        'total_face_frames': len(face_frames),
        'max_faces_in_frame': max(len(fa.face_positions) for fa in face_frames),
        'face_motion': face_motion
    }

def extract_scene_segments(frame_analyses: List[FrameAnalysis]) -> List[Dict]:
    """Extract scene segments from analysis"""
    segments = []
    current_segment = {'start': 0, 'frames': []}
    
    for i, fa in enumerate(frame_analyses):
        if fa.scene_change and i > 0:
            # End current segment
            current_segment['end'] = i - 1
            current_segment['duration'] = (current_segment['end'] - current_segment['start'] + 1)
            segments.append(current_segment)
            
            # Start new segment
            current_segment = {'start': i, 'frames': []}
        
        current_segment['frames'].append(i)
    
    # Add final segment
    if current_segment['frames']:
        current_segment['end'] = len(frame_analyses) - 1
        current_segment['duration'] = (current_segment['end'] - current_segment['start'] + 1)
        segments.append(current_segment)
    
    # Analyze each segment
    for segment in segments:
        frames = [frame_analyses[i] for i in segment['frames']]
        
        # Calculate segment characteristics
        segment['characteristics'] = {
            'average_motion': float(np.mean([f.motion_magnitude for f in frames])),
            'dominant_hue': int(np.median([f.dominant_hue for f in frames])),
            'quality': float(np.mean([f.quality_score for f in frames])),
            'has_faces': any(f.face_detected for f in frames)
        }
    
    return segments

def find_motion_peaks(frame_analyses: List[FrameAnalysis], threshold_percentile: float = 90) -> List[Dict]:
    """Find peaks in motion activity"""
    motions = [fa.motion_magnitude for fa in frame_analyses]
    
    if not motions:
        return []
    
    threshold = np.percentile(motions, threshold_percentile)
    peaks = []
    
    for i, fa in enumerate(frame_analyses):
        if fa.motion_magnitude > threshold:
            peaks.append({
                'frame': i,
                'timestamp': fa.timestamp,
                'magnitude': fa.motion_magnitude,
                'direction': fa.motion_vector
            })
    
    return peaks

def find_color_transitions(frame_analyses: List[FrameAnalysis], threshold: float = 30) -> List[Dict]:
    """Find significant color transitions"""
    transitions = []
    
    for i in range(1, len(frame_analyses)):
        prev_hue = frame_analyses[i-1].dominant_hue
        curr_hue = frame_analyses[i].dominant_hue
        
        # Calculate hue difference (circular)
        diff = abs(curr_hue - prev_hue)
        diff = min(diff, 360 - diff)
        
        if diff > threshold:
            transitions.append({
                'frame': i,
                'timestamp': frame_analyses[i].timestamp,
                'from_hue': prev_hue,
                'to_hue': curr_hue,
                'difference': diff
            })
    
    return transitions

def calculate_recommended_fps(temporal_coherence: Dict, metadata: VideoMetadata) -> float:
    """Calculate recommended FPS for holographic rendering"""
    base_fps = 30.0
    
    # Adjust based on motion
    if temporal_coherence['motion_energy'] > 0.7:
        base_fps = 60.0
    elif temporal_coherence['motion_energy'] < 0.3:
        base_fps = 24.0
    
    # Consider beat frequency
    if temporal_coherence['beat_frequency'] > 0:
        # Align with beat frequency if reasonable
        beat_fps = temporal_coherence['beat_frequency'] * 2  # Nyquist
        if 15 <= beat_fps <= 120:
            base_fps = beat_fps
    
    return float(np.clip(base_fps, 15, 60))

def suggest_depth_layers(motion_trajectory: Dict) -> int:
    """Suggest number of depth layers based on motion complexity"""
    motion_type = motion_trajectory['type']
    
    if motion_type == 'static':
        return 3
    elif motion_type in ['linear', 'horizontal', 'vertical']:
        return 5
    else:  # complex
        return 8

def calculate_particle_density(temporal_coherence: Dict) -> float:
    """Calculate particle density based on video characteristics"""
    # Base density on motion energy and stability
    energy = temporal_coherence['motion_energy']
    stability = temporal_coherence['motion_stability']
    
    # High energy + low stability = more particles
    density = energy * (2 - stability)
    
    return float(np.clip(density, 0.1, 1.0))

def estimate_volumetric_complexity(frame_analyses: List[FrameAnalysis]) -> float:
    """Estimate complexity for volumetric rendering"""
    if not frame_analyses:
        return 0.5
    
    # Factors: motion variation, color diversity, quality variation
    motion_var = np.var([fa.motion_magnitude for fa in frame_analyses])
    color_diversity = len(set(fa.dominant_hue for fa in frame_analyses)) / 180
    quality_var = np.var([fa.quality_score for fa in frame_analyses])
    
    # Combine factors
    complexity = (
        np.tanh(motion_var * 10) * 0.4 +
        color_diversity * 0.3 +
        np.tanh(quality_var * 5) * 0.3
    )
    
    return float(np.clip(complexity, 0, 1))

# Batch processing function
def batch_process_videos(video_paths: List[str], sample_fps: int = 1,
                        max_workers: int = 4) -> List[Dict]:
    """Process multiple videos in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(ingest_video, path, sample_fps): path 
            for path in video_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                result['source_path'] = path
                results.append(result)
                logger.info(f"Completed processing: {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    'source_path': path,
                    'error': str(e)
                })
    
    return results

# Export main function
__all__ = ['ingest_video', 'batch_process_videos', 'VideoMetadata', 'FrameAnalysis', 'handle']

# ---------------------------------------------------------------------------
# Router-style entry-point
# ---------------------------------------------------------------------------

async def handle(file_path: str, mime_type: str = "", **kwargs) -> dict:
    """
    Pipeline wrapper - demuxes & analyses video, returns IngestResult.
    """
    import asyncio
    from pathlib import Path
    
    # Extract kwargs
    sample_fps = kwargs.get("sample_fps", 1)
    doc_id = kwargs.get("doc_id", "router")
    
    # Run synchronous function in thread pool
    loop = asyncio.get_event_loop()
    result_dict = await loop.run_in_executor(None, ingest_video, file_path, sample_fps)
    
    # Import the result class
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up to project root
    from ingest_pdf.pipeline.ingest_common.result import IngestResult
    
    # Extract concepts from the analysis
    concepts = []
    
    # Add scene concepts
    for scene in result_dict.get("scenes", []):
        concepts.append({
            "name": f"scene_{scene.get('type', 'unknown')}",
            "confidence": scene.get("confidence", 0.5),
            "type": "scene",
            "source": "video_analysis",
            "timestamp": scene.get("timestamp", 0)
        })
    
    # Add motion concepts
    motion = result_dict.get("motion_analysis", {})
    if motion.get("avg_motion", 0) > 0.5:
        concepts.append({
            "name": "high_motion",
            "confidence": min(motion.get("avg_motion", 0), 1.0),
            "type": "motion",
            "source": "video_analysis"
        })
    
    # Add audio concepts if present
    audio_concepts = result_dict.get("audio_analysis", {}).get("concepts", [])
    concepts.extend(audio_concepts)
    
    # Create IngestResult
    return IngestResult(
        filename=Path(file_path).name,
        file_path=file_path,
        media_type="video",
        transcript=result_dict.get("audio_analysis", {}).get("transcript", ""),
        concepts=concepts,
        concept_count=len(concepts),
        concept_names=[c.get("name", "") for c in concepts],
        psi_state=result_dict.get("audio_analysis", {}).get("psi_state", {}),
        spectral_features=result_dict.get("audio_analysis", {}).get("spectral_features", {}),
        duration_seconds=result_dict.get("metadata", {}).get("duration_seconds"),
        frame_count=result_dict.get("metadata", {}).get("frame_count"),
        fps=result_dict.get("metadata", {}).get("fps"),
        resolution=result_dict.get("metadata", {}).get("resolution"),
        file_size_bytes=result_dict.get("metadata", {}).get("file_size", 0),
        file_size_mb=result_dict.get("metadata", {}).get("file_size", 0) / (1024 * 1024),
        sha256=result_dict.get("metadata", {}).get("sha256", "unknown")
    ).to_dict()