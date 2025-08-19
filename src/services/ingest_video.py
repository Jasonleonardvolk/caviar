"""
ingest_video.py - Visual input pipeline for TORI's conceptual graph
Analyzes visual content and motion for concept extraction
"""

import asyncio
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

# Mock imports for production dependencies
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using mock object detection")

@dataclass
class VisualSpectralMetadata:
    """Visual spectral analysis for holographic integration"""
    dominant_colors: List[Tuple[int, int, int]]
    color_temperature: float
    brightness_mean: float
    contrast_level: float
    motion_intensity: float
    edge_density: float
    spatial_complexity: float
    chromatic_signature: Dict[str, float]

@dataclass
class VideoConceptDiff:
    """Extracted concepts from video with metadata"""
    concept_id: str
    confidence: float
    visual_source: str
    spectral_signature: VisualSpectralMetadata
    temporal_position: Tuple[float, float]  # start, end in seconds
    spatial_region: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    motion_context: Dict[str, float]

@dataclass
class VideoIngestionResult:
    """Complete result of video processing"""
    session_id: str
    concepts: List[VideoConceptDiff]
    duration: float
    frame_count: int
    spectral_summary: VisualSpectralMetadata
    motion_timeline: List[Dict[str, float]]
    detected_objects: List[Dict[str, Any]]
    phase_signature: str
    holographic_hints: Dict[str, Any]

class VisualEmotionDetector:
    """Analyzes visual characteristics for emotional/contextual state"""
    
    def __init__(self):
        self.emotion_indicators = {
            'calm': {'motion': (0, 0.2), 'brightness': (0.3, 0.7), 'contrast': (0.2, 0.6)},
            'chaotic': {'motion': (0.7, 1.0), 'brightness': (0.0, 1.0), 'contrast': (0.6, 1.0)},
            'focused': {'motion': (0, 0.3), 'brightness': (0.4, 0.8), 'contrast': (0.4, 0.8)},
            'unstable': {'motion': (0.4, 0.8), 'brightness': (0.0, 0.4), 'contrast': (0.7, 1.0)}
        }
    
    def analyze_visual_state(self, visual_features: VisualSpectralMetadata) -> Dict[str, float]:
        """Classify visual emotional state from features"""
        states = {}
        
        for state, ranges in self.emotion_indicators.items():
            motion_match = self._in_range(visual_features.motion_intensity, ranges['motion'])
            brightness_match = self._in_range(visual_features.brightness_mean, ranges['brightness'])
            contrast_match = self._in_range(visual_features.contrast_level, ranges['contrast'])
            
            # Combine scores
            states[state] = (motion_match + brightness_match + contrast_match) / 3
        
        # Normalize scores
        total = sum(states.values())
        if total > 0:
            states = {k: v / total for k, v in states.items()}
        
        return states
    
    def _in_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate how well a value fits within a range (0-1 score)"""
        min_val, max_val = range_tuple
        if min_val <= value <= max_val:
            center = (min_val + max_val) / 2
            return 1.0 - abs(value - center) / (max_val - min_val)
        else:
            if value < min_val:
                distance = min_val - value
            else:
                distance = value - max_val
            return max(0, 1.0 - distance / max_val)

class VisualSpectralAnalyzer:
    """Extracts visual spectral features for holographic mapping"""
    
    def __init__(self):
        self.previous_frame = None
    
    def analyze_frame(self, frame: np.ndarray) -> VisualSpectralMetadata:
        """Extract comprehensive visual features from a frame"""
        
        # Color analysis
        dominant_colors = self._extract_dominant_colors(frame)
        color_temperature = self._calculate_color_temperature(frame)
        chromatic_signature = self._analyze_chromatic_signature(frame)
        
        # Brightness and contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_mean = np.mean(gray) / 255.0
        contrast_level = np.std(gray) / 255.0
        
        # Motion analysis
        motion_intensity = self._calculate_motion_intensity(gray)
        
        # Spatial complexity
        edge_density = self._calculate_edge_density(gray)
        spatial_complexity = self._calculate_spatial_complexity(gray)
        
        return VisualSpectralMetadata(
            dominant_colors=dominant_colors,
            color_temperature=color_temperature,
            brightness_mean=brightness_mean,
            contrast_level=contrast_level,
            motion_intensity=motion_intensity,
            edge_density=edge_density,
            spatial_complexity=spatial_complexity,
            chromatic_signature=chromatic_signature
        )
    
    def _extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering"""
        data = frame.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to integers and return as tuples
        centers = np.uint8(centers)
        return [tuple(map(int, center)) for center in centers]
    
    def _calculate_color_temperature(self, frame: np.ndarray) -> float:
        """Calculate approximate color temperature"""
        # Simplified color temperature calculation
        mean_color = np.mean(frame, axis=(0, 1))
        b, g, r = mean_color
        
        # Simple approximation: higher red = warmer, higher blue = cooler
        if b > 0:
            temperature_ratio = r / b
            # Convert to approximate Kelvin (simplified)
            return 2000 + (temperature_ratio * 3000)  # Range: 2000K - 8000K
        return 5000  # Neutral temperature
    
    def _analyze_chromatic_signature(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze chromatic properties for emotional mapping"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        return {
            'saturation_mean': np.mean(s) / 255.0,
            'hue_variance': np.var(h) / (180 * 180),  # Normalized by max variance
            'value_range': (np.max(v) - np.min(v)) / 255.0,
            'warm_cool_balance': self._calculate_warm_cool_balance(frame)
        }
    
    def _calculate_warm_cool_balance(self, frame: np.ndarray) -> float:
        """Calculate balance between warm and cool colors"""
        # Convert to RGB and analyze color balance
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)
        
        warm_intensity = np.mean(r) + np.mean(g) * 0.5  # Red and some green
        cool_intensity = np.mean(b) + np.mean(g) * 0.5  # Blue and some green
        
        if warm_intensity + cool_intensity > 0:
            return warm_intensity / (warm_intensity + cool_intensity)
        return 0.5  # Neutral
    
    def _calculate_motion_intensity(self, gray: np.ndarray) -> float:
        """Calculate motion intensity compared to previous frame"""
        if self.previous_frame is None:
            self.previous_frame = gray.copy()
            return 0.0
        
        # Calculate optical flow or frame difference
        diff = cv2.absdiff(self.previous_frame, gray)
        motion_intensity = np.mean(diff) / 255.0
        
        self.previous_frame = gray.copy()
        return motion_intensity
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def _calculate_spatial_complexity(self, gray: np.ndarray) -> float:
        """Calculate spatial complexity using texture analysis"""
        # Simple complexity measure using local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        complexity = np.mean(local_variance) / (255 * 255)
        return complexity

class VideoObjectDetector:
    """Detects and classifies objects in video frames"""
    
    def __init__(self):
        self.model = self._init_model()
        self.common_objects = [
            'person', 'book', 'laptop', 'mouse', 'keyboard', 'cell phone',
            'chair', 'desk', 'monitor', 'bottle', 'cup', 'paper', 'pen'
        ]
    
    def _init_model(self):
        """Initialize object detection model"""
        if TORCH_AVAILABLE:
            try:
                # Load a pre-trained ResNet model for basic classification
                model = resnet50(pretrained=True)
                model.eval()
                return model
            except Exception as e:
                logging.warning(f"Failed to load model: {e}")
                return None
        return None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame"""
        if self.model and TORCH_AVAILABLE:
            return self._detect_with_model(frame)
        else:
            return self._mock_detection(frame)
    
    def _detect_with_model(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Actual object detection with ML model"""
        # Simplified detection - in production would use proper object detection
        # For now, return mock results
        return self._mock_detection(frame)
    
    def _mock_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Mock object detection for development"""
        # Analyze frame properties to make educated guesses
        height, width = frame.shape[:2]
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        detections = []
        
        # Mock detection based on frame characteristics
        if brightness > 100:  # Bright frame - likely indoor scene
            detections.extend([
                {'object': 'person', 'confidence': 0.7, 'bbox': (50, 50, 200, 300)},
                {'object': 'laptop', 'confidence': 0.6, 'bbox': (width//2, height//2, 300, 200)}
            ])
        
        if brightness < 80:  # Dark frame
            detections.append({'object': 'monitor', 'confidence': 0.5, 'bbox': (100, 100, 400, 300)})
        
        return detections

class VideoConceptExtractor:
    """Extracts concepts from detected objects and visual patterns"""
    
    def __init__(self):
        self.concept_mappings = {
            'person': ['human-presence', 'collaboration', 'communication'],
            'laptop': ['programming', 'work', 'technology'],
            'book': ['learning', 'knowledge', 'study'],
            'whiteboard': ['planning', 'brainstorming', 'visual-thinking'],
            'error': ['debugging', 'problem-solving', 'troubleshooting'],
            'code': ['programming', 'development', 'logic']
        }
        
        self.motion_concepts = {
            'high_motion': ['activity', 'energy', 'chaos'],
            'low_motion': ['focus', 'calm', 'stability'],
            'rapid_changes': ['excitement', 'urgency', 'multitasking']
        }
    
    def extract_concepts(self, detections: List[Dict[str, Any]], 
                        visual_metadata: VisualSpectralMetadata) -> List[str]:
        """Extract concept IDs from detections and visual analysis"""
        concepts = set()
        
        # Object-based concepts
        for detection in detections:
            obj = detection['object']
            if obj in self.concept_mappings:
                concepts.update(self.concept_mappings[obj])
        
        # Motion-based concepts
        motion_level = visual_metadata.motion_intensity
        if motion_level > 0.7:
            concepts.update(self.motion_concepts['high_motion'])
        elif motion_level < 0.2:
            concepts.update(self.motion_concepts['low_motion'])
        
        # Visual pattern concepts
        if visual_metadata.edge_density > 0.3:
            concepts.add('visual-complexity')
        
        if visual_metadata.contrast_level > 0.7:
            concepts.add('high-contrast')
        
        # Color-based emotional concepts
        chromatic = visual_metadata.chromatic_signature
        if chromatic['warm_cool_balance'] > 0.7:
            concepts.add('warm-environment')
        elif chromatic['warm_cool_balance'] < 0.3:
            concepts.add('cool-environment')
        
        return list(concepts)

class VideoProcessor:
    """Main video ingestion processor"""
    
    def __init__(self):
        self.spectral_analyzer = VisualSpectralAnalyzer()
        self.emotion_detector = VisualEmotionDetector()
        self.object_detector = VideoObjectDetector()
        self.concept_extractor = VideoConceptExtractor()
    
    async def process_video_file(self, file_path: str, sample_rate: int = 1) -> VideoIngestionResult:
        """Process video file and extract concepts"""
        session_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        try:
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            concepts = []
            motion_timeline = []
            all_detections = []
            spectral_data = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on sample_rate
                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps if fps > 0 else frame_idx
                    
                    # Analyze frame
                    spectral_meta = self.spectral_analyzer.analyze_frame(frame)
                    spectral_data.append(spectral_meta)
                    
                    # Detect objects
                    detections = self.object_detector.detect_objects(frame)
                    all_detections.extend(detections)
                    
                    # Extract concepts
                    frame_concepts = self.concept_extractor.extract_concepts(detections, spectral_meta)
                    
                    # Create concept diffs
                    for concept_id in frame_concepts:
                        visual_state = self.emotion_detector.analyze_visual_state(spectral_meta)
                        
                        concept_diff = VideoConceptDiff(
                            concept_id=concept_id,
                            confidence=0.7,
                            visual_source=f"frame_{frame_idx}",
                            spectral_signature=spectral_meta,
                            temporal_position=(timestamp, timestamp + 1/fps if fps > 0 else 1),
                            spatial_region=None,  # Could be refined with object localization
                            motion_context=visual_state
                        )
                        concepts.append(concept_diff)
                    
                    # Track motion timeline
                    motion_timeline.append({
                        'timestamp': timestamp,
                        'motion_intensity': spectral_meta.motion_intensity,
                        'brightness': spectral_meta.brightness_mean,
                        'contrast': spectral_meta.contrast_level
                    })
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate overall spectral summary
            if spectral_data:
                spectral_summary = self._calculate_spectral_summary(spectral_data)
            else:
                spectral_summary = VisualSpectralMetadata([], 5000, 0.5, 0.5, 0, 0, 0, {})
            
            # Determine phase signature
            phase_signature = self._determine_phase_signature(spectral_summary, motion_timeline)
            
            # Generate holographic hints
            holographic_hints = self._generate_holographic_hints(spectral_summary, motion_timeline)
            
            return VideoIngestionResult(
                session_id=session_id,
                concepts=concepts,
                duration=duration,
                frame_count=frame_count,
                spectral_summary=spectral_summary,
                motion_timeline=motion_timeline,
                detected_objects=all_detections,
                phase_signature=phase_signature,
                holographic_hints=holographic_hints
            )
            
        except Exception as e:
            logging.error(f"Video processing failed: {e}")
            return VideoIngestionResult(
                session_id=session_id,
                concepts=[],
                duration=0,
                frame_count=0,
                spectral_summary=VisualSpectralMetadata([], 5000, 0.5, 0.5, 0, 0, 0, {}),
                motion_timeline=[],
                detected_objects=[],
                phase_signature="error",
                holographic_hints={}
            )
    
    def _calculate_spectral_summary(self, spectral_data: List[VisualSpectralMetadata]) -> VisualSpectralMetadata:
        """Calculate summary statistics from all frames"""
        if not spectral_data:
            return VisualSpectralMetadata([], 5000, 0.5, 0.5, 0, 0, 0, {})
        
        # Aggregate dominant colors
        all_colors = []
        for data in spectral_data:
            all_colors.extend(data.dominant_colors)
        
        # Calculate means
        mean_brightness = np.mean([d.brightness_mean for d in spectral_data])
        mean_contrast = np.mean([d.contrast_level for d in spectral_data])
        mean_motion = np.mean([d.motion_intensity for d in spectral_data])
        mean_edge_density = np.mean([d.edge_density for d in spectral_data])
        mean_complexity = np.mean([d.spatial_complexity for d in spectral_data])
        mean_temperature = np.mean([d.color_temperature for d in spectral_data])
        
        # Aggregate chromatic signatures
        chromatic_summary = {}
        if spectral_data[0].chromatic_signature:
            for key in spectral_data[0].chromatic_signature:
                chromatic_summary[key] = np.mean([d.chromatic_signature.get(key, 0) for d in spectral_data])
        
        return VisualSpectralMetadata(
            dominant_colors=all_colors[:10],  # Keep top 10 colors
            color_temperature=mean_temperature,
            brightness_mean=mean_brightness,
            contrast_level=mean_contrast,
            motion_intensity=mean_motion,
            edge_density=mean_edge_density,
            spatial_complexity=mean_complexity,
            chromatic_signature=chromatic_summary
        )
    
    def _determine_phase_signature(self, spectral: VisualSpectralMetadata, 
                                 motion_timeline: List[Dict[str, float]]) -> str:
        """Determine phase signature from visual characteristics"""
        if not motion_timeline:
            return 'neutral'
        
        avg_motion = np.mean([m['motion_intensity'] for m in motion_timeline])
        motion_variance = np.var([m['motion_intensity'] for m in motion_timeline])
        
        if avg_motion > 0.7:
            return 'entropy'  # High motion = chaotic
        elif avg_motion < 0.2 and motion_variance < 0.1:
            return 'coherence'  # Low, stable motion = coherent
        elif motion_variance > 0.3:
            return 'drift'  # Variable motion = drifting
        else:
            return 'resonance'  # Moderate, rhythmic motion
    
    def _generate_holographic_hints(self, spectral: VisualSpectralMetadata,
                                  motion_timeline: List[Dict[str, float]]) -> Dict[str, Any]:
        """Generate hints for holographic visualization"""
        if not spectral.dominant_colors:
            primary_color = (128, 128, 128)  # Gray default
        else:
            primary_color = spectral.dominant_colors[0]
        
        # Convert RGB to wavelength approximation
        r, g, b = primary_color
        if r > g and r > b:
            wavelength = 650  # Red
        elif g > r and g > b:
            wavelength = 520  # Green
        elif b > r and b > g:
            wavelength = 470  # Blue
        else:
            wavelength = 550  # Default
        
        return {
            'wavelength': wavelength,
            'intensity': spectral.brightness_mean,
            'pulse_rate': spectral.motion_intensity * 5,
            'glow_radius': 20 + spectral.spatial_complexity * 40,
            'color_hint': f"rgb({r},{g},{b})",
            'motion_signature': {
                'average_motion': np.mean([m['motion_intensity'] for m in motion_timeline]) if motion_timeline else 0,
                'motion_variance': np.var([m['motion_intensity'] for m in motion_timeline]) if motion_timeline else 0,
                'brightness_trend': 'stable'  # Could be calculated from timeline
            },
            'visual_complexity': spectral.spatial_complexity,
            'color_temperature': spectral.color_temperature
        }

# Main API function
async def ingest_video(file_path: str, sample_rate: int = 1) -> Dict[str, Any]:
    """
    Main entry point for video ingestion
    
    Args:
        file_path: Path to video file
        sample_rate: Process every nth frame (1 = every frame)
    
    Returns:
        Dictionary containing concept diffs and metadata
    """
    processor = VideoProcessor()
    result = await processor.process_video_file(file_path, sample_rate)
    
    # Convert to dictionary for API response
    return {
        'session_id': result.session_id,
        'concepts': [asdict(concept) for concept in result.concepts],
        'duration': result.duration,
        'frame_count': result.frame_count,
        'spectral_summary': asdict(result.spectral_summary),
        'motion_timeline': result.motion_timeline,
        'detected_objects': result.detected_objects,
        'phase_signature': result.phase_signature,
        'holographic_hints': result.holographic_hints,
        'timestamp': datetime.now().isoformat()
    }

# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python ingest_video.py <video_file_path> [sample_rate]")
            return
        
        file_path = sys.argv[1]
        sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        
        result = await ingest_video(file_path, sample_rate)
        
        print(json.dumps(result, indent=2, default=str))