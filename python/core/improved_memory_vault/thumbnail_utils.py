"""
Thumbnail and Fingerprint Utilities for Multimodal Memory Vault
================================================================

Provides functions for generating thumbnails, fingerprints, and compact
representations of images, audio, and other multimodal data for efficient
storage and retrieval in the memory vault.

Features:
- Image thumbnail generation with perceptual hashing
- Audio fingerprinting using spectral features
- Video keyframe extraction
- Document preview generation
- Efficient compression and encoding
"""

import logging
import hashlib
import base64
import json
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
from datetime import datetime

# Optional imports for image processing
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Optional imports for audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

# Optional imports for video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

logger = logging.getLogger("ThumbnailUtils")

# Configuration
DEFAULT_THUMBNAIL_SIZE = (128, 128)
DEFAULT_AUDIO_DURATION = 30  # seconds
DEFAULT_VIDEO_KEYFRAMES = 5
PERCEPTUAL_HASH_SIZE = 8  # 8x8 = 64 bit hash

class ThumbnailGenerator:
    """
    Generates thumbnails and compact representations for multimodal data.
    """
    
    def __init__(self, 
                 thumbnail_size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
                 audio_duration: float = DEFAULT_AUDIO_DURATION,
                 video_keyframes: int = DEFAULT_VIDEO_KEYFRAMES):
        """
        Initialize thumbnail generator with configuration.
        
        Args:
            thumbnail_size: Target size for image thumbnails
            audio_duration: Maximum audio duration to process (seconds)
            video_keyframes: Number of keyframes to extract from videos
        """
        self.thumbnail_size = thumbnail_size
        self.audio_duration = audio_duration
        self.video_keyframes = video_keyframes
        
        # Check available libraries
        self.capabilities = {
            'image': PIL_AVAILABLE,
            'audio': LIBROSA_AVAILABLE,
            'video': CV2_AVAILABLE
        }
        
        logger.info(f"ThumbnailGenerator initialized. Capabilities: {self.capabilities}")
    
    def generate_image_thumbnail(self, 
                                image_path: Optional[str] = None,
                                image_data: Optional[bytes] = None,
                                image_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate thumbnail and perceptual hash for an image.
        
        Args:
            image_path: Path to image file
            image_data: Raw image bytes
            image_array: Numpy array of image data
            
        Returns:
            Dictionary with thumbnail data and metadata
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - cannot generate image thumbnail")
            return self._create_placeholder_thumbnail('image')
        
        try:
            # Load image
            if image_path:
                img = Image.open(image_path)
            elif image_data:
                import io
                img = Image.open(io.BytesIO(image_data))
            elif image_array is not None:
                img = Image.fromarray(image_array)
            else:
                raise ValueError("No image data provided")
            
            # Store original dimensions
            original_size = img.size
            original_mode = img.mode
            
            # Generate thumbnail
            thumbnail = img.copy()
            thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB for consistent processing
            if thumbnail.mode != 'RGB':
                thumbnail = thumbnail.convert('RGB')
            
            # Generate perceptual hash
            phash = self._calculate_perceptual_hash(thumbnail)
            
            # Convert thumbnail to base64
            import io
            buffer = io.BytesIO()
            thumbnail.save(buffer, format='JPEG', quality=85)
            thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Calculate basic statistics
            thumb_array = np.array(thumbnail)
            stats = {
                'mean_rgb': thumb_array.mean(axis=(0, 1)).tolist(),
                'std_rgb': thumb_array.std(axis=(0, 1)).tolist(),
                'dominant_colors': self._extract_dominant_colors(thumb_array, n=5)
            }
            
            return {
                'type': 'image',
                'thumbnail': thumbnail_b64,
                'thumbnail_size': thumbnail.size,
                'original_size': original_size,
                'original_mode': original_mode,
                'perceptual_hash': phash,
                'statistics': stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating image thumbnail: {e}")
            return self._create_placeholder_thumbnail('image', error=str(e))
    
    def generate_audio_fingerprint(self,
                                  audio_path: Optional[str] = None,
                                  audio_data: Optional[np.ndarray] = None,
                                  sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate audio fingerprint using spectral features.
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio samples
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with audio fingerprint and metadata
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available - cannot generate audio fingerprint")
            return self._create_placeholder_thumbnail('audio')
        
        try:
            # Load audio
            if audio_path:
                y, sr = librosa.load(audio_path, duration=self.audio_duration)
            elif audio_data is not None and sample_rate:
                y = audio_data[:int(sample_rate * self.audio_duration)]
                sr = sample_rate
            else:
                raise ValueError("No audio data provided")
            
            # Extract features
            features = {}
            
            # Mel-frequency cepstral coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
            features['mfcc_std'] = mfcc.std(axis=1).tolist()
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(spectral_centroid.mean())
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = float(zcr.mean())
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_frames'] = len(beats)
            
            # Generate compact spectogram thumbnail
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            # Downsample spectrogram for thumbnail
            thumbnail_spec = cv2.resize(D, (64, 64)) if CV2_AVAILABLE else D[:64, :64]
            
            # Create audio fingerprint hash
            fingerprint = self._hash_features(features)
            
            return {
                'type': 'audio',
                'fingerprint': fingerprint,
                'features': features,
                'spectrogram_thumbnail': base64.b64encode(thumbnail_spec.tobytes()).decode('utf-8'),
                'duration': len(y) / sr,
                'sample_rate': sr,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating audio fingerprint: {e}")
            return self._create_placeholder_thumbnail('audio', error=str(e))
    
    def generate_video_keyframes(self,
                                video_path: str) -> Dict[str, Any]:
        """
        Extract keyframes from video for thumbnail generation.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with keyframes and metadata
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - cannot extract video keyframes")
            return self._create_placeholder_thumbnail('video')
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate keyframe positions
            keyframe_indices = np.linspace(0, frame_count - 1, self.video_keyframes, dtype=int)
            
            keyframes = []
            for idx in keyframe_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame for thumbnail
                    thumbnail = cv2.resize(frame, self.thumbnail_size)
                    # Convert BGR to RGB
                    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', thumbnail)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    keyframes.append({
                        'frame_index': int(idx),
                        'timestamp': idx / fps if fps > 0 else 0,
                        'thumbnail': frame_b64
                    })
            
            cap.release()
            
            # Generate video fingerprint from keyframes
            if keyframes:
                fingerprint = self._generate_video_fingerprint(keyframes)
            else:
                fingerprint = None
            
            return {
                'type': 'video',
                'keyframes': keyframes,
                'fingerprint': fingerprint,
                'frame_count': frame_count,
                'fps': fps,
                'duration': duration,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting video keyframes: {e}")
            return self._create_placeholder_thumbnail('video', error=str(e))
    
    def generate_document_preview(self,
                                 document_path: Optional[str] = None,
                                 text_content: Optional[str] = None,
                                 max_preview_length: int = 500) -> Dict[str, Any]:
        """
        Generate preview and metadata for text documents.
        
        Args:
            document_path: Path to document file
            text_content: Direct text content
            max_preview_length: Maximum length of preview text
            
        Returns:
            Dictionary with document preview and metadata
        """
        try:
            # Load content
            if document_path:
                with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                doc_name = Path(document_path).name
            elif text_content:
                content = text_content
                doc_name = "untitled"
            else:
                raise ValueError("No document data provided")
            
            # Generate preview
            preview = content[:max_preview_length]
            if len(content) > max_preview_length:
                preview += "..."
            
            # Extract basic statistics
            lines = content.split('\n')
            words = content.split()
            
            stats = {
                'total_length': len(content),
                'line_count': len(lines),
                'word_count': len(words),
                'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1),
                'unique_words': len(set(words))
            }
            
            # Generate content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Extract keywords (simple frequency-based)
            word_freq = {}
            for word in words:
                word_lower = word.lower().strip('.,!?;:"')
                if len(word_lower) > 3:  # Skip short words
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'type': 'document',
                'preview': preview,
                'statistics': stats,
                'content_hash': content_hash,
                'keywords': [kw[0] for kw in keywords],
                'document_name': doc_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating document preview: {e}")
            return self._create_placeholder_thumbnail('document', error=str(e))
    
    def _calculate_perceptual_hash(self, image: Image.Image) -> str:
        """
        Calculate perceptual hash of an image for similarity detection.
        
        Args:
            image: PIL Image object
            
        Returns:
            Hex string of perceptual hash
        """
        # Resize and convert to grayscale
        img = image.resize((PERCEPTUAL_HASH_SIZE, PERCEPTUAL_HASH_SIZE), Image.Resampling.LANCZOS)
        img = img.convert('L')
        
        # Get pixel data
        pixels = np.array(img)
        
        # Calculate average
        avg = pixels.mean()
        
        # Calculate hash
        hash_bits = (pixels > avg).flatten()
        
        # Convert to hex
        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)
        
        return hex(hash_int)[2:].zfill(16)
    
    def _extract_dominant_colors(self, image_array: np.ndarray, n: int = 5) -> List[List[int]]:
        """
        Extract dominant colors from image using k-means clustering.
        
        Args:
            image_array: Numpy array of image
            n: Number of dominant colors to extract
            
        Returns:
            List of RGB color values
        """
        try:
            # Reshape to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers as dominant colors
            colors = kmeans.cluster_centers_.astype(int).tolist()
            return colors
        except:
            # Fallback to simple averaging
            return [image_array.mean(axis=(0, 1)).astype(int).tolist()]
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """
        Generate hash from feature dictionary.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Hash string
        """
        # Convert features to stable string representation
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(feature_str.encode('utf-8')).hexdigest()[:16]
    
    def _generate_video_fingerprint(self, keyframes: List[Dict[str, Any]]) -> str:
        """
        Generate fingerprint from video keyframes.
        
        Args:
            keyframes: List of keyframe data
            
        Returns:
            Video fingerprint hash
        """
        # Combine keyframe thumbnails for fingerprinting
        combined = ''.join(kf['thumbnail'][:100] for kf in keyframes)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _create_placeholder_thumbnail(self, media_type: str, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Create placeholder thumbnail when processing fails.
        
        Args:
            media_type: Type of media
            error: Error message if any
            
        Returns:
            Placeholder thumbnail data
        """
        return {
            'type': media_type,
            'placeholder': True,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }

# Singleton instance
_thumbnail_generator = None

def get_thumbnail_generator() -> ThumbnailGenerator:
    """Get or create singleton thumbnail generator instance."""
    global _thumbnail_generator
    if _thumbnail_generator is None:
        _thumbnail_generator = ThumbnailGenerator()
    return _thumbnail_generator

# Convenience functions
def generate_image_thumbnail(image_path: str) -> Dict[str, Any]:
    """Generate thumbnail for image file."""
    return get_thumbnail_generator().generate_image_thumbnail(image_path=image_path)

def generate_audio_fingerprint(audio_path: str) -> Dict[str, Any]:
    """Generate fingerprint for audio file."""
    return get_thumbnail_generator().generate_audio_fingerprint(audio_path=audio_path)

def generate_video_keyframes(video_path: str) -> Dict[str, Any]:
    """Extract keyframes from video file."""
    return get_thumbnail_generator().generate_video_keyframes(video_path=video_path)

def generate_document_preview(document_path: str) -> Dict[str, Any]:
    """Generate preview for document file."""
    return get_thumbnail_generator().generate_document_preview(document_path=document_path)

if __name__ == "__main__":
    # Test thumbnail generation
    import sys
    
    generator = ThumbnailGenerator()
    logger.info(f"Thumbnail generator capabilities: {generator.capabilities}")
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        file_ext = Path(test_file).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            result = generator.generate_image_thumbnail(image_path=test_file)
            logger.info(f"Image thumbnail generated: hash={result.get('perceptual_hash')}")
        elif file_ext in ['.mp3', '.wav', '.flac', '.m4a']:
            result = generator.generate_audio_fingerprint(audio_path=test_file)
            logger.info(f"Audio fingerprint generated: {result.get('fingerprint')}")
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            result = generator.generate_video_keyframes(test_file)
            logger.info(f"Video keyframes extracted: {len(result.get('keyframes', []))} frames")
        elif file_ext in ['.txt', '.md', '.py', '.js', '.html']:
            result = generator.generate_document_preview(document_path=test_file)
            logger.info(f"Document preview generated: {result.get('statistics')}")
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
    else:
        logger.info("Thumbnail utilities loaded successfully")
