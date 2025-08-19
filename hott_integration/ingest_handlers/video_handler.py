"""
Video Ingest Handler
Processes video files by extracting frames and audio
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import tempfile
import shutil
import subprocess
import json

from hott_integration.ingest_handlers.base_handler import BaseIngestHandler
from hott_integration.ingest_handlers.image_handler import ImageIngestHandler
from hott_integration.ingest_handlers.audio_handler import AudioIngestHandler
from hott_integration.psi_morphon import (
    PsiMorphon, PsiStrand, HolographicMemory,
    ModalityType, StrandType
)

logger = logging.getLogger(__name__)

# Check for ffmpeg
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

class VideoIngestHandler(BaseIngestHandler):
    """
    Handler for ingesting video files
    Extracts key frames and audio, then processes them separately
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        self.modality_type = ModalityType.VIDEO
        
        # Video processing settings
        self.keyframe_interval = config.get('keyframe_interval', 5.0) if config else 5.0  # seconds
        self.max_frames = config.get('max_frames', 20) if config else 20
        self.frame_size = config.get('frame_size', (640, 480)) if config else (640, 480)
        
        # Initialize sub-handlers
        self.image_handler = ImageIngestHandler(config)
        self.audio_handler = AudioIngestHandler(config)
        
        if not FFMPEG_AVAILABLE:
            logger.warning("FFmpeg not available - video processing will be limited")
    
    async def extract_morphons(self, file_path: Path,
                             metadata: Optional[Dict[str, Any]]) -> List[PsiMorphon]:
        """Extract morphons from video file"""
        morphons = []
        
        # Create main video morphon
        video_info = await self._get_video_info(file_path)
        video_morphon = PsiMorphon(
            modality=ModalityType.VIDEO,
            content=str(file_path),
            metadata={
                **self.extract_metadata(file_path),
                **video_info
            },
            salience=1.0
        )
        morphons.append(video_morphon)
        
        if not FFMPEG_AVAILABLE:
            logger.warning("Skipping frame/audio extraction - FFmpeg not available")
            return morphons
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract keyframes
            frame_paths = await self._extract_keyframes(file_path, temp_path)
            logger.info(f"Extracted {len(frame_paths)} keyframes")
            
            # Process each frame
            for i, frame_path in enumerate(frame_paths):
                # Get timestamp for this frame
                timestamp = i * self.keyframe_interval
                
                # Process frame with image handler
                frame_memory = await self.image_handler.ingest(
                    frame_path,
                    video_morphon.tenant_scope,
                    video_morphon.tenant_id,
                    metadata={"frame_index": i, "timestamp": timestamp}
                )
                
                # Add frame morphons with temporal info
                for morphon in frame_memory.morphons:
                    morphon.temporal_index = timestamp
                    morphon.metadata["from_video"] = str(file_path)
                    morphon.metadata["frame_index"] = i
                    morphons.append(morphon)
            
            # Extract audio track
            audio_path = await self._extract_audio(file_path, temp_path)
            if audio_path and audio_path.exists():
                logger.info("Processing extracted audio track")
                
                # Process audio with audio handler
                audio_memory = await self.audio_handler.ingest(
                    audio_path,
                    video_morphon.tenant_scope,
                    video_morphon.tenant_id,
                    metadata={"from_video": str(file_path)}
                )
                
                # Add audio morphons
                for morphon in audio_memory.morphons:
                    morphon.metadata["from_video"] = str(file_path)
                    morphons.append(morphon)
        
        return morphons
    
    async def _get_video_info(self, file_path: Path) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        if not FFMPEG_AVAILABLE:
            return {"format": "video", "duration": 0}
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract relevant info
                info = {
                    "duration": float(data['format'].get('duration', 0)),
                    "bit_rate": int(data['format'].get('bit_rate', 0)),
                    "format_name": data['format'].get('format_name', 'unknown')
                }
                
                # Get video stream info
                for stream in data.get('streams', []):
                    if stream['codec_type'] == 'video':
                        info.update({
                            "width": stream.get('width', 0),
                            "height": stream.get('height', 0),
                            "fps": eval(stream.get('r_frame_rate', '0/1')),
                            "codec": stream.get('codec_name', 'unknown')
                        })
                        break
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
        
        return {"format": "video", "duration": 0}
    
    async def _extract_keyframes(self, video_path: Path, output_dir: Path) -> List[Path]:
        """Extract keyframes from video using ffmpeg"""
        frame_paths = []
        
        try:
            # Calculate frame extraction rate
            video_info = await self._get_video_info(video_path)
            duration = video_info.get('duration', 0)
            
            if duration <= 0:
                logger.warning("Could not determine video duration")
                return frame_paths
            
            # Determine number of frames to extract
            num_frames = min(
                int(duration / self.keyframe_interval),
                self.max_frames
            )
            
            if num_frames <= 0:
                num_frames = 1
            
            # Extract frames
            for i in range(num_frames):
                timestamp = i * self.keyframe_interval
                output_path = output_dir / f"frame_{i:04d}.jpg"
                
                cmd = [
                    'ffmpeg',
                    '-ss', str(timestamp),
                    '-i', str(video_path),
                    '-vframes', '1',
                    '-s', f"{self.frame_size[0]}x{self.frame_size[1]}",
                    '-f', 'image2',
                    str(output_path),
                    '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0 and output_path.exists():
                    frame_paths.append(output_path)
                else:
                    logger.warning(f"Failed to extract frame at {timestamp}s")
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frame_paths
    
    async def _extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """Extract audio track from video"""
        audio_path = output_dir / "audio.mp3"
        
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'mp3',
                '-ab', '128k',
                '-ar', '16000',  # 16kHz for speech processing
                str(audio_path),
                '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and audio_path.exists():
                return audio_path
            else:
                logger.warning("No audio track extracted or extraction failed")
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
        
        return None
    
    async def create_strands(self, memory: HolographicMemory) -> List[PsiStrand]:
        """Create strands between video components"""
        strands = []
        
        # Find main video morphon
        video_morphons = memory.get_morphon_by_modality(ModalityType.VIDEO)
        if not video_morphons:
            return strands
        
        main_video = video_morphons[0]
        
        # Connect video to all extracted components
        for morphon in memory.morphons:
            if morphon.id == main_video.id:
                continue
                
            # Connect to frames
            if morphon.metadata.get('from_video') == main_video.content:
                strand = PsiStrand(
                    source_morphon_id=main_video.id,
                    target_morphon_id=morphon.id,
                    strand_type=StrandType.TEMPORAL,
                    strength=0.9,
                    evidence=f"Extracted from video at {morphon.temporal_index or 0:.1f}s"
                )
                strands.append(strand)
        
        # Create synesthetic connections between audio and visual
        audio_morphons = [m for m in memory.morphons 
                         if m.modality == ModalityType.AUDIO]
        image_morphons = [m for m in memory.morphons 
                         if m.modality == ModalityType.IMAGE]
        
        for audio in audio_morphons:
            for image in image_morphons:
                # If they're temporally close, create synesthetic link
                if (image.temporal_index is not None and
                    audio.temporal_index is not None):
                    
                    time_diff = abs(image.temporal_index - audio.temporal_index)
                    if time_diff < 2.0:  # Within 2 seconds
                        strand = PsiStrand(
                            source_morphon_id=audio.id,
                            target_morphon_id=image.id,
                            strand_type=StrandType.SYNESTHETIC,
                            strength=max(0.3, 1.0 - time_diff / 2.0),
                            bidirectional=True,
                            evidence=f"Temporal proximity: {time_diff:.1f}s",
                            temporal_offset=time_diff
                        )
                        strands.append(strand)
        
        return strands
