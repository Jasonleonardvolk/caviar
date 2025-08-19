"""
Image Ingest Handler
Processes images using CLIP for embeddings and concept extraction
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from PIL import Image
import numpy as np
import torch

from hott_integration.ingest_handlers.base_handler import BaseIngestHandler
from hott_integration.psi_morphon import (
    PsiMorphon, PsiStrand, HolographicMemory,
    ModalityType, StrandType
)

logger = logging.getLogger(__name__)

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    logger.warning("CLIP not available - using mock embeddings")
    CLIP_AVAILABLE = False
    clip = None

class ImageIngestHandler(BaseIngestHandler):
    """
    Handler for ingesting image files
    Uses CLIP for visual embeddings and concept extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        self.modality_type = ModalityType.IMAGE
        
        # CLIP settings
        self.clip_model_name = config.get('clip_model', 'ViT-B/32') if config else 'ViT-B/32'
        self.device = config.get('device', 'cpu') if config else 'cpu'
        self.batch_size = config.get('batch_size', 1) if config else 1
        
        # Initialize CLIP if available
        self.clip_model = None
        self.clip_preprocess = None
        if CLIP_AVAILABLE:
            self._init_clip()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name, device=self.device)
            logger.info(f"✅ CLIP model {self.clip_model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.clip_model = None
    
    async def extract_morphons(self, file_path: Path, 
                             metadata: Optional[Dict[str, Any]]) -> List[PsiMorphon]:
        """Extract morphons from image"""
        morphons = []
        
        # Load image
        try:
            image = Image.open(file_path).convert('RGB')
            logger.info(f"Loaded image: {image.size} from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise
        
        # Create main image morphon
        image_morphon = PsiMorphon(
            modality=ModalityType.IMAGE,
            content=str(file_path),  # Store path
            metadata={
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                **self.extract_metadata(file_path)
            },
            salience=1.0
        )
        morphons.append(image_morphon)
        
        # Extract visual concepts using CLIP
        if self.clip_model:
            concepts = await self._extract_visual_concepts(image)
            
            # Create text morphons for detected concepts
            for concept, confidence in concepts:
                concept_morphon = PsiMorphon(
                    modality=ModalityType.TEXT,
                    content=concept,
                    metadata={
                        "source": "clip_detection",
                        "confidence": confidence,
                        "from_image": str(file_path)
                    },
                    salience=confidence
                )
                morphons.append(concept_morphon)
        
        # Extract EXIF data if available
        exif_data = self._extract_exif(image)
        if exif_data:
            exif_morphon = PsiMorphon(
                modality=ModalityType.TEXT,
                content=f"EXIF metadata: {exif_data}",
                metadata={"exif": exif_data},
                salience=0.3
            )
            morphons.append(exif_morphon)
        
        return morphons
    
    async def _extract_visual_concepts(self, image: Image) -> List[Tuple[str, float]]:
        """Extract visual concepts using CLIP"""
        if not self.clip_model:
            return []
        
        try:
            # Prepare image
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Common concept prompts
            concepts = [
                "a photo of a person",
                "a photo of an animal",
                "a photo of a landscape",
                "a photo of a building",
                "a photo of food",
                "a photo of a vehicle",
                "a photo of nature",
                "a photo of technology",
                "a photo of art",
                "a photo of text or writing",
                "an indoor scene",
                "an outdoor scene",
                "a close-up photo",
                "a wide-angle photo",
                "a black and white photo",
                "a colorful photo"
            ]
            
            # Encode concepts
            with torch.no_grad():
                text_tokens = clip.tokenize(concepts).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = self.clip_model.encode_image(image_tensor)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarities[0].topk(5)
            
            # Return top concepts
            detected = []
            for value, idx in zip(values, indices):
                if value.item() > 0.1:  # Confidence threshold
                    detected.append((concepts[idx], value.item()))
            
            return detected
            
        except Exception as e:
            logger.error(f"CLIP concept extraction failed: {e}")
            return []
    
    async def _generate_embedding(self, morphon: PsiMorphon) -> Optional[np.ndarray]:
        """Generate CLIP embedding for morphon"""
        if morphon.modality == ModalityType.IMAGE and self.clip_model:
            try:
                # Load and preprocess image
                image = Image.open(morphon.content).convert('RGB')
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # Get CLIP embedding
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = image_features.cpu().numpy().squeeze()
                return embedding.astype(np.float32)
                
            except Exception as e:
                logger.error(f"Failed to generate CLIP embedding: {e}")
        
        # Fall back to base implementation
        return await super()._generate_embedding(morphon)
    
    async def create_strands(self, memory: HolographicMemory) -> List[PsiStrand]:
        """Create strands between image and detected concepts"""
        strands = []
        
        # Find the main image morphon
        image_morphons = memory.get_morphon_by_modality(ModalityType.IMAGE)
        if not image_morphons:
            return strands
        
        main_image = image_morphons[0]
        
        # Connect to all detected concept morphons
        for morphon in memory.morphons:
            if (morphon.modality == ModalityType.TEXT and 
                morphon.metadata.get('source') == 'clip_detection'):
                
                strand = PsiStrand(
                    source_morphon_id=main_image.id,
                    target_morphon_id=morphon.id,
                    strand_type=StrandType.VISUAL_INSTANCE,
                    strength=morphon.metadata.get('confidence', 0.5),
                    evidence=f"CLIP detected '{morphon.content}' in image",
                    confidence=morphon.metadata.get('confidence', 0.5)
                )
                strands.append(strand)
        
        return strands
    
    def _extract_exif(self, image: Image) -> Optional[Dict[str, Any]]:
        """Extract EXIF data from image"""
        try:
            exif = image.getexif()
            if exif:
                # Convert to readable format
                exif_data = {}
                for tag, value in exif.items():
                    if tag in Image.ExifTags.TAGS:
                        tag_name = Image.ExifTags.TAGS[tag]
                        exif_data[tag_name] = str(value)
                return exif_data
        except Exception as e:
            logger.debug(f"No EXIF data or extraction failed: {e}")
        
        return None
