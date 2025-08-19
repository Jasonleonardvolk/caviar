# losses.py - Phase correction loss functions with phasor arithmetic
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def phasor_loss(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
    """
    Phasor-based phase loss that handles wrapping elegantly.
    Loss = 1 - cos(Δφ) ∈ [0, 2]
    
    Args:
        pred_phase: Predicted phase in radians
        tgt_phase: Target phase in radians
    
    Returns:
        Scalar loss value
    """
    dphi = pred_phase - tgt_phase
    return (1.0 - torch.cos(dphi)).mean()


def complex_mag_loss(pred_complex: torch.Tensor, tgt_complex: torch.Tensor) -> torch.Tensor:
    """
    Magnitude consistency loss for complex fields.
    Ensures amplitude is preserved during phase correction.
    
    Args:
        pred_complex: Predicted complex field (real, imag)
        tgt_complex: Target complex field (real, imag)
    
    Returns:
        L1 loss between magnitudes
    """
    pred_mag = torch.abs(pred_complex)
    tgt_mag = torch.abs(tgt_complex)
    return F.l1_loss(pred_mag, tgt_mag)


def tv_loss(delta_phase: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss to suppress phase sparkle/noise.
    Encourages smooth phase corrections.
    
    Args:
        delta_phase: Phase correction map [B, C, H, W]
    
    Returns:
        TV regularization loss
    """
    # Compute gradients
    dx = delta_phase[:, :, 1:, :] - delta_phase[:, :, :-1, :]
    dy = delta_phase[:, :, :, 1:] - delta_phase[:, :, :, :-1]
    
    # L1 norm of gradients
    return (dx.abs().mean() + dy.abs().mean())


def phase_correction_loss(
    pred_phase: torch.Tensor,
    tgt_phase: torch.Tensor,
    pred_complex: torch.Tensor,
    tgt_complex: torch.Tensor,
    delta_phase: Optional[torch.Tensor] = None,
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.02)
) -> torch.Tensor:
    """
    Combined loss for phase correction network training.
    
    Args:
        pred_phase: Predicted corrected phase
        tgt_phase: Target clean phase
        pred_complex: Predicted complex field (for magnitude check)
        tgt_complex: Target complex field
        delta_phase: Optional phase correction map (for TV regularization)
        weights: (phase_weight, mag_weight, tv_weight)
    
    Returns:
        Combined scalar loss
    """
    lp = phasor_loss(pred_phase, tgt_phase)
    lm = complex_mag_loss(pred_complex, tgt_complex)
    
    total = weights[0] * lp + weights[1] * lm
    
    if delta_phase is not None:
        lt = tv_loss(delta_phase)
        total += weights[2] * lt
    
    return total


def complex_field_loss(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    weights: Tuple[float, float] = (1.0, 2.0)
) -> torch.Tensor:
    """
    Direct complex field loss that preserves both magnitude and phase.
    Avoids phase wrapping issues by working in complex domain.
    
    Args:
        pred_field: Predicted complex field [B, 2, H, W] where dim=1 is (real, imag)
        tgt_field: Target complex field
        weights: (magnitude_weight, complex_weight)
    
    Returns:
        Combined loss
    """
    # Magnitude loss
    pred_complex = torch.complex(pred_field[:, 0], pred_field[:, 1])
    tgt_complex = torch.complex(tgt_field[:, 0], tgt_field[:, 1])
    mag_loss = F.l1_loss(pred_complex.abs(), tgt_complex.abs())
    
    # Complex domain loss (preserves phase relationships)
    complex_loss = F.l1_loss(pred_field[:, 0], tgt_field[:, 0]) + \
                   F.l1_loss(pred_field[:, 1], tgt_field[:, 1])
    
    return weights[0] * mag_loss + weights[1] * complex_loss


def perceptual_phase_loss(
    pred_phase: torch.Tensor,
    tgt_phase: torch.Tensor,
    depth_map: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perceptually-weighted phase loss that emphasizes important regions.
    
    Args:
        pred_phase: Predicted phase
        tgt_phase: Target phase
        depth_map: Optional depth map for importance weighting
    
    Returns:
        Weighted phase loss
    """
    # Base phasor loss
    dphi = pred_phase - tgt_phase
    loss_map = 1.0 - torch.cos(dphi)
    
    if depth_map is not None:
        # Weight by inverse depth (closer objects more important)
        importance = 1.0 / (depth_map + 0.1)
        importance = importance / importance.mean()  # Normalize
        loss_map = loss_map * importance
    
    return loss_map.mean()


class PhaseCorrrectionCriterion(torch.nn.Module):
    """
    Complete criterion for training phase correction networks.
    """
    
    def __init__(
        self,
        phase_weight: float = 1.0,
        mag_weight: float = 0.5,
        tv_weight: float = 0.02,
        max_correction: float = 0.2  # Maximum phase correction in radians
    ):
        super().__init__()
        self.phase_weight = phase_weight
        self.mag_weight = mag_weight
        self.tv_weight = tv_weight
        self.max_correction = max_correction
    
    def forward(
        self,
        pred_correction: torch.Tensor,
        artifacted_phase: torch.Tensor,
        clean_phase: torch.Tensor,
        artifacted_complex: torch.Tensor,
        clean_complex: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for phase correction network.
        
        Args:
            pred_correction: Predicted phase correction Δφ
            artifacted_phase: Input artifacted phase
            clean_phase: Target clean phase
            artifacted_complex: Input complex field
            clean_complex: Target complex field
        
        Returns:
            (total_loss, loss_components_dict)
        """
        # Clamp correction to prevent over-correction
        pred_correction = torch.clamp(pred_correction, -self.max_correction, self.max_correction)
        
        # Apply correction
        corrected_phase = artifacted_phase + pred_correction
        
        # Apply correction to complex field
        correction_complex = torch.exp(1j * pred_correction)
        corrected_complex = artifacted_complex * correction_complex
        
        # Compute losses
        lp = phasor_loss(corrected_phase, clean_phase)
        lm = complex_mag_loss(corrected_complex, clean_complex)
        lt = tv_loss(pred_correction)
        
        # Total loss
        total = self.phase_weight * lp + self.mag_weight * lm + self.tv_weight * lt
        
        # Return components for logging
        components = {
            'phase_loss': lp.item(),
            'mag_loss': lm.item(),
            'tv_loss': lt.item(),
            'total_loss': total.item(),
            'mean_correction': pred_correction.abs().mean().item()
        }
        
        return total, components


# Helper functions for training

def generate_synthetic_artifacts(
    clean_field: torch.Tensor,
    artifact_type: str = 'discontinuity'
) -> torch.Tensor:
    """
    Generate synthetic artifacts for training data.
    
    Args:
        clean_field: Clean complex field
        artifact_type: Type of artifact to introduce
    
    Returns:
        Artifacted field
    """
    if artifact_type == 'discontinuity':
        # Add phase jumps at random locations
        mask = torch.rand_like(clean_field[:, 0]) > 0.95
        phase_jump = torch.randn_like(clean_field[:, 0]) * 0.5
        phase_jump = phase_jump * mask.float()
        
        # Apply to complex field
        artifacted = clean_field.clone()
        phase_shift = torch.exp(1j * phase_jump)
        artifacted_complex = torch.complex(clean_field[:, 0], clean_field[:, 1]) * phase_shift
        artifacted[:, 0] = artifacted_complex.real
        artifacted[:, 1] = artifacted_complex.imag
        
    elif artifact_type == 'noise':
        # Add phase noise
        noise = torch.randn_like(clean_field[:, 0]) * 0.1
        phase_shift = torch.exp(1j * noise)
        artifacted_complex = torch.complex(clean_field[:, 0], clean_field[:, 1]) * phase_shift
        artifacted = torch.stack([artifacted_complex.real, artifacted_complex.imag], dim=1)
        
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    return artifacted


def prepare_training_batch(
    clean_fields: torch.Tensor,
    artifact_types: list = ['discontinuity', 'noise']
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a training batch with synthetic artifacts.
    
    Args:
        clean_fields: Batch of clean complex fields [B, 2, H, W]
        artifact_types: List of artifact types to apply
    
    Returns:
        (artifacted_fields, clean_fields)
    """
    batch_size = clean_fields.shape[0]
    artifacted = []
    
    for i in range(batch_size):
        # Random artifact type
        artifact_type = artifact_types[i % len(artifact_types)]
        artifacted_field = generate_synthetic_artifacts(
            clean_fields[i:i+1],
            artifact_type
        )
        artifacted.append(artifacted_field)
    
    artifacted_batch = torch.cat(artifacted, dim=0)
    
    return artifacted_batch, clean_fields


if __name__ == "__main__":
    # Test the loss functions
    B, H, W = 2, 256, 256
    
    # Create synthetic data
    pred_phase = torch.randn(B, 1, H, W)
    tgt_phase = torch.randn(B, 1, H, W)
    pred_complex = torch.randn(B, 2, H, W)  # (real, imag)
    tgt_complex = torch.randn(B, 2, H, W)
    
    # Test individual losses
    print(f"Phasor loss: {phasor_loss(pred_phase, tgt_phase):.4f}")
    print(f"Complex mag loss: {complex_mag_loss(pred_complex, tgt_complex):.4f}")
    print(f"TV loss: {tv_loss(pred_phase):.4f}")
    
    # Test combined loss
    total_loss = phase_correction_loss(
        pred_phase, tgt_phase,
        pred_complex, tgt_complex,
        delta_phase=pred_phase
    )
    print(f"Total loss: {total_loss:.4f}")
    
    # Test criterion
    criterion = PhaseCorrrectionCriterion()
    correction = torch.randn(B, 1, H, W) * 0.1
    loss, components = criterion(
        correction,
        pred_phase.squeeze(1),
        tgt_phase.squeeze(1),
        torch.complex(pred_complex[:, 0], pred_complex[:, 1]),
        torch.complex(tgt_complex[:, 0], tgt_complex[:, 1])
    )
    print(f"Criterion loss: {loss:.4f}")
    print(f"Components: {components}")
