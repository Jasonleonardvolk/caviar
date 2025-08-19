import numpy as np
from alan_backend.koopman import koopman_modes

def spectral_embed(features: np.ndarray, k: int) -> np.ndarray:
    """Project features into Koopman spectral space (top k modes)."""
    data = features.T  # (d × m)
    _, eigvecs, _ = koopman_modes(data)
    modes = np.real(eigvecs[:, :k])
    if modes.shape[0] == features.shape[1]:
        emb = features @ modes
    elif modes.shape[0] == features.shape[0]:
        emb = modes
    else:
        common = min(features.shape[1], modes.shape[0])
        emb = features[:, :common] @ modes[:common, :]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb
